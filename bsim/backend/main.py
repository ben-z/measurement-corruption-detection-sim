import asyncio
from copy import deepcopy
import traceback
from urllib.parse import parse_qsl
import discrete_kinematic_bicycle as dkb
import continuous_kinematic_bicycle as ckb
import json
import lookahead_lqr
import mergedeep
import numpy as np
import path_following_kmpc as pfkmpc
import re
import secrets
import sys
import websockets

INITIAL_WORLD_STATE = {
    't': 0,
    'DT': 0.01,
    'entities': {}
}

world_state = deepcopy(INITIAL_WORLD_STATE)

ENTITY_PATH_REGEX = re.compile(r'^/entities/(?P<entity_id>\w+)$')
CREATE_ENTITY_REGEX = re.compile(r'^create_entity: (?P<entity_type>\w+)(?: (?P<entity_id>\w+))?(?: (?P<entity_options>[\w=%,-]+))?$')
ENTITY_UPDATE_STATE_REGEX = re.compile(r'^update_state: (?P<new_state>.+)$')


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Exception):
            return repr(obj)
        return json.JSONEncoder.default(self, obj)

# Derived from: https://til.simonwillison.net/python/json-floating-point
def round_floats(o, decimals=6):
    if isinstance(o, float):
        return round(o, decimals)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(round_floats(x) for x in o)
    if isinstance(o, np.ndarray):
        return o.round(decimals)
    return o

def strip_internal_vars(o):
    if isinstance(o, dict):
        return {k: strip_internal_vars(v) for k, v in o.items() if not k.startswith('_')}
    if isinstance(o, (list, tuple)):
        return type(o)(strip_internal_vars(x) for x in o)
    return o

def get_handler(path: str):
    if path == '/world':
        return world_handler
    if ENTITY_PATH_REGEX.match(path):
        entity_id = ENTITY_PATH_REGEX.match(path).group('entity_id')
        return get_entity_handler(entity_id)
    
    raise Exception(f"ERROR: Unknown path: '{path}'")


def get_entity_handler(entity_id: str):
    entity = world_state['entities'].get(entity_id)
    if not entity:
        raise Exception(f"ERROR: Unknown entity: '{entity_id}'")
    
    return entity['_handler']


def world_handler(command: str):
    global world_state

    if command == 'terminate':
        sys.exit(0)
    if command == 'reset':
        world_state = deepcopy(INITIAL_WORLD_STATE)
        # world_handler('create_entity: ego ego1') # create an initial ego vehicle
    elif command == 'state':
        pass # noop, just return the current state
    elif command == 'tick':
        world_state['t'] += world_state['DT']
        for entity in world_state['entities'].values():
            entity['_handler']('_tick')
    elif CREATE_ENTITY_REGEX.match(command):
        entity_type = CREATE_ENTITY_REGEX.match(command).group('entity_type')
        entity_id = CREATE_ENTITY_REGEX.match(command).group('entity_id')
        entity_options = CREATE_ENTITY_REGEX.match(command).group('entity_options')
        print(entity_options)

        if not entity_id:
            entity_id = f"{entity_type}_{secrets.token_urlsafe(8)}"
        
        if entity_id in world_state['entities']:
            raise Exception(f"ERROR: entity with ID '{entity_id}' already exists")

        user_options = dict(parse_qsl(entity_options, separator=",", strict_parsing=True)) if entity_options else {}

        if entity_type == 'ego':
            default_options = {
                'controller': 'manual',
                'L': 2.9,
                # coordinates of the waypoints in meters. This will form a closed path
                'target_path': np.array([ [-10,3], [12,-5], [10,-5], [7, -8], [0,-10], [-10,-3] ]),
                'target_speed': 1, # m/s
                'sensor': 'state_with_corruption',
                'estimator': 'sensor',
            }

            additional_allowed_options = set(['controller_options'])
            unknown_option_keys = set(user_options.keys()) - set(default_options.keys()) - additional_allowed_options
            if unknown_option_keys:
                raise Exception(f"ERROR: unknown options: {unknown_option_keys}")

            options = {**default_options, **user_options}
            if isinstance(options['target_path'], str):
                options['target_path'] = np.array(json.loads(options['target_path']))

            if options['sensor'] == 'state':
                sensor_state = {
                    'sensor': 'state',
                    '_sensor_fn': lambda sstate, state: (state, sstate, {}),
                    '_sensor_state': None,
                    'sensor_debug_output': {},
                }
            elif options['sensor'] == 'state_with_corruption':
                sensor_state = {
                    'sensor': 'state',
                    '_sensor_fn': lambda sstate, state: (state*sstate['multiplicative_corruption']+sstate['additive_corruption'], sstate, {'sensor_state': sstate}),
                    '_sensor_state': {
                        'multiplicative_corruption': [1,1,1,1,1],
                        'additive_corruption': [0,0,0,0,0],
                    },
                    'sensor_debug_output': {},
                }
            else:
                raise Exception(f"ERROR: unknown sensor: '{options['sensor']}'")

            if options['estimator'] == 'sensor':
                estimator_state = {
                    'estimator': 'sensor',
                    '_estimator_fn': lambda est_state, measurement: (measurement, est_state, {}),
                    '_estimator_state': None,
                    'estimator_debug_output': {},
                }
            else:
                raise Exception(f"ERROR: unknown estimator: '{options['estimator']}'")

            if options['controller'] == 'manual':
                controller_state = {
                    'controller': 'manual',
                    '_controller_fn': lambda cstate, _estimate: (cstate, cstate, {}),
                    '_controller_state': dkb.get_noop_action(),
                    'controller_debug_output': {},
                }
            elif options['controller'] == 'path_following_kmpc':
                controller_state = {
                    'controller': 'path_following_kmpc',
                    '_controller_fn': pfkmpc.path_following_kmpc,
                    '_controller_state': pfkmpc.get_initial_state(target_path=options['target_path'], dt=world_state['DT'], L=options['L']),
                    'controller_debug_output': {},
                }
            elif options['controller'] == 'lookahead_lqr':
                tuning_options = json.loads(options['controller_options']) if 'controller_options' in options else {}

                controller_state = {
                    'controller': 'lookahead_lqr',
                    '_controller_fn': lookahead_lqr.lookahead_lqr,
                    '_controller_state': lookahead_lqr.get_initial_state(
                        target_path=options['target_path'],
                        target_speed=options['target_speed'],
                        dt=world_state['DT'],
                        L=options['L'],
                        **tuning_options,
                    ),
                    'controller_debug_output': {},
                }
            else:
                raise Exception(f"ERROR: unknown controller: {options['controller']}")

            world_state['entities'][entity_id] = {
                'type': 'ego',
                'state': dkb.get_initial_state(),
                'action': dkb.get_noop_action(),
                'L': options['L'],
                'target_path': options['target_path'],
                '_handler': make_ego_handler(entity_id),
                # '_model': ckb.make_continuous_kinematic_bicycle_model(options['L']),
                '_model': dkb.make_discrete_kinematic_bicycle_model(options['L'], world_state['DT']),
                **sensor_state,
                **estimator_state,
                **controller_state,
            }
        else:
            raise Exception(f"ERROR: Unknown entity type '{entity_type}'")
    else:
        raise Exception(f"ERROR: Unknown command: {command}")
    
    return world_state

def make_ego_handler(entity_id: str):
    def ego_handler(command: str):
        entity = world_state['entities'][entity_id]

        if command == 'reset':
            entity['state'] = dkb.get_initial_state()
            entity['action'] = dkb.get_noop_action()
        elif command.startswith('action: '):
            if entity['controller'] != 'manual':
                raise Exception(f"ERROR: entity '{entity_id}' is not in manual mode")
            entity['_controller_state'] = np.fromstring(command[len('action: '):], dtype=float, sep=' ')
        elif command == 'state':
            pass # noop, just return the current state
        elif command == '_tick':
            # TODO: add sensor, estimation, and controller code here
            state = entity['state']

            entity['measurement'], entity['_sensor_state'], entity['sensor_debug_output'] = \
                entity['_sensor_fn'](entity['_sensor_state'], state)
            entity['estimate'], entity['_estimator_state'], entity['estimator_debug_output'] = \
                entity['_estimator_fn'](entity['_estimator_state'], entity['measurement'])
            # model = control.sample_system(entity['_model'].linearize(estimate, entity['action']), world_state['DT'])
            model = entity['_model']

            # calculate control action
            entity['action'], entity['_controller_state'], entity['controller_debug_output'] = \
                entity['_controller_fn'](entity['_controller_state'], entity['estimate'])

            entity['state'] = model.dynamics(0, entity['state'], entity['action'])
        elif ENTITY_UPDATE_STATE_REGEX.match(command):
            new_state = json.loads(ENTITY_UPDATE_STATE_REGEX.match(command).group('new_state'))
            mergedeep.merge(entity, new_state, strategy=mergedeep.Strategy.TYPESAFE_REPLACE)
        else:
            raise Exception(f"ERROR: Unknown command: {command}")
        
        return entity

    return ego_handler


async def new_connection(websocket, path: str):
    print(f"New connection: {path}")

    handler = get_handler(path)
    if handler is None:
        print(f"ERROR: Unknown path: {path}")
        return

    while True:
        try:
            raw_request = await websocket.recv()
        except websockets.exceptions.ConnectionClosedOK:
            print(f"Connection closed: {path}")
            break

        try:
            request = json.loads(raw_request)
            request_id = request['id']
            command = request['command']
        except:
            print(f"ERROR: Cannot parse request: {raw_request}")
            await websocket.send(json.dumps({'error': 'Cannot parse request'}))
            break

        print(f"[{request_id}] < {path}: {request}")
        try:
            response = {'id': request_id, 'response': handler(command)}
        except Exception:
            print(f"[{request_id}] ERROR: exception thrown while handling command")
            traceback.print_exc()
            response = {'id': request_id, 'error': traceback.format_exc()}

        serialized_response = json.dumps(
            round_floats(strip_internal_vars(response)), cls=MyEncoder)

        await websocket.send(serialized_response)
        print(f"[{request_id}] > {path}: {serialized_response}")

start_server = websockets.serve(new_connection, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
