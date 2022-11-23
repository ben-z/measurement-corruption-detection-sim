import asyncio
import control
from copy import deepcopy
import traceback
import discrete_kinematic_bicycle as dkb
import continuous_kinematic_bicycle as ckb
import json
import lookahead_lqr
import mergedeep
from MyEstimator import MyEstimator
import numpy as np
import path_following_kmpc as pfkmpc
import re
import secrets
import sys
from static_slice_planner import StaticSlicePlanner
from subdivision_planner import SubdivisionPlanner
from lateral_profile_planner import LateralProfilePlanner
from urllib.parse import parse_qsl
from utils import JSONNumpyDecoder, ensure_options_are_known
import websockets
from typing import Dict, Any

INITIAL_WORLD_STATE = {
    't': 0,
    'DT': 0.01,
    'entities': {}
}

world_state = deepcopy(INITIAL_WORLD_STATE)

ENTITY_PATH_REGEX = re.compile(r'^/entities/(?P<entity_id>\w+)$')
CREATE_ENTITY_REGEX = re.compile(r'^create_entity: (?P<entity_type>\w+)(?: (?P<entity_id>\w+))?(?: (?P<entity_options>[\w=%,-\.]+))?$')
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

        user_options: Dict[str, Any] = dict(parse_qsl(entity_options, separator=",", strict_parsing=True)) if entity_options else {}

        if entity_type == 'ego':
            default_options = {
                'controller': 'manual',
                'L': 2.9,
                # coordinates of the waypoints in meters. This will form a closed path
                'global_ref_path': np.array([ [-10,3], [12,-5], [10,-5], [7, -8], [0,-10], [-10,-3] ]),
                'target_speed': 1., # m/s
                'sensor': 'model_output_with_corruption',
                'estimator': 'l1_optimizer',
                # 'estimator': 'first_n',
                # 'planner': 'static_slice',
                # 'planner': 'subdivision',
                'planner': 'lateral_profile',
                # Default options for these modules are defined in their respective handlers
                'plant_options': {},
                'controller_options': {},
                'planner_options': {},
            }

            # check for invalid options
            ensure_options_are_known(user_options, default_options, entity_id)

            # unpack user options that are stringified in transit
            if isinstance(user_options.get('global_ref_path'), str):
                user_options['global_ref_path'] = np.array(json.loads(user_options['global_ref_path']))
            if isinstance(user_options.get('target_speed'), str):
                user_options['target_speed'] = float(user_options['target_speed'])
            if isinstance(user_options.get('plant_options'), str):
                user_options['plant_options'] = json.loads(user_options['plant_options'], cls=JSONNumpyDecoder)
            if isinstance(user_options.get('controller_options'), str):
                user_options['controller_options'] = json.loads(user_options['controller_options'], cls=JSONNumpyDecoder)
            if isinstance(user_options.get('planner_options'), str):
                user_options['planner_options'] = json.loads(user_options['planner_options'], cls=JSONNumpyDecoder)

            # merge the user options with the default options
            options = mergedeep.merge({}, default_options, user_options, strategy=mergedeep.Strategy.TYPESAFE_REPLACE)
            
            plant_options = options['plant_options']

            # create the vehicle
            # plant_factory = ckb
            # plant_model = plant_factory.make_model(options['L'])
            plant_factory = dkb
            plant_model = plant_factory.make_model(options['L'], world_state['DT'])
            plant_initial_state = plant_options.get('initial_state', plant_factory.get_initial_state())
            plant_initial_action = plant_options.get('initial_action', plant_factory.get_noop_action())
            plant_state_normalizer = plant_factory.normalize_state

            # Sensor
            if options['sensor'] == 'model_output':
                sensor_state = {
                    'sensor': 'model_output',
                    '_sensor_fn': lambda sstate, x, u: (sstate['_model'].output(0, x, u), sstate, {}),
                    '_sensor_state': {'_model': plant_model},
                    'sensor_debug_output': {},
                }
            elif options['sensor'] == 'model_output_with_corruption':
                sensor_state = {
                    'sensor': 'model_output_with_corruption',
                    '_sensor_fn': lambda sstate, x, u: (sstate['_model'].output(0, x, u)*sstate['multiplicative_corruption']+sstate['additive_corruption'], sstate, {'sensor_state': sstate}),
                    '_sensor_state': {
                        'multiplicative_corruption': np.ones(plant_model.noutputs),
                        'additive_corruption': np.zeros(plant_model.noutputs),
                        '_model': plant_model,
                    },
                    'sensor_debug_output': {},
                }
            else:
                raise Exception(f"ERROR: unknown sensor: '{options['sensor']}'")

            # Estimator
            if options['estimator'] == 'first_n':
                estimator_state = {
                    'estimator': 'sensor',
                    '_estimator_fn': lambda est_state, measurement, prev_inputs, _true_state: (measurement[0:plant_model.nstates], est_state, {}),
                    '_estimator_state': None,
                    'estimator_debug_output': {},
                }
            elif options['estimator'] == 'l1_optimizer':
                estimator = MyEstimator(options['L'], world_state['DT'])
                estimator_state = {
                    'estimator': 'l1_optimizer',
                    '_estimator_fn': estimator.tick,
                    '_estimator_state': {
                        'target_speed': options['target_speed'], # m/s
                        # TODO: this should be coming from the planner
                        # 'target_path': options['target_path'],
                    },
                    'estimator_debug_output': {},
                }
            else:
                raise Exception(f"ERROR: unknown estimator: '{options['estimator']}'")

            # Planner
            if options['planner'] == 'static_slice':
                planner = StaticSlicePlanner(options['global_ref_path'], **options['planner_options'])
                planner_state = {
                    'planner': 'static_slice',
                    '_planner_fn': planner.tick,
                    '_planner_state': {},
                    'planner_debug_output': {},
                }
            elif options['planner'] == 'subdivision':
                planner = SubdivisionPlanner(options['global_ref_path'], **options['planner_options'])
                planner_state = {
                    'planner': 'subdivision',
                    '_planner_fn': planner.tick,
                    '_planner_state': {},
                    'planner_debug_output': {},
                }
            elif options['planner'] == 'lateral_profile':
                planner = LateralProfilePlanner(options['global_ref_path'], options['target_speed'], **options['planner_options'])
                planner_state = {
                    'planner': 'lateral_profile',
                    '_planner_fn': planner.tick,
                    '_planner_state': {},
                    'planner_debug_output': {},
                }
            else:
                raise Exception(f"ERROR: unknown planner: '{options['planner']}'")

            # Controller
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
                    '_controller_state': pfkmpc.get_initial_state(target_path=options['global_ref_path'], dt=world_state['DT'], L=options['L']),
                    'controller_debug_output': {},
                }
            elif options['controller'] == 'lookahead_lqr':
                controller_state = {
                    'controller': 'lookahead_lqr',
                    '_controller_fn': lookahead_lqr.lookahead_lqr,
                    '_controller_state': lookahead_lqr.get_initial_state(
                        target_speed=options['target_speed'],
                        dt=world_state['DT'],
                        L=options['L'],
                        **options['controller_options'],
                    ),
                    'controller_debug_output': {},
                }
            else:
                raise Exception(f"ERROR: unknown controller: {options['controller']}")

            world_state['entities'][entity_id] = {
                'type': 'ego',
                'state': plant_initial_state,
                'action': plant_initial_action,
                'L': options['L'],
                'global_ref_path': options['global_ref_path'],
                '_handler': make_ego_handler(entity_id),
                '_model': plant_model,
                '_model_state_normalizer': plant_state_normalizer,
                **sensor_state,
                **estimator_state,
                **planner_state,
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
            entity['state'] = ckb.get_initial_state()
            entity['action'] = dkb.get_noop_action()
        elif command.startswith('action: '):
            if entity['controller'] != 'manual':
                raise Exception(f"ERROR: entity '{entity_id}' is not in manual mode")
            entity['_controller_state'] = np.fromstring(command[len('action: '):], dtype=float, sep=' ')
        elif command == 'state':
            pass # noop, just return the current state
        elif command == '_tick':
            # sensor
            entity['measurement'], entity['_sensor_state'], entity['sensor_debug_output'] = \
                entity['_sensor_fn'](entity['_sensor_state'], entity['state'], entity['action'])
            # estimator
            entity['estimate'], entity['_estimator_state'], entity['estimator_debug_output'] = \
                entity['_estimator_fn'](entity['_estimator_state'], entity['measurement'], entity['action'], entity['state'])
            # planner
            entity['_planner_state']['t'] = world_state['t']
            entity['planner_output'], entity['_planner_state'], entity['planner_debug_output'] = \
                entity['_planner_fn'](entity['_planner_state'], entity['estimate'], entity['action'])

            entity['_controller_state']['target_path'] = entity['planner_output']['target_path']
            entity['_estimator_state']['target_path'] = entity['planner_output']['target_path']

            model = entity['_model']
            # model = control.sample_system(entity['_model'], world_state['DT'])
            # model = control.sample_system(entity['_model'].linearize(
            #     [0.0, 0.0, 0.7853981633974483, 5.0, 0.0], [0,0]), world_state['DT'])
            # model = control.sample_system(entity['_model'].linearize([0,0,entity['state'][2],5,0], [0, 0]), world_state['DT'])

            # calculate control action
            entity['action'], entity['_controller_state'], entity['controller_debug_output'] = \
                entity['_controller_fn'](entity['_controller_state'], entity['estimate'])

            # calculate new plant state
            if model.isdtime():
                entity['state'] = entity['_model_state_normalizer'](model.dynamics(0, entity['state'], entity['action']))
            else:
                # forward euler
                entity['state'] = entity['_model_state_normalizer'](model.dynamics(0, entity['state'], entity['action']) * world_state['DT'] + entity['state'])

        elif ENTITY_UPDATE_STATE_REGEX.match(command):
            new_state = json.loads(ENTITY_UPDATE_STATE_REGEX.match(
                command).group('new_state'), cls=JSONNumpyDecoder)
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

        # print(f"[{request_id}] < {path}: {request}")
        try:
            response = {'id': request_id, 'response': handler(command)}
        except Exception:
            print(f"[{request_id}] ERROR: exception thrown while handling command")
            traceback.print_exc()
            response = {'id': request_id, 'error': traceback.format_exc()}

        serialized_response = json.dumps(
            round_floats(strip_internal_vars(response)), cls=MyEncoder)

        await websocket.send(serialized_response)
        # print(f"[{request_id}] > {path}: {serialized_response}")

start_server = websockets.serve(new_connection, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
print("Starting backend event loop...")
asyncio.get_event_loop().run_forever()
