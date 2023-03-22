import asyncio
import control
from copy import deepcopy
import traceback
import discrete_kinematic_bicycle as dkb
import continuous_kinematic_bicycle as ckb
import json
import lookahead_lqr
import mergedeep
from MyDetector import MyDetector
import numpy as np
import path_following_kmpc as pfkmpc
import re
import secrets
import sys
from static_slice_planner import StaticSlicePlanner
from subdivision_planner import SubdivisionPlanner
from lateral_profile_planner import LateralProfilePlanner
from urllib.parse import parse_qsl
from utils import JSONNumpyDecoder, ensure_options_are_known, AutoPerfCounter
import websockets
from typing import Dict, Any
import time
import urllib

INITIAL_WORLD_STATE = {
    't': 0,
    'DT': 0.01,
    'entities': {},
    'execution_times': {},
}

world_state = deepcopy(INITIAL_WORLD_STATE)

ENTITY_PATH_REGEX = re.compile(r'^/entities/(?P<entity_id>\w+)$')
CREATE_ENTITY_REGEX = re.compile(r'^create_entity: (?P<entity_type>\w+)(?: (?P<entity_id>\w+))?(?: (?P<entity_options>[^ ]+))?$')
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
    if isinstance(o, np.ndarray) and o.dtype in (np.float32, np.float64):
        return o.round(decimals)
    return o

def strip_internal_vars(o):
    if isinstance(o, dict):
        return {k: strip_internal_vars(v) for k, v in o.items() if not k.startswith('_')}
    if isinstance(o, (list, tuple)):
        return type(o)(strip_internal_vars(x) for x in o)
    return o

def strip_nans(o):
    if isinstance(o, dict):
        return {k: strip_nans(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(strip_nans(x) for x in o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, np.ndarray) and o.dtype in (np.float32, np.float64):
        return strip_nans(o.tolist())
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

    world_state['execution_times'] = {}

    start_perf_counter = time.perf_counter()

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
        print(f"Creating entity {entity_id} of type {entity_type}")

        if not entity_id:
            entity_id = f"{entity_type}_{secrets.token_urlsafe(8)}"
        
        if entity_id in world_state['entities']:
            raise Exception(f"ERROR: entity with ID '{entity_id}' already exists")

        user_options: Dict[str, Any] = json.loads(urllib.parse.unquote(entity_options), cls=JSONNumpyDecoder) if entity_options else {}

        if entity_type == 'ego':
            default_options = {
                'controller': 'manual',
                'L': 2.9,
                # coordinates of the waypoints in meters. This will form a closed path
                'global_ref_path': np.array([ [-10,3], [12,-5], [10,-5], [7, -8], [0,-10], [-10,-3] ]),
                'target_speed': 1., # m/s
                'sensor': 'model_output_with_corruption',
                'detector': 'l1_optimizer',
                'estimator': 'first_n',
                'estimator': 'average_valid',
                # 'planner': 'static_slice',
                # 'planner': 'subdivision',
                'planner': 'lateral_profile',
                # Default options for these modules are defined in their respective handlers
                'plant_options': {},
                'controller_options': {},
                'planner_options': {},
                'detector_options': {},
            }

            # check for invalid options
            ensure_options_are_known(user_options, default_options, entity_id)

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

            # Detector
            if options['detector'] == 'none':
                # simply pass through the sensor output
                detector_state = {
                    'detector': 'none',
                    '_detector_fn': lambda det_state, measurement, *args, **kwargs: (measurement, det_state, {}),
                    '_detector_state': {},
                    'detector_debug_output': {},
                }
            elif options['detector'] == 'l1_optimizer':
                detector = MyDetector(options['L'], world_state['DT'], **options["detector_options"])
                detector_state = {
                    'detector': 'l1_optimizer',
                    '_detector_fn': detector.tick,
                    '_detector_state': {
                        'target_speed': options['target_speed'], # m/s
                        # TODO: this should be coming from the planner
                        # 'target_path': options['target_path'],
                    },
                    'detector_debug_output': {},
                }
            else:
                raise Exception(f"ERROR: unknown detector: '{options['detector']}'")

            # Estimator
            if options['estimator'] == 'first_n':
                def first_n_estimator_fn(estimator_state, measurement, _prev_inputs, _true_state):
                    estimate = measurement[0:plant_model.nstates]
                    if np.nan in estimate:
                        raise Exception(f"ERROR: this estimator cannot handle NaN's: {estimate}")
                    return estimate, estimator_state, {}

                estimator_state = {
                    'estimator': 'first_n',
                    '_estimator_fn': first_n_estimator_fn,
                    '_estimator_state': None,
                    'estimator_debug_output': {},
                }
            elif options["estimator"] == "average_valid":
                # average the valid (non-NaN) measurements
                def average_valid_estimator_fn(estimator_state, measurement, _prev_inputs, _true_state):
                    estimate = np.zeros(plant_model.nstates)

                    for i in range(plant_model.nstates):
                        estimate[i] = np.nanmean(measurement[ckb.OUTPUT_TO_STATE_MAP == i])

                    return estimate, estimator_state, {}

                estimator_state = {
                    'estimator': 'average_valid',
                    '_estimator_fn': average_valid_estimator_fn,
                    '_estimator_state': None,
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
                **detector_state,
                **estimator_state,
                **planner_state,
                **controller_state,
            }
        else:
            raise Exception(f"ERROR: Unknown entity type '{entity_type}'")
    else:
        raise Exception(f"ERROR: Unknown command: {command}")
    
    world_state['execution_times']['request_handler'] = time.perf_counter() - start_perf_counter
    
    return world_state

def make_ego_handler(entity_id: str):
    def ego_handler(command: str):
        entity = world_state['entities'][entity_id]
        entity['execution_times'] = {}
        start_perf_counter = time.perf_counter()

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
            with AutoPerfCounter(entity['execution_times'], 'sensor'):
                entity['measurement'], entity['_sensor_state'], entity['sensor_debug_output'] = \
                    entity['_sensor_fn'](entity['_sensor_state'], entity['state'], entity['action'])
                entity['measurement'].setflags(write=False)
            # detector
            with AutoPerfCounter(entity['execution_times'], 'detector'):
                entity['valid_measurement'], entity['_detector_state'], entity['detector_debug_output'] = \
                    entity['_detector_fn'](
                        entity['_detector_state'],
                        entity['measurement'],
                        prev_input=entity['action'],
                        prev_estimate=entity.get('estimate'),
                        true_state=entity['state'],
                    )
                entity['valid_measurement'].setflags(write=False)
            # estimator
            with AutoPerfCounter(entity['execution_times'], 'estimator'):
                entity['estimate'], entity['_estimator_state'], entity['estimator_debug_output'] = \
                    entity['_estimator_fn'](entity['_estimator_state'], entity['valid_measurement'], entity['action'], entity['state'])
                entity['estimate'].setflags(write=False)
            # planner
            with AutoPerfCounter(entity['execution_times'], 'planner'):
                entity['_planner_state']['t'] = world_state['t']
                entity['planner_output'], entity['_planner_state'], entity['planner_debug_output'] = \
                    entity['_planner_fn'](entity['_planner_state'], entity['estimate'], entity['action'])

            entity['_controller_state']['target_path'] = entity['planner_output']['target_path']
            entity['_detector_state']['target_path'] = entity['planner_output']['target_path']

            model = entity['_model']
            # model = control.sample_system(entity['_model'], world_state['DT'])
            # model = control.sample_system(entity['_model'].linearize(
            #     [0.0, 0.0, 0.7853981633974483, 5.0, 0.0], [0,0]), world_state['DT'])
            # model = control.sample_system(entity['_model'].linearize([0,0,entity['state'][2],5,0], [0, 0]), world_state['DT'])

            # controller
            with AutoPerfCounter(entity['execution_times'], 'controller'):
                entity['action'], entity['_controller_state'], entity['controller_debug_output'] = \
                    entity['_controller_fn'](entity['_controller_state'], entity['estimate'])

            # calculate new plant state
            with AutoPerfCounter(entity['execution_times'], 'plant'):
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
        
        entity['execution_times']['request_handler'] = time.perf_counter() - start_perf_counter
        
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

        serialized_response = json.dumps(strip_nans(round_floats(strip_internal_vars(response))), allow_nan=False, cls=MyEncoder)

        await websocket.send(serialized_response)
        # print(f"[{request_id}] > {path}: {serialized_response}")

start_server = websockets.serve(new_connection, host="0.0.0.0", port=8766)

loop = asyncio.get_event_loop_policy().get_event_loop()
loop.run_until_complete(start_server)
print("Starting backend event loop...")
loop.run_forever()
