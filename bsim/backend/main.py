#!/usr/bin/env python

# WS server example

import asyncio
from discrete_kinematic_bicycle import get_initial_state, discrete_kinematic_bicycle_model, get_noop_action
import numpy as np
import websockets
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(self, obj)

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



def get_handler(path: str):
    if path == '/world':
        return  world_handler
    if path == '/ego':
        return  ego_handler
    
    return None


world_state = {
    't': 0,
    'DT': 0.01,
    'ego_state': get_initial_state(),
    'ego_action': get_noop_action(),
    'ego_L': 2.9,
}


def world_handler(command: str):
    global world_state

    if command == 'reset':
        ego_handler('reset')
    elif command == 'state':
        pass # noop, just return the current state
    elif command == 'tick':
        world_state['t'] += world_state['DT']
        world_state['ego_state'] = discrete_kinematic_bicycle_model(world_state['ego_state'], world_state['ego_action'], world_state['DT'], world_state['ego_L'])
    else:
        raise Exception(f"ERROR: Unknown command: {command}")
    
    return world_state


def ego_handler(command: str):
    global world_state

    if command == 'reset':
        world_state['ego_state'] = get_initial_state()
        world_state['ego_action'] = get_noop_action()
    elif command.startswith('action: '):
        world_state['ego_action'] = np.fromstring(command[len('action: '):], dtype=float, sep=' ')
    else:
        return f"ERROR: Unknown command: {command}"
    
    return {'result': 'OK'}


async def new_connection(websocket, path: str):
    print(f"New connection: {path}")

    handler = get_handler(path)
    if handler is None:
        print(f"ERROR: Unknown path: {path}")
        return

    while True:
        try:
            command = await websocket.recv()
        except websockets.exceptions.ConnectionClosedOK:
            print(f"Connection closed: {path}")
            break
    
        print(f"< {path}: {command}")
        try:
            response = handler(command)
        except Exception as e:
            response = {'error': e}

        serialized_response = json.dumps(round_floats(response), cls=NumpyEncoder)      

        await websocket.send(serialized_response)
        print(f"> {path}: {serialized_response}")

start_server = websockets.serve(new_connection, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
