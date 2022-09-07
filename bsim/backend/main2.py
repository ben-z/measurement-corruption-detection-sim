from copy import deepcopy
import re
import uuid
import discrete_kinematic_bicycle as dkb

INITIAL_WORLD_STATE = {
    't': 0,
    'DT': 0.01,
    'entities': {
        'ego1': {
            'type': 'ego',
            'state': dkb.get_initial_state(),
            'action': dkb.get_noop_action(),
            'controller': 'manual',
            'L': 2.9,
        }
    }
}

world_state = deepcopy(INITIAL_WORLD_STATE)

ENTITY_PATH_REGEX = re.compile(r'^/entities/(?P<entity_id>\w+)$')
CREATE_ENTITY_REGEX = re.compile(r'^create_entity: (?P<entity_type>\w+)(?: (?P<entity_id>\w+))?(?: (?P<entity_options>\w+))?$')

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
    
    return entity['handler']


def world_handler(command: str):
    global world_state

    if command == 'reset':
        world_state = deepcopy(INITIAL_WORLD_STATE)
    elif command == 'state':
        pass # noop, just return the current state
    elif command == 'tick':
        world_state['t'] += world_state['DT']
        for entity in world_state['entities'].values():
            entity['handler']('tick')
    elif CREATE_ENTITY_REGEX.match(command):
        entity_type = CREATE_ENTITY_REGEX.match(command).group('entity_type')
        entity_id = CREATE_ENTITY_REGEX.match(command).group('entity_id')
        entity_options = CREATE_ENTITY_REGEX.match(command).group('entity_options')

        if not entity_id:
            entity_id = f"{entity_type}_{str(uuid.uuid4())[:8]}"
        
        if entity_id in world_state['entities']:
            raise Exception(f"ERROR: entity with ID '{entity_id}' already exists")

        if entity_type == 'ego':
            if entity_options:
                raise Exception(f"ERROR: entity_options not supported yet: '{entity_options}'")

            world_state['entities'][entity_id] = {
                'type': 'ego',
                'state': dkb.get_initial_state(),
                'action': dkb.get_noop_action(),
                'controller': 'manual',
                'L': 2.9,
                'handler': make_ego_handler(entity_id),
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
            action = command.split('action: ')[1]
            entity['action'] = action
        elif command == 'state':
            pass # noop, just return the current state
        elif command == 'tick':
            # TODO: add sensor, estimation, and controller code here
            # ego_state = world_state['ego_state']
            # ego_measurement = ego_state
            # ego_estimate = ego_measurement
            # control_action = world_state['ego_controller'](ego_estimate)
            # world_state['ego_state'] = discrete_kinematic_bicycle_model(world_state['ego_state'], control_action, world_state['DT'], world_state['ego_L'])
            entity['state'] = dkb.discrete_kinematic_bicycle_model(entity['state'], entity['action'], world_state['DT'], entity['L'])
        else:
            raise Exception(f"ERROR: Unknown command: {command}")
        
        return entity

    return ego_handler
