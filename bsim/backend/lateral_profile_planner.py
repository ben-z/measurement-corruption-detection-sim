from copy import deepcopy
from math import ceil
import numpy as np
from numpy.linalg import norm
from utils import generate_segment_info, move_along_path, frenet2global_point
from itertools import chain

def get_lateral_deviation(lateral_deviation_profile, t):
    """
    Returns the lateral deviation at a given time t
    """
    if lateral_deviation_profile['periodic']:
        t = t % lateral_deviation_profile['points'][-1]['t']
        assert lateral_deviation_profile['points'][0]['lateral_deviation'] == lateral_deviation_profile['points'][-1]['lateral_deviation'], \
            'Periodic lateral deviation profile must have the same lateral deviation at the start and end'

    if t < lateral_deviation_profile['points'][0]['t']:
        return lateral_deviation_profile['points'][0]['lateral_deviation']
    
    if t > lateral_deviation_profile['points'][-1]['t']:
        return lateral_deviation_profile['points'][-1]['lateral_deviation']


    # Find the two points that bound the current time
    for i in range(len(lateral_deviation_profile['points'])-1):
        if t >= lateral_deviation_profile['points'][i]['t'] and t <= lateral_deviation_profile['points'][i+1]['t']:
            p0 = lateral_deviation_profile['points'][i]
            p1 = lateral_deviation_profile['points'][i+1]
            break
    else:
        raise Exception(f'Could not find lateral deviation profile point for time {t}')

    # Interpolate between the two points
    if lateral_deviation_profile['interpolation'] == 'step':
        return p0['lateral_deviation']
    
    if lateral_deviation_profile['interpolation'] == 'linear':
        return p0['lateral_deviation'] + (t - p0['t']) / (p1['t'] - p0['t']) * (p1['lateral_deviation'] - p0['lateral_deviation'])
    
    raise Exception(f'Unsupported interpolation method {lateral_deviation_profile["interpolation"]}')

DEFAULT_LATERAL_DEVIATION_PROFILE = {
    'interpolation': 'linear',
    'periodic': True,
    'points': [
        {'t': 0, 'lateral_deviation': 0},
        {'t': 5, 'lateral_deviation': 0},
        {'t': 10, 'lateral_deviation': 3},
        {'t': 20, 'lateral_deviation': 3},
        {'t': 30, 'lateral_deviation': 0},
        {'t': 35, 'lateral_deviation': 0},
        {'t': 40, 'lateral_deviation': -3},
        {'t': 50, 'lateral_deviation': -3},
        {'t': 60, 'lateral_deviation': 0},
    ],
}

class LateralProfilePlanner:
    """
    Planner that follows a lateral deviation profile.
    """

    def __init__(self, global_ref_path, target_speed, lookahead_m=10., subdivision_m=1., lateral_deviation_profile=DEFAULT_LATERAL_DEVIATION_PROFILE):
        self.global_ref_path = global_ref_path
        self.lookahead_m = lookahead_m
        self.target_speed = target_speed
        self.subdivision_m = subdivision_m
        self.lateral_deviation_profile = lateral_deviation_profile

    def tick(self, ext_state, estimate, input):

        debug_output = {}
        
        t0 = ext_state['t']

        pos = estimate[:2]
        # TODO: we should be able to reuse the segment info from the previous tick somehow. Reduced search space for the closest segment?
        segment_info = generate_segment_info(pos, self.global_ref_path)
        current_path_segment_idx = np.argmin(
            [norm(info.closest_point - pos) for info in segment_info])
        current_path_segment = segment_info[current_path_segment_idx]

        target_path = []
        target_lateral_deviations = []
        target_path_generator = chain(
            [(current_path_segment, current_path_segment_idx)],
            move_along_path(deepcopy(segment_info),
                            current_path_segment_idx, self.subdivision_m)
        )
        for i in range(0, ceil(self.lookahead_m / self.subdivision_m)+1):
            t = t0 + i * self.subdivision_m / self.target_speed
            segment = next(target_path_generator)[0]
            latdev = get_lateral_deviation(self.lateral_deviation_profile, t)
            target_point = frenet2global_point(segment.closest_point[0], segment.closest_point[1], segment.heading, 0, latdev)
            target_path.append(target_point)
            target_lateral_deviations.append(latdev)

        debug_output['target_lateral_deviations'] = target_lateral_deviations

        return {
            'target_path': target_path,
        }, ext_state, debug_output
