from copy import deepcopy
import numpy as np
from numpy.linalg import norm
from utils import generate_segment_info, move_along_path

class StaticSlicePlanner:
    """
    Planner that returns a slice of the static target path
    """

    def __init__(self, global_ref_path, lookahead_m=10., lookbehind_m=0.):
        self.global_ref_path = global_ref_path
        self.lookahead_m = lookahead_m
        self.lookbehind_m = lookbehind_m
    
    def tick(self, ext_state, estimate, input):

        debug_output = {}
        
        pos = estimate[:2]
        # TODO: we should be able to reuse the segment info from the previous tick somehow. Reduced search space for the closest segment?
        segment_info = generate_segment_info(pos, self.global_ref_path)
        current_path_segment_idx = np.argmin([norm(info.closest_point - pos) for info in segment_info])

        lookahead_path_segment, lookahead_path_segment_idx = next(move_along_path(deepcopy(segment_info), current_path_segment_idx, self.lookahead_m))
        lookbehind_path_segment, lookbehind_path_segment_idx = next(move_along_path(deepcopy(segment_info), current_path_segment_idx, -self.lookbehind_m))

        # The total number of segments in the resulting path
        res_num_segments = (lookahead_path_segment_idx - lookbehind_path_segment_idx + 1) % len(segment_info)

        assert res_num_segments <= len(segment_info), "The target path is too short for the lookahead distance"

        debug_output['current_path_segment_idx'] = int(current_path_segment_idx)
        debug_output['lookahead_path_segment_idx'] = int(lookahead_path_segment_idx)
        debug_output['lookbehind_path_segment_idx'] = int(lookbehind_path_segment_idx)
        debug_output['res_num_segments'] = int(res_num_segments)
        debug_output['lookahead_point'] = lookahead_path_segment.closest_point
        debug_output['lookbehind_point'] = lookbehind_path_segment.closest_point

        # The resulting path
        target_path = np.roll(self.global_ref_path, -lookbehind_path_segment_idx, axis=0)[:res_num_segments+1]

        return {
            'target_path': target_path,
        }, ext_state, debug_output
