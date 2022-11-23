from copy import deepcopy
from math import ceil
import numpy as np
from numpy.linalg import norm
from utils import generate_segment_info, move_along_path
from itertools import chain


class SubdivisionPlanner:
    """
    Planner that returns a subdivided slice of the static target path
    """

    def __init__(self, global_ref_path, lookahead_m=10., subdivision_m=1.):
        self.global_ref_path = global_ref_path
        self.lookahead_m = lookahead_m
        self.subdivision_m = subdivision_m

    def tick(self, ext_state, estimate, input):

        debug_output = {}

        pos = estimate[:2]
        # TODO: we should be able to reuse the segment info from the previous tick somehow. Reduced search space for the closest segment?
        segment_info = generate_segment_info(pos, self.global_ref_path)
        current_path_segment_idx = np.argmin(
            [norm(info.closest_point - pos) for info in segment_info])
        current_path_segment = segment_info[current_path_segment_idx]

        target_path = []
        target_path_generator = chain(
            [(current_path_segment, current_path_segment_idx)],
            move_along_path(deepcopy(segment_info), current_path_segment_idx, self.subdivision_m)
        )
        for i in range(0, ceil(self.lookahead_m / self.subdivision_m)+1):
            target_path.append(next(target_path_generator)[0].closest_point)

        return {
            'target_path': target_path,
        }, ext_state, debug_output
