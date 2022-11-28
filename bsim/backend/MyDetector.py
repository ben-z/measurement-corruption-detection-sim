from copy import deepcopy
from itertools import chain
import control
import continuous_kinematic_bicycle as ckb
import numpy as np
from numpy.linalg import eig, matrix_power, norm
from scipy.linalg import block_diag
import time
from utils import calc_input_effects_on_output, optimize_l1, \
    optimize_l0, distance_to_line_segment, wrap_to_pi, \
    get_l0_state_estimation_l2_bound, s_sparse_observability, \
    get_error_estimation_l2_bounds, \
    does_l1_state_estimation_error_analytical_bound_hypothesis_hold_for_K, \
    is_l1_state_estimation_error_bounded, get_l1_state_estimation_l2_bound, \
    closest_point_on_line, generate_segment_info, move_along_path, EndOfPathError

np.set_printoptions(suppress=True, precision=4)

PATH_MEMORY_EVICT_MULTIPLIER = 10

def calc_desired_state_trajectory_on_a_line(linearization_state, N, dt):
    """
    returns a n-by-N matrix of desired state trajectory.
    """
    x0, y0, theta0, v0, delta0 = linearization_state
    n = len(linearization_state)

    desired_trajectory = np.zeros((n, N))
    for t in range(N):
        desired_trajectory[:, t] = np.array([
            v0*np.cos(theta0)*t*dt + x0,
            v0*np.sin(theta0)*t*dt + y0,
            theta0,
            v0,
            delta0
        ])
    
    return desired_trajectory


class MyDetector:
    def __init__(self, L, dt, N=10, min_ticks_per_solve=50):
        """
        N: time horizon, the number of time steps
        min_ticks_per_solve: minimum number of ticks between solver invocations
        """
        self.N = N
        self.dt = dt
        self.min_ticks_per_solve = min_ticks_per_solve
        self.L = L
        self.model = ckb.make_continuous_kinematic_bicycle_model(L)
        self._prev_solve_tick = -min_ticks_per_solve
        self._sensor_validity_map = np.ones(self.model.noutputs, dtype=bool)
        self._tick_count = 0
        self._path_points: np.ndarray = np.array([]) # keep a running list of historical path points so that we can look back
        self._clear_data()

    def _clear_data(self):
        # for type checking
        assert self.model.nstates is not None
        assert self.model.noutputs is not None
        assert self.model.ninputs is not None

        self._measurements = np.zeros((self.model.noutputs, self.N))
        self._inputs = np.zeros((self.model.ninputs, self.N))
        self._true_states = np.zeros((self.model.nstates, self.N))
        self._num_data_points = 0

    def _solve(self, ext_state, debug_output, segment_info, current_path_segment_idx):
        # solve the estimation problem
        # returns the sensor validity map

        print('solving estimation problem')

        sensor_validity_map = np.ones(self.model.noutputs, dtype=bool)

        # for type checking
        assert self.model.nstates is not None
        assert self.model.noutputs is not None
        assert self.model.ninputs is not None

        current_path_segment = segment_info[current_path_segment_idx]

        m_per_step = ext_state['target_speed'] * self.dt

        # a generator to move along the path backwards, starting from the current state
        desired_trajectory_generator = chain(
            [(current_path_segment, current_path_segment_idx)],
            move_along_path(deepcopy(segment_info), current_path_segment_idx, -m_per_step)
        )
        desired_state_trajectory = np.zeros((self.model.nstates, self.N))
        for k in range(self.N-1, -1, -1):
            info, _ = next(desired_trajectory_generator)
            desired_state_trajectory[:2, k] = info.closest_point
            desired_state_trajectory[2, k] = info.heading
            desired_state_trajectory[3, k] = ext_state['target_speed']
            desired_state_trajectory[4, k] = 0

        continuous_linear_models = []
        linear_models = []
        for k in range(self.N):
            linearization_state = desired_state_trajectory[:, k]
            linearization_input = np.zeros(self.model.ninputs)
            sysc = ckb.get_linear_model_straight_line_ref(*linearization_state, *linearization_input, L=self.L)

            # avoid repeated calculations of the same model
            if k > 0 and all([np.array_equal(new, old) for new, old in zip(sysc, continuous_linear_models[-1])]):
                continuous_linear_models.append(continuous_linear_models[-1])
                linear_models.append(linear_models[-1])
                continue

            continuous_linear_models.append(sysc)
            linear_models.append(control.sample_system(control.ss(*sysc), self.dt))

        # Calculate a signature, mainly used for debugging.
        # The array denotes the number of same models used in a row.
        # e.g. [5,21] means that in the first 5 time steps, the one model was used,
        # and in the next 21 time steps, another model was used.
        time_varying_model_signature = [1]
        for k in range(1, self.N):
            if np.array_equal(linear_models[k], linear_models[k-1]):
                time_varying_model_signature[-1] += 1
            else:
                time_varying_model_signature.append(1)

        debug_output["time_varying_model_signature"] = time_varying_model_signature

        # # Prevent the solver from using multiple models
        # if len(time_varying_model_signature) > 1:
        #     print(f"There are more than one model used in the estimation problem (signature {time_varying_model_signature})."
        #             " This is not supported yet. Deferring to a future time step.")
        #     return sensor_validity_map

        self._prev_solve_tick = self._tick_count

        evolution_matrix = np.zeros((self.model.nstates*self.N, self.model.nstates))
        evolution_matrix[0:self.model.nstates, :] = np.eye(self.model.nstates)
        for k in range(1, self.N):
            evolution_matrix[k*self.model.nstates:(k+1)*self.model.nstates, :] = linear_models[k-1].A @ evolution_matrix[(k-1)*self.model.nstates:k*self.model.nstates, :]
        
        input_effect_matrix_As = block_diag(*[np.eye(self.model.nstates) for _ in range(self.N)])
        for k in range(1, self.N):
            for j in range(k):
                input_effect_matrix_As[k*self.model.nstates:(k+1)*self.model.nstates, j*self.model.nstates:(j+1)*self.model.nstates] = \
                    linear_models[k].A @ input_effect_matrix_As[(k-1)*self.model.nstates:k*self.model.nstates, j*self.model.nstates:(j+1)*self.model.nstates]
        
        input_effect_matrix_Bs = block_diag(*[linear_models[k].B for k in range(self.N)])
        
        Phi = block_diag(*[ckb.get_C()]*self.N) @ evolution_matrix

        input_effects = (block_diag(*[ckb.get_C()]*self.N) @ input_effect_matrix_As @ input_effect_matrix_Bs @ self._inputs.reshape((self.model.ninputs*self.N,1), order='F')).reshape((self.model.noutputs, self.N), order='F')
        desired_trajectory = ckb.get_C() @ desired_state_trajectory
        measurements = self._measurements - input_effects - desired_trajectory
        # measurements = self._measurements - desired_trajectory
        # normalize angular measurements
        measurements[ckb.get_angular_outputs_mask(), :] = wrap_to_pi(measurements[ckb.get_angular_outputs_mask(), :])

        # normalize measurements
        # Y is measurements stacked vertically
        Y = np.reshape(measurements, (self.model.noutputs*self.N,), order='F')

        # s_sparse_observability(A,C)
        sensor_errors = np.array([0.15, 0.15, 0.8, 0.1, 0.3, 0.15, 0.15, 0.8, 0.1])
        # Dx = get_l0_state_estimation_l2_bound(A, C, sensor_errors, 1, self.N)
        # De = get_error_estimation_l2_bounds(A, C, Dx, sensor_errors, self.N)

        # does_l1_state_estimation_error_analytical_bound_hypothesis_hold_for_K(A, C, np.array([3]), self.N)
        # is_l1_state_estimation_error_bounded(A, C, np.array([3]), self.N)
        # get_l1_state_estimation_l2_bound(A, C, sensor_errors, np.array([3]), self.N)

        solve_start = time.time()
        # prob, x0_hat = optimize_l0(self.model.nstates, self.model.noutputs, self.N, Phi, Y, sensor_errors)
        prob, x0_hat = optimize_l1(self.model.nstates, self.model.noutputs, self.N, Phi, Y)
        solve_end = time.time()
        
        linearization_state_x0 = desired_state_trajectory[:, 0]
        linearization_state_xf = desired_state_trajectory[:, -1]
        diff_from_true_x0 = ckb.normalize_state(
            x0_hat.value + linearization_state_x0 - self._true_states[:, 0])
        
        estimated_states = (evolution_matrix @ x0_hat.value).reshape((self.model.nstates, self.N), order='F') + desired_state_trajectory
        diff_from_true = estimated_states - self._true_states
        for t in range(self.N):
            diff_from_true[:,t] = ckb.normalize_state(diff_from_true[:,t])
        
        xf_hat = evolution_matrix[-self.model.nstates:, :] @ x0_hat.value + linearization_state_xf
        
        assert(np.allclose(x0_hat.value + linearization_state_x0, estimated_states[:,0]))
        assert(np.allclose(xf_hat, estimated_states[:,-1]))
        debug_output["true_states"] = self._true_states.T
        debug_output["estimated_states"] = estimated_states.T
        debug_output["diff_from_true"] = diff_from_true.T
        
        print("time_varying_model_signature", time_varying_model_signature)
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal state (x0)", x0_hat.value + linearization_state_x0)
        print("true state (x0)", self._true_states[:,0])
        print("state estimation error (x0)", diff_from_true_x0)
        print(f"state estimation l2 error (x0): {norm(diff_from_true_x0):.4f}")
        # print(f"state estimation l2 error bound: {Dx:.4f}")
        # print(f"state estimation l2 error bound violated: {norm(diff_from_true_x0) > Dx}")

        debug_output["state_estimation_error_x0"] = diff_from_true_x0
        debug_output["state_estimation_l2_error_x0"] = norm(diff_from_true_x0)

        diff_from_true_xf = ckb.normalize_state(xf_hat - self._true_states[:, -1])

        print("estimated state (xf)", xf_hat)
        print("true state (xf)", self._true_states[:,-1])
        print("state estimation error (xf)", diff_from_true_xf)
        print(f"state estimation l2 error (xf): {norm(diff_from_true_xf):.4f}")

        debug_output["state_estimation_error_xf"] = diff_from_true_xf
        debug_output["state_estimation_l2_error_xf"] = norm(diff_from_true_xf)

        print(f"solve time: {solve_end - solve_start:.4f}s")

        # attack_vector is a pxT matrix
        attack_vector = (Y - np.matmul(Phi, x0_hat.value)).reshape((self.model.noutputs, self.N), order='F')
        attack_vector_norms = norm(attack_vector, axis=1)
        print(f"attack vector norms (l2 norm, all time steps): {attack_vector_norms}")
        De = 1 # arbitrary threshold
        print(f"attack vector norms threshold: {De}")
        mean_attack_vector = np.mean(attack_vector, axis=1)
        sensors_under_attack = attack_vector_norms > De
        sensor_validity_map = np.logical_not(sensors_under_attack)
        num_sensors_under_attack = np.sum(sensors_under_attack)
        print("mean attack vector:", mean_attack_vector)
        print("Sensors under attack:", sensors_under_attack)
        print("Number of sensors under attack:", num_sensors_under_attack)
        print("=================================")

        return sensor_validity_map

    def tick(self, ext_state, measurement, prev_inputs, true_state=None):
        """
        ext_state: state managed by the client. This usually stores values that change during runtime.
        measurement: the measurement from the sensors.
        prev_inputs: the inputs from the previous time step.
        true_state: the true state, used for debugging (e.g. evaluating estimation error).
        """
        self._tick_count += 1

        debug_output = {}

        # TODO: use measurement instead of true_state. However, we don't know what the
        # true state is without first solving the estimation problem. This seems like a
        # chicken and egg problem.
        # TODO: use something other than the true state as the initial guess, perhaps the previous estimate
        # projected forward in time or the target state along the path (the latter requires a feedback from
        # the controller)?
        # Linearize, assuming the reference is a line. (See 2022-09-15 and 2022-09-22 notes for derivations)
        target_path = ext_state.get('target_path')
        if target_path is None:
            return measurement, ext_state, debug_output

        # This is used to infer the nominal trajectory and linearization
        # TODO: use estimated x and y instead of true x and y
        x = true_state[0]
        y = true_state[1]

        pos = np.array([x,y])
        target_path_segment_info = generate_segment_info(pos, target_path, wrap=False)
        target_path_current_segment_idx = np.argmin([norm(info.closest_point - pos) for info in target_path_segment_info])

        path_memory_segment_info = generate_segment_info(pos, self._path_points, wrap=False)
        if len(path_memory_segment_info) == 0:
            # we don't have any paths in memory
            path_memory_current_segment_idx = target_path_current_segment_idx
            self._path_points = target_path
            path_memory_segment_info = target_path_segment_info
        else:
            # search in reverse, so that when segments overlap, we take the latest segment
            # TODO: we really want to search from the segment index from the previous tick (assuming the vehicle does not teleport).
            # But for now it's okay to assume the latest closest segment is the correct one.
            path_memory_current_segment_idx = len(path_memory_segment_info) - 1 - np.argmin([norm(info.closest_point - pos) for info in reversed(path_memory_segment_info)])
            # We want to append the remainder of the target path to the path memory.
            # To handle different path lengths, we need to find the segment on the target path that's just after the current segment in the path memory.
            path_memory_current_segment_dist_remaining = path_memory_segment_info[path_memory_current_segment_idx].distance_remaining
            target_path_segment_idx_at_the_end_of_path_memory_segment = next(move_along_path(deepcopy(target_path_segment_info), target_path_current_segment_idx, path_memory_current_segment_dist_remaining))[1]
            # append the target path starting from the second point of the next segment
            # this prevents sharp changes in heading in the resulting path.
            target_path_starting_idx = target_path_segment_idx_at_the_end_of_path_memory_segment+2
            # use the path memory up to and including the current segment
            self._path_points = np.concatenate(
                (self._path_points[:path_memory_current_segment_idx+2], target_path[target_path_starting_idx:]), axis=0)

            # assertion to make sure there are no duplicate path points. This is slow so we comment it out
            # assert all(np.sum(np.diff(self._path_points, axis=0), axis=1) != 0)

        debug_output["target_path_current_segment_idx"] = int(target_path_current_segment_idx)
        debug_output["path_memory"] = self._path_points
        debug_output["path_memory_segment_idx"] = int(path_memory_current_segment_idx)

        # Collect measurements
        self._measurements = np.roll(self._measurements, -1, axis=1)
        self._measurements[:,-1] = measurement
        self._inputs = np.roll(self._inputs, -1, axis=1)
        self._inputs[:,-1] = prev_inputs
        self._true_states = np.roll(self._true_states, -1, axis=1)
        self._true_states[:,-1] = true_state
        self._num_data_points += 1

        debug_output["num_data_points"] = self._num_data_points

        if self._num_data_points >= self.N and self._tick_count - self._prev_solve_tick >= self.min_ticks_per_solve:
            new_sensor_validity_map = self._solve(
                ext_state, debug_output, path_memory_segment_info, path_memory_current_segment_idx)
            
            if not np.array_equal(new_sensor_validity_map, self._sensor_validity_map):
                print(f"sensor_validity_map changed from {self._sensor_validity_map} to {new_sensor_validity_map}!")
            
            self._sensor_validity_map = new_sensor_validity_map

        debug_output["tick_count"] = self._tick_count
        debug_output["prev_solve_tick"] = self._prev_solve_tick
        debug_output["sensor_validity_map"] = self._sensor_validity_map

        # evict old path memory (arbitrarily set to be a multiplier on the number of states needed for the solver)
        lookbehind_m = ext_state["target_speed"] * self.N * self.dt * PATH_MEMORY_EVICT_MULTIPLIER
        try:
            eviction_threshold_segment_idx = next(move_along_path(deepcopy(path_memory_segment_info), path_memory_current_segment_idx, -lookbehind_m, wrap=False))[1]
        except EndOfPathError:
            eviction_threshold_segment_idx = 0
        self._path_points = self._path_points[eviction_threshold_segment_idx:]
        debug_output["eviction_threshold_segment_idx"] = int(eviction_threshold_segment_idx)

        # overwrite invalid sensors
        valid_measurement = measurement.copy()
        valid_measurement[np.invert(self._sensor_validity_map)] = np.nan

        return valid_measurement, ext_state, debug_output
