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
    closest_point_on_line, generate_segment_info, move_along_path

np.set_printoptions(suppress=True, precision=4)

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


class MyEstimator:
    def __init__(self, L, dt, N=30, ticks_per_solve=100):
        """
        N: time horizon, the number of time steps
        ticks_per_solve: number of ticks between solves
        """
        self.N = N
        self.dt = dt
        self.ticks_per_solve = ticks_per_solve
        self._tick_count = 0
        self.L = L
        self.model = ckb.make_continuous_kinematic_bicycle_model(L)
        # for type checking
        assert self.model.nstates is not None
        assert self.model.noutputs is not None
        assert self.model.ninputs is not None
        self._measurements = np.zeros((self.model.noutputs, N))
        self._inputs = np.zeros((self.model.ninputs, N))
        self._true_states = np.zeros((self.model.nstates, N))

    def tick(self, ext_state, measurement, prev_inputs, true_state=None):
        """
        ext_state: state managed by the client. This usually stores values that change during runtime.
        measurement: the measurement from the sensors.
        prev_inputs: the inputs from the previous time step.
        true_state: the true state, used for debugging (e.g. evaluating estimation error).
        """
        self._tick_count += 1

        # TODO: This is currently passing through the first set of sensors. Use the estimate from the actual estimator instead.
        estimate = measurement[0:self.model.nstates]
        debug_output = {}

        # Collect measurements
        self._measurements = np.roll(self._measurements, -1, axis=1)
        self._measurements[:,-1] = measurement
        self._inputs = np.roll(self._inputs, -1, axis=1)
        self._inputs[:,-1] = prev_inputs
        self._true_states = np.roll(self._true_states, -1, axis=1)
        self._true_states[:,-1] = true_state

        # for type checking
        assert self.model.nstates is not None
        assert self.model.noutputs is not None
        assert self.model.ninputs is not None

        # TODO: use measurement instead of true_state. However, we don't know what the
        # true state is without first solving the estimation problem. This seems like a
        # chicken and egg problem.
        # TODO: use something other than the true state as the initial guess, perhaps the previous estimate
        # projected forward in time or the target state along the path (the latter requires a feedback from
        # the controller)?
        # Linearize, assuming the reference is a line. (See 2022-09-15 and 2022-09-22 notes for derivations)
        target_path = ext_state['target_path']

        # This is used to infer the nominal trajectory and linearization
        # TODO: use estimated x and y instead of true x and y
        x = true_state[0]
        y = true_state[1]

        pos = np.array([x,y])
        segment_info = generate_segment_info(pos, target_path)
        current_path_segment_idx = np.argmin([norm(info.closest_point - pos) for info in segment_info])

        debug_output['current_path_segment_idx'] = int(current_path_segment_idx)

        if self._tick_count >= self.N and self._tick_count % self.ticks_per_solve == 0:
            # solve the estimation problem
            print('solving estimation problem')

            current_path_segment = segment_info[current_path_segment_idx]

            m_per_step = ext_state['target_speed'] * self.dt

            # a generator to move along the path backwards, starting from the current state
            desired_trajectory_generator = chain(
                [current_path_segment],
                move_along_path(deepcopy(segment_info), current_path_segment_idx, -m_per_step)
            )
            desired_state_trajectory = np.zeros((self.model.nstates, self.N))
            for k in range(self.N-1, -1, -1):
                info = next(desired_trajectory_generator)
                desired_state_trajectory[:2, k] = info.closest_point
                desired_state_trajectory[2, k] = info.heading
                desired_state_trajectory[3, k] = ext_state['target_speed']
                desired_state_trajectory[4, k] = 0

            linear_models = []
            for k in range(self.N):
                linearization_state = desired_state_trajectory[:, k]
                linearization_input = np.zeros(self.model.ninputs)
                Ac, Bc, Cc, Dc = ckb.get_linear_model_straight_line_ref(*linearization_state, *linearization_input, L=self.L)
                linear_models.append(control.sample_system(control.ss(Ac, Bc, Cc, Dc), self.dt))

            evolution_matrix = np.zeros((self.model.nstates*self.N, self.model.nstates))
            evolution_matrix[0:self.model.nstates, :] = np.eye(self.model.nstates)
            for k in range(1, self.N):
                evolution_matrix[k*self.model.nstates:(k+1)*self.model.nstates, :] = evolution_matrix[(k-1)*self.model.nstates:k*self.model.nstates, :] @ linear_models[k-1].A
            
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
            diff_from_true = ckb.normalize_state(
                x0_hat.value + linearization_state_x0 - self._true_states[:, 0])
            
            xf_hat = evolution_matrix[-self.model.nstates:, :] @ x0_hat.value + linearization_state_xf
            
            print("status:", prob.status)
            print("optimal value", prob.value)
            print("optimal state (x0)",
                  x0_hat.value + linearization_state_x0)
            print("true state (x0)", self._true_states[:,0])
            print("state estimation error", diff_from_true)
            print(f"state estimation l2 error: {norm(diff_from_true):.4f}")
            # print(f"state estimation l2 error bound: {Dx:.4f}")
            # print(f"state estimation l2 error bound violated: {norm(diff_from_true) > Dx}")

            debug_output["state_estimation_l2_error_x0"] = norm(diff_from_true)

            print("estimated state (xf)", xf_hat)
            print("true state (xf)", self._true_states[:,-1])
            print(f"solve time: {solve_end - solve_start:.4f}s")

            debug_output["state_estimation_l2_error_xf"] = norm(xf_hat - self._true_states[:,-1])

            # attack_vector is a pxT matrix
            attack_vector = (Y - np.matmul(Phi, x0_hat.value)).reshape((self.model.noutputs, self.N), order='F')
            attack_vector_norms = norm(attack_vector, axis=1)
            print(f"attack vector norms (l2 norm, all time steps): {attack_vector_norms}")
            # print(f"attack vector norms threshold: {De}")
            mean_attack_vector = np.mean(attack_vector, axis=1)
            # sensors_under_attack = attack_vector_norms > De
            # num_sensors_under_attack = np.sum(sensors_under_attack)
            print("mean attack vector:", mean_attack_vector)
            # print("Sensors under attack:", sensors_under_attack)
            # print("Number of sensors under attack:", num_sensors_under_attack)
            print("=================================")
            pass

        return estimate, ext_state, debug_output
