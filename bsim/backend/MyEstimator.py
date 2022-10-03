import control
import continuous_kinematic_bicycle as ckb
import numpy as np
from numpy.linalg import eig, matrix_power, norm
import time
from utils import calc_input_effects_on_output, optimize_l1, optimize_l0, distance_to_line_segment, wrap_to_pi

np.set_printoptions(suppress=True, precision=4)

def calc_desired_state_trajectory(linearization_state, T, dt):
    """
    returns a n-by-T matrix of desired state trajectory.
    """
    x0, y0, theta0, v0, delta0 = linearization_state
    n = len(linearization_state)

    desired_trajectory = np.zeros((n, T))
    for t in range(T):
        desired_trajectory[:, t] = np.array([
            v0*np.cos(theta0)*t*dt + x0,
            v0*np.sin(theta0)*t*dt + y0,
            theta0,
            v0,
            delta0
        ])
    
    return desired_trajectory


class MyEstimator:
    def __init__(self, L, dt, T=2, ticks_per_solve=50):
        """
        T: time horizon, the number of time steps
        ticks_per_solve: number of ticks between solves
        """
        self.T = T
        self.dt = dt
        self.ticks_per_solve = ticks_per_solve
        self._tick_count = 0
        self.L = L
        self.model = ckb.make_continuous_kinematic_bicycle_model(L)
        self._measurements = np.zeros((self.model.noutputs, T))
        self._inputs = np.zeros((self.model.ninputs, T))
        self._true_states = np.zeros((self.model.nstates, T))

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

        if self._tick_count >= self.T and self._tick_count % self.ticks_per_solve == 0:
            # solve the estimation problem
            # TODO: use measurement instead of true_state. However, we don't know what the
            # true state is without first solving the estimation problem. This seems like a
            # chicken and egg problem.
            print('solving estimation problem')
            # TODO: use something other than the true state as the initial guess, perhaps the previous estimate
            # projected forward in time or the target state along the path (the latter requires a feedback from
            # the controller)?
            # Linearize, assuming the reference is a line. (See 2022-09-15 and 2022-09-22 notes for derivations)
            target_path = ext_state['target_path']
            # TODO: use estimated x and y instead of true x and y
            x = true_state[0]
            y = true_state[1]

            path_segments = np.stack(
                [target_path, np.roll(target_path, -1, axis=0)], axis=1)

            current_path_segment_index = np.argmin([
                distance_to_line_segment(np.array([x, y]), p1, p2) for p1, p2 in path_segments
            ])
            current_path_segment = path_segments[current_path_segment_index]
            current_path_heading = np.arctan2(
                current_path_segment[1, 1] - current_path_segment[0, 1], current_path_segment[1, 0] - current_path_segment[0, 0])

            linearization_state = np.array([x, y, current_path_heading, ext_state['target_speed'], 0])
            # linearization_state = np.array([0, 0, current_path_heading, true_state[4], 0])
            linearization_input = np.zeros(self.model.ninputs)
            # FIXME: these linearizations should be the same. Verify that they are and remove the duplicate code.
            # linsys = control.sample_system(self.model.linearize(linearization_state, linearization_input), self.dt)
            # A = linsys.A
            # B = linsys.B
            # C = linsys.C
            Ac, Bc, Cc, Dc = ckb.get_linear_model_straight_line_ref(*linearization_state, *linearization_input, L=self.L)
            linsys = control.sample_system(control.ss(Ac,Bc,Cc,Dc), self.dt)
            A = linsys.A
            B = linsys.B
            C = linsys.C
            n = A.shape[0]
            p = C.shape[0]
            Phi = np.zeros((p*self.T, n))
            for i in range(self.T):
                row_begin = i * p
                row_end = row_begin + p
                Phi[row_begin:row_end, :] = np.matmul(C, matrix_power(A, i))
            
            # input_effects has p rows and T columns
            input_effects = calc_input_effects_on_output(A, B, C, self._inputs)
            desired_trajectory = C @ calc_desired_state_trajectory(linearization_state, self.T, self.dt)
            # linearization_traj = self.model.dynamics(0, linearization_state, linearization_input)
            # trajectory_effects = ((C @ linearization_traj) * self.dt).reshape((p, 1)) @ np.arange(0, self.T).reshape((1, self.T))
            # measurements = self._measurements - input_effects - trajectory_effects - (C @ linearization_state).reshape((p, 1))
            measurements = self._measurements - input_effects - desired_trajectory

            # normalize angular measurements
            measurements[ckb.get_angular_outputs_mask(), :] = wrap_to_pi(measurements[ckb.get_angular_outputs_mask(), :])

            # normalize measurements
            # Y is measurements stacked vertically
            Y = np.reshape(measurements, (p*self.T,), order='F')

            solve_start = time.time()
            # prob_l1, x0_hat_l1 = optimize_l0(n, p, self.T, Phi, Y, [0.15, 0.15, 0.035, 0.1, 0.3, 0.15, 0.15, 0.1])
            prob_l1, x0_hat_l1 = optimize_l1(n, p, self.T, Phi, Y)
            solve_end = time.time()
            
            diff_from_true = ckb.normalize_state(
                x0_hat_l1.value + linearization_state - self._true_states[:, 0])
            
            print("status:", prob_l1.status)
            print("optimal value", prob_l1.value)
            print("optimal var (x0)",
                  x0_hat_l1.value + linearization_state)
            print("true var (x0)", self._true_states[:,0])
            print("opt - true (x0)", diff_from_true)
            print(f"dist to true var (x0): {norm(diff_from_true):.4f}")
            print(f"solve time: {solve_end - solve_start:.4f}s")

            # attack_vector is a pxT matrix
            attack_vector = (Y - np.matmul(Phi, x0_hat_l1.value)).reshape((p, self.T), order='F')
            mean_attack_vector = np.mean(attack_vector, axis=1)
            sensors_under_attack = np.abs(mean_attack_vector) > 0.05 # some arbitrary threshold, above which the sensor is faulty, below which is modelling error/noise, can set this per-sensor using experimental data.
            num_sensors_under_attack = np.sum(sensors_under_attack)
            print("mean attack vector:", mean_attack_vector)
            print("Sensors under attack:", sensors_under_attack)
            print("Number of sensors under attack:", num_sensors_under_attack)
            print("=================================")
            pass

        return estimate, ext_state, debug_output
