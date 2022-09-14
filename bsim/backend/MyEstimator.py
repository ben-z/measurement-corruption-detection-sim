from utils import calc_input_effects_on_output, optimize_l1
import continuous_kinematic_bicycle as ckb
import numpy as np
from numpy.linalg import eig, matrix_power, norm
import time

class MyEstimator:
    def __init__(self, L, T=200, ticks_per_solve=200):
        """
        T: time horizon, the number of time steps
        ticks_per_solve: number of ticks between solves
        """
        self.T = T
        self.ticks_per_solve = ticks_per_solve
        self._tick_count = 0
        self.model = ckb.make_continuous_kinematic_bicycle_model(L)
        self._measurements = np.zeros((self.model.noutputs, T))
        self._inputs = np.zeros((self.model.ninputs, T))
        self._true_states = np.zeros((self.model.nstates, T))

    def tick(self, _ext_state, measurement, prev_inputs, true_state=None):
        """
        _ext_state: state managed by the client. This is not used because we
                    can keep state ourselves in the object.
        measurement: the measurement from the sensors.
        prev_inputs: the inputs from the previous time step.
        true_state: the true state, used for debugging (e.g. evaluating estimation error).
        """
        self._tick_count += 1

        estimate = measurement
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
            # TODO: try linearizing at a set point, also change the actual model to that
            linsys = self.model.linearize(true_state, [0, 0])
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
            
            input_effects = calc_input_effects_on_output(A, B, C, self._inputs)

            measurements_without_input_effects = self._measurements - input_effects
            l1_start = time.time()
            prob_l1, x0_hat_l1 = optimize_l1(
                n, p, self.T, Phi, np.reshape(np.transpose(measurements_without_input_effects), (p*self.T,)))
            l1_end = time.time()
            
            print("status:", prob_l1.status)
            print("optimal value", prob_l1.value)
            print("optimal var", x0_hat_l1.value)
            print("true var", self._true_states[:,0])
            print(f"dist to true var: {np.linalg.norm(self._true_states[:,0] - x0_hat_l1.value):.6f}")
            print(f"l1 solve time: {l1_end - l1_start:.4f}s")
            print("=================================")
            pass

        return estimate, _ext_state, debug_output
