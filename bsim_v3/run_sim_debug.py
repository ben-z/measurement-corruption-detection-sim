# %%

import numpy as np
from run_sim import run_single_simulation, create_fault_fn

dt = 0.01
fault_fn, fault_params = create_fault_fn("bias", 3)
fault_generators = [
    fault_fn(10 / dt)
]
detector_eps = np.array([1.5, 1.5, 0.3, 1.5, 1.5, 0.3])


res = run_single_simulation(dt, fault_generators, detector_eps, sim_time=15)
