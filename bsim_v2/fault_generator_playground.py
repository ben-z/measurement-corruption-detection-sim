# This file is used to test the fault generators in fault_generators.py
# %%
import numpy as np
import matplotlib.pyplot as plt
from fault_generators import (
    sensor_bias_fault,
    intermittent_fault,
    complete_failure,
    drift_fault,
    random_noise_fault,
    spike_fault,
)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
np.set_printoptions(suppress=True)

# %%

# Plot example faults
ts = np.linspace(0, 10, 1000)
plt.figure()
# bias_generator = sensor_bias_fault(3, 0, 2)
# plt.plot(ts, [bias_generator(t, [0])[0] for t in ts], label="Sensor Bias")
# plt.show()

# drift_generator = drift_fault(3, 0, 2)
# plt.plot(ts, [drift_generator(t, [0])[0] for t in ts], label="Drift Fault")
# plt.show()

# noise_generator = random_noise_fault(3, 0, 2)
# plt.plot(ts, [noise_generator(t, [0])[0] for t in ts], label="Random Noise Fault")
# plt.show()

# spike_generator = spike_fault(3, 0, 2, 2)
# plt.plot(ts, [spike_generator(t, [0])[0] for t in ts], label="Spike Fault")
# plt.show()

fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Plot sensor bias
bias_mag = 2
bias_generator = sensor_bias_fault(2, 0, bias_mag)
axs[0].plot(ts, [bias_generator(t, [0])[0] for t in ts])
axs[0].set_title(f"Bias (magnitude = {bias_mag})")

# Plot spike fault
spike_mag = 2
spike_duration = 0.1
spike_generator = spike_fault(2, 0, spike_mag, spike_duration)
axs[1].plot(ts, [spike_generator(t, [0])[0] for t in ts])
axs[1].set_title(f"Spike (magnitude = {spike_mag}, duration = {spike_duration} s)")

# Plot random noise fault
noise_mag = 2
noise_generator = random_noise_fault(2, 0, noise_mag)
axs[2].plot(ts, [noise_generator(t, [0])[0] for t in ts])
axs[2].set_title(f"Random Noise (magnitude = {noise_mag})")

# Plot drift fault
drift_rate = 2
drift_generator = drift_fault(2, 0, drift_rate)
axs[3].plot(ts, [drift_generator(t, [0])[0] for t in ts])
axs[3].set_title(f"Drift (rate = {drift_rate})")

plt.tight_layout()
plt.xlabel("Time (s)")
fig.text(0, 0.5, "Fault Magnitude", va="center", rotation="vertical")
plt.savefig("fault-examples.pdf", format="pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()

# intermittent_generator = intermittent_fault(3, 0, 2, 2)
# plt.plot(ts, [intermittent_generator(t, [0])[0] for t in ts], label="Intermittent Fault")
# plt.show()

# failure_generator = complete_failure(3, 0, 2)
# plt.plot(ts, [failure_generator(t, [0])[0] for t in ts], label="Complete Failure")
# plt.show()
