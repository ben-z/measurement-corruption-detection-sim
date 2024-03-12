import matplotlib.pyplot as plt
import numpy as np

from plants.kinematic_bicycle import KinematicBicycle5StateRearWheelRef

plant = KinematicBicycle5StateRearWheelRef([0, 0, 0, 0, 0], 0.1)

# Set the inputs
plant.set_inputs(0.5, 0.1)

# Simulate the plant
x = []
for _ in range(100):
    plant.next()
    x.append(plant.get_state())

x = np.array(x)

# Plot the results
plt.figure()
plt.plot(x[:, 0], x[:, 1])
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Trajectory")
plt.axis("equal")
plt.show()

