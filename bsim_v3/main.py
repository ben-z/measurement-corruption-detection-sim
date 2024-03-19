import matplotlib.pyplot as plt
import numpy as np

from lib.plants.kinematic_bicycle import KinematicBicycle5StateRearWheelRefPlant
from lib.sensors.kinematic_bicycle_race_day import KinematicBicycleRaceDaySensor

plant = KinematicBicycle5StateRearWheelRefPlant([0, 0, 0, 0, 0], 0.1)
sensor = KinematicBicycleRaceDaySensor()

# Set the inputs
plant.set_inputs([0.5, 0.1])

# Simulate the plant
x = []
z = []
for _ in range(100):
    plant.next()
    x.append(plant.get_state())
    z.append(sensor.get_output(plant.get_state()))

x = np.array(x)

# Plot the results
plt.figure()
plt.plot(x[:, 0], x[:, 1], label="True")
plt.plot([z_[0] for z_ in z], [z_[1] for z_ in z], label="Measured")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Trajectory")
plt.axis("equal")
plt.show()
