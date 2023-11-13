import unittest
from fault_generators import (
    sensor_bias_fault,
    intermittent_fault,
    complete_failure,
    drift_fault,
    random_noise_fault,
    spike_fault,
)

class TestAttackGenerators(unittest.TestCase):
    def test_sensor_bias_fault(self):
        generator = sensor_bias_fault(start_t=5, bias=1, sensor_idx=2)
        output = [0] * 6
        for t in range(10):
            output = generator(t, output)
        self.assertEqual(output[2], 1 * (10 - 5 - 1))

    def test_intermittent_fault(self):
        generator = intermittent_fault(
            start_t=4, anomaly_value=10, interval=2, sensor_idx=3
        )
        output = [0] * 6
        for t in range(10):
            output = generator(t, output)
        self.assertEqual(output[3], 10)

    def test_complete_failure(self):
        generator = complete_failure(start_t=3, failure_value=0, sensor_idx=1)
        output = [0] * 6
        for t in range(10):
            output = generator(t, output)
        self.assertEqual(output[1], 0)

    def test_drift_fault(self):
        generator = drift_fault(start_t=2, drift_rate=0.5, sensor_idx=4)
        output = [0] * 6
        for t in range(10):
            output = generator(t, output)
        expected_drift = sum(0.5 * (t - 2) for t in range(2, 10))
        self.assertAlmostEqual(output[4], expected_drift, places=3)

    def test_random_noise_fault(self):
        generator = random_noise_fault(start_t=1, noise_level=5, sensor_idx=5)
        output = [0] * 6
        for t in range(10):
            output = generator(t, output)
        self.assertNotEqual(output[5], 0)

    def test_spike_fault(self):
        generator = spike_fault(
            start_t=6, spike_value=3, duration=2, sensor_idx=0
        )
        output = [0] * 6
        for t in range(10):
            output = generator(t, output)
        self.assertEqual(output[0], 3)

if __name__ == "__main__":
    unittest.main()