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
        generator = sensor_bias_fault(start_t=5, sensor_idx=2, bias=1)
        self.assertEqual(generator(4, [0] * 6), [0] * 6)
        self.assertEqual(generator(5, [0] * 6), [0, 0, 1, 0, 0, 0])
        self.assertEqual(generator(6, [0] * 6), [0, 0, 1, 0, 0, 0])


    def test_intermittent_fault(self):
        generator = intermittent_fault(start_t=4, sensor_idx=3, anomaly_value=10, interval=3)
        self.assertEqual(generator(3, [0] * 6), [0, 0, 0, 0, 0, 0])
        self.assertEqual(generator(4, [0] * 6), [0, 0, 0, 10, 0, 0])
        self.assertEqual(generator(5, [0] * 6), [0, 0, 0, 0, 0, 0])
        self.assertEqual(generator(6, [0] * 6), [0, 0, 0, 0, 0, 0])
        self.assertEqual(generator(7, [0] * 6), [0, 0, 0, 10, 0, 0])
        self.assertEqual(generator(8, [0] * 6), [0, 0, 0, 0, 0, 0])
        self.assertEqual(generator(9, [0] * 6), [0, 0, 0, 0, 0, 0])
        self.assertEqual(generator(10, [0] * 6), [0, 0, 0, 10, 0, 0])




    def test_complete_failure(self):
        generator = complete_failure(start_t=3, sensor_idx=1, failure_value=0)
        self.assertEqual(generator(2, [1] * 6), [1] * 6)
        self.assertEqual(generator(3, [1] * 6), [1, 0, 1, 1, 1, 1])

    def test_drift_fault(self):
        generator = drift_fault(start_t=2, sensor_idx=4, drift_rate=0.5)
        output_t2 = generator(2, [0] * 6)
        self.assertEqual(output_t2, [0] * 6)
        output_t3 = generator(3, [0] * 6)
        self.assertEqual(output_t3, [0, 0, 0, 0, 0.5, 0])
        output_t4 = generator(4, [0] * 6)
        self.assertEqual(output_t4, [0, 0, 0, 0, 1, 0])
        output_t5 = generator(5, [0] * 6)
        self.assertEqual(output_t5, [0, 0, 0, 0, 1.5, 0])

    def test_random_noise_fault(self):
        generator = random_noise_fault(start_t=1, sensor_idx=5, noise_level=5)
        self.assertEqual(generator(0, [0] * 6), [0] * 6)
        output_t1 = generator(1, [0] * 6)
        self.assertNotEqual(output_t1[5], 0)
        self.assertEqual(output_t1[:5], [0] * 5)

    def test_spike_fault(self):
        generator = spike_fault(start_t=6, sensor_idx=0, spike_value=3, duration=2)
        self.assertEqual(generator(5, [0] * 6), [0] * 6)
        self.assertEqual(generator(6, [0] * 6), [3, 0, 0, 0, 0, 0])
        self.assertEqual(generator(7, [0] * 6), [3, 0, 0, 0, 0, 0])
        self.assertEqual(generator(8, [0] * 6), [0] * 6)

if __name__ == "__main__":
    unittest.main()