import random
import numpy as np
from collections import deque

# Fault generators. Derived from the following:
# - https://chat.openai.com/share/51cabea9-ebf9-4ae9-85fa-0285eac496bc
# - https://chatgpt.com/share/679b38d0-fe14-8010-a076-9eb3a4d88d7a

# Defining attack generators
def sensor_bias_fault(start_t, sensor_idx, bias):
    def fault_generator(t, output):
        if t >= start_t:
            output[sensor_idx] += bias
        return output

    return fault_generator


def intermittent_fault(start_t, sensor_idx, anomaly_value, interval):
    def fault_generator(t, output):
        if t >= start_t and int(t - start_t) % interval == 0:
            output[sensor_idx] = anomaly_value
        return output

    return fault_generator


def complete_failure(start_t, sensor_idx, failure_value):
    def fault_generator(t, output):
        if t >= start_t:
            output[sensor_idx] = failure_value
        return output

    return fault_generator


def drift_fault(start_t, sensor_idx, drift_rate):
    def fault_generator(t, output):
        if t >= start_t:
            output[sensor_idx] += drift_rate * (t - start_t)
        return output

    return fault_generator


def random_noise_fault(start_t, sensor_idx, noise_level):
    def fault_generator(t, output):
        if t >= start_t:
            output[sensor_idx] += random.uniform(-noise_level, noise_level)
        return output

    return fault_generator


def spike_fault(start_t, sensor_idx, spike_value, duration):
    def fault_generator(t, output):
        if start_t <= t < start_t + duration:
            output[sensor_idx] += spike_value
        return output

    return fault_generator


def stuck_at_fault(start_t, sensor_idx, stuck_value):
    """Sensor gets stuck at a constant value after start_t."""

    def fault_generator(t, output):
        if t >= start_t:
            output[sensor_idx] = stuck_value
        return output

    return fault_generator


def scaling_fault(start_t, sensor_idx, scale_factor):
    """Sensor outputs values consistently scaled after start_t."""

    def fault_generator(t, output):
        if t >= start_t:
            output[sensor_idx] *= scale_factor
        return output

    return fault_generator


def delayed_fault(start_t, sensor_idx, delay_steps):
    """Sensor reports old values, creating a delay in response."""
    buffer = deque([0] * delay_steps, maxlen=delay_steps)

    def fault_generator(t, output):
        buffer.append(output[sensor_idx])
        if t >= start_t:
            output[sensor_idx] = buffer[0]  # Output delayed value
        return output

    return fault_generator


def intermittent_dropout(start_t, sensor_idx, period, duration):
    """Sensor turns off for `duration` time steps every `period` steps."""

    def fault_generator(t, output):
        if t >= start_t and ((t - start_t) % period) < duration:
            output[sensor_idx] = None  # Simulating sensor dropout
        return output

    return fault_generator


def correlated_fault(start_t, sensor_idxs, drift_rate):
    """Multiple sensors experience a similar drift together."""

    def fault_generator(t, output):
        if t >= start_t:
            for idx in sensor_idxs:
                output[idx] += drift_rate * (t - start_t)
        return output

    return fault_generator


def oscillation_fault(start_t, sensor_idx, amplitude, frequency):
    """Sensor oscillates at a given frequency and amplitude."""

    def fault_generator(t, output):
        if t >= start_t:
            output[sensor_idx] += amplitude * np.sin(2 * np.pi * frequency * t)
        return output

    return fault_generator


def awgn_fault(start_t, sensor_idx, noise_std):
    """Adds additive white Gaussian noise (AWGN) to the sensor output."""

    def fault_generator(t, output):
        if t >= start_t:
            output[sensor_idx] += np.random.normal(0, noise_std)
        return output

    return fault_generator


def drifting_with_recovery(start_t, sensor_idx, drift_rate, recovery_time):
    """Sensor drifts but recovers over `recovery_time` seconds."""

    def fault_generator(t, output):
        if t >= start_t:
            recovery_factor = max(0, recovery_time - (t - start_t))
            output[sensor_idx] += drift_rate * recovery_factor
        return output

    return fault_generator


def noop():
    def fault_generator(t, output):
        return output
    return fault_generator
