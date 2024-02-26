import random

# Fault generators. Derived from the following:
# - https://chat.openai.com/share/51cabea9-ebf9-4ae9-85fa-0285eac496bc

# Defining attack generators
def sensor_bias_fault(start_t, sensor_idx, bias):
    def attack_generator(t, output):
        if t >= start_t:
            output[sensor_idx] += bias
        return output

    return attack_generator


def intermittent_fault(start_t, sensor_idx, anomaly_value, interval):
    def attack_generator(t, output):
        if t >= start_t and int(t - start_t) % interval == 0:
            output[sensor_idx] = anomaly_value
        return output

    return attack_generator


def complete_failure(start_t, sensor_idx, failure_value):
    def attack_generator(t, output):
        if t >= start_t:
            output[sensor_idx] = failure_value
        return output

    return attack_generator


def drift_fault(start_t, sensor_idx, drift_rate):
    def attack_generator(t, output):
        if t >= start_t:
            output[sensor_idx] += drift_rate * (t - start_t)
        return output

    return attack_generator


def random_noise_fault(start_t, sensor_idx, noise_level):
    def attack_generator(t, output):
        if t >= start_t:
            output[sensor_idx] += random.uniform(-noise_level, noise_level)
        return output

    return attack_generator


def spike_fault(start_t, sensor_idx, spike_value, duration):
    def attack_generator(t, output):
        if start_t <= t < start_t + duration:
            output[sensor_idx] += spike_value
        return output

    return attack_generator

def noop():
    def attack_generator(t, output):
        return output
    return attack_generator