import random

# Fault generators. Derived from the following:
# - https://chat.openai.com/share/51cabea9-ebf9-4ae9-85fa-0285eac496bc

# Defining attack generators
def sensor_bias_fault(start_t, bias, sensor_idx):
    def attack_generator(t, output):
        if t > start_t:
            output[sensor_idx] += bias
        return output

    return attack_generator


def intermittent_fault(start_t, anomaly_value, interval, sensor_idx):
    def attack_generator(t, output):
        if t > start_t and int(t) % interval == 0:
            output[sensor_idx] = anomaly_value
        return output

    return attack_generator


def complete_failure(start_t, failure_value, sensor_idx):
    def attack_generator(t, output):
        if t > start_t:
            output[sensor_idx] = failure_value
        return output

    return attack_generator


def drift_fault(start_t, drift_rate, sensor_idx):
    def attack_generator(t, output):
        if t > start_t:
            output[sensor_idx] += drift_rate * (t - start_t)
        return output

    return attack_generator


def random_noise_fault(start_t, noise_level, sensor_idx):
    def attack_generator(t, output):
        if t > start_t:
            output[sensor_idx] += random.uniform(-noise_level, noise_level)
        return output

    return attack_generator


def spike_fault(start_t, spike_value, duration, sensor_idx):
    def attack_generator(t, output):
        if start_t <= t < start_t + duration:
            if "spike_added" not in attack_generator.__dict__:
                output[sensor_idx] += spike_value
                attack_generator.spike_added = True
        return output

    return attack_generator
