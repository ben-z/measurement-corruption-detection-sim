import numpy as np

#################################################################
# General utility functions
#################################################################


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def clamp(x, lower, upper):
    return np.maximum(lower, np.minimum(x, upper))
