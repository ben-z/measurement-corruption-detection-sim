import numpy as np
from numbers import Number

#################################################################
# General utility functions
#################################################################


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


Scalar = (
    Number
    | bool
    | int
    | float
    | complex
    | str
    | bytes
    | memoryview
    | None
)
