import numpy as np
from math import sqrt


def walk_trajectory_by_durations(path_points, velocities, starting_idx, durations):
    """
    Walks a trajectory by the given durations. Returns the indices of the path points that were travelled to.
    Parameters:
        path_points: list[tuple[float, float]] - list of path points denoting x and y coordinates over time
        velocities: list[float] - list of velocities at each path point
        starting_idx: int - the index of the starting path point
        durations: list[float] - list of durations to travel for each segment. This is non-cumulative.
    Returns:
        indices: list[int] - list of indices of path points that were travelled to
    """
    assert len(path_points) == len(
        velocities
    ), f"path_points and velocities must have the same length. {len(path_points)=}, {len(velocities)=}"
    assert starting_idx >= 0 and starting_idx < len(
        path_points
    ), "starting_idx must be a valid index"
    assert len(durations) > 0, "Must have a duration"
    assert all(d >= 0 for d in durations), "durations must be non-negative"

    idx = starting_idx - 1
    remaining_segment_duration = 0.0
    indices = []
    for duration in durations:
        # first travel the untravelled portion of the current segment
        remaining_duration = max(0.0, duration - remaining_segment_duration)
        remaining_segment_duration -= duration - remaining_duration

        # at least one of two scenarios are possible here:
        # - we have used up the remaining segment duration, thus go into the loop to go onto the next segment
        # - we have not used up te remaining segment duration, so we stay on the current segment.
        # If both are true, we arbitrarily choose to record the existing segment index instead of the new one.
        # This is okay because we are already doing floating point calculations (minor errors are tolerable).
        while remaining_duration > 1e-15:
            # if this assertion is violated, then we have travelled more than the remaining duration, which is not possible
            assert (
                -1e-15 < remaining_segment_duration < 1e-15
            ), "remaining segment duration must be zero in the beginning of this loop"
            if remaining_segment_duration < 1e-15:
                # we have used up the remaining segment duration, progress onto the next segment
                idx = (idx + 1) % len(path_points)

            segment_dist = sqrt(
                (path_points[(idx + 1) % len(path_points)][0] - path_points[idx][0])
                ** 2
                + (path_points[(idx + 1) % len(path_points)][1] - path_points[idx][1])
                ** 2
            )
            segment_duration = segment_dist / velocities[idx]

            remaining_segment_duration = max(0, segment_duration - remaining_duration)
            remaining_duration -= segment_duration - remaining_segment_duration

        indices.append(idx)
    return indices


def calc_input_effects_on_output(As, Bs, Cs, inputs):
    """
    calculates the input effects on the output
    Parameters:
    As: numpy.ndarray[] - list of N-1 matrices of size (n,n)
    Bs: numpy.ndarray[] - list of N-1 matrices of size (n,p)
    Cs: numpy.ndarray[] - list of N matrices of size (q,n)
    inputs: numpy.ndarray[] - list of N-1 input vectors of size p
    Returns:
    effects: numpy.ndarray[] - the effects of the inputs on the output, a list of N matrices of size (q,)
    """

    n = As[0].shape[0]
    N = len(Cs)
    assert len(As) == N - 1, f"As must have length N-1 ({N-1}). Got {len(As)}"
    assert len(Bs) == N - 1, f"Bs must have length N-1 ({N-1}). Got {len(Bs)}"
    assert len(inputs) == N - 1, f"inputs must have length N-1 ({N-1}). Got {len(inputs)}"

    # Algorithm (for LTV systems):
    # zeros
    # B[0]u[0]
    # B[1]u[1] + A[0](B[0]u[0])
    # B[2]u[2] + A[1](B[1]u[1] + A[0](B[0]u[0]))
    # ...
    # Then pass everything through C
    state_effects = [np.zeros((n,)), np.matmul(Bs[0], inputs[0])]
    for i in range(1, N - 1):
        state_effects.append(
            np.matmul(Bs[i], inputs[i]) + np.matmul(As[i - 1], state_effects[-1])
        )

    effects = [Cs[i] @ state_effects[i] for i in range(N)]

    return effects


def get_state_evolution_tensor(As: list[np.ndarray]):
    """
    Given a list of $N-1$ state transition matrices, returns a tensor of size $(N,n,n)$ that describes the evolution of the state over time,
    where $n$ is the size of the state. The first element of the tensor is the identity matrix.
    Parameters:
        As: list[np.ndarray] - list of state transition matrices, size (n, n) each, where n is the number of states
    Returns:
        Evo: numpy.ndarray - tensor of size $(N, n, n)$ that describes the evolution of the state over time. When multiplied with the
            initial state, it produces the state at each time step.
    """
    assert len(As) > 0, "Must have at least one state transition matrix"

    N = len(As) + 1
    n = As[0].shape[0]
    Evo = np.zeros((N, n, n))
    Evo[0] = np.eye(n)
    for i in range(1, N):
        Evo[i] = np.matmul(As[i - 1], Evo[i - 1])
    return Evo


def get_output_evolution_tensor(Cs: list[np.ndarray], Evo: np.ndarray):
    """
    Given a list of $N$ output matrices and the state evolution tensor, returns a tensor of size
    $(N,q,n)$ that describes the evolution of the output over time, where $q$ is the number of
    outputs and $n$ is the number of states.

    Parameters:
        Cs: list[np.ndarray] - list of output matrices, size (q, n) each, where q is the number of outputs and n is the number of states
        Evo: numpy.ndarray - tensor of size $(N, n, n)$ that describes the evolution of the state over time. When multiplied with the
            initial state, it produces the state at each time step.
    Returns:
        Phi: numpy.ndarray - tensor of size $(N, q, n)$ that describes the evolution of the output over time.
    """
    assert len(Cs) > 0, "Must have at least one output matrix"
    assert Evo.shape[1] == Evo.shape[2], f"State evolution matrices must be square. Got {Evo.shape}"
    N = len(Cs)
    assert (
        N == Evo.shape[0]
    ), "The number of output matrices must be equal to the size of the first dimension of the state evolution tensor"
    q = Cs[0].shape[0]
    n = Cs[0].shape[1]
    assert (
        n == Evo.shape[1]
    ), "The number of states must match between the output matrices and the state evolution tensor"

    Phi = np.zeros((N, q, n))

    for i in range(N):
        Phi[i] = np.matmul(Cs[i], Evo[i])

    return Phi

def calc_invalid_spans(validities, idx):
    """
    Given a list of validities, returns the spans of invalid values.
    Parameters:
        validities: list[bool] - list of validities
        idx: int - the index to check
    Returns:
        spans: list[tuple[int, int]] - list of tuples of start and end indices of invalid spans
    """
    ret = []

    start = None
    for i, v in enumerate(validities):
        if v[idx] == False:
            if start is None:
                start = i
        else:
            if start is not None:
                ret.append((start, i))
                start = None
    if start is not None:
        ret.append((start, len(validities)))

    return ret

