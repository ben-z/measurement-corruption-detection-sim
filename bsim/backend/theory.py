import math
import numpy as np
from numpy import sin, cos, tan
from itertools import combinations
from utils import \
    powerset \
    , get_observability_matrix as old_get_observability_matrix \
    , s_sparse_observability as old_get_s_sparse_observability \
    , clamp, max_binomial_coeff, flatten
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "monospace",
    'text.latex.preamble': r'\usepackage{amsfonts}',
    # autolayout and constrained_layout seem to be similar
    # "figure.autolayout": True,
    "figure.constrained_layout.use": True,
})


RED = "tomato"
GREEN = "green"
LIGHT_GREEN = "limegreen"
BLUE = "cornflowerblue"
BLACK = "black"

def np_make_mask(n, I):
    """
    Creates a mask of size n with 1s at the positions in I.
    """
    mask = np.full(n, False)
    mask[list(I)] = True
    return mask

def toBinVec(S, n):
    """
    Converts a set S to a binary vector of length n.
    """
    return np_make_mask(n, S)

def toSet(binVec):
    """
    Converts a binary vector to a set.
    """
    return frozenset(np.where(binVec)[0])

def setToStr(S, special_empty_set=None):
    """
    Converts a set S to a string.
    """
    if special_empty_set and len(S) == 0:
        return special_empty_set

    inner = ",".join(str(s) for s in S)
    return f"$\\{{{inner}\\}}$"

def get_observability_matrix(A, C, N, S):
    """
    Calculates the observability matrix of a system with observability index N.
    Parameters:
        A: state transition matrix
        C: output matrix
        N: observability index (the number of time steps)
        S: {0,1}^p vector indicating which sensors are available (1 means available, 0 means missing)
    """

    n = A.shape[0]
    p = C.shape[0]

    # Matrix that sets the attacked sensors to zero
    I_S = np.diag(S)

    O = np.zeros((p*N, n))
    for i in range(N):
        O[i*p:(i+1)*p, :] = I_S @ C @ np.linalg.matrix_power(A,i)
    
    return O

def get_s_sparse_observability(A,C,N,P):
    """
    Calculates the s-sparse observability of a system.
    Arguments:
        A: state transition matrix
        C: output matrix
        N: observability index (the number of time steps)
        P: {0,1}^p vector indicating whether a sensor is protected (1 means protected, 0 means unprotected)
    """
    n = A.shape[0]
    p = C.shape[0]

    s = -1
    for K_tuple in powerset(range(p)):
        # K is a {0,1}^p vector indicating which sensors are missing
        K = np_make_mask(p, K_tuple)
        
        # Skip if a missing sensor is protected
        if np.any(np.logical_and(K, P)):
            continue

        # construct the sensor availability vector S
        S = np.ones(p)
        S[K] = 0

        O_K = get_observability_matrix(A, C, N, S)
        
        O_K_rank = np.linalg.matrix_rank(O_K)

        if O_K_rank < n:
            # found the first non-full-rank entry, which means
            # that the system no longer has full observability
            s = K.sum() - 1
            break

    if s == -1:
        # the system is fully observable
        s = p - P.sum()

    return s

def is_observable(A, C, N, S):
    """
    Checks if a system is observable.
    Parameters:
        A: state transition matrix
        C: output matrix
        N: observability index (the number of time steps)
        S: {0,1}^p vector indicating which sensors are available (1 means available, 0 means missing)
    """
    O = get_observability_matrix(A, C, N, S)
    return np.linalg.matrix_rank(O) == A.shape[0]

def get_s_sparse_observability2(A,C,N,P):
    """
    Calculates the s-sparse observability of a system.
    Arguments:
        A: state transition matrix
        C: output matrix
        N: observability index (the number of time steps)
        P: {0,1}^p vector indicating whether a sensor is protected (1 means protected, 0 means unprotected)
    Note: this is the same as get_s_sparse_observability, but it uses a different method
    """

    n = A.shape[0]
    p = C.shape[0]

    unprotected_sensors = np.where(P == 0)[0]

    # In each iteration, check if the system is observable when s_candidate sensors are missing.
    # If the system is observable, then the system has s_candidate-sparse observability.
    # If the system is not observable, then we return the largest s_candidate found.
    for s_candidate in range(1, len(unprotected_sensors)+1):
        # all combinations of s_candidate sensors that can go missing
        Ks = np.array([np_make_mask(p, setK) for setK in combinations(unprotected_sensors, s_candidate)])

        observable_mask = np.array([is_observable(A, C, N, ~K) for K in Ks])

        if not np.all(observable_mask):
            # found the first non-full-rank entry, which means
            # that the system no longer has full observability
            s = s_candidate - 1
            # sensors that when removed, the system is no longer observable
            important_Ks = Ks[~observable_mask]
            # sensors that when removed, the system is still observable
            unimportant_Ks = Ks[observable_mask]
            break
    else:
        # the system is fully sparse-observable, i.e. any unprotected sensor can be removed
        # without affecting the observability of the system
        s = len(unprotected_sensors)
        important_Ks = np.array([])
        unimportant_Ks = np.array([])
    
    return s, important_Ks, unimportant_Ks

def get_s_sparse_observability2_visualization_data(A,C,N,P):
    """
    Visualizes the s-sparse observability of a system.
    Arguments:
        A: state transition matrix
        C: output matrix
        N: observability index (the number of time steps)
        P: {0,1}^p vector indicating whether a sensor is protected (1 means protected, 0 means unprotected)
    Note: this is the same as get_s_sparse_observability2, but the code is structured such that
    it can be used to visualize the entire s-sparse observability of a system.
    """

    n = A.shape[0]
    q = C.shape[0]

    unprotected_sensors = np.where(P == 0)[0]

    # Initialize s to an invalid value
    s = None
    important_Ks_list = []
    unimportant_Ks_list = []
    importance_scores = np.zeros(q)

    for s_candidate in range(0, q+1):
        # all combinations of s_candidate sensors that can go missing
        Ks = np.array([np_make_mask(q, setK) for setK in combinations(unprotected_sensors, s_candidate)]).reshape(-1, q)

        observable_mask = np.array([is_observable(A, C, N, ~K) for K in Ks], dtype=bool)

        # sensors that when removed, the system is no longer observable
        important_Ks_list.append(Ks[~observable_mask])
        # sensors that when removed, the system is still observable
        unimportant_Ks_list.append(Ks[observable_mask])

        if len(Ks) and np.all(observable_mask):
            # the system is observable with s_candidate sensors missing
            s = s_candidate
        elif s is None:
            s = s_candidate - 1
    
    assert s is not None, "This scenario is impossible when q > 0" # make type checker happy

    for important_K in flatten(important_Ks_list):
        if len(important_K) == 0:
            continue

        assert len(important_K) == q, "important_K must have length 0 or q"

        importance_scores += important_K

    return s, important_Ks_list, unimportant_Ks_list, importance_scores

def np_is_row(X, row):
    """
    Checks if a row exists in a numpy array.
    """
    return np.any(np.all(X == row, axis=1))

def convert_set_indices_for_paper(set_indices):
    """
    Converts a set of indices from 0-based to 1-based.
    """
    return set([i+1 for i in set_indices])

def visualize_s_sparse_observability(A,C,N,P,output_filename,show_title=True):
    visualization_data = get_s_sparse_observability2_visualization_data(A,C,N,P)
    s, important_Ks_list, unimportant_Ks_list, importance_scores = visualization_data

    GENERIC_ANNOTATION_FONTSIZE = 12
    CIRCLE_RADIUS = 0.3
    CIRCLE_ANNOTATION_FONT_SIZE_UNITS_PER_INCH = 100 # hand-tuned
    MAX_CIRCLE_ANNOTATION_FONT_SIZE = CIRCLE_RADIUS / 2 * CIRCLE_ANNOTATION_FONT_SIZE_UNITS_PER_INCH
    MIN_CIRCLE_ANNOTATION_FONT_SIZE = CIRCLE_RADIUS / 5 * CIRCLE_ANNOTATION_FONT_SIZE_UNITS_PER_INCH

    p = C.shape[0]

    plot_width = math.comb(p, math.ceil(p/2)) # maximum number of sensor combinations for any s
    xlim = (-1, plot_width)
    xlim_length = xlim[1] - xlim[0]
    ylim = (-0.8, p+0.8+0.3) # +0.3 to account for the legend
    ylim_length = ylim[1] - ylim[0]
    yticks = np.arange(0, p+1)

    # THe size of the plot is experimentally determined
    fig = plt.figure(figsize=(xlim_length, ylim_length))
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel('number of missing sensors', fontsize=GENERIC_ANNOTATION_FONTSIZE)
    ax.invert_yaxis()
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    if show_title:
        ax.set_title(f"The system's tolerance to missing sensors ($\\mathbb{{P}}=\\{{{','.join(str(p) for p in toSet(P))}\\}}$)")

    has_important = False
    has_unimportant = False
    has_impossible = False

    # Can also calculate this using max_r(nCr) where n is the number of sensors and r is in {0, ..., n}
    # max_num_cases = max(len(iK) + len(uK) for iK,uK in zip(important_Ks_list, unimportant_Ks_list))
    max_num_cases = max_binomial_coeff(p)

    for s_candidate, (important_Ks, unimportant_Ks) in enumerate(zip(important_Ks_list, unimportant_Ks_list)):
        Ks = np.array([np_make_mask(p, setK) for setK in combinations(range(p), s_candidate)])
        num_cases = len(Ks)
        left_offset = (max_num_cases - num_cases) / 2

        for i, K in enumerate(Ks):
            circle_color = None
            if np_is_row(important_Ks, K):
                # this sensor combination is important
                circle_color = RED
                has_important = True
            elif np_is_row(unimportant_Ks, K):
                # this sensor combination is unimportant
                circle_color = LIGHT_GREEN
                has_unimportant = True
            else:
                # this sensor combination is impossible
                circle_color = BLUE
                has_impossible = True
            circle = mpatches.Circle((i+left_offset, s_candidate), CIRCLE_RADIUS, color=circle_color)
            ax.add_patch(circle)
            
            # Draw the sensor combination
            K_set = convert_set_indices_for_paper(toSet(K))
            K_str = setToStr(K_set, special_empty_set="$\\emptyset$")
            fontsize = clamp(
                CIRCLE_RADIUS / max(1, len(K_set)) * CIRCLE_ANNOTATION_FONT_SIZE_UNITS_PER_INCH,
                MIN_CIRCLE_ANNOTATION_FONT_SIZE,
                MAX_CIRCLE_ANNOTATION_FONT_SIZE
            )
            ax.annotate(K_str, (i+left_offset, s_candidate),
                        ha='center', va='center',
                        fontsize=fontsize)
            pass

    # Draw legend
    legend_handles = []
    if has_unimportant:
        legend_handles.append(mpatches.Patch(color=LIGHT_GREEN, label='removal does not affect observability'))
    if has_important:
        legend_handles.append(mpatches.Patch(color=RED, label='removal makes the system unobservable'))
    if has_impossible:
        legend_handles.append(mpatches.Patch(color=BLUE, label="impossible scenario (contains protected sensors)"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=GENERIC_ANNOTATION_FONTSIZE)

    # Draw a line to indicate the s value
    ax.plot(xlim, [s+0.5, s+0.5], color=BLACK, linestyle='--')
    ax.arrow(xlim[0]+0.2, s+0.5, 0, -0.5, head_width=0.1, head_length=0.1, facecolor=BLACK, edgecolor=BLACK)
    ax.arrow(xlim[1]-0.2, s+0.5, 0, -0.5, head_width=0.1, head_length=0.1, facecolor=BLACK, edgecolor=BLACK)
    ax.text(xlim_length/2+xlim[0], s+0.5, f"the system is maximally {s}-sparse observable", ha='center', va='bottom', color=BLACK, fontsize=GENERIC_ANNOTATION_FONTSIZE)

    fig.savefig(output_filename, dpi=300)

    return visualization_data
    

def dev_tests():
    # Tests written during development (can be incomplete and messy)

    # test that get_observability_matrix is equivalent to old_get_observability_matrix
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]
    # available sensors
    # R = list(range(C.shape[0]))
    R = [0]
    # available sensors as a vector
    S = np.zeros(C.shape[0])
    S[R] = 1

    O = get_observability_matrix(A, C,  N, S)
    O_old = old_get_observability_matrix(A, C, R, N)
    # remove zero rows from O and compare it with O_old
    assert np.allclose(O[~np.all(O == 0, axis=1)], O_old)

    # test that get_s_sparse_observability is equivalent to old_get_s_sparse_observability
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]

    s = get_s_sparse_observability(A, C, N, np.zeros(C.shape[0]))
    s_old = old_get_s_sparse_observability(A, C)

    assert s == s_old

    # This system's s-sparse observability increases if we protect sensor 0
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]
    assert get_s_sparse_observability(A, C, N, np.array([0, 0])) == 0
    assert get_s_sparse_observability(A, C, N, np.array([1, 0])) == 1
    assert get_s_sparse_observability(A, C, N, np.array([0, 1])) == 0
    assert get_s_sparse_observability(A, C, N, np.array([1, 1])) == 0

    # Test that get_s_sparse_observability2 is equivalent to get_s_sparse_observability
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]
    P = np.array([0, 0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    s_old = get_s_sparse_observability(A, C, N, P)
    assert s == s_old

    # Test that get_s_sparse_observability2 is equivalent to get_s_sparse_observability
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]
    P = np.array([1, 0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    s_old = get_s_sparse_observability(A, C, N, P)
    assert s == s_old

    # Test that get_s_sparse_observability2 is equivalent to get_s_sparse_observability
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]
    P = np.array([0, 1])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    s_old = get_s_sparse_observability(A, C, N, P)
    assert s == s_old

    # Test that get_s_sparse_observability2 is equivalent to get_s_sparse_observability
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]
    P = np.array([1, 1])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    s_old = get_s_sparse_observability(A, C, N, P)
    assert s == s_old

    pass

def test_s_sparse_observability2():
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    # observability index, or the number of time steps
    N = A.shape[0]

    # protected sensors
    P = toBinVec({}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 0
    assert set(toSet(K) for K in important_Ks) == set([frozenset({0})])
    assert set(toSet(K) for K in unimportant_Ks) == set([frozenset({1})])

    P = toBinVec({0}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 1
    assert set(toSet(K) for K in important_Ks) == set()
    assert set(toSet(K) for K in unimportant_Ks) == set()

    P = toBinVec({1}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 0
    assert set(toSet(K) for K in important_Ks) == set([frozenset({0})])
    assert set(toSet(K) for K in unimportant_Ks) == set()

    P = toBinVec({0,1}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 0
    assert set(toSet(K) for K in important_Ks) == set()
    assert set(toSet(K) for K in unimportant_Ks) == set()

    # Tests using a kinematic bicycle model
    # Kinematic bicycle
    theta = np.pi / 4
    delta = np.pi / 8
    L = 2.9
    v = 5

    A = np.array([
        [0, 0, -v*sin(theta), cos(theta), 0],
        [0, 0, v*cos(theta), sin(theta), 0],
        [0, 0, 0, tan(delta)/L, v/(L*cos(theta) ** 2)],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    # observability index, or the number of time steps
    N = A.shape[0]

    # Identity matrix
    C = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    P = toBinVec({}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 0
    assert set(toSet(K) for K in important_Ks) == set([frozenset({0}), frozenset({1})])
    assert set(toSet(K) for K in unimportant_Ks) == set([frozenset({2}), frozenset({3}), frozenset({4})])
    P = toBinVec({0}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 0
    assert set(toSet(K) for K in important_Ks) == set([frozenset({1})])
    assert set(toSet(K) for K in unimportant_Ks) == set([frozenset({2}), frozenset({3}), frozenset({4})])
    P = toBinVec({0,1}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 3 # full s-sparse observability!
    assert set(toSet(K) for K in important_Ks) == set()
    assert set(toSet(K) for K in unimportant_Ks) == set()

    # Redundancy on x
    C = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
    ])
    P = toBinVec({}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 0
    assert set(toSet(K) for K in important_Ks) == set([frozenset({1})])
    assert set(toSet(K) for K in unimportant_Ks) == set([frozenset({0}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})])

    # Redundancy on x and y
    C = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ])
    P = toBinVec({}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 1
    assert set(toSet(K) for K in important_Ks) == set([frozenset({0,5}), frozenset({1,6})])
    assert set(toSet(K) for K in unimportant_Ks) == set([
        frozenset({0, 1}), frozenset({0, 2}), frozenset({0, 3}), frozenset({0, 4}), frozenset({0, 6}),
        frozenset({1, 2}), frozenset({1, 3}), frozenset({1, 4}), frozenset({1, 5}),
        frozenset({2, 3}), frozenset({2, 4}), frozenset({2, 5}), frozenset({2, 6}),
        frozenset({3, 4}), frozenset({3, 5}), frozenset({3, 6}),
        frozenset({4, 5}), frozenset({4, 6}),
        frozenset({5, 6}),
    ])

    pass

def dev_visualizations():
    def kinematic_bicycle():
        # Kinematic bicycle
        theta = np.pi / 4
        delta = np.pi / 8
        L = 2.9
        v = 5

        A = np.array([
            [0, 0, -v*sin(theta), cos(theta), 0],
            [0, 0, v*cos(theta), sin(theta), 0],
            [0, 0, 0, tan(delta)/L, v/(L*cos(theta) ** 2)],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        # observability index, or the number of time steps
        N = A.shape[0]

        # Identity matrix
        C = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ])
        P = toBinVec({}, C.shape[0])
        visualize_s_sparse_observability(A, C, N, P, "x_y_theta_v_delta-no-protection.png")
        P = toBinVec({0}, C.shape[0])
        visualize_s_sparse_observability(A, C, N, P, "x_y_theta_v_delta-protection_on_0.png")
        P = toBinVec({1}, C.shape[0])
        visualize_s_sparse_observability(A, C, N, P, "x_y_theta_v_delta-protection_on_1.png")
        P = toBinVec({0,1}, C.shape[0])
        visualize_s_sparse_observability(A, C, N, P, "x_y_theta_v_delta-protection_on_0_1.png")


    def kinematic_bicycle_with_redundant_x_y():
        # Kinematic bicycle
        theta = np.pi / 4
        delta = np.pi / 8
        L = 2.9
        v = 5

        A = np.array([
            [0, 0, -v*sin(theta), cos(theta), 0],
            [0, 0, v*cos(theta), sin(theta), 0],
            [0, 0, 0, tan(delta)/L, v/(L*cos(theta) ** 2)],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        # observability index, or the number of time steps
        N = A.shape[0]

        # Identity matrix
        C = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ])
        P = toBinVec({}, C.shape[0])
        visualize_s_sparse_observability(A, C, N, P, "x_y_theta_v_delta_x_y-no-protection.png")
        P = toBinVec({0}, C.shape[0])
        visualize_s_sparse_observability(A, C, N, P, "x_y_theta_v_delta_x_y-protection_on_0.png")
        P = toBinVec({1}, C.shape[0])
        visualize_s_sparse_observability(A, C, N, P, "x_y_theta_v_delta_x_y-protection_on_1.png")
        P = toBinVec({0,1}, C.shape[0])
        visualize_s_sparse_observability(A, C, N, P, "x_y_theta_v_delta_x_y-protection_on_0_1.png")

    def academic_example():
        # academic example
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])
        C = np.eye(A.shape[0])

        P = toBinVec({}, C.shape[0])
        visualization_data = visualize_s_sparse_observability(A, C, A.shape[0], P, "academic-example-no-protection.png", show_title=False)
        print("Importance scores:")
        print(visualization_data[3])

        P = toBinVec({0}, C.shape[0])
        visualization_data = visualize_s_sparse_observability(A, C, A.shape[0], P, "academic-example-protection_on_0.png", show_title=False)
        print("Importance scores:")
        print(visualization_data[3])
    
    def academic_example_2():
        # academic example
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])
        C = np.array([
            [1,0,0,0],
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
        ])

        P = toBinVec({}, C.shape[0])
        visualization_data = visualize_s_sparse_observability(A, C, A.shape[0], P, "academic-example-2-no-protection.png", show_title=False)
        print("Importance scores:")
        print(visualization_data[3])

        P = toBinVec({0}, C.shape[0])
        visualization_data = visualize_s_sparse_observability(A, C, A.shape[0], P, "academic-example-2-protection_on_0.png", show_title=False)
        print("Importance scores:")
        print(visualization_data[3])
    
    def experiment_example():
        # Differential drive robot with states (x, y, theta, v, omega)
        # and inputs (a, d_omega)

        # linearization point
        v = 0.5
        theta = np.pi / 4

        A = np.array([
            [0, 0, -v*np.sin(theta), np.cos(theta), 0],
            [0, 0, v*np.cos(theta), np.sin(theta), 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ])
        C = np.array([
            [1, 0, 0, 0, 0], # Global pose
            [0, 1, 0, 0, 0], # Global pose
            [0, 0, 1, 0, 0], # Global pose
            [0, 0, 0, 1, 0], # Wheel odometry
            [0, 0, 0, 0, 1], # Wheel odometry
            [0, 0, 0, 0, 1], # IMU
        ])
        P = toBinVec({}, C.shape[0])
        visualization_data = visualize_s_sparse_observability(A, C, A.shape[0], P, "experiment-system-no-protection.png", show_title=True)
        print("Importance scores:")
        print(visualization_data[3])

        P = toBinVec({0}, C.shape[0])
        visualization_data = visualize_s_sparse_observability(A, C, A.shape[0], P, "experiment-system-protection_on_0.png", show_title=True)
        print("Importance scores:")
        print(visualization_data[3])
    
    # kinematic_bicycle()
    # kinematic_bicycle_with_redundant_x_y()
    # academic_example()
    # academic_example_2()
    experiment_example()




def main():
    # dev_tests()
    # test_s_sparse_observability2()
    dev_visualizations()

if __name__ == "__main__":
    main()
    print("All tests passed!")
