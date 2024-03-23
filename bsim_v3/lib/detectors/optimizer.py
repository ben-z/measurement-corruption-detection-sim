import time
from itertools import repeat
from typing import Any, Iterable, List, NamedTuple, Optional, Tuple

import cvxpy as cp
import numpy as np
from cvxpy.problems.problem import SolverStats
from more_itertools import powerset
from numpy.typing import NDArray


class CvxpyProblem(NamedTuple):
    """
    Stripped down cvxpy.Problem that is serializable
    """

    status: str
    solver_stats: SolverStats
    compilation_time: float | None
    solve_time: float | None
    value: Any


class OptimizerCaseResult(NamedTuple):
    x0_hat: NDArray[np.float64] | None
    prob: CvxpyProblem
    metadata: dict


class OptimizerOutput(NamedTuple):
    soln: Optional[OptimizerCaseResult]
    solns: List[OptimizerCaseResult]
    metadata: dict


def optimize_l0_case(
    S: Iterable[int],
    q: int,
    prob: cp.Problem,
    x0_hat: cp.Variable,
    can_corrupt: cp.Parameter,
    solver_args: dict = {},
) -> OptimizerCaseResult:
    r"""
    Solves the l0 minimization problem for a given set of uncorrupted sensors $S$.
    Parameters:
        S: Iterable[int] - the set of uncorrupted sensors
        q: int - the number of outputs
        prob: cp.Problem - the optimization problem
        x0_hat: cp.Variable - the variable to optimize
        can_corrupt: cp.Parameter - a parameter that indicates which sensors can be corrupted
        solver_args: dict - arguments to pass to the solver
    Returns:
        x0_hat: numpy.ndarray - estimated state, size $n$
        prob: CvxpyProblem - the optimization problem
        metadata: dict - metadata about the optimization problem. Please see the code for the exact contents.
    """
    additional_metadata = {}

    # K is the sensors that can be corrupted (i.e. the sensors that are not in S)
    K = list(set(range(q)) - set(S))

    can_corrupt.value = np.ones(q)
    for j in S:
        can_corrupt.value[j] = False

    start = time.perf_counter()
    try:
        prob.solve(**solver_args)
    except cp.SolverError as e:
        print(f"Solver error when solving for {S=}: {e}")
        additional_metadata["solver_error"] = str(e)
    except Exception as e:
        print(f"Unknown error when solving for {S=}: {e}")
        additional_metadata["solver_error"] = str(e)
    end = time.perf_counter()

    return OptimizerCaseResult(
        x0_hat=x0_hat.value,
        prob=CvxpyProblem(
            status=prob.status,
            solver_stats=prob.solver_stats,
            compilation_time=prob.compilation_time,
            solve_time=prob._solve_time,
            value=prob.value,
        ),
        metadata={
            "K": K,
            "S": S,
            "solve_time": end - start,
            **additional_metadata,
        },
    )


class Optimizer:
    def __init__(self, N: int, q: int, n: int, solver: str = cp.CLARABEL):
        self.N = N
        self.q = q
        self.n = n
        self.eps_param = cp.Parameter((q,))
        self.cvx_Y_param = cp.Parameter((self.N * self.q,))
        self.cvx_Phi_param = cp.Parameter((self.N * self.q, self.n))
        self.solver = solver

        self.x0_hat = cp.Variable(self.n)
        optimizer = cp.reshape(
            self.cvx_Y_param - self.cvx_Phi_param @ self.x0_hat, (self.q, self.N)
        )
        optimizer_final = cp.mixed_norm(optimizer, p=2, q=1)

        self.can_corrupt = cp.Parameter(self.q, boolean=True)
        self.can_corrupt.value = np.ones(self.q)
        slack = cp.Variable(self.q)
        constraints = []
        for j in range(self.q):
            for k in range(self.N):
                constraints.append(
                    cp.abs(optimizer[j][k])
                    <= self.eps_param[j] + cp.multiply(self.can_corrupt[j], slack[j])
                )
        self.prob = cp.Problem(cp.Minimize(optimizer_final), constraints)
        # Warm up problem data cache. This should make compilation much faster see (prob.compilation_time)
        self.prob.get_problem_data(self.solver)

    def optimize_l0_v4(
        self,
        Phi: np.ndarray,
        Y: np.ndarray,
        eps: NDArray[np.float64] | float = 1e-15,
        S_list: Optional[Iterable[Iterable[int]]] = None,
        solver_args: dict = {},
        early_exit: bool = True,
    ) -> OptimizerOutput:
        r"""
        solves the l0 minimization problem. i.e. attempt to explain the output $Y$ using the model $\Phi$ (`Phi`) and return
        the most-likely initial state $\hat{x}_0$ (`x0_hat`) and the corrupted sensors (`K`).
        Parameters:
            Phi: numpy.ndarray - tensor of size (N, q, n) that describes the evolution of the output over time.
                $N$ is the number of time steps, $q$ is the number of outputs, and $n$ is the number of states
            Y: numpy.ndarray - measured outputs, with input effects subtracted, size $(N, q)$
            eps: numpy.ndarray - noise tolerance for each output, size $(1,)$ or $(q,)$
            S_list: Optional[Iterable[Iterable[int]]] - list of sensor combinations to try. If None, then all possible sensor combinations are tried.
                This is useful when you know that some sensors are not corrupted.
            solver_args: dict - arguments to pass to the cp.Problem().solve function
        Returns:
            x0_hat: numpy.ndarray - estimated state, size $n$
            prob: MyCvxpyProblem- the optimization problem
            metadata: dict - metadata about the optimization problem. Please see the code for the exact contents.
            solns: list - list of solutions for each possible set of corrupted sensors that was tried
        """
        metadata = {}
        start = time.perf_counter()

        N, q, n = Phi.shape
        assert Y.shape == (N, q), f"{Y.shape=} must be equal to {(N, q)=}"
        assert N == self.N, f"{N=} must be equal to {self.N=}"
        assert q == self.q, f"{q=} must be equal to {self.q=}"
        assert n == self.n, f"{n=} must be equal to {self.n=}"

        cvx_Y = Y.reshape((N * q,))  # groups of sensor measurements stacked vertically
        cvx_Phi = Phi.reshape(
            (N * q, n)
        )  # groups of transition+output matrices stacked vertically

        # sort the list of sensor combinations by size, largest to smallest
        # this is because the optimization algorithm needs to minimize the number of corrupt sensors
        S_list = sorted(
            S_list or powerset(range(q)), key=lambda S: len(list(S)), reverse=True
        )

        # Support scalar or vector eps
        self.eps_param.value = np.broadcast_to(eps, (q,))
        self.cvx_Y_param.value = cvx_Y
        self.cvx_Phi_param.value = cvx_Phi

        end = time.perf_counter()
        metadata["setup_time"] = end - start

        start = time.perf_counter()
        map_args = [
            S_list,
            repeat(q),
            repeat(self.prob),
            repeat(self.x0_hat),
            repeat(self.can_corrupt),
            repeat({"solver": self.solver, **solver_args}),
        ]
        soln_generator = map(optimize_l0_case, *map_args)
        # with Pool(min(MAX_POOL_SIZE, os.cpu_count() or 1)) as pool:
        #   soln_generator = pool.starmap(optimize_l0_case, zip(*map_args))

        metadata["solutions_with_errors"] = []

        ret = None
        solns = []
        try:
            for s in soln_generator:
                s_x0_hat, s_prob, s_metadata = s
                solns.append(s)
                if s_metadata.get("solver_error"):
                    metadata["solutions_with_errors"].append(s_metadata)
                if ret is None and s_prob.status in ["optimal", "optimal_inaccurate"]:
                    ret = s
                    if early_exit:
                        break
        finally:
            end = time.perf_counter()
            metadata["solve_time"] = end - start

        return OptimizerOutput(
            soln=ret,
            solns=solns,
            metadata=metadata,
        )
