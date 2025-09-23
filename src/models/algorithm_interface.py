import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod

from .optimization_problem import OptimizationProblem

@dataclass
class RunResult:
    zz_traj: np.ndarray      # (K, N, d)
    grad_traj: np.ndarray    # (K, d)
    cost_traj: np.ndarray    # (K,)
    aux: dict                # anything extra (ss, vv, zbar, diagnostics, etc.)

    # TODO: return also a dictionary and run tests on it!
    def summary(self):
        """
        Print last values of cost, gradients, agent states and aux variables.
        """
        print("=== Final RunResult Summary ===")

        # cost trajectory
        if self.cost_traj is not None and len(self.cost_traj) > 0:
            print(f"Final cost: {self.cost_traj[-1]}")

        # gradient trajectory
        if self.grad_traj is not None and len(self.grad_traj) > 0:
            grad_last = self.grad_traj[-1]
            norm_grad_last = np.linalg.norm(grad_last)
            print(f"Final gradient norm: {norm_grad_last}")

        # agent states
        if self.zz_traj is not None and len(self.zz_traj) > 0:
            zz_last = self.zz_traj[-1]
            print(f"Final agent states (zz): {zz_last}")

        # anything in aux
        if self.aux and isinstance(self.aux, dict):
            print("--- Aux ---")
            for key, traj in self.aux.items():
                if traj is None or len(traj) == 0:
                    continue
                val = traj[-1]
                if isinstance(val, np.ndarray):
                    norm_val = np.linalg.norm(val)
                    print(f"{key}: last={val}, norm={norm_val}")
                else:
                    print(f"{key}: last={val}")

        print("===============================")


class TrajectoryCollector:
    def __init__(self, name, max_iter, shape):
        """
        name         : identifier (e.g. 'z', 'grad', 'cost')
        max_iter     : number of iterations
        shape        : tuple (dimension of the logged variable per iteration)
        """
        self.name = name
        self.data = np.zeros((max_iter,) + shape) # (500,)+(10,2) = (500,10,2)

    def log(self, k, value):
        self.data[k] = value

    def get(self):
        return self.data

class Algorithm(ABC):
    @dataclass(frozen=True)
    
    class AlgorithmParams(ABC):
        max_iter: int
        stepsize: float
        seed: int
    
    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
    
    @abstractmethod
    def run(self, params: AlgorithmParams) -> RunResult: 
        ...