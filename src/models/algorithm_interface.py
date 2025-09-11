import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod


# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from .optimization_problem import OptimizationProblem


@dataclass
class RunResult:
    zz_traj: np.ndarray      # (K, N, d)
    grad_traj: np.ndarray    # (K, d)
    cost_traj: np.ndarray    # (K,)
    aux: dict                # anything extra (ss, vv, zbar, diagnostics, etc.)


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