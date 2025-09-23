import numpy as np
from abc import ABC, abstractmethod
# from dataclasses import dataclass


class Constraint(ABC):
    ...
    # @dataclass
    # class CostParams(ABC):
    #     ...
    
    @abstractmethod
    def check():
        ...
    
    @abstractmethod
    def matrices():
        ...
    
    @abstractmethod
    def residual():
        ...
    


class LinearConstraint(Constraint):
    # $$ Az \le b $$

    def __init__(self, B : np.ndarray, b: np.ndarray):
        self.A = B
        self.b = b
        
        #TODO: check dimensions

        super().__init__()

    def check(self, zz):
        return self.residual(zz) <= 0
    
    def matrices(self):
        return self.A, self.b

    def residual(self, zz):
        return self.A @ zz - self.b

    # def grad(self, zz):
    #     return self.A
    
