import numpy as np
from abc import ABC, abstractmethod

class AggregationContributionFunction(ABC):
    """ $$ \phi_i(\cdot): \mathbb{R}^{n_i} \rightarrow \ \mathbb{R}^{d} $$"""
    def __init__(self, n_i, d):
        super().__init__()
        self.n_i = n_i
        self.d = d

    @abstractmethod
    def eval(self, zz):
        ...
    
    @abstractmethod
    def nabla(self):
        ...

class IdentityFunction(AggregationContributionFunction):
    """ $$ \text{Simplifying hypothesis: } n_i = d,  \forall i \in [N] $$ """
    """ so: $$ \phi_i*: \mathbb{R}^{d} \rightarrow \ \mathbb{R}^{d} $$"""

    def __init__(self, d):
        super().__init__(d,d)

    def eval(self, zz):
        return zz
    
    def nabla(self):
        return np.eye(self.d)