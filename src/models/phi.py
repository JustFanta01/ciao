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
    def nabla(self, zz):
        ...

class IdentityFunction(AggregationContributionFunction):
    """ $$ \text{Simplifying hypothesis: } n_i = d,  \forall i \in [N] $$ """
    """ so: $$ \phi_i*: \mathbb{R}^{d} \rightarrow \ \mathbb{R}^{d} $$"""

    def __init__(self, d):
        super().__init__(d,d)

    def eval(self, zz):
        return zz
    
    def nabla(self, zz):
        return np.eye(self.d)
    
class LinearFunction(AggregationContributionFunction):
    """ $$ \text{Simplifying hypothesis: } n_i = d,  \forall i \in [N] $$ """
    """ so: $$ \phi_i*: \mathbb{R}^{d} \rightarrow \ \mathbb{R}^{d} $$"""

    def __init__(self, d, Q):
        super().__init__(d,d)
        assert Q.shape == (d,d), f"phi: wrong shape of Q: {Q.shape}, expecting {(d,d)}"
        self.Q = Q

    # $$ \phi_i(z_i) = Qz_i $$
    def eval(self, zz):
        if zz.ndim == 1:  # (d,)
            return self.Q @ zz
        
        if zz.ndim > 1 and zz.shape[-1] != self.d: # (H, W)
            zz = np.expand_dims(zz, axis=-1)  # (H, W, 1)
        res = np.tensordot(zz, self.Q.T, axes=([-1], [0])).squeeze(-1)
        return res
    
    def nabla(self, zz):
        return self.Q
    
class SquareComponentWiseFunction(AggregationContributionFunction):
    """ $$ \text{Simplifying hypothesis: } n_i = d,  \forall i \in [N] $$ """
    """ so: $$ \phi_i*: \mathbb{R}^{d} \rightarrow \ \mathbb{R}^{d} $$"""

    def __init__(self, d):
        super().__init__(d,d)

    def eval(self, zz):
        return np.square(zz)
    
    def nabla(self, zz):
        jac = np.zeros((self.d,self.d))
        np.fill_diagonal(jac, 2 * zz)
        return jac