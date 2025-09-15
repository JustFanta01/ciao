import numpy as np
from .cost import CostFunction
from models.agent import Agent

class OptimizationProblem():
    def __init__(self, agents:list[Agent], adj:np.ndarray, seed : int):
        self.agents = agents
        self.adj = adj
        if not self._check_doubly_stochastic(self.adj):
            raise ValueError("Adjacency matrix isn't doubly stochastic!")
        
        self.seed = seed

        self.N = len(agents)
        self.d = agents[0].zz.shape

    def sigma(self):
        
        """ $$ \sigma(\textbf{z}) := \frac{\sum_{i=0}^{N} \phi_i(z_i)}{N}  $$ """

        sigma = 0
        for agent_i in self.agents:
            sigma += agent_i.phi()
        sigma /= self.N

        # print(f"sigma = avg_offload_ratio = {sigma}")
        return sigma

    def centralized_cost_fn(self):
        
        # $$ \ell(\textbf{z}) = \sum_{i=0}^{N} \ell_i(z_i, \sigma(\textbf{z}))$$
        
        # $$ \textbf{z}\in R^{Nd}$$
        
        s = self.sigma()
        cost = 0
        for i, agent_i in enumerate(self.agents): \
            cost += agent_i.cost_fn(s) # $$ \ell_i(z_i, \sigma(\textbf{z})) $$
        return cost



    def _check_doubly_stochastic(self, adj: np.ndarray, atol: float = 1e-12) -> bool:
        rows_ok = np.allclose(adj.sum(axis=1), 1.0, atol=atol)
        cols_ok = np.allclose(adj.sum(axis=0), 1.0, atol=atol)
        nonneg_ok = np.all(adj >= -atol)
        return rows_ok and cols_ok and nonneg_ok

