import numpy as np
from .cost import CostFunction
from models.agent import Agent

class OptimizationProblem():
    def sigma(self):
        
        """ $$ \sigma(\textbf{z}) := \frac{\sum_{i=0}^{N} \phi_i(z_i)}{N}  $$ """

        sigma = 0
        for agent_i in self.agents:
            sigma += agent_i.phi()
        sigma *= 1/self.N

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

    def __init__(self, agents:list[Agent], adj:np.ndarray, seed : int):
        self.agents = agents
        self.adj = adj
        self.seed = seed

        self.N = len(agents)
        self.d = agents[0].zz.shape

