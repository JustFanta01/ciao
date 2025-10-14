import numpy as np
from .agent import Agent
from abc import ABC, abstractmethod

class OptimizationProblem():
    def __init__(self, agents:list[Agent], adj:np.ndarray, seed : int):
        self.agents = agents
        self.adj = adj
        if not self._check_doubly_stochastic(self.adj):
            raise ValueError("Adjacency matrix isn't doubly stochastic!")
        
        self.seed = seed

        self.N = len(agents)
        self.d = agents[0].zz.shape[0] # flatten dimension!

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
        for i, agent_i in enumerate(self.agents):
            cost += agent_i.cost_fn(s) # $$ \ell_i(z_i, \sigma(\textbf{z})) $$
        return cost

    def centralized_gradient(self) -> np.ndarray:
        
        # $$ \nabla \ell(z, \sigma(z)) = \sum_{i=0}^{N} \nabla \ell_i(z_i, \sigma (z)) $$

        total_gradient = np.zeros((self.N, self.d))
        for i, agent_i in enumerate(self.agents):
            total_gradient[i] = self.centralized_gradient_of_cost_fn_of_agent(i)

        return total_gradient
        

    def centralized_gradient_of_cost_fn_of_agent(self, idx):
        
        # $$ \text{grad\_i\_f} = \nabla_1 \ell_i(z_i^k, \sigma^k) + \frac{1}{N} \nabla \phi_i(z_i^k) \sum_{j=1}^N \nabla_2 \ell_j(z_j^k, \sigma^k) $$

        sigma = self.sigma()
        # TODO: maybe ask this to agent i, and if centralized you receive the real global info
        agent_i = self.agents[idx]

        # [ compute gradients ]
        nabla_2_sum = np.zeros(self.d)
        for agent_j in self.agents:
            nabla_2_sum += agent_j.nabla_2(agent_j["zz"] , sigma)
        
        nabla_1 = agent_i.nabla_1(agent_i["zz"], sigma)
        nabla_phi = agent_i.nabla_phi(agent_i["zz"])
        grad_i_f = nabla_1 + 1/self.N * nabla_phi @ nabla_2_sum

        return grad_i_f

    def _check_doubly_stochastic(self, adj: np.ndarray, atol: float = 1e-12) -> bool:
        rows_ok = np.allclose(adj.sum(axis=1), 1.0, atol=atol)
        cols_ok = np.allclose(adj.sum(axis=0), 1.0, atol=atol)
        nonneg_ok = np.all(adj >= -atol)
        return rows_ok and cols_ok and nonneg_ok

class ConstrainedOptimizationProblem(OptimizationProblem, ABC):
    def __init__(self, agents:list[Agent], adj:np.ndarray, seed : int):
        super().__init__(agents, adj, seed)

    @abstractmethod
    def check():
        pass

    @abstractmethod
    def global_residual():
        pass

    @abstractmethod
    def global_matrices():
        pass


class AffineCouplingProblem(ConstrainedOptimizationProblem):
    
    # Affine coupling constraint: $$ \sum_{i=1}^{N}B_i z_i \le b_i $$

    def __init__(self, agents:list[Agent], adj:np.ndarray, seed : int):
        super().__init__(agents, adj, seed)

        for i, ag in enumerate(self.agents):
            if ag.local_constraint is None:
                raise ValueError(f"Agent-{i} has no constraint, but problem requires constraints")

        B_list = []
        b_list = []
        for ag in self.agents:
            B_i, b_i = ag.local_constraint.matrices()
            B_list.append(B_i)
            b_list.append(b_i)
        
        # number of constraints
        m = B_list[0].shape[0]
        
        self.B_local = self._block_diag(B_list)         # shape: (N*m, N*d)
        self.b_local = np.concatenate(b_list, axis=0)   # shape: (N*m, 1)
        
        self.B_global = np.hstack(B_list)                   # shape (m, N*d)
        self.b_global = np.sum(b_list, axis=0)              # shape (m, 1)

        assert self.B_global.shape == (m, self.N*self.d), "" f"B_global.shape was {self.B_global.shape}, expected {(m, self.N*self.d)}"
        assert self.b_global.shape == (m,), f"b_global.shape was {self.b_global.shape}, expected {(m,)}"

        # [ N=5, d=2, m=3
        #  [0. 0. 1. 1. 2. 2. 3. 3. 4. 4.]
        #  [0. 0. 1. 1. 2. 2. 3. 3. 4. 4.]
        #  [0. 0. 1. 1. 2. 2. 3. 3. 4. 4.]
        # ]

        self.m = self.B_global.shape[0]

    def check(self):
        return self.global_residual() <= 0

    def global_residual(self):
        
        # it returns $$ \sum_{i=0}^{N}B_i z_i - \sum_{i=0}^{N}b_i $$

        z_tot = np.concatenate([ag["zz"] for ag in self.agents])  # shape (N*d,)
        assert z_tot.shape == (self.N*self.d,)
        return self.B_global @ z_tot - self.b_global
    
    def _block_diag(self, matrices):
        """Build block diagonal matrix from list of 2D arrays."""
        shapes = np.array([M.shape for M in matrices])
        m, n = shapes[:,0].sum(), shapes[:,1].sum()
        out = np.zeros((m, n))
        i, j = 0, 0
        for M in matrices:
            mi, ni = M.shape
            out[i:i+mi, j:j+ni] = M
            i += mi
            j += ni
        return out

    def global_matrices(self):
        return self.B_global, self.b_global