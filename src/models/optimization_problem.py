import numpy as np
from .agent import Agent
from abc import ABC, abstractmethod
from multipledispatch import dispatch

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
    def check() -> bool:
        pass

    @abstractmethod
    def global_residual():
        pass

    @abstractmethod
    def global_matrices():
        pass
    
    @abstractmethod
    def draw_constraints():
        pass

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

    @dispatch()
    def check(self):
        z_tot = np.concatenate([ag["zz"] for ag in self.agents])  # shape (N*d,)
        assert z_tot.shape == (self.N*self.d,)
        return self.global_residual(z_tot) <= 0

    @dispatch(np.ndarray, np.ndarray)
    def check(self, zz, ss):
        res = self.global_residual(zz)
        feasible = np.all(res <= 0, axis=-1) # (H, W) boolean mask
        return feasible

    def global_residual(self, zz: np.ndarray) -> np.ndarray:
        # zz: (H, W, N*d), N=2, d=1
        # zz: (N*d,)
        # B: (m, N*d)
        # b: (m,)
        B, b = self.B_global, self.b_global
        assert zz.shape[-1] == B.shape[1], f"expected last dim {B.shape[1]}, got {zz.shape[-1]}"
        res = np.tensordot(zz, B.T, axes=([-1],[0])) - b  # (..., m)
        return res

    def global_matrices(self):
        return self.B_global, self.b_global
    
    def draw_constraints(self, ax):
        assert self.B_global.shape[1] == 2, "N=2, d=1 for drawing lines"
        xs = np.linspace(-0.1, 1.1, 400)
        ys = np.linspace(-0.1, 1.1, 400)

        for m in range(self.m):
            a1, a2, b = self.B_global[m,0], self.B_global[m,1], self.b_global[m]
            
            if abs(a2) > 1e-12:
                y_line = (b - a1 * xs) / a2
                ax.plot(xs, y_line, 'k--', label=f"{a1}·z1 + {a2}·z2 = {b}")
            else:
                if abs(a1) < 1e-12:
                    continue
                x_line = np.full_like(ys, b / a1)
                ax.plot(x_line, ys, 'k--', label=f"{a1}·z1 = {b}")

class ConstrainedSigmaProblem(ConstrainedOptimizationProblem):
    def __init__(self, agents:list[Agent], adj:np.ndarray, seed : int, c : np.ndarray):
        super().__init__(agents, adj, seed)

        # $$ \sigma(z) \le c $$
        shape = self.sigma().shape
        assert shape == c.shape, f"Wrong shape of c, expected {shape} but got {c.shape}"
        self.c = c
        self.m = shape[0]

    @dispatch()
    def check(self):
        return self.global_residual(self.sigma()) <= 0

    @dispatch(np.ndarray, np.ndarray)
    def check(self, zz, ss):
        return self.global_residual(ss) <= 0

    def global_residual(self, ss):
        return ss - self.c

    def global_matrices(self):
        return self.c
    
    def draw_constraints(self, ax):
        assert self.N == 2 and self.d == 1, "N=2, d=1 for drawing 2d plot"
        # xs = np.linspace(-0.1, 1.1, 400)
        # ys = np.linspace(-0.1, 1.1, 400)
        pass