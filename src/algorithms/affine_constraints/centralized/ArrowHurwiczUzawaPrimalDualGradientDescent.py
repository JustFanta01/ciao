import numpy as np
from models.algorithm_interface import RunResult, Algorithm, TrajectoryCollector
from models.optimization_problem import ConstrainedOptimizationProblem
from typing import cast

class ArrowHurwiczUzawaPrimalDualGradientDescent(Algorithm):
    def __init__(self, problem: ConstrainedOptimizationProblem):
        super().__init__(problem)
        self.lamda = None
        self.problem = cast(ConstrainedOptimizationProblem, self.problem)

    class AlgorithmParams(Algorithm.AlgorithmParams):
        def __init__(self, max_iter: int, stepsize: float, seed: int, dual_stepsize = None):
            super().__init__(max_iter, stepsize, seed)
            self.dual_stepsize = dual_stepsize if dual_stepsize is not None else stepsize
    
    def run(self, params: AlgorithmParams):
        K = params.max_iter
        stepsize = params.stepsize
        dual_stepsize = params.dual_stepsize
        rng = np.random.default_rng(params.seed)

        A, b = self.problem.global_matrices()
        # TODO: c = np.einsum('ijk,jk -> i', A, z) with A=(m,N,d) and z=(N,d)
        # also 'm N ... , N ... -> m'
        
        N = self.problem.N
        d = self.problem.d

        m = A.shape[0] # number of constraints per agent

        assert A.shape[0] == b.shape[0], "A and b mismatch"
        assert A.shape[1] == N*d, "A has wrong number of columns"

        # [ collectors ]
        collector_zz    = TrajectoryCollector("zz", K, (N,d))
        collector_lamda = TrajectoryCollector("lambda", K, m)
        
        collector_cost  = TrajectoryCollector("cost", K, ())
        collector_grad_f = TrajectoryCollector("total gradient of f(x)", K, (N,d))
        
        
        # $$ \nabla f_0(\mathbf{x}^*) + \sum_{i=1}^{m} \lambda_i \nabla f_i(\mathbf{x}^*) + \sum_{j=1}^{p} \nu_j \nabla h_j(\mathbf{x}^*) = 0$$
        
        # collector_KKT_stationarity = TrajectoryCollector("KKT stationarity", K, (n_tot,))
        
        collector_grad_L_x = TrajectoryCollector("gradient of L(x,l) wrt. x", K, (N,d))
        collector_grad_L_l = TrajectoryCollector("gradient of L(x,l) wrt. l", K, m)
        collector_kkt = TrajectoryCollector("kkt", K, 3)
        collector_sigma = TrajectoryCollector("sigma", K, d)

        # [ init ]
        self.lamda = np.zeros(m)
        # self.lamda = -self.problem.global_residual()
        assert np.all(self.lamda >= 0), "Negative lambda"

        def lagrangian():
            # f(x) + (A@xx-b).T @ ll
            return self.problem.centralized_cost_fn() + self.problem.global_residual().T @ self.lamda

        for k in range(K):
            # [ snapshot ]
            zz_k = np.zeros((N,d))
            for i, ag in enumerate(self.problem.agents):
                zz_k[i] = np.copy(ag["zz"])
            assert zz_k.shape == (N,d), f"zz_k wrong shape {zz_k.shape}, expected {(N,d)}"
            
            lamda_k = np.copy(self.lamda)
            
            sigma = self.problem.sigma()
            # cost = self.problem.centralized_cost_fn() # $$ f(x)$$
            cost = lagrangian() # values at current iteration

            # $$ \nabla f(x) $$
            grad_f_list = []
            for i, agent_i in enumerate(self.problem.agents):
                grad_i_f = self.problem.centralized_gradient_of_cost_fn_of_agent(i) # shape: (d,)
                grad_f_list.append(grad_i_f)
            total_grad_f = np.array(grad_f_list) # shape (N,d)
            assert total_grad_f.shape == (N,d), f"total_grad_f shape {total_grad_f.shape}, expected {(N,d)}"
            
            
            # $$ \sum_{i=0}^{m} \lambda_i \nabla g(x) $$
            
            grad_lamda_times_constr = np.zeros((N*d,))
            # TODO: double check this part
            for i in range(m):
                t = lamda_k[i] * A[i].T # scalar * (N*d,)
                assert t.shape == (N*d,), f"t shape {t.shape}, expected {(N*d,)}"
                grad_lamda_times_constr += t
            
            grad_lamda_times_constr = grad_lamda_times_constr.reshape((N,d))
            total_grad_L_in_x = total_grad_f + grad_lamda_times_constr
            assert total_grad_L_in_x.shape == (N,d), f"total_grad_L_in_x shape {total_grad_L_in_x.shape}, expected {(N,d)}"
            zz_k_plus_1 = zz_k - stepsize * (total_grad_L_in_x)
            zz_k_plus_1 = np.clip(zz_k_plus_1, 0, 1)   # apply constraint [0,1]

            # ------[ lamba update / gradient ascent ]------
            zz_k_flatten = zz_k.reshape((N*d,))
            violation = A @ zz_k_flatten - b
            lambda_k_plus_1 = np.maximum(0.0, lamda_k + dual_stepsize * violation)
            
            total_grad_L_in_lamda = violation
    
            # stationarity: ||∇f(x) + A^T λ||
            stationarity = np.linalg.norm(total_grad_f + grad_lamda_times_constr)

            # primal feasibility: ||(Ax - b)_+||_∞
            primal_violation = np.max(np.maximum(violation, 0.0))

            # complementarity: max |λ_j * r_j| (max to highlight the "less satisfied constraint")
            complementarity = np.max(np.abs(self.lamda * violation))

            # ------[ collector ]------
            collector_zz.log(k, zz_k)
            collector_cost.log(k, cost)
            collector_grad_f.log(k, total_grad_f)
            collector_lamda.log(k, self.lamda)
            collector_sigma.log(k, sigma)
            collector_grad_L_x.log(k, total_grad_L_in_x)
            collector_grad_L_l.log(k, total_grad_L_in_lamda)
            collector_kkt.log(k, np.array([stationarity, primal_violation, complementarity], dtype=float))
            
            # ------[ writeback ]------
            for i, ag in enumerate(self.problem.agents):
                ag["zz"] = zz_k_plus_1[i]
            self.lamda = lambda_k_plus_1

        result = RunResult(
            algorithm_name="AHUPDGD",
            algorithm_fullname=type(self).__name__,
            zz_traj = collector_zz.get(),
            grad_traj = collector_grad_f.get(),
            cost_traj = collector_cost.get(),
            aux = {
                "sigma_traj": collector_sigma.get(),
                "lambda_traj": collector_lamda.get(),
                "grad_L_x_traj": collector_grad_L_x.get(),
                "grad_L_l_traj": collector_grad_L_l.get(),
                "kkt_traj": collector_kkt.get(),
            }
        )
        return result