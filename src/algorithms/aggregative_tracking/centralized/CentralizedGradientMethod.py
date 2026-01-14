import numpy as np
from models.algorithm_interface import RunResult, OptimizationProblem, Algorithm, TrajectoryCollector

class CentralizedGradientMethod(Algorithm):
    class AlgorithmParams(Algorithm.AlgorithmParams):
        def __init__(self, max_iter: int, stepsize: float, seed: int):
            super().__init__(max_iter, stepsize, seed)

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)

    def run(self, params : AlgorithmParams):
        max_iter = params.max_iter
        stepsize = params.stepsize
        seed = params.seed

        N = self.problem.N
        d = self.problem.d
        
        # [ define collectors ]
        collector_zz = TrajectoryCollector("zz", max_iter, (N,d))
        collector_cost = TrajectoryCollector("cost", max_iter, ())
        collector_grad = TrajectoryCollector("grad", max_iter, (N,d))
        collector_sigma = TrajectoryCollector("sigma", max_iter, d)
    
        for k in range(max_iter):
            sigma = self.problem.sigma()
            cost = self.problem.centralized_cost_fn()
            
            # This list contains each i-th 
            # block of the gradient, denoted as: $$ [\nabla_{z_i} \sum_{j=1}^{N} \ell_j(z_j, \sigma(z_1, \dots, z_N))]_i \Bigg|_{z_1 = z_1^k,\, \dots,\, z_N = z_N^k} \in \mathbb{R}^{d} $$
            total_grad_list = []

            # [ new states ]
            zz_k_plus_1 = np.zeros((N,d))

            for i, agent_i in enumerate(self.problem.agents):
                nabla_2_sum = np.zeros(d)

                for agent_j in self.problem.agents:
                    nabla_2_sum += agent_j.nabla_2(agent_j["zz"], sigma)

                # [ compute gradients ]
                nabla_1 = agent_i.nabla_1(agent_i["zz"], sigma)
                nabla_phi = agent_i.nabla_phi(agent_i["zz"])
                
                
                # $$[\nabla_{z_i} \ell(z, \sigma(z))]_i$$ is equal to: $$  \nabla_1 \ell_i(z_i, \sigma) \Bigg|_{z_i = z_i^k, \sigma = \frac{1}{N}\sum_{j=1}^N \phi_j(z_j^k)} + \frac{1}{N}\nabla \phi_i(z_i) \Bigg|_{z_i = z_i^k} \cdot \sum_{j=1}^N \nabla_2 \ell_j(z_j, \sigma) \Bigg|_{z_j = z_j^k, \sigma = \frac{1}{N}\sum_{j=1}^N \phi_j(z_j^k)} $$
                
                
                grad_i = nabla_1 + 1/N * nabla_phi @ nabla_2_sum
                assert grad_i.shape == (d,), f"grad_i shape {grad_i.shape}, expected {(d,)}"
                # grad_i_f = self.problem.centralized_gradient_of_cost_fn_of_agent(i) # shape: (d,)
                

                # [ update zz_i ]
                zz_k_plus_1[i] = agent_i["zz"] - stepsize * grad_i
                zz_k_plus_1[i] = np.clip(zz_k_plus_1[i], 0, 1)   # vincolo [0,1]
                
                # total_grad += grad_i
                total_grad_list.append(grad_i)
            
            # [ collectors ]
            collector_zz.log(k, np.array([ag["zz"] for ag in self.problem.agents]))
            collector_cost.log(k, cost)

            total_grad = np.array(total_grad_list)
            collector_grad.log(k, total_grad)
            collector_sigma.log(k, sigma)
            
            # [ writeback ]
            for i, agent_i in enumerate(self.problem.agents):
                agent_i["zz"] = zz_k_plus_1[i]

        result = RunResult(
            algorithm_name="GM",
            algorithm_fullname=type(self).__name__,
            zz_traj=collector_zz.get(),
            cost_traj=collector_cost.get(), 
            grad_traj=collector_grad.get(),
            aux = {
                "sigma_traj": collector_sigma.get()
            }
        )
        return result