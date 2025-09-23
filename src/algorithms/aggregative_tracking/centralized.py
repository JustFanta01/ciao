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
        d = self.problem.d # tuple!
        
        # [ define collectors ]
        collector_zz = TrajectoryCollector("zz", max_iter, (N,) + d)
        collector_cost = TrajectoryCollector("cost", max_iter, ())
        collector_grad = TrajectoryCollector("grad", max_iter, d)
        collector_sigma = TrajectoryCollector("sigma", max_iter, d)
    
        for k in range(max_iter):
            sigma = self.problem.sigma()
            cost = self.problem.centralized_cost_fn()
            total_grad = np.zeros(d)
            
            # [ new states ]
            zz_k_plus_1 = np.zeros((N,) + d)

            for i, agent_i in enumerate(self.problem.agents):
                gradient_sum = np.zeros(d)

                for agent_j in self.problem.agents:
                    gradient_sum += agent_j.nabla_2(agent_j["zz"], sigma)

                # [ compute gradients ]
                nabla_1 = agent_i.nabla_1(agent_i["zz"], sigma)
                nabla_phi = agent_i.nabla_phi(agent_i["zz"])
                grad_i = nabla_1 + 1/N * nabla_phi @ gradient_sum
                
                # [ update zz_i ]
                zz_k_plus_1[i] = agent_i["zz"] - stepsize * grad_i
                zz_k_plus_1[i] = np.clip(zz_k_plus_1[i], 0, 1)   # vincolo [0,1]
                
                total_grad += grad_i
            
            # [ writeback ]
            for i, agent_i in enumerate(self.problem.agents):
                agent_i["zz"] = zz_k_plus_1[i]

            # [ collectors ]
            collector_zz.log(k, np.array([ag["zz"] for ag in self.problem.agents]))
            collector_cost.log(k, cost)
            collector_grad.log(k, total_grad)
            collector_sigma.log(k, sigma)

        result = RunResult(
            zz_traj=collector_zz.get(),
            cost_traj=collector_cost.get(), 
            grad_traj=collector_grad.get(),
            aux = {
                "sigma_traj": collector_sigma.get()
            }
        )
        return result