from models.algorithm_interface import RunResult, OptimizationProblem, Algorithm, TrajectoryCollector
from models.agent import Agent
import numpy as np

class DistributedAggregativeTracking(Algorithm):
    class AlgorithmParams(Algorithm.AlgorithmParams):
        def __init__(self, max_iter: int, stepsize: float, seed: int):
            super().__init__(max_iter, stepsize, seed)

    adj : np.ndarray
    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)


    def run(self, params : AlgorithmParams):
        max_iter = params.max_iter
        stepsize = params.stepsize
        seed = params.seed
        
        N = self.problem.N
        d = self.problem.d
        
        # [ define collectors ]
        collector_zz = TrajectoryCollector("zz", max_iter, (N,) + d)

        collector_ss = TrajectoryCollector("ss", max_iter, (N,) + d) # proxy of $$\sigma(\textbf{z}^k)$$
        
        collector_vv = TrajectoryCollector("vv", max_iter, (N,) + d) # proxy of $$\frac{1}{N}\sum_{j=1}^{N}\nabla_2\ell_j(z_j^k, \sigma(\textbf{z}^k))$$
        
        collector_cost = TrajectoryCollector("cost", max_iter, ())
        collector_grad = TrajectoryCollector("grad", max_iter, d)

        # [ init zz ]
        # Agents are already initialized!
        
        for agent_i in self.problem.agents:
            # [ init ss ]
            agent_i["ss"] = agent_i.phi() # $$ s_i^0 = \phi_i(z_i^0 ) $$

            # [ init vv ]
            agent_i["vv"] = agent_i.nabla_2(agent_i["ss"]) # $$ v_i^0 = \nabla_2 \ell_i(z_i^0, s_i^0) $$

        for k in range(max_iter): # TODO: why max-1?
            total_cost = 0
            total_grad = np.zeros(d)

            for i, agent_i in enumerate(self.problem.agents):
                # NOTE: usage of the tracker ss instead of avg_offload_ratio
                cost_i = agent_i.cost_fn(sigma=agent_i["ss"])
                
                # [ zz update ]
                nabla_1 = agent_i.nabla_1(agent_i["ss"])
                nabla_phi = agent_i.nabla_phi()

                grad_i = nabla_1 + nabla_phi @ agent_i["vv"]
                old_zz = agent_i["zz"]

                agent_i["zz"] = agent_i["zz"] - stepsize * grad_i
                agent_i["zz"] = np.clip(agent_i.zz, 0, 1)   # apply constraint [0,1]
                    
                # [ ss update ]
                adj_i = self.problem.adj[i] # (1,N)
                ss = np.array([ag["ss"] for ag in self.problem.agents]) # (N,d)
                ss_consensus = ss.T @ adj_i.T
                ss_local_innovation = agent_i["zz"] - old_zz # $$\phi(z_i^{k+1}) - \phi(z_i^{k})$$
                old_ss = agent_i["ss"]
                agent_i["ss"] = ss_consensus + ss_local_innovation

                # [ vv update ]
                vv = np.array([ag["vv"] for ag in self.problem.agents]) # (N,d)
                vv_consensus = vv.T @ adj_i.T
                nabla_2_k = agent_i.nabla_2(old_ss)
                nabla_2_k_plus_1 = agent_i.nabla_2(agent_i["ss"])
                vv_local_innovation = nabla_2_k_plus_1 - nabla_2_k
                agent_i["vv"] = vv_consensus + vv_local_innovation

                total_cost += cost_i
                total_grad += grad_i
            
            
            # [ collector ]
            collector_zz.log(k, np.array([ag["zz"] for ag in self.problem.agents]))
            collector_ss.log(k, np.array([ag["ss"] for ag in self.problem.agents]))
            collector_vv.log(k, np.array([ag["vv"] for ag in self.problem.agents]))

            collector_cost.log(k, total_cost)
            collector_grad.log(k, total_grad)

        return RunResult (
            zz_traj = collector_zz.get(),
            grad_traj = collector_grad.get(),
            cost_traj = collector_cost.get(),
            aux = {
                # "ss_traj": collector_ss.get(),
                "vv_traj": collector_vv.get(),
                "sigma_traj": collector_ss.get()
            }
        )