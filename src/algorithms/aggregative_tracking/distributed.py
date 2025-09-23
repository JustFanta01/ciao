import numpy as np
from models.algorithm_interface import RunResult, OptimizationProblem, Algorithm, TrajectoryCollector

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
        d = self.problem.d # tuple!
        
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
            agent_i["ss"] = agent_i.phi(agent_i["zz"]) # $$ s_i^0 = \phi_i(z_i^0 ) $$

            # [ init vv ]
            agent_i["vv"] = agent_i.nabla_2(agent_i["zz"], agent_i["ss"]) # $$ v_i^0 = \nabla_2 \ell_i(z_i^0, s_i^0) $$
        
        
        for k in range(max_iter):
            total_cost = 0
            total_grad = np.zeros(d)

            # [ new states ] $$ z_i^{k+1}, s_i^{k+1}, v_i^{k+1} $$
            zz_k_plus_1 = np.zeros((N,) + d)
            ss_k_plus_1 = np.zeros((N,) + d)
            vv_k_plus_1 = np.zeros((N,) + d)

            for i, agent_i in enumerate(self.problem.agents):
                # NOTE: usage of the tracker ss instead of sigma
                cost_i = agent_i.cost_fn(agent_i["zz"], agent_i["ss"])
                
                # [ zz update ]
                nabla_1 = agent_i.nabla_1(agent_i["zz"], agent_i["ss"])
                nabla_phi = agent_i.nabla_phi(agent_i["zz"])
                grad_i = nabla_1 + nabla_phi @ agent_i["vv"]
                zz_k_plus_1[i] = agent_i["zz"] - stepsize * grad_i
                zz_k_plus_1[i] = np.clip(zz_k_plus_1[i], 0, 1)   # apply constraint [0,1]

                # [ adj matrix ]
                adj_i = self.problem.adj[i] # (1,N)
                
                # [ ss update ]
                ss = np.array([ag["ss"] for ag in self.problem.agents]) # (N,d)
                ss_consensus = ss.T @ adj_i.T
                ss_local_innovation = agent_i.phi(zz_k_plus_1[i]) - agent_i.phi(agent_i["zz"]) # $$\phi_i(z_i^{k+1}) - \phi_i(z_i^{k})$$
                ss_k_plus_1[i] = ss_consensus + ss_local_innovation

                # [ vv update ]
                vv = np.array([ag["vv"] for ag in self.problem.agents]) # (N,d)
                vv_consensus = vv.T @ adj_i.T
                nabla_2_k = agent_i.nabla_2(agent_i["zz"], agent_i["ss"])
                nabla_2_k_plus_1 = agent_i.nabla_2(zz_k_plus_1[i], ss_k_plus_1[i])
                vv_local_innovation = nabla_2_k_plus_1 - nabla_2_k
                vv_k_plus_1[i] = vv_consensus + vv_local_innovation

                total_cost += cost_i
                total_grad += grad_i

            for i, agent_i in enumerate(self.problem.agents):
                agent_i["zz"] = zz_k_plus_1[i]
                agent_i["ss"] = ss_k_plus_1[i]
                agent_i["vv"] = vv_k_plus_1[i]

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