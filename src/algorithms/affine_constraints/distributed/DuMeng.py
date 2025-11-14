import numpy as np
from models.algorithm_interface import RunResult, Algorithm, TrajectoryCollector
from models.optimization_problem import ConstrainedOptimizationProblem
from typing import cast

class DuMeng(Algorithm):
    class AlgorithmParams(Algorithm.AlgorithmParams):
        def __init__(self, max_iter: int, stepsize: float, seed: int, beta: float, gamma: float):
            super().__init__(max_iter, stepsize, seed)
            self.beta = beta
            self.gamma = gamma
    
    def __init__(self, problem: ConstrainedOptimizationProblem):
        super().__init__(problem)
        self.problem = cast(ConstrainedOptimizationProblem, self.problem)

    def run(self, params : AlgorithmParams):
        K = params.max_iter
        alpha = params.stepsize
        beta = params.beta
        gamma = params.gamma
        rng = np.random.default_rng(params.seed)
        
        N = self.problem.N
        d = self.problem.d
        m = self.problem.m

        BB, bb = self.problem.global_matrices()

        def augmented_lagrangian(ll_k, zz_k):
            lapl = 0.5 * (np.eye(N) - self.problem.adj)
            lapl = np.kron(lapl, np.eye(m))

            # $$ L(z,\lambda) = \ell(z) + \sum_{i=0}^{m}\lambda_i g_i(z) - \frac{1}{2} \lambda^T L^2 \lambda $$
            
            # for the "constr", lambda should be (m,1)
            # for the "regularizer", lambda should be (N*m, 1)
            cost_f = self.problem.centralized_cost_fn()
            
            # using the mean at that iteration to compute the global constraint violation
            ll_k_bar = np.mean(ll_k.reshape((N,m)), axis=0)
            constr = ll_k_bar.T @ (BB @ zz_k - bb)
            regularizer = - 0.5 * ll_k.T @ lapl @ ll_k
            return cost_f + constr + regularizer


        # [ define collectors ]
        collector_zz = TrajectoryCollector("zz", K, (N,d)) # decision variable

        collector_ss = TrajectoryCollector("ss", K, (N,d)) # proxy of $$\sigma(\textbf{z}^k)$$
        
        collector_vv = TrajectoryCollector("vv", K, (N,d)) # proxy of $$\frac{1}{N}\sum_{j=1}^{N}\nabla_2\ell_j(z_j^k, \sigma(\textbf{z}^k))$$
        
        collector_aa = TrajectoryCollector("aa", K, (N,m)) # aux

        collector_ll = TrajectoryCollector("ll", K, (N,m)) # $\lambda^k$
        
        collector_cost = TrajectoryCollector("cost", K, (2,))
        

        collector_grad_ell = TrajectoryCollector("total gradient of l(x)", K, (N,d))
        collector_grad_L_z = TrajectoryCollector("gradient of L(z,l) wrt. z", K, (N,d))
        collector_grad_L_l = TrajectoryCollector("gradient of L(z,l) wrt. l", K, (m,))

        collector_kkt = TrajectoryCollector("kkt", K, (3,))

        # [ init zz ]
        # $$ z_i^0 \text{ arbitrarily (from init\_state)} $$

        for agent_i in self.problem.agents:
            # [ init ss ]
            agent_i["ss"] = agent_i.phi(agent_i["zz"]) # $$ s_i^0 = \phi_i(z_i^0 ) $$

            # [ init vv ]
            agent_i["vv"] = agent_i.nabla_2(agent_i["zz"], agent_i["ss"]) # $$ v_i^0 = \nabla_2 \ell_i(z_i^0, s_i^0) $$
        
            # [ init aa ]
            agent_i["aa"] = np.zeros(m) # $$ a_i^0 = \mathbf{0} $$

            # [ init ll ]
            agent_i["llm1"] = np.zeros(m) # $$ \lambda_i^{-1} = \mathbf{0} $$
            agent_i["ll"] = np.zeros(m) # $$ \lambda_i^0 = \mathbf{0}, \ \text{arbitrarily}$$

        for k in range(K):
            total_cost = 0
            total_grad_ell = np.zeros((N,d))
            total_grad_L_in_z = np.zeros((N,d))
            total_grad_L_in_l = np.zeros(m)

            # [ new states ] $$ z_i^{k+1}, s_i^{k+1}, v_i^{k+1}, a_i^{k+1}, \lambda_i^{k+1} $$
            zz_k_plus_1 = np.zeros((N,d))
            ss_k_plus_1 = np.zeros((N,d))
            vv_k_plus_1 = np.zeros((N,d))
            aa_k_plus_1 = np.zeros((N,m))
            ll_k_plus_1 = np.zeros((N,m))
            

            # point of view of agent-i
            for i, agent_i in enumerate(self.problem.agents):

                cost_i = agent_i.cost_fn(agent_i["zz"], agent_i["ss"]) # $$\ell_i(z_i^k, s_i^k), s_i^k \text{ tracker of } \sigma(z) $$
    
                # [ local constraint matrices ]
                B_i, b_i = agent_i.local_constraint.matrices()
               
                # [ adj matrix ]
                adj_i = self.problem.adj[i] # (1,N)
                
                # [ laplacian matrix ]
                ee_i = np.zeros_like(adj_i)
                ee_i[i] = 1
                rr_i = 0.5 * (ee_i - adj_i)

                # [ zz update ]
                
                # $$ z_i^{k+1} = z_i^k - \alpha(\nabla_1 \ell_i(z_i^k, s_i^k) + \nabla \phi_i(z_i^k) v_i^k + B_i^T \lambda_i^k ) $$

                nabla_1 = agent_i.nabla_1(agent_i["zz"], agent_i["ss"])
                nabla_phi = agent_i.nabla_phi(agent_i["zz"])
                grad_ell = nabla_1 + nabla_phi @ agent_i["vv"]
                nabla_constr = B_i.T @ agent_i["ll"]
                grad_L_in_z = grad_ell + nabla_constr
                zz_k_plus_1[i] = agent_i["zz"] - alpha * grad_L_in_z
                zz_k_plus_1[i] = np.clip(zz_k_plus_1[i], 0, 1)   # apply constraint [0,1]

                # [ ss update ]
                
                # $$ s_i^{k+1} = \sum_{j=1}^{N} a_{ij} s_j^{k} + \phi_i(z_i^{k+1}) - \phi_i(z_i^{k}) $$
                
                ss_k = np.array([ag["ss"] for ag in self.problem.agents]) # (N,d)
                ss_consensus = ss_k.T @ adj_i.T # (d,N)x(N,1)=(d,1)
                ss_local_innovation = agent_i.phi(zz_k_plus_1[i]) - agent_i.phi(agent_i["zz"])
                ss_k_plus_1[i] = ss_consensus + ss_local_innovation

                # [ vv update ]
                
                # $$ v_i^{k+1} = \sum_{j=1}^{N} a_{ij} v_j^{k} + \nabla_2 \ell_i(z_i^{k+1}, s_i^{k+1}) - \nabla_2 \ell_i(z_i^{k}, s_i^{k}) $$
                
                vv_k = np.array([ag["vv"] for ag in self.problem.agents]) # (N,d)
                vv_consensus = vv_k.T @ adj_i.T # (d,N)x(N,1)=(d,1)
                nabla_2_k = agent_i.nabla_2(agent_i["zz"], agent_i["ss"])
                nabla_2_k_plus_1 = agent_i.nabla_2(zz_k_plus_1[i], ss_k_plus_1[i]) 
                vv_local_innovation = nabla_2_k_plus_1 - nabla_2_k
                vv_k_plus_1[i] = vv_consensus + vv_local_innovation

                # [ aa update ]
                
                # $$ a_i^{k+1} = a_i^{k} - \gamma \sum_{j=0}^{N}r_{ij} a_{j}^k - \sum_{j=0}^{N}r_{ij}(\lambda_j^k - \lambda_j^{k-1}) + (\lambda_i^k - \lambda_i^{k-1}) + \beta B_i(z_i^{k+1} - z_i^{k}) $$
                
                aa_k         = np.array([agent_j["aa"]      for agent_j in self.problem.agents])   # (N,m)
                ll_k_minus_1 = np.array([agent_j["llm1"]    for agent_j in self.problem.agents])   # (N,m)
                ll_k         = np.array([agent_j["ll"]      for agent_j in self.problem.agents])   # (N,m)
                aa_consensus        = aa_k.T @ rr_i.T                      # (m,N)x(N,)=(m,)
                ll_diff_consensus   = (ll_k - ll_k_minus_1).T @ rr_i.T     # (m,N)x(N,)=(m,)
                ll_innovation       = agent_i["ll"] - agent_i["llm1"]      # (m,)
                zz_innovation       = zz_k_plus_1[i] - agent_i["zz"]       # (d,)

                aa_k_plus_1[i]  = agent_i["aa"] - gamma * aa_consensus\
                                - ll_diff_consensus + ll_innovation\
                                + beta * (B_i @ zz_innovation)

                # HYBRID - centralized update:
                # zz_k_flat = np.array([ag["zz"] for ag in self.problem.agents]).reshape((N*d,))
                # aa_k_plus_1[i] = agent_i["ll"] + beta * (BB @ zz_k_flat - bb)

                # [ ll update ]
                
                # $$ \lambda_i^{k+1} = \mathit{\Pi}_{\mathbb{R}_+^m}[v_i^{k+1}] $$
                
                ll_k_plus_1[i] = np.maximum(0.0, aa_k_plus_1[i]) # ReLU

                total_cost += cost_i
                
                total_grad_ell[i] = grad_ell
                total_grad_L_in_z[i] = grad_L_in_z # (N,d)

            # global primal residual: r = A x - b
            zz_k_flat = np.array([ag["zz"] for ag in self.problem.agents]).reshape((N*d,))
            r = BB @ zz_k_flat - bb # (m,)
            total_grad_L_in_l = r
            
            # stationarity: ||∇f(x) + A^T λ||
            ll_k = np.array([ag["ll"] for ag in self.problem.agents])       # (N, m)
            ll_k_flat = ll_k.reshape((N*m,))
            ll_bar = np.mean(ll_k, axis=0) # TODO: the mean?
            # BB.T @ ll_bar
            # eq (10a): $$ \mathbf{0} = \nabla_1 f(w^*, z^*) + \nabla h(w^*)\mu^* + \Lambda^T\lambda^* $$
            stationarity = np.linalg.norm(total_grad_ell.reshape(-1) + self.problem.B_local.T @ ll_k_flat)

            # primal feasibility: ||(Ax - b)_+||_∞
            primal_violation = np.max(np.maximum(r, 0.0))

            # complementarity: max |λ_j * r_j| (max to highlight the "less satisfied constraint")
            complementarity = np.max(np.abs(ll_bar * r))
    
            # [ collector ]
            #  -- cost
            cost = augmented_lagrangian(ll_k_flat, zz_k_flat)
            collector_cost.log(k, np.array([total_cost, cost]))
            
            #  -- states
            collector_zz.log(k, np.array([ag["zz"] for ag in self.problem.agents]))
            collector_ss.log(k, np.array([ag["ss"] for ag in self.problem.agents]))
            collector_vv.log(k, np.array([ag["vv"] for ag in self.problem.agents]))
            collector_aa.log(k, np.array([ag["aa"] for ag in self.problem.agents]))
            collector_ll.log(k, np.array([ag["ll"] for ag in self.problem.agents]))
            
            # -- kkt
            collector_kkt.log(k, np.array([stationarity, primal_violation, complementarity], dtype=float))
            
            # -- grads
            collector_grad_ell.log(k, total_grad_ell)
            collector_grad_L_z.log(k, total_grad_L_in_z)
            collector_grad_L_l.log(k, total_grad_L_in_l)   

            # [ write back ]
            for i, agent_i in enumerate(self.problem.agents):
                agent_i["zz"] = zz_k_plus_1[i]
                agent_i["ss"] = ss_k_plus_1[i]
                agent_i["vv"] = vv_k_plus_1[i]
                agent_i["aa"] = aa_k_plus_1[i]
                agent_i["llm1"] = agent_i["ll"] #     old  <--  current 
                agent_i["ll"] = ll_k_plus_1[i]  # current  <--  new

        return RunResult (
            algorithm_name=type(self).__name__,
            zz_traj = collector_zz.get(),
            grad_traj = collector_grad_ell.get(),
            cost_traj = collector_cost.get(),
            aux = {
                # "ss_traj": collector_ss.get(),
                "sigma_traj": collector_ss.get(),
                "vv_traj": collector_vv.get(),
                "aa_traj": collector_aa.get(), 
                "lambda_traj": collector_ll.get(),
                "grad_L_x_traj": collector_grad_L_z.get(),                
                "grad_L_l_traj": collector_grad_L_l.get(),
                "kkt_traj": collector_kkt.get()
            }
        )