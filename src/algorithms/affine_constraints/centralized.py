import numpy as np
from models.algorithm_interface import RunResult, Algorithm, TrajectoryCollector
from models.optimization_problem import ConstrainedOptimizationProblem

class AugmentedPrimalDualGradientDescent(Algorithm):
    def __init__(self, problem: ConstrainedOptimizationProblem):
        super().__init__(problem)
        
        # this is the state of the algorithm, it shouldn't be included in any agent!
        # so it's an attribute
        self.lamda = None

    class AlgorithmParams(Algorithm.AlgorithmParams):
        def __init__(self, max_iter: int, stepsize: float, seed: int, rho : float, dual_stepsize: float):
            super().__init__(max_iter, stepsize, seed)
            self.dual_stepsize = dual_stepsize
            self.rho = rho


    def H(self, axmb, ll, rho):
        # $$H_\rho(\cdot, \cdot) \text{ is a penalty function on constraint violation} $$
        # defined as follows:

        # $$ \text{If } \rho(\mathbb{a}_j^T\mathbb{x} - b_j) + \lambda_j \ge 0 \text{:}$$
        if rho * axmb + ll >= 0:
            
            #  $$ H_\rho(\mathbb{a}_j^T\mathbb{x} - b_j, \lambda_j) = (\mathbb{a}_j^T\mathbb{x} - b_j) \lambda_j + \frac{\rho}{2}(\mathbb{a}_j^T\mathbb{x} - b_j)^2$$
            
            return axmb * ll + 0.5 * rho * axmb ** 2
        
        else: # \rho(\mathbb{a}_j^T\mathbb{x} - b_j) + \lambda_j \lt 0 \text{:}
        
            # $$     H_\rho(\mathbb{a}_j^T\mathbb{x} - b_j, \lambda_j) = -\frac{1}{2}\frac{\lambda_j^2}{\rho} $$
            
            return -0.5 * ll ** 2 / rho
    
    def H_nabla_1_jth(self, a_j, xx, b_j, l_j, rho, j):
        
        # $$ \nabla_\mathbb{x} H_\rho(\mathbb{a}_j^T\mathbb{x} - b_j, \lambda_j) = \max(\rho(\mathbb{a}_j^T\mathbb{x} - b_j) + \lambda_j, 0)\mathbb{a}_j $$
        
        # TODO: very bad check...
        assert a_j.shape == (self.N * self.d[0],), "aj should be a column that represents the j-th row of the A matrix"
        # aj is a column that represents the j-th row of the A matrix
        # aj.T is a row
        r_j = rho * (a_j.T@xx - b_j)
        s_j = np.max(r_j + l_j, 0)
        return  s_j * a_j

    def H_nabla_2_jth(self, a_j, xx, b_j, l_j, rho, j):
        
        # $$ \nabla_\mathbb{x} H_\rho(\mathbb{a}_j^T\mathbb{x} - b_j, \lambda_j) = \frac{1}{\rho} \left( \max(\rho(\mathbb{a}_j^T\mathbb{x} - b_j) + \lambda_j, 0) - \lambda_j \right)\mathbb{e}_j $$
        
        e_j = np.zeros(self.m)
        e_j[j] = 1
        r_j = rho * (a_j.T@xx - b_j)
        s_j = np.max(r_j + l_j, 0) - l_j
        return 1/rho * s_j * e_j

    def augmented_lagrangian(self, rho):

        # $$ \mathcal{L}(x, \lambda) = f(x) + \sum_{j=1}^{m}H_\rho(\mathbb{a}_j^T\mathbb{x} - b_j, \lambda_j) $$ 
        
        # $$\rho \text{ is a free parameter} $$
        
        cost = self.problem.centralized_cost_fn() 
        A, b = self.problem.global_matrices()
        zz = np.concatenate([ag["zz"] for ag in self.problem.agents])
        assert zz.shape == (self.N * self.d[0],), "zz bad shape!"
        residual = A @ zz - b # shape (m,)
        
        val = cost
        for j in range(self.m): # for each row
            val += self.H(residual[j], self.lamda[j], rho)
        return val

    def run(self, params: AlgorithmParams):
        K = params.max_iter
        stepsize = params.stepsize  
        dual_stepsize  = params.dual_stepsize
        rho = params.rho
        rng = np.random.default_rng(params.seed)

        A, b = self.problem.global_matrices()
        
        self.N = self.problem.N
        self.d = self.problem.d              # tuple
        n_tot = sum(ag["zz"].size for ag in self.problem.agents)
        self.m = A.shape[0] # number of constraints per agent

        assert A.shape[0] == b.shape[0], "A and b mismatch"
        assert A.shape[1] == n_tot, "A has wrong number of columns"

        # [ collectors ]
        collector_zz    = TrajectoryCollector("zz", K, (self.N,) + self.d)
        collector_lamda = TrajectoryCollector("lambda", K, (self.m,))
        
        collector_cost  = TrajectoryCollector("cost", K, ())
        collector_grad_f = TrajectoryCollector("total gradient of f(x)", K, (n_tot,))
        
        
        # $$ \nabla f_0(\mathbf{x}^*) + \sum_{i=1}^{m} \lambda_i \nabla f_i(\mathbf{x}^*) + \sum_{j=1}^{p} \nu_j \nabla h_j(\mathbf{x}^*) = 0$$
        
        # collector_KKT_stationarity = TrajectoryCollector("KKT stationarity", K, (n_tot,))
        
        collector_grad_L_x = TrajectoryCollector("gradient of L(x,l) wrt. x", K, (n_tot,))
        collector_grad_L_l = TrajectoryCollector("gradient of L(x,l) wrt. l", K, (self.m,))

        collector_kkt = TrajectoryCollector("kkt", K, (3,))

        collector_sigma = TrajectoryCollector("sigma", K, self.d)

        # [ init ]
        # TODO: what is the right init value of lambda?
        # at least positive and then the dynamics will keep it positive...
        self.lamda = np.zeros(self.m)

        for k in range(K):
            # [ snapshot ]
            zz_k = np.concatenate([ np.copy(ag["zz"]) for ag in self.problem.agents] )
            assert zz_k.shape == (n_tot,), f"zz_k wrong shape {zz_k.shape}, expected {(n_tot,)}"
            lamda_k = np.copy(self.lamda)
            
            sigma = self.problem.sigma()
            # cost = self.problem.centralized_cost_fn() # $$ f(x)$$
            cost = self.augmented_lagrangian(rho)

            grad_f_list = []
            for i, agent_i in enumerate(self.problem.agents):
                grad_i_f = self.problem.centralized_gradient_of_cost_fn_of_agent(i)
                grad_f_list.append(grad_i_f)
            grad_f = np.concatenate(grad_f_list) # shape (N*d,)
            assert grad_f.shape == (n_tot,), f"grad_f shape {grad_f.shape}, expected {(n_tot,)}"
            
            H_nabla_1s_list = [] # m elemtents, each (N*d,)
            for j in range(self.m):
                H_j_nabla_1 = self.H_nabla_1_jth(A[j], zz_k, b[j], self.lamda[j], rho, j)
                H_nabla_1s_list.append(H_j_nabla_1)
            H_nabla_1 = np.sum(H_nabla_1s_list, axis=0) # (N*d,)
            assert H_nabla_1.shape == (n_tot,), f"H_nabla_1 wrong shape {H_nabla_1.shape}"
            
            # [ z update ] : gradient descent
            # fixed time constant to 1

            # $$ \dot{x} = -\nabla_x L(x, \lambda) = -\nabla f(x) - \sum_{j=1}^{m} \nabla_x H_\rho(a_j^T x - b_j, \lambda_j) = -\nabla f(x) - \sum_{j=1}^{m} \max(\rho(a_j^T x - b_j), 0) a_j $$

            zz_k_plus_1 = zz_k - stepsize * (grad_f + H_nabla_1)


            # [ lamba update ] : gradient ascent
            # $$\eta$$ time constant for dual (absorbed in dual_stepsize?)

            H_nabla_2s_list = [] # m elements, each (m,1)
            for j in range(self.m):
                H_i_nabla_2 = self.H_nabla_2_jth(A[j], zz_k, b[j], self.lamda[j], rho, j)
                H_nabla_2s_list.append(H_i_nabla_2)
            H_nabla_2 = np.sum(H_nabla_2s_list, axis=0)
            assert H_nabla_2.shape == (self.m,), f"H_nabla_2 wrong shape {H_nabla_2.shape}"


            # $$\dot{\lambda} = \eta \nabla_{\lambda} \mathcal{L}(x, \lambda) = \eta \sum_{j=1}^{m} \nabla_{\lambda} H_{\rho}(a_j^T x - b_j, \lambda_j) = \eta \sum_{j=1}^{m} \frac{\max(\rho(a_j^T x - b_j) + \lambda_j, 0) - \lambda_j}{\rho} e_j$$
            
            
            lambda_k_plus_1 = lamda_k + dual_stepsize * H_nabla_2


            # $$ \nabla_x \mathcal{L}(x, \lambda) = \nabla f(x) + \sum_{j=1}^{m} \nabla_x H_\rho(a_j^T x - b_j, \lambda_j) $$
            
            # Monitor that: che $$ \norm{\nabla_x \mathcal{L}(x, \lambda)} \rightarrow 0 $$ e $$ \norm{\nabla_\lambda \mathcal{L}(x, \lambda)} \rightarrow 0 $$
            # TODO: (?!?!?)

            
            # total_grad_L_in_x = np.sum(grad_f_list, axis=0) + H_nabla_1
            total_grad_L_in_x = grad_f + H_nabla_1 # TODO: is it the right thing? sum of \ell_i... ?
            total_grad_L_in_lamda = H_nabla_2
            total_grad_f = np.sum(grad_f_list)

            # [ KKT conditions ]

            # $$ \nabla f(\mathbf{x}^*) + \sum_{i=1}^{m} \lambda_i \nabla h_i(\mathbf{x}^*) + \sum_{j=1}^{p} \nu_j \nabla g_j(\mathbf{x}^*) = 0$$
            
            
            kkt_stationarity = total_grad_f
            for i in range(self.m):
                grad_constraint_h_i = A[i]
                kkt_stationarity += lamda_k[i] * grad_constraint_h_i # TODO: missing equality constrain var.

            # ---- KKT residuals ----
            # primal residual r = A x - b
            r = A @ zz_k - b
            # print(f"A: {A}")
            # print(f"zz_k: {zz_k}")
            # print(f"b: {b}")
            # print(f"residual: {r}")

            # input("Press Enter to continue...")

            # stationarity: ||∇f(x) + A^T λ||
            stationarity = np.linalg.norm(grad_f + A.T @ self.lamda)

            # primal feasibility: ||(Ax - b)_+||_∞
            primal_violation = np.max(np.maximum(r, 0.0))

            # complementarity: max |λ_j * r_j| (max t highlight the one that is not zero! the "less satisfied constraint")
            complementarity = np.max(np.abs(self.lamda * r))


            # [ writeback ]
            off = 0
            for ag in self.problem.agents:
                d_i = ag.zz.size
                ag["zz"] = zz_k_plus_1[off:off+d_i].reshape(ag["zz"].shape)
                off += d_i
            self.lamda = lambda_k_plus_1
            

            # [ collector ]
            collector_zz.log(k, np.array([ag["zz"] for ag in self.problem.agents]))
            collector_cost.log(k, cost)
            collector_grad_f.log(k, total_grad_f)
            collector_lamda.log(k, self.lamda)
            collector_sigma.log(k, sigma)
            collector_grad_L_x.log(k, total_grad_L_in_x)
            collector_grad_L_l.log(k, total_grad_L_in_lamda)
            collector_kkt.log(k, np.array([stationarity, primal_violation, complementarity], dtype=float))
            

        result = RunResult(
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