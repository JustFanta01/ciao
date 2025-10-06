import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from models.algorithm_interface import RunResult
from algorithms.affine_constraints.centralized import AugmentedPrimalDualGradientDescent, ArrowHurwiczUzawaPrimalDualGradientDescent
from models.phi import IdentityFunction
from models.optimization_problem import AffineCouplingProblem
from models.cost import LocalCloudTradeoffCostFunction, QuadraticCostFunction
from models.agent import Agent
from models.constraints import LinearConstraint
from viz import plots, animation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import graph_utils


def main():
    N = 2  # number of agents
    m = 1  # number of affine constraints per agent, B_i is a matrix (m,d)
    d = 1  # dimension of the state space
    seed = 3
    rng = np.random.default_rng(seed)

    # [ define \ell_i ] 
    rng = np.random.default_rng(seed)
    alpha = np.ones(N)
    beta = np.ones(N)
    energy_task = np.ones(N)*4
    time_task = np.ones(N)*3
    energy_tx = np.ones(N)*3
    time_rtt = np.ones(N)*4

    # init_state = rng.uniform(0, 0.1, size=(N, d))
    init_state = np.array([[0.1],[0.3]])
    # init_state = np.zeros(shape=(N,d))
    # print(f"init_state: {init_state}")

    def setup_problem():
        agents = []
        for i in range(N):
            # cost_params = LocalCloudTradeoffCostFunction.CostParams(
            #     alpha[i], beta[i], energy_tx[i], energy_task[i], time_task[i], time_rtt[i]
            # )
            # cost_fn = LocalCloudTradeoffCostFunction(cost_params)
            cost_params = QuadraticCostFunction.CostParams(cc=2 * 0.5) # optimum: z_i = 0.5
            cost_fn = QuadraticCostFunction(cost_params)
            
            B_i = np.ones((m,d))
            b_i = np.ones((m,))*0.4
            
            lc = LinearConstraint(B_i, b_i)

            phi_fn = IdentityFunction(d)
            # init_state = rng.uniform(0, 1, size=d)
            agent_i = Agent(i, cost_fn, phi_fn, init_state[i], local_constraint=lc)
            
            agents.append(agent_i)

        args = {'edge_probability': 0.45, 'seed': seed}
        graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.ERDOS_RENYI, args)
    
        # fig, axs = plt.subplots(figsize=(7, 4), nrows=1, ncols=2)
        # title = f"Graph and Adj Matrix"
        # fig.suptitle(title)
        # fig.canvas.manager.set_window_title(title)
        # plots.show_graph_and_adj_matrix(fig, axs, graph, adj)
        # plots.show_and_wait(fig)
    
        problem = AffineCouplingProblem(agents, adj, seed)
        return problem

    # -----------------------
    # |     CENTRALIZED     |
    # -----------------------
    if True:
        problem = setup_problem()

        centralized = AugmentedPrimalDualGradientDescent(problem)
        args = {
            "max_iter": 2500, 
            "stepsize": 0.01,
            "seed": seed,
            "rho": 3,
            "dual_stepsize": 0.05
        }
        algo_params = AugmentedPrimalDualGradientDescent.AlgorithmParams(**args)
        result = centralized.run(algo_params)

        A = []
        b = 0
        for ag in problem.agents:
            A_i, b_i = ag.local_constraint.matrices()
            A.append(A_i[0,0])
            b += b_i
        plots.plot_trajectory_plane_with_affine_constraint(result, A, b)

        result.summary()

        plotter = plots.ConstrainedRunResultPlotter(result)
        plotter\
            .plot_cost()\
            .plot_grad_norm()\
            .plot_agents_trajectories()\
            .plot_sigma_trajectory()\
            .plot_Lagr_stationarity()\
            .plot_kkt_conditions(semilogy=False)\
            .plot_lambda(semilogy=False)\
            .show()
        # animation.animate_offloading_with_mean(result, problem.agents, interval=80)

    # -----------------------
    # |     CENTRALIZED     |
    # -----------------------
    if True:
        problem = setup_problem()
        centralized = ArrowHurwiczUzawaPrimalDualGradientDescent(problem)
        args = {
            "max_iter": 2500, 
            "stepsize": 0.01,
            "seed": seed,
            "dual_stepsize": 0.01
        }
        algo_params = ArrowHurwiczUzawaPrimalDualGradientDescent.AlgorithmParams(**args)
        result = centralized.run(algo_params)

        A = []
        b = 0
        for ag in problem.agents:
            A_i, b_i = ag.local_constraint.matrices()
            A.append(A_i[0,0])
            b += b_i
        plots.plot_trajectory_plane_with_affine_constraint(result, A, b)

        result.summary()

        plotter = plots.ConstrainedRunResultPlotter(result)
        plotter\
            .plot_cost()\
            .plot_grad_norm()\
            .plot_agents_trajectories()\
            .plot_sigma_trajectory()\
            .plot_Lagr_stationarity()\
            .plot_kkt_conditions(semilogy=False)\
            .plot_lambda(semilogy=False)\
            .show()
        # animation.animate_offloading_with_mean(result, problem.agents, interval=80)

    # -----------------------
    # |     DISTRIBUTED     |
    # -----------------------
    # problem = setup_problem()
    # distributed = DistributedAggregativeTracking(problem)
    # algo_params = DistributedAggregativeTracking.AlgorithmParams(max_iter=500, stepsize=0.01, seed=seed)
    # result = distributed.run(algo_params)

    # plotter = plots.RunResultPlotter(result, semilogy=True)
    # plotter.plot_cost().plot_grad_norm().plot_agents_trajectories().plot_sigma_trajectory().show()
    # animation.animate_offloading_with_mean(result, problem.agents, interval=80)

    # TODO: bias del tracker s rispetto alla vera media, puo' essere utile!!!
    # sbar_bias = np.linalg.norm(np.mean(ss_distr[-1, :, 0]) - np.mean(zz_distr[-1, :, 0]))
    # print(f"[CHECK] s-tracker bias on zbar: {sbar_bias:.3e}")


if __name__ == "__main__":
    main()
        
