import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Optional

from models.algorithm_interface import RunResult
from algorithms.sigma_constraints.distributed import VarianceConstraint
from models.phi import IdentityFunction, LinearFunction, SquareComponentWiseFunction
from models.optimization_problem import ConstrainedSigmaProblem
from models.cost import LocalCloudTradeoffCostFunction, QuadraticCostFunction, LocalCloudTradeoffCostFunction2
from models.agent import Agent
from models.constraints import LinearConstraint
from plots import plots_old, animation, plots

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import graph_utils

import numpy as np

SHOW_CENTRALIZED = False
SHOW_DISTRIBUTED = True # False
SHOW_COMPARISON  = False

SAVE_CENTRALIZED = False # True
SAVE_DISTRIBUTED = False # True
SAVE_COMPARISON  = False # True

def main():
    N = 5  # number of agents
    d = 1  # dimension of the state space
    constraint = np.array([0.03])
    seed = 3
    # rng = np.random.default_rng(seed)

    # [ define \ell_i ] 
    rng = np.random.default_rng(seed)
    alpha = rng.normal(1, 0, N)
    beta = rng.normal(1, 0, N)
    energy_task = rng.normal(3.5, 0.2, N)
    time_task = rng.normal(2.5, 0.2, N)
    energy_tx = rng.normal(2.5, 0.2, N)
    time_rtt = rng.normal(3.5, 0.2, N)
    gamma = rng.normal(1, 0.2, N)

    # [ define initial condition ]
    init_state = rng.uniform(0, 1, size=(N, d))
    # init_state = np.zeros(shape=(N,d))
    # init_state = np.array([[0.1],[0.2], []])
    # print(f"init_state: {init_state}")

    def setup_problem():
        agents = []
        for i in range(N):
            cost_params = LocalCloudTradeoffCostFunction2.CostParams(
                alpha[i], beta[i], energy_tx[i], energy_task[i], time_task[i], time_rtt[i], gamma[i]
            )
            cost_fn = LocalCloudTradeoffCostFunction2(cost_params)
            
            # cost_params = QuadraticCostFunction.CostParams(cc=np.ones(shape=(d,))) # optimum: z_i = 0.5 (= 1/2 * c)
            # cost_fn = QuadraticCostFunction(cost_params)
            
            phi_fn = SquareComponentWiseFunction(d)
            # phi_fn = IdentityFunction(d)
            # phi_fn = LinearFunction(d, np.eye(d) * (2*i + 1))
            agent_i = Agent(i, cost_fn, phi_fn, init_state[i])
            
            agents.append(agent_i)

        args = {'edge_probability': 0.45, 'seed': seed}
        graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.ERDOS_RENYI, args)
    
        if SHOW_DISTRIBUTED:
            fig, axs = plt.subplots(figsize=(7, 4), nrows=1, ncols=2)
            title = f"Graph and Adj Matrix"
            fig.suptitle(title)
            fig.canvas.manager.set_window_title(title)
            plots.show_graph_and_adj_matrix(fig, axs, graph, adj)
            plots.show_and_wait(fig)
    
        
        problem = ConstrainedSigmaProblem(agents, adj, seed, constraint)
        return problem


    # -----------------------
    # |     DISTRIBUTED     |
    # -----------------------
    problem = setup_problem()
    distributed = VarianceConstraint(problem)
    args = {
        "seed": seed,
        "max_iter": 3000,
        "stepsize": 0.01,
        "beta": 0.05,
        "gamma": 0.05
    }
    algo_params = VarianceConstraint.AlgorithmParams(**args)
    result_distributed = distributed.run(algo_params)

    result_distributed.summary()

    
    plotter = plots.ConstrainedRunResultPlotter(problem, result_distributed)
    filename = plots.plot_filename_from_result(result_distributed, "perf")
    plotter\
        .clear()\
        \
        .plot_cost()\
        .plot_grad_norm()\
        .plot_lambda_trajectory(semilogy=False)\
        \
        .plot_agents_trajectories()\
        .plot_sigma_trajectory()\
        .plot_aux(["aa_traj", "nn_traj", "pp_traj"], semilogy=False)\
        \
        .plot_lagr_stationarity()\
        .plot_consensus_error(["lambda_traj", "sigma_traj", "vv_traj"])\
        .plot_kkt_conditions()
    if SHOW_DISTRIBUTED:
        plotter.show()
    if SAVE_DISTRIBUTED:
        plotter.save(filename)

    
    if N == 2 and d == 1:
        filename = plots.plot_filename_from_result(result_distributed, "phase2d")
        plotter\
            .clear()\
            .plot_phase2d()
        if SHOW_DISTRIBUTED:
            plotter.show()
        if SAVE_DISTRIBUTED:
            plotter.save(filename)
    # animation.animate_offloading_with_mean(result, problem.agents, interval=80)


    # -----------------------
    # |     COMPARISON      |
    # -----------------------
    # results = [result_centralized, result_distributed]
    # grid_plotter = plots.ComparisonConstrainedRunResultPlotter(problem, results, layout="grid")
    # filename = plots.plot_filename_comparison_from_results(results, "perf1")
    # grid_plotter\
    # .clear()\
    #     .plot_cost()\
    #     .plot_grad_norm()\
    #     .plot_lambda_trajectory(semilogy=False)
    # if SHOW_COMPARISON:
    #     grid_plotter.show()
    # if SAVE_COMPARISON:
    #     grid_plotter.save(filename)
    
    # filename = plots.plot_filename_comparison_from_results(results, "perf2")
    # grid_plotter\
    #     .clear()\
    #     .plot_agents_trajectories()\
    #     .plot_sigma_trajectory()\
    #     .plot_lagr_stationarity()\
    #     .plot_kkt_conditions()
    # if SHOW_COMPARISON:
    #     grid_plotter.show()
    # if SAVE_COMPARISON:
    #     grid_plotter.save(filename)
    
    # if N == 2 and d == 1:
    #     horizontal_plotter = plots.ComparisonConstrainedRunResultPlotter(problem, [result_centralized, result_distributed], layout="horizontal")
    #     filename = plots.plot_filename_comparison_from_results(results, "phase2d")
    #     horizontal_plotter\
    #         .clear()\
    #         .plot_phase2d()
    #     if SHOW_COMPARISON:
    #         horizontal_plotter.show()
    #     if SAVE_COMPARISON:
    #         horizontal_plotter.save(filename)
    

if __name__ == "__main__":
    main()
        
