import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from models.algorithm_interface import RunResult
from algorithms.aggregative_tracking.centralized import CentralizedGradientMethod
from algorithms.aggregative_tracking.distributed import AggregativeTracking
from models.phi import IdentityFunction
from models.optimization_problem import OptimizationProblem
from models.cost import LocalCloudTradeoffCostFunction, QuadraticCostFunction
from models.agent import Agent
from viz import plots, animation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import graph_utils


def main():
    N = 5  # number of agents
    d = 1  # dimension of the state space
    seed = 3
    rng = np.random.default_rng(seed)

    # [ define \ell_i ] 
    rng = np.random.default_rng(seed)
    alpha = np.ones(N)
    beta = np.ones(N)
    energy_task = np.ones(N)*4
    # energy_task[-1] += 1                # last agent should prefer cloud [TO ZERO]
    # energy_task[-1] += 2              # last agent should prefer cloud [NOT TO ZERO]
    time_task = np.ones(N)*3
    energy_tx = np.ones(N)*3
    # energy_tx[0] += 1               # first agent should prefer local
    time_rtt = np.ones(N)*4

    # alpha = rng.uniform(0,5, size=(N,))
    # beta = rng.uniform(0,5, size=(N,))
    # energy_task = rng.uniform(0,5, size=(N,))
    # time_task = rng.uniform(0,5, size=(N,))
    # energy_tx = rng.uniform(0,5, size=(N,))
    # time_rtt = rng.uniform(0,5, size=(N,))
    
    init_state = rng.uniform(0, 1, size=(N, d))
    # print(f"init_state: {init_state}")

    def setup_problem():
        agents = []
        for i in range(N):
            cost_params = LocalCloudTradeoffCostFunction.CostParams(
                alpha[i], beta[i], energy_tx[i], energy_task[i], time_task[i], time_rtt[i]
            )
            cost_fn = LocalCloudTradeoffCostFunction(cost_params)
            # cost_params = QuadraticCostFunction.CostParams(cc=2 * 0.5) # optimum: z_i = 0.5
            # cost_fn = QuadraticCostFunction(cost_params)
            phi_fn = IdentityFunction(d)
            # init_state = rng.uniform(0, 1, size=d)
            agent_i = Agent(i, cost_fn, phi_fn, init_state[i])
            
            agents.append(agent_i)

        args = {'edge_probability': 0.45, 'seed': seed}
        graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.ERDOS_RENYI, args)
    
        fig, axs = plt.subplots(figsize=(7, 4), nrows=1, ncols=2)
        title = f"Graph and Adj Matrix"
        fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)
        plots.show_graph_and_adj_matrix(fig, axs, graph, adj)
        plots.show_and_wait(fig)
    
        problem = OptimizationProblem(agents, adj, seed)
        return problem

    # -----------------------
    # |     CENTRALIZED     |
    # -----------------------
    problem = setup_problem()
    centralized = CentralizedGradientMethod(problem)
    args = {
        "max_iter": 1000,
        "stepsize": 0.01, 
        "seed":seed
    }
    algo_params = CentralizedGradientMethod.AlgorithmParams(**args)
    result = centralized.run(algo_params)
    
    plotter = plots.BaseRunResultPlotter(result)
    plotter.plot_cost().plot_grad_norm().plot_agents_trajectories().plot_sigma_trajectory().show()
    # animation.animate_offloading_with_mean(result, problem.agents, interval=80)


    # -----------------------
    # |     DISTRIBUTED     |
    # -----------------------
    problem = setup_problem()
    distributed = AggregativeTracking(problem)
    args = {
        "max_iter": 1000,
        "stepsize": 0.01, 
        "seed":seed
    }
    algo_params = AggregativeTracking.AlgorithmParams(**args)
    result = distributed.run(algo_params)

    plotter = plots.BaseRunResultPlotter(result)
    plotter.plot_cost().plot_grad_norm().plot_agents_trajectories().plot_sigma_trajectory().show()
    # animation.animate_offloading_with_mean(result, problem.agents, interval=80)

    # TODO: bias del tracker s rispetto alla vera media, puo' essere utile!!!
    # sbar_bias = np.linalg.norm(np.mean(ss_distr[-1, :, 0]) - np.mean(zz_distr[-1, :, 0]))
    # print(f"[CHECK] s-tracker bias on zbar: {sbar_bias:.3e}")


if __name__ == "__main__":
    main()
        
