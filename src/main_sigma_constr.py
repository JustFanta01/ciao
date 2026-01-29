import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Optional

from models.algorithm_interface import RunResult
from algorithms.sigma_constraints.centralized import SigmaConstraint
from algorithms.sigma_constraints.distributed import CIAO
from models.phi import IdentityFunction, LinearFunction, SquareComponentWiseFunction
from models.optimization_problem import ConstrainedSigmaProblem
from models.cost import LocalCloudTradeoffCostFunction, QuadraticCostFunction
from models.agent import Agent
from models.constraints import LinearConstraint
from plots import plots_old, animation, plots

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import graph_utils

import numpy as np

SHOW_CENTRALIZED = False
SHOW_DISTRIBUTED = False
SHOW_COMPARISON = False

SAVE_CENTRALIZED = True
SAVE_DISTRIBUTED = True
SAVE_COMPARISON = True

def generate_B_b_simple(
    # Per-agent params (shape (N,))
    W_nom_GF: np.ndarray,          # job size in MFLOPs (without offload)
    M_job_GB: np.ndarray,          # memory footprint on cloud (MB) if offloaded
    S_GB: np.ndarray,              # fixed MB per offload request
    gamma_GB_per_GF: np.ndarray,   # MB per MFLOP for offloaded data
    # Cloud/global budgets
    T: float,                      # time window [s]
    Pcloud_GFps: float,            # cloud peak compute [MFLOPs/s]
    Mem_cloud_GB: float,           # cloud memory budget [MB]
    BWcloud_GBps: Optional[float] = None,  # cloud ingress bandwidth [MB/s]; None => no net limit
    # Private RHS split (equal by default; pass weights to customize)
    equal_split: bool = True,
    w_comp: Optional[np.ndarray] = None,   # weights per metric (sum to 1)
    w_net:  Optional[np.ndarray] = None,
    w_mem:  Optional[np.ndarray] = None,
):
    """
    Returns:
      B_list: list of N arrays, each (3,1): [compute_GF; net_GB; mem_GB] for z_i=1
      b_list: list of N arrays, each (3,)  : private RHS vectors per agent
      b_sum : (3,) aggregate RHS = sum_i b_i
    Coupling (component-wise):
        sum_i (B_i * z_i) <= sum_i b_i
    """
    # --- sanitize inputs ---
    W_nom_GF = np.asarray(W_nom_GF, dtype=float)
    M_job_GB = np.asarray(M_job_GB, dtype=float)
    S_GB     = np.asarray(S_GB,     dtype=float)
    gamma_GB_per_GF = np.asarray(gamma_GB_per_GF, dtype=float)
    N = W_nom_GF.size

    # --- per-agent B_i (3x1), purely physical ---
    # 1) compute MFLOPs to execute on cloud if z_i=1
    B_comp_GF = W_nom_GF
    # 2) network
    B_net_GB  = S_GB + gamma_GB_per_GF * W_nom_GF
    # 3) memory
    B_mem_GB  = M_job_GB

    B_list = [np.array([[B_comp_GF[i]], [B_net_GB[i]], [B_mem_GB[i]]], dtype=float)
              for i in range(N)]

    # --- global budgets (not exposed directly to agents) ---
    b_comp_global = float(Pcloud_GFps) * float(T)                 # MFLOPs
    b_net_global  = (float(BWcloud_GBps) * float(T)) if (BWcloud_GBps is not None) else np.inf
    b_mem_global  = float(Mem_cloud_GB)

    # --- private RHS split (equal or weighted) ---
    def _weights_or_uniform(w):
        if equal_split or w is None:
            return np.ones(N, dtype=float) / N
        w = np.asarray(w, dtype=float)
        if w.shape != (N,) or not np.isclose(w.sum(), 1.0):
            raise ValueError("Weights must be shape (N,) and sum to 1.")
        return w

    w_comp = _weights_or_uniform(w_comp)
    w_net  = _weights_or_uniform(w_net)
    w_mem  = _weights_or_uniform(w_mem)

    b_list = []
    for i in range(N):
        b_i = np.array([
            w_comp[i] * b_comp_global,   # MFLOPs
            w_net[i]  * b_net_global,    # MB
            w_mem[i]  * b_mem_global     # MB
        ], dtype=float)
        b_list.append(b_i)

    b_sum = np.sum(np.vstack(b_list), axis=0)  # (3,)

    return B_list, b_list, b_sum   

def main():
    N = 2  # number of agents
    d = 1  # dimension of the state space
    constraint = np.ones(d) * 0.25
    seed = 3
    # rng = np.random.default_rng(seed)

    # [ define \ell_i ] 
    rng = np.random.default_rng(seed)
    alpha = np.ones(N)
    beta = np.ones(N)
    energy_task = np.ones(N)*4
    time_task = np.ones(N)*3
    energy_tx = np.ones(N)*3
    time_rtt = np.ones(N)*4

    # [ define initial condition ]
    init_state = rng.uniform(0, 1, size=(N, d))
    # init_state = np.zeros(shape=(N,d))
    # init_state = np.array([[0.1],[0.2], []])
    # print(f"init_state: {init_state}")

    # [ define constraints ]
    
    def setup_problem():
        agents = []
        for i in range(N):
            cost_params = LocalCloudTradeoffCostFunction.CostParams(
                alpha[i], beta[i], energy_tx[i], energy_task[i], time_task[i], time_rtt[i]
            )
            cost_fn = LocalCloudTradeoffCostFunction(cost_params)
            
            # cost_params = QuadraticCostFunction.CostParams(cc=np.ones(shape=(d,))) # optimum: z_i = 0.5 (= 1/2 * c)
            # cost_fn = QuadraticCostFunction(cost_params)
            
            phi_fn = SquareComponentWiseFunction(d)
            # phi_fn = IdentityFunction(d)
            # phi_fn = LinearFunction(d, np.eye(d) * (2*i + 1))
            agent_i = Agent(i, cost_fn, phi_fn, init_state[i])
            
            agents.append(agent_i)

        args = {'edge_probability': 0.45, 'seed': seed}
        graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.ERDOS_RENYI, args)
    
        # fig, axs = plt.subplots(figsize=(7, 4), nrows=1, ncols=2)
        # title = f"Graph and Adj Matrix"
        # fig.suptitle(title)
        # fig.canvas.manager.set_window_title(title)
        # plots.show_graph_and_adj_matrix(fig, axs, graph, adj)
        # plots.show_and_wait(fig)
    
        
        problem = ConstrainedSigmaProblem(agents, adj, seed, constraint)
        return problem
    

    # -----------------------
    # |     CENTRALIZED     |
    # -----------------------
    problem = setup_problem()
    distributed = SigmaConstraint(problem)
    args = {
        "seed": seed,
        "max_iter": 2000,
        "stepsize": 0.01,
        "dual_stepsize": 0.01
    }
    algo_params = SigmaConstraint.AlgorithmParams(**args)
    result_centralized = distributed.run(algo_params)
    result_centralized.summary()

    plotter = plots.ConstrainedRunResultPlotter(problem, result_centralized)
    filename = plots.plot_filename_from_result(result_centralized, "perf")
    plotter\
        .clear()\
        \
        .plot_cost()\
        .plot_grad_norm()\
        .plot_lambda_trajectory(semilogy=False)\
        \
        .plot_agents_trajectories()\
        .plot_sigma_trajectory()\
        .plot_lagr_stationarity()\
        \
        .plot_kkt_conditions()
    if SHOW_CENTRALIZED:
        plotter.show()
    if SAVE_CENTRALIZED:
        plotter.save(filename)


    if N == 2 and d == 1:
        filename = plots.plot_filename_from_result(result_centralized, "phase2d")
        plotter\
            .clear()\
            .plot_phase2d()
        if SHOW_CENTRALIZED:
            plotter.show()
        if SAVE_CENTRALIZED:
            plotter.save(filename)        


    # -----------------------
    # |     DISTRIBUTED     |
    # -----------------------
    problem = setup_problem()
    distributed = CIAO(problem)
    args = {
        "seed": seed,
        "max_iter": 2000,
        "stepsize": 0.01,
        "beta": 0.05,
        "gamma": 0.05
    }
    algo_params = CIAO.AlgorithmParams(**args)
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
        .plot_aux(["aa_traj", "nn_traj"], semilogy=False)\
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
    results = [result_centralized, result_distributed]
    grid_plotter = plots.ComparisonConstrainedRunResultPlotter(problem, results, layout="grid")
    filename = plots.plot_filename_comparison_from_results(results, "perf1")
    grid_plotter\
    .clear()\
        .plot_cost()\
        .plot_grad_norm()\
        .plot_lambda_trajectory(semilogy=False)
    if SHOW_COMPARISON:
        grid_plotter.show()
    if SAVE_COMPARISON:
        grid_plotter.save(filename)
    
    filename = plots.plot_filename_comparison_from_results(results, "perf2")
    grid_plotter\
        .clear()\
        .plot_agents_trajectories()\
        .plot_sigma_trajectory()\
        .plot_lagr_stationarity()\
        .plot_kkt_conditions()
    if SHOW_COMPARISON:
        grid_plotter.show()
    if SAVE_COMPARISON:
        grid_plotter.save(filename)
    
    horizontal_plotter = plots.ComparisonConstrainedRunResultPlotter(problem, [result_centralized, result_distributed], layout="horizontal")
    filename = plots.plot_filename_comparison_from_results(results, "phase2d")
    horizontal_plotter\
        .clear()\
        .plot_phase2d()
    if SHOW_COMPARISON:
        horizontal_plotter.show()
    if SAVE_COMPARISON:
        horizontal_plotter.save(filename)
    

if __name__ == "__main__":
    main()
        
