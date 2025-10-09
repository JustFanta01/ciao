import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Optional

from models.algorithm_interface import RunResult
from algorithms.affine_constraints.centralized import AugmentedPrimalDualGradientDescent, ArrowHurwiczUzawaPrimalDualGradientDescent
from algorithms.affine_constraints.distributed import DuMeng
from models.phi import IdentityFunction
from models.optimization_problem import AffineCouplingProblem
from models.cost import LocalCloudTradeoffCostFunction, QuadraticCostFunction
from models.agent import Agent
from models.constraints import LinearConstraint
from viz import plots, animation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import graph_utils

import numpy as np

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

    # [ define initial condition ]
    init_state = rng.uniform(0, 1, size=(N, d))
    # init_state = np.zeros(shape=(N,d))
    # print(f"init_state: {init_state}")

    # [ define constraints ]
    # time window
    T = 1

    # Agents (N=5) — all in MF/MB
    W_nom_GF = np.array([3, 4, 2, 2, 2], dtype=float)   # MFLOPs
    S_GB            = np.full(W_nom_GF.shape, 1)
    gamma_GB_per_GF = np.full(W_nom_GF.shape, 0.3)
    M_job_GB        = np.array([5, 2, 4, 5, 1], dtype=float)

    # Cloud capacities
    Pcloud_GFps = 3
    BWcloud_GBps = 3
    Mem_cloud_GB = 3

    B_list, b_list, b_sum = generate_B_b_simple(
        W_nom_GF=W_nom_GF[0:N],
        M_job_GB=M_job_GB[0:N],
        S_GB=S_GB[0:N],
        gamma_GB_per_GF=gamma_GB_per_GF[0:N],
        T=T,
        Pcloud_GFps=Pcloud_GFps,
        Mem_cloud_GB=Mem_cloud_GB,
        BWcloud_GBps=BWcloud_GBps,
        equal_split=True  # set False and pass w_* to customize splits
    )

    def setup_problem():
        agents = []
        for i in range(N):
            cost_params = LocalCloudTradeoffCostFunction.CostParams(
                alpha[i], beta[i], energy_tx[i], energy_task[i], time_task[i], time_rtt[i]
            )
            cost_fn = LocalCloudTradeoffCostFunction(cost_params)
            
            # cost_params = QuadraticCostFunction.CostParams(cc=2 * 0.5) # optimum: z_i = 0.5
            # cost_fn = QuadraticCostFunction(cost_params)
            
            lc = LinearConstraint(B_list[i], b_list[i])

            phi_fn = IdentityFunction(d)
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

        i1, i2 = 0, 1  # indices of the agents you want to visualize
        constraints = [
            ((B_list[i1][k, 0], B_list[i2][k, 0]), b_list[i1][k] + b_list[i2][k])
            for k in range(B_list[i1].shape[0])
        ]
        plots.plot_trajectory_plane_with_affine_constraints(result, constraints)

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
    if False:
        problem = setup_problem()
        centralized = ArrowHurwiczUzawaPrimalDualGradientDescent(problem)
        args = {
            "max_iter": 2500, 
            "stepsize": 0.001,
            "seed": seed,
            "dual_stepsize": 0.001
        }
        algo_params = ArrowHurwiczUzawaPrimalDualGradientDescent.AlgorithmParams(**args)
        result = centralized.run(algo_params)

        i1, i2 = 0, 1  # indices of the agents you want to visualize
        constraints = [
            ((B_list[i1][k, 0], B_list[i2][k, 0]), b_list[i1][k] + b_list[i2][k])
            for k in range(B_list[i1].shape[0])
        ]
        plots.plot_trajectory_plane_with_affine_constraints(result, constraints)

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
    if True:
        problem = setup_problem()
        distributed = DuMeng(problem)
        args = {
            "max_iter": 2500, 
            "stepsize": 0.01,
            "seed": seed,
            "beta": 0.05,
            "gamma": 0.05
        }
        algo_params = DuMeng.AlgorithmParams(**args)
        result = distributed.run(algo_params)

        i1, i2 = 0, 1  # indices of the agents you want to visualize
        constraints = [
            ((B_list[i1][k, 0], B_list[i2][k, 0]), b_list[i1][k] + b_list[i2][k])
            for k in range(B_list[i1].shape[0])
        ]
        plots.plot_trajectory_plane_with_affine_constraints(result, constraints)

        result.summary()

        plotter = plots.ConstrainedRunResultPlotter(result)
        plotter\
            .plot_cost()\
            .plot_grad_norm()\
            .plot_lambda(semilogy=False)\
            .plot_agents_trajectories()\
            .plot_sigma_trajectory()\
            .plot_aux()\
            .plot_Lagr_stationarity()\
            .plot_kkt_conditions(semilogy=False)\
            .plot_consensus_error(["lambda_traj", "sigma_traj"])\
            .show()
        # animation.animate_offloading_with_mean(result, problem.agents, interval=80)


if __name__ == "__main__":
    main()
        
