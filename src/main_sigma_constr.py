import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Optional

from models.algorithm_interface import RunResult
from algorithms.sigma_constraints.centralized import SigmaConstraint
from algorithms.sigma_constraints.distributed import CIAO
from models.phi import IdentityFunction, LinearFunction, SquareComponentWiseFunction, WeightedQuadraticContribution
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
SHOW_COMPARISON  = False

SAVE_CENTRALIZED = True
SAVE_DISTRIBUTED = True
SAVE_COMPARISON  = True

def main():
    N = 7  # number of agents
    m = 1  # number of affine constraints per agent, B_i is a matrix (m,d)
    d = 1  # dimension of the state space
    constraint = np.ones(d) * 1.35

    seed = 7
    rng = np.random.default_rng(seed)

    # [ define \ell_i ] 
    # ============================================================
    # [ design an interior stationary point z_target ]
    # ============================================================
    #
    # We want to choose the parameters so that the UNCONSTRAINED
    # global objective has stationary point exactly at
    #
    # $$ z^\star = z_{\text{target}} \in (0,1)^N. $$
    #
    # The global cost is
    #
    # $$ f(z) = \sum_{i=1}^N \ell_i(z_i,\sigma(z)) $$
    #
    # with
    #
    # $$ \sigma(z) = \frac1N \sum_{j=1}^N \phi_j(z_j), $$
    #
    # $$ \phi_i(z_i) = w_i \big(z_i + \rho_i z_i^2\big), $$
    #
    # and
    #
    # $$ \ell_i(z_i,\sigma) = (1-z_i)\ell_i^{loc} + z_i\ell_i^{cloud} + \frac1N\Big(\sigma + \frac{\kappa_i}{2}\sigma^2\Big) + \frac{\gamma_i}{2} z_i^2. $$
    #
    # The exact i-th component of the global gradient used by the code is
    #
    # $$ \nabla_i f(z) = \nabla_1 \ell_i(z_i,\sigma) + \frac1N \nabla \phi_i(z_i)\sum_{j=1}^N \nabla_2 \ell_j(z_j,\sigma). $$
    #
    # Here:
    #
    # $$ \nabla_1 \ell_i(z_i,\sigma) = -\ell_i^{loc} + \ell_i^{cloud} + \gamma_i z_i, $$
    #
    # $$ \nabla_2 \ell_j(z_j,\sigma) = \frac{1+\kappa_j \sigma}{N}, $$
    #
    # $$ \nabla \phi_i(z_i) = w_i(1+2\rho_i z_i). $$
    #
    # Therefore at z_target we must enforce
    #
    # $$ 0 = -\ell_i^{loc} + \ell_i^{cloud} + \gamma_i z_i^\star + \frac1N w_i(1+2\rho_i z_i^\star)\sum_{j=1}^N \frac{1+\kappa_j\sigma^\star}{N}. $$
    #
    # If we define
    #
    # $$ \Delta_i := \ell_i^{loc} - \ell_i^{cloud}, $$
    #
    # then the exact condition is
    #
    # $$ \Delta_i = \gamma_i z_i^\star + \frac1N w_i(1+2\rho_i z_i^\star)\sum_{j=1}^N \frac{1+\kappa_j\sigma^\star}{N}. $$
    #
    # After computing Delta, we choose cloud baselines freely and set
    #
    # $$ \ell_i^{loc} = \ell_i^{cloud} + \Delta_i. $$
    #
    # This makes z_target an exact stationary point of the unconstrained problem.
    # ============================================================

    # ------------------------------------------------------------
    # 1) choose an interior target
    # ------------------------------------------------------------
    # Keep it safely away from the box boundary for nicer phase plots.
    z_target = rng.uniform(0.45, 0.85, size=N)

    # ------------------------------------------------------------
    # 2) choose the contribution-function parameters
    # ------------------------------------------------------------
    # $$ \phi_i(z_i) = w_i ( z_i + \rho_i z_i^2 ) $$
    workload = rng.uniform(1.6, 2.4, size=N)   # $$ w_i $$
    rho      = rng.uniform(0.4, 1.0, size=N)   # $$ \rho_i $$

    # ------------------------------------------------------------
    # 3) choose the cost-shape parameters
    # ------------------------------------------------------------
    # $$ \ell_i(z_i,\sigma) = ... + \frac1N(\sigma + \frac{\kappa_i}{2}\sigma^2)
    #    + \frac{\gamma_i}{2} z_i^2 $$
    #
    # gamma > 0 is what gives us strong convexity in z.
    kappa = rng.uniform(0.4, 0.9, size=N)      # $$ \kappa_i $$
    gamma = rng.uniform(2.5, 4.0, size=N)      # $$ \gamma_i > 0 $$

    # ------------------------------------------------------------
    # 4) compute sigma at the target
    # ------------------------------------------------------------
    
    # $$ \sigma^\star = \frac1N \sum_i w_i(z_i^\star + \rho_i (z_i^\star)^2) $$
    
    sigma_star = (1.0 / N) * np.sum(
        workload * (z_target + rho * z_target**2)
    )

    # ------------------------------------------------------------
    # 5) compute the exact chain-rule factor
    # ------------------------------------------------------------
    
    # $$ S^\star := \sum_{j=1}^N \nabla_2 \ell_j(z_j^\star,\sigma^\star) = \sum_{j=1}^N \frac{1+\kappa_j \sigma^\star}{N}. $$
    
    nabla2_at_target = (1.0 + kappa * sigma_star) / N     # shape (N,)
    S_star = np.sum(nabla2_at_target)                     # scalar

    # ------------------------------------------------------------
    # 6) compute the exact gap Delta_i = ell_loc_i - ell_cloud_i
    # ------------------------------------------------------------
    # Exact stationarity equation:
    #
    # $$ \Delta_i = \gamma_i z_i^\star + \frac1N w_i(1+2\rho_i z_i^\star) S^\star. $$
    Delta = (
        gamma * z_target
        + (1.0 / N) * workload * (1.0 + 2.0 * rho * z_target) * S_star
    )

    # ------------------------------------------------------------
    # 7) choose cloud baseline costs freely
    # ------------------------------------------------------------
    # These are only the "cloud" linear part:
    #
    # $$ \ell_i^{cloud} = E_i^{tx} + T_i^{tx}. $$
    #
    # Keep them moderate and positive.
    energy_tx = rng.uniform(0.8, 1.2, size=N)
    time_tx   = rng.uniform(0.8, 1.2, size=N)

    ell_cloud = energy_tx + time_tx

    # ------------------------------------------------------------
    # 8) recover the local baseline cost from Delta
    # ------------------------------------------------------------
    # $$ \ell_i^{loc} = \ell_i^{cloud} + \Delta_i. $$
    ell_loc = ell_cloud + Delta

    # Split local cost however you want; only the sum matters for the model.
    energy_task = 0.5 * ell_loc
    time_task   = 0.5 * ell_loc

    # ------------------------------------------------------------
    # 9) sanity checks
    # ------------------------------------------------------------
    print("z_target            =", z_target)
    print("sigma_star          =", sigma_star)
    print("S_star              =", S_star)
    print("Delta               =", Delta)
    print("ell_cloud           =", ell_cloud)
    print("ell_loc             =", ell_loc)

    # ------------------------------------------------------------
    # 10) verify the exact stationarity numerically
    # ------------------------------------------------------------
    # We evaluate the same gradient formula used by
    # OptimizationProblem.centralized_gradient_of_cost_fn_of_agent, but directly.
    #
    # For each i:
    #
    # $$ g_i(z^\star) = -\ell_i^{loc} + \ell_i^{cloud} + \gamma_i z_i^\star + \frac1N w_i(1+2\rho_i z_i^\star) S^\star. $$
    
    grad_at_target = (
        -ell_loc
        + ell_cloud
        + gamma * z_target
        + (1.0 / N) * workload * (1.0 + 2.0 * rho * z_target) * S_star
    )

    print("grad_at_target      =", grad_at_target)
    print("||grad_at_target||  =", np.linalg.norm(grad_at_target))

    # Optional: this should be very close to zero numerically.
    assert np.allclose(grad_at_target, 0.0, atol=1e-10), \
        "z_target is NOT stationary for the unconstrained global objective."

    # ------------------------------------------------------------
    # 11) initial condition
    # ------------------------------------------------------------
    init_state = rng.uniform(0.0, 1.0, size=(N, d))
    # init_state = np.zeros(shape=(N,d))
    # init_state = np.array([[0.25],[0.45]])
    # print(f"init_state: {init_state}")

    # init_state = np.zeros(shape=(N,d))
    # init_state = np.array([[0.1],[0.2], []])
    # print(f"init_state: {init_state}")

    def setup_problem():
        agents = []
        for i in range(N):
            cost_params = LocalCloudTradeoffCostFunction.CostParams(
                N, energy_task[i], time_task[i], energy_tx[i], time_tx[i], kappa[i], gamma[i]
            )
            cost_fn = LocalCloudTradeoffCostFunction(cost_params)
            
            # cost_params = QuadraticCostFunction.CostParams(cc=np.ones(shape=(d,))) # optimum: z_i = 0.5 (= 1/2 * c)
            # cost_fn = QuadraticCostFunction(cost_params)
            
            phi_fn = WeightedQuadraticContribution(d, workload[i], rho[i])
            # phi_fn = IdentityFunction(d)
            # phi_fn = LinearFunction(d, np.eye(d) * (2*i + 1))
            agent_i = Agent(i, cost_fn, phi_fn, init_state[i])
            
            agents.append(agent_i)

        args = {'edge_probability': 0.45, 'rng': rng}
        graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.ERDOS_RENYI, args)
    
        fig, axs = plt.subplots(figsize=(7, 4), nrows=1, ncols=2)
        title = f"Graph and Adj Matrix"
        fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)
        plots.show_graph_and_adj_matrix(fig, axs, graph, adj)
        if SHOW_DISTRIBUTED:
            plots.show_and_wait(fig)
        if SAVE_DISTRIBUTED and N>2:
            fig.savefig(f"../output/sigma_constraints_network_{N}.png")
    
        
        problem = ConstrainedSigmaProblem(agents, adj, seed, constraint)
        return problem
    

    # -----------------------
    # |     CENTRALIZED     |
    # -----------------------
    # [ start feasible - update init_state ]
    search_zone = 1
    iteration = 0
    problem = setup_problem()
    while not problem.check():
        print("unfeasible init...")
        init_state = rng.uniform(0, search_zone, size=(N, d))
        print(f"init_state: {init_state}")
        problem = setup_problem()
        search_zone *= 0.5
        iteration += 1
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
        "beta": 0.01,
        "gamma": 0.01,
        "theta": 0.0001
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
        .plot_aux(["qq_traj", "nn_traj"], semilogy=False)\
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
    # |     DISTRIBUTED     |
    # -----------------------
    problem = setup_problem()
    distributed = CIAO(problem)
    args = {
        "seed": seed,
        "max_iter": 2000,
        "stepsize": 0.01,
        "beta": 0.01,
        "gamma": 0.01,
        "theta": 0.000001
    }
    algo_params = CIAO.AlgorithmParams(**args)
    result_distributed_theta = distributed.run(algo_params)

    result_distributed_theta.summary()

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
    
    if N == 2 and d == 1:
        horizontal_plotter = plots.ComparisonConstrainedRunResultPlotter(problem, [result_centralized, result_distributed], layout="horizontal")
        filename = plots.plot_filename_comparison_from_results(results, "phase2d")
        horizontal_plotter\
            .clear()\
            .plot_phase2d()
        if SHOW_COMPARISON:
            horizontal_plotter.show()
        if SAVE_COMPARISON:
            horizontal_plotter.save(filename)

    results = [result_centralized, result_distributed]
    horizontal_plotter = plots.ComparisonConstrainedRunResultPlotter(problem, [result_distributed, result_distributed_theta], layout="horizontal")
    filename = plots.plot_filename_comparison_from_results(results, "theta")
    horizontal_plotter\
        .clear()\
        .plot_lagr_stationarity()
    if SHOW_COMPARISON:
        horizontal_plotter.show()
    if SAVE_COMPARISON:
        horizontal_plotter.save(filename)
    

if __name__ == "__main__":
    main()
        
