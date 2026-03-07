import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Optional, List, Tuple

from models.algorithm_interface import RunResult
from algorithms.affine_constraints.centralized import AugmentedPrimalDualGradientDescent, ArrowHurwiczUzawaPrimalDualGradientDescent
from algorithms.affine_constraints.distributed import DuMeng, DuMeng2, DuMeng3, DuMeng4, DuMeng5, DuMeng6
from models.phi import IdentityFunction, WeightedQuadraticContribution
from models.optimization_problem import AffineCouplingProblem
from models.cost import LocalCloudTradeoffCostFunction, QuadraticCostFunction
from models.agent import Agent
from models.constraints import LinearConstraint
from plots import plots, animation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import graph_utils

import numpy as np

SHOW_CENTRALIZED = True
SHOW_DISTRIBUTED = True
SHOW_COMPARISON  = True

SAVE_CENTRALIZED = False
SAVE_DISTRIBUTED = False
SAVE_COMPARISON  = False

def generate_B_b(
    # Per-agent params (shape (N,))
    W_nom_GF: np.ndarray,          # job size in GFLOPs (without offload)
    M_job_GB: np.ndarray,          # memory footprint on cloud (GB) if offloaded
    S_GB: np.ndarray,              # fixed GB per offload request
    gamma_GB_per_GF: np.ndarray,   # GB per GFLOP for offloaded data
    
    # Cloud/global budgets
    T: float,                      # time window [s]
    Pcloud_GFps: float,            # cloud peak compute [GFLOPs/s]
    Mem_cloud_GB: float,           # cloud memory budget [GB]
    BWcloud_GBps: float,  # cloud ingress bandwidth [GB/s]
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
    B_net_GB  = gamma_GB_per_GF
    # 3) memory
    B_mem_GB  = M_job_GB

    B_list = [np.array([[B_comp_GF[i]], [B_net_GB[i]], [B_mem_GB[i]]], dtype=float) for i in range(N)]

    # --- global budgets (not exposed directly to agents) ---
    b_comp_global = Pcloud_GFps * T
    b_net_global  = BWcloud_GBps * T
    b_mem_global  = Mem_cloud_GB

    b_list = []
    for i in range(N):
        b_i = np.array([
            # equal split
            1/N * b_comp_global,   # MFLOPs
            1/N * b_net_global,    # MB
            1/N * b_mem_global     # MB
        ], dtype=float)
        b_list.append(b_i)

    b_sum = np.sum(np.vstack(b_list), axis=0)  # (3,)

    return B_list, b_list, b_sum   

def generate_B_b_demand_driven(
    z_star: np.ndarray,         # unconstrained optimum (N,)
    computation: np.ndarray,    # compute demand per unit z
    network: np.ndarray,        # network per unit z
    memory: np.ndarray,         # memory per unit z
    seed: int = 0,
):
    """
    Demand-driven physically interpretable cloud constraints.

    Coupling constraint:

        sum_i B_i z_i <= sum_i b_i

    where B_i ∈ R^{3×1}, b_i ∈ R^3.
    """

    rng = np.random.default_rng(seed)
    N = len(z_star)

    # -----------------------------
    # 1) Build per-agent matrices B_i
    # -----------------------------

    B_list = []
    for i in range(N):
        B_i = np.array([
            [computation[i]],    
            [network[i]],
            [memory[i]]        
        ], dtype=float)

        # ensures rank(B_i)=1 (nonzero column)
        B_list.append(B_i)

    # -----------------------------
    # 2) Compute demand at z_star
    # -----------------------------
    demand_comp = np.sum(computation * z_star)
    demand_net  = np.sum(network     * z_star)
    demand_mem  = np.sum(memory      * z_star)

    # -----------------------------
    # 3) Tighten capacities
    # -----------------------------
    theta = rng.uniform(0.5, 0.8, size=3)

    C_comp = theta[0] * demand_comp
    C_net  = theta[1] * demand_net
    C_mem  = theta[2] * demand_mem

    b_global = np.array([C_comp, C_net, C_mem], dtype=float)

    # -----------------------------
    # 4) Split equally among agents
    # -----------------------------
    b_list = [(1.0/N) * b_global for _ in range(N)]
    b_sum = b_global.copy()

    return B_list, b_list, b_sum

def generate_B_b_full_row_rank(
    z_star: np.ndarray,           # (N,)
    workload: np.ndarray,         # (N,)
    memory: np.ndarray,         # (N,)
    network: np.ndarray,  # (N,)
    seed: int = 0,
    theta_range=(0.60, 0.85),     # how much we tighten relative to demand at z_star
    weights=(1.0, 1.0, 1.0),      # weights for (compute, net, mem) to make a single scalar constraint
):
    """
    m=d=1:
        sum_i B_i z_i <= b_sum
    with B_i in R^{1x1}. Full row rank holds as long as B_i != 0.
    """
    rng = np.random.default_rng(seed)
    N = len(z_star)

    wC, wN, wM = weights

    # per-agent scalar "resource per unit offload"
    # B_i := wC*W_i + wN*gamma_i + wM*M_i  (positive => rank 1)
    B_scalar = wC * workload + wN * network + wM * memory
    B_scalar = np.maximum(B_scalar, 1e-6)  # ensure nonzero

    # demand at z_star
    demand = float(np.sum(B_scalar * z_star))

    # tighten so z_star violates: b_sum = theta * demand, theta<1
    theta = rng.uniform(*theta_range)
    b_sum = np.array([theta * demand], dtype=float)  # (1,)

    # package into required structure
    B_list = [np.array([[B_scalar[i]]], dtype=float) for i in range(N)]  # (1,1) each
    b_list = [(1.0 / N) * b_sum for _ in range(N)]                       # each (1,)
    return B_list, b_list, b_sum

def generate_B_b_cloud(
    z_star: np.ndarray,         # (N,)
    workload: np.ndarray,       # (N,)
    memory: np.ndarray,         # (N,)
    network: np.ndarray,        # (N,)
    seed: int = 0,
    theta_range=(0.60, 0.95),     # tighten each resource budget relative to demand at z_star
    ensure_active_all=True,       # if True, make ALL 3 constraints violated by z_star
):
    """
    m=3, d=1:
        sum_i B_i z_i <= b_sum
    with B_i in R^{3x1}. This violates "full row rank" (rank<=1), but is interpretable.

    Constraint rows correspond to:
      1) compute  sum_i W_i z_i <= C_comp
      2) net      sum_i gamma_i z_i <= C_net
      3) memory   sum_i M_i z_i <= C_mem
    """
    rng = np.random.default_rng(seed)
    N = len(z_star)

    # Build B_i (3x1) per agent
    B_list = [
        np.array([[workload[i]],
                  [network[i]],
                  [memory[i]]], dtype=float)
        for i in range(N)
    ]

    # demands at z_star
    demand_comp = float(np.sum(workload * z_star))
    demand_net  = float(np.sum(network * z_star))
    demand_mem  = float(np.sum(memory * z_star))
    demand_vec = np.array([demand_comp, demand_net, demand_mem], dtype=float)
    
    # choose theta per resource (all <1 to cut)
    if ensure_active_all:
        theta = rng.uniform(theta_range[0], theta_range[1], size=m)
    else:
        # optional: only one binds hard, others looser (still <1 if you want cuts)
        theta = np.array([rng.uniform(*theta_range), 0.95, 0.95], dtype=float)

    b_sum = theta * demand_vec  # (3,)
    print(f"Cloud capacity: computation={b_sum[0]:.2f}, network={b_sum[1]:.2f}, memory={b_sum[2]:.2f}")

    # split equally among agents
    b_list = [(1.0 / N) * b_sum for _ in range(N)]

    return B_list, b_list, b_sum

def main():
    N = 5  # number of agents
    m = 1  # number of affine constraints per agent, B_i is a matrix (m,d)
    d = 1  # dimension of the state space
    seed = 3
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

    if False:
        # ------------------------------------------------------------
        # Target heterogeneous interior solution
        # ------------------------------------------------------------
        z_target = rng.uniform(0.65, 0.9, N)

        # ------------------------------------------------------------
        # Aggregative contribution parameters
        # ------------------------------------------------------------
        workload = rng.uniform(1.8, 2.3, N)   # w_i
        rho      = rng.uniform(1.5, 2.12, N)
        gamma    = rng.uniform(1.8, 2.4,  N)

        # congestion curvature parameter
        kappa = np.ones(N) * 0.5

        # ------------------------------------------------------------
        # Compute sigma at target
        # ------------------------------------------------------------
        sigma_star = (1.0/N) * np.sum(
            workload * (z_target + rho * z_target**2)
        )

        # effective coupling at equilibrium
        K_eff = (1.0/N) * (1.0 + 2.0 * kappa * sigma_star)

        # ------------------------------------------------------------
        # Construct Δ_i so z_target is stationary
        # ------------------------------------------------------------
        Delta = (
            workload * K_eff
            + z_target * (gamma + 2 * rho * workload * K_eff)
        )

        # ------------------------------------------------------------
        # Build baseline costs
        # ------------------------------------------------------------
        energy_tx = rng.uniform(0.7, 1.1, N)
        time_tx   = rng.uniform(0.7, 1.1, N)

        ell_cloud = energy_tx + time_tx
        ell_loc   = ell_cloud + Delta

        energy_task = 0.5 * ell_loc
        time_task   = 0.5 * ell_loc

    if False:
        # ------------------------------------------------------------
        # Target heterogeneous interior solution
        # ------------------------------------------------------------
        # We choose a desired equilibrium profile
        #
        # $$ z_i^{\star} \in (0.65, 0.90) $$
        #
        # centered around 0.5 to create a genuine local/cloud tradeoff.
        #
        z_target = rng.uniform(0.65, 0.90, N)


        # ------------------------------------------------------------
        # Aggregative contribution parameters
        # ------------------------------------------------------------
        # Contribution function:
        #
        # $$ \phi_i(z_i) = w_i (z_i + \rho_i z_i^2) $$
        #
        # where:
        #   w_i   controls the impact of agent i on congestion,
        #   rho_i controls nonlinear congestion sensitivity.
        #
        workload = rng.uniform(1, 3, N)   # w_i
        rho      = rng.uniform(0.1, 0.7, N)
        gamma    = rng.uniform(0.8, 2.5,  N)  # strong convexity


        # ------------------------------------------------------------
        # Congestion intensity
        # ------------------------------------------------------------
        # Cost structure:
        #
        # $$ \ell_i(z_i,\sigma) = (1-z_i)\ell_i^{loc} + z_i\ell_i^{cloud} + \frac{\kappa}{N}\sigma + \frac{\gamma_i}{2}z_i^2 $$
        #
        # Trackers estimate the AVERAGE of:
        #
        # $$ \nabla_2 \ell_i = \frac{\kappa}{N} $$
        #
        # Therefore the tracker converges to:
        #
        # $$ v_i \to \frac{\kappa}{N}. $$
        #
        kappa_scalar = 0.5
        kappa = np.ones(N) * kappa_scalar


        # Effective congestion gain appearing in the gradient:
        #
        # Since nabla_phi returns φ_i'(z_i) (without 1/N),
        # and v_i → κ/N,
        #
        # the effective term in the gradient is:
        #
        # $$ \phi_i'(z_i) \cdot \frac{\kappa}{N} $$
        #
        # so we define:
        #
        # $$ K_{\text{eff}} = \frac{\kappa}{N}. $$
        #
        K_eff = kappa_scalar / N


        # ------------------------------------------------------------
        # Stationary condition of the implemented algorithm
        # ------------------------------------------------------------
        #
        # The algorithm solves:
        #
        # $$ 0 = -\ell_i^{loc} + \ell_i^{cloud} + \gamma_i z_i + K_{\text{eff}} w_i (1 + 2\rho_i z_i). $$
        #
        # Define:
        #
        # $$ \Delta_i := \ell_i^{loc} - \ell_i^{cloud}. $$
        #
        # Rearranging gives:
        #
        # $$ (\gamma_i + 2 K_{\text{eff}} w_i \rho_i) z_i = \Delta_i - K_{\text{eff}} w_i. $$
        #
        # Hence:
        #
        # $$ z_i^\star = \frac{\Delta_i - K_{\text{eff}} w_i} {\gamma_i + 2 K_{\text{eff}} w_i \rho_i}. $$
        #
        # ------------------------------------------------------------
        # Invert the formula to impose z_i^* = z_target
        # ------------------------------------------------------------
        #
        # We solve for Δ_i:
        #
        # $$ \Delta_i = K_{\text{eff}} w_i + z_i^{target} (\gamma_i + 2 K_{\text{eff}} w_i \rho_i). $$
        #
        Delta = (
            K_eff * workload
            + z_target * (gamma + 2*K_eff*workload*rho)
        )


        # ------------------------------------------------------------
        # Build baseline local/cloud costs
        # ------------------------------------------------------------
        #
        # By definition:
        #
        # $$ \Delta_i = \ell_i^{loc} - \ell_i^{cloud}. $$
        #
        # We choose cloud costs arbitrarily and enforce:
        #
        # $$ \ell_i^{loc} = \ell_i^{cloud} + \Delta_i. $$
        #
        energy_tx = rng.uniform(0.7, 1.1, N)
        time_tx   = rng.uniform(0.7, 1.1, N)

        ell_cloud = energy_tx + time_tx
        ell_loc   = ell_cloud + Delta

        # Split local cost into energy/time components
        energy_task = 0.5 * ell_loc
        time_task   = 0.5 * ell_loc


        # ------------------------------------------------------------
        # Verify predicted equilibrium
        # ------------------------------------------------------------
        #
        # Plug parameters into closed-form expression:
        #
        z_star = (
            (Delta - K_eff*workload)
            / (gamma + 2*K_eff*workload*rho)
        )

    # [ define constraints ]
    memory  = rng.uniform(1,3,N)
    network = rng.uniform(1,3,N)

    # B_list, b_list, b_sum = generate_B_b_full_row_rank(z_target, workload, memory, network)

    # [ simple constraints ]
    B_list = np.ones((N,m,d))
    B_list[0] = np.array([5])
    B_list[1] = np.array([3])
    b_list = np.ones((N,m)) * 3 / N

    def setup_problem():
        agents = []
        for i in range(N):
            cost_params = LocalCloudTradeoffCostFunction.CostParams(
                N, energy_task[i], time_task[i], energy_tx[i], time_tx[i], kappa[i], gamma[i]
            )
            cost_fn = LocalCloudTradeoffCostFunction(cost_params)
            
            # cost_params = QuadraticCostFunction.CostParams(cc=np.ones(shape=(d,))) # optimum: z_i = 0.5 (= 1/2 * c)
            # cost_fn = QuadraticCostFunction(cost_params)
            
            lc = LinearConstraint(B_list[i], b_list[i])

            # phi_fn = IdentityFunction(d)
            phi_fn = WeightedQuadraticContribution(d, workload[i], rho[i])
            agent_i = Agent(i, cost_fn, phi_fn, init_state[i], local_constraint=lc)
            
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
    
        problem = AffineCouplingProblem(agents, adj, seed)
        return problem

    # -----------------------
    # |     CENTRALIZED     |
    # -----------------------    
    # [ start feasible - update init_state ]
    if True:
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

        centralized = AugmentedPrimalDualGradientDescent(problem)
        args = {
            "max_iter": 2000, 
            "stepsize": 0.01,
            "seed": seed,
            "rho": 3,
            "dual_stepsize": N * 0.01
        }
        algo_params = AugmentedPrimalDualGradientDescent.AlgorithmParams(**args)
        result_centralized = centralized.run(algo_params)
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
                
    # animation.animate_offloading_with_mean(result, problem.agents, interval=80)

    if False:
        # -----------------------
        # |     CENTRALIZED     |
        # -----------------------
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

        centralized = ArrowHurwiczUzawaPrimalDualGradientDescent(problem)
        args = {
            "max_iter": 2000,
            "stepsize": 0.01,
            "seed": seed,
            "dual_stepsize": 0.01
        }
        algo_params = ArrowHurwiczUzawaPrimalDualGradientDescent.AlgorithmParams(**args)
        result_centralized = centralized.run(algo_params)
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
            filename = plots.plot_filename_from_result(result_centralized, "perf")
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
    problem = setup_problem() # reuse same init_state
    distributed = DuMeng2(problem)
    args = {
        "max_iter": 2000,
        "stepsize": 0.01,
        "seed": seed,
        "beta":  0.01,
        "gamma": 0.05
    }
    algo_params = DuMeng.AlgorithmParams(**args)
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
        .plot_aux(["aa_traj", "yy_traj"], semilogy=False)\
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
        # animation.animate_offloading_with_mean(result_distributed, problem.agents, interval=80)

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
        .plot_agents_trajectories()\
        .plot_lambda_trajectory(semilogy=False)
    if SHOW_COMPARISON:
        grid_plotter.show()
    if SAVE_COMPARISON:
        grid_plotter.save(filename)

    filename = plots.plot_filename_comparison_from_results(results, "perf2")
    grid_plotter\
        .clear()\
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


if __name__ == "__main__":
    main()
        
