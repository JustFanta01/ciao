# tests/conftest.py
import os, sys
import numpy as np
import pytest

import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT  = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.agent import Agent
from src.models.cost import LocalCloudTradeoffCostFunction, QuadraticCostFunction
from src.models.phi import IdentityFunction
from src.models.constraints import LinearConstraint
from src.models.optimization_problem import OptimizationProblem, AffineCouplingProblem
from utils import graph_utils

# ---- Parametri globali ----
_DEFAULT_SEED = 3

# @pytest.fixture(scope="function") # so it always gives the SAME FIRST NUMBERS starting from that seed.
# def rng():
#     return np.random.default_rng(_DEFAULT_SEED)

@pytest.fixture(scope="session", params=[1, 5])
def N(request):
    return request.param

@pytest.fixture(scope="session", params=[1, 3])
def d(request):
    return request.param

@pytest.fixture(scope="session", params=[1, 3])
def m(request):
    return request.param

@pytest.fixture(scope="session")
def adj(N):
    # ER + Metropolis-Hastings
    graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(
        N,
        graph_utils.GraphType.ERDOS_RENYI,
        {'edge_probability': 0.45, 'seed': _DEFAULT_SEED}
    )
    return adj

@pytest.fixture(params=["Quadratic", "LocalCloudTradeoff"])
def cost_kind(request):
    return request.param

def make_cost_fn(name):
    if name == "LocalCloudTradeoff":
        alpha       = 1
        beta        = 1
        energy_task = 4
        time_task   = 3
        energy_tx   = 3
        time_rtt    = 4
        cost_params = LocalCloudTradeoffCostFunction.CostParams(
            alpha, beta, energy_tx, energy_task, time_task, time_rtt
        )
        cost_function = LocalCloudTradeoffCostFunction(cost_params)
    elif name == "Quadratic":
        cost_params = QuadraticCostFunction.CostParams(cc=2 * 0.5) # optimum: z_i = 0.5
        cost_function = QuadraticCostFunction(cost_params)
    return cost_function

    
# -------- Problema aggregativo (no vincoli) --------
@pytest.fixture
def problem_agg(N, d, adj, cost_kind):
    if d > 1 and cost_kind == "LocalCloudTradeoff":
        pytest.skip("LocalCloudTradeoffCostFunction è definita solo per d == 1.")
    
    rng = np.random.default_rng(_DEFAULT_SEED)
    init_state = rng.uniform(0, 1, size=(N, d))

    agents = []
    for i in range(N):
        cost_fn = make_cost_fn(cost_kind)
        phi_fn = IdentityFunction(d)
        ag = Agent(i, cost_fn, phi_fn, init_state[i])
        agents.append(ag)

    return OptimizationProblem(agents, adj, seed=_DEFAULT_SEED)

@pytest.fixture
def problem_agg_factory(N, d, adj, cost_kind):
    def _make():
        if d > 1 and cost_kind == "LocalCloudTradeoff":
            pytest.skip("LocalCloudTradeoffCostFunction è definita solo per d == 1.")

        rng = np.random.default_rng(_DEFAULT_SEED)
        init_state = rng.uniform(0, 1, size=(N, d))
        print(f"init_state: {init_state}")

        agents = []
        for i in range(N):
            cost_fn = make_cost_fn(cost_kind)
            phi_fn = IdentityFunction(d)
            ag = Agent(i, cost_fn, phi_fn, init_state[i])
            agents.append(ag)

        return OptimizationProblem(agents, adj, seed=_DEFAULT_SEED)

    return _make

# -------- Problema con vincoli affini --------
@pytest.fixture
def problem_affine(adj):
    # Config minimale come nel tuo main_affine_constr.py:
    N = 2; d = 1; m = 1
    init_state = np.array([[0.1],[0.3]])

    agents = []
    for i in range(N):
        # costi quadratici semplici (opt z_i = 0.5)
        cost_params = QuadraticCostFunction.CostParams(cc=2 * 0.5)
        cost_fn = QuadraticCostFunction(cost_params)
        # vincoli locali che sommano in un vincolo globale
        if i == 0:
            A = np.ones((m, d)); b = np.ones((m))*0.5
        else:
            A = np.ones((m, d))*2; b = np.ones((m))*0.5
        lc = LinearConstraint(A, b)
        phi_fn = IdentityFunction(d)
        ag = Agent(i, cost_fn, phi_fn, init_state[i], local_constraint=lc)
        agents.append(ag)

    return AffineCouplingProblem(agents, adj, seed=_DEFAULT_SEED)

# -------- Factory algoritmi + parametri --------
@pytest.fixture(params=[
    ("GD",  {"max_iter": 2500, "stepsize": 0.01, "seed": _DEFAULT_SEED}),
    ("AT",  {"max_iter": 2500, "stepsize": 0.01, "seed": _DEFAULT_SEED}),
])
def algo_agg(request):
    name, kw = request.param
    if name == "GD":
        from src.algorithms.aggregative_tracking.centralized.CentralizedGradientMethod import CentralizedGradientMethod as Algo
    elif name == "AT":
        from src.algorithms.aggregative_tracking.distributed.AggregativeTracking import AggregativeTracking as Algo
    params = Algo.AlgorithmParams(**kw)
    return Algo, params

@pytest.fixture(params=[
    ("AHU",  {"max_iter": 1200, "stepsize": 0.01, "dual_stepsize": 0.01, "seed": _DEFAULT_SEED}),
    ("APDGD",{"max_iter": 1200, "stepsize": 0.01, "dual_stepsize": 0.05, "rho": 3.0, "seed": _DEFAULT_SEED}),
])
def algo_affine(request):
    name, kw = request.param
    if name == "AHU":
        from src.algorithms.affine_constraints.centralized.ArrowHurwiczUzawaPrimalDualGradientDescent import ArrowHurwiczUzawaPrimalDualGradientDescent as Algo
    elif name == "APDGD":
        from src.algorithms.affine_constraints.centralized.AugmentedPrimalDualGradientDescent import AugmentedPrimalDualGradientDescent as Algo
    params = Algo.AlgorithmParams(**kw)
    return Algo, params
