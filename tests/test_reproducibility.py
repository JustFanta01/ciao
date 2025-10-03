# tests/test_reproducibility.py
import numpy as np

def test_same_seed_same_trajectory(problem_agg_factory):
    from src.algorithms.aggregative_tracking.centralized.CentralizedGradientMethod import CentralizedGradientMethod as CGM
    # A algo.run() will leave the Agents inside the Problem with a "dirty" state.
    # The second run will start from previous run's last state.

    algo1 = CGM(problem_agg_factory())
    algo2 = CGM(problem_agg_factory())

    p = CGM.AlgorithmParams(max_iter=2000, stepsize=0.01, seed=123)
    res1 = algo1.run(p)
    res2 = algo2.run(p)

    assert np.allclose(res1.zz_traj[-1], res2.zz_traj[-1])
    assert np.allclose(res1.cost_traj[-1], res2.cost_traj[-1])

def test_bad_adj_raises():
    import numpy as np
    from src.models.agent import Agent
    from src.models.cost import QuadraticCostFunction
    from src.models.phi import IdentityFunction
    from src.models.optimization_problem import OptimizationProblem

    N, d = 3, 1
    agents = []
    for i in range(N):
        params = QuadraticCostFunction.CostParams(cc=1.0)
        cost = QuadraticCostFunction(params)
        ag = Agent(i, cost, IdentityFunction(d), np.array([0.2]))
        agents.append(ag)

    # Non doubly-stochastic (riga 0 non somma a 1)
    bad_adj = np.array([[0.0,1.0,0.0],
                        [0.5,0.0,0.5],
                        [0.5,0.5,0.0]])

    import pytest
    with pytest.raises(ValueError):
        OptimizationProblem(agents, bad_adj, seed=0)
