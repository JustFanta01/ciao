# tests/test_algorithms_basic.py
import numpy as np
import pytest

_TOL_COST_MONO   = 1e-6       # tolleranza micro su monotonia
_TOL_FINAL_CLOSE = 1e-2
_TOL_GRAD_DEC    = 1e-3
_TOL_RESIDUAL    = 5e-2

# -------------------- Vincoli affini: feasibility + stabilità --------------------

def _global_residual(problem):
    # come in AffineCouplingProblem.global_residual()
    z_tot = np.concatenate([ag["zz"] for ag in problem.agents])
    B, b = problem.global_matrices()
    return B @ z_tot - b

def test_affine_algorithms_feasibility(problem_affine, algo_affine):
    Algo, params = algo_affine
    algo = Algo(problem_affine)
    res = algo.run(params)

    # costo ben definito
    assert np.isfinite(res.cost_traj).all()

    # residuo globale <= tolleranza (<=0 è feasibility; permettiamo piccolo >0 per numerica)
    residual = _global_residual(problem_affine)
    assert residual.shape == (1,)
    assert residual <= _TOL_RESIDUAL

def test_affine_stationarity_proxy(problem_affine, algo_affine):
    # Verifica che la norma gradiente tenda giù anche con vincoli (proxy KKT)
    Algo, params = algo_affine
    algo = Algo(problem_affine)
    grad = algo.run(params).grad_traj

    dims = len(grad.shape)
    # the norm of the whole matrix / vector present at each iteration
    # so if grad.shape = (K, N, d1, d2, d3)
    # grad_norm[k] is the norm of the (N, d1, d2, d3) matrix at grad[k]
    grad_norm = np.linalg.norm(grad, axis=tuple(range(1, dims)))
    K = grad_norm.shape[0]

    third_of_the_way = grad_norm[: max(2, len(grad_norm)//3)]
    assert grad_norm[-1] <= np.median(third_of_the_way) + 5e-3
