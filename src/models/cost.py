from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

class CostFunction(ABC):
    @dataclass
    class CostParams(ABC):
        ...

    @abstractmethod
    def cost_fn(zz_i, sigma) -> int:...
    
    @abstractmethod
    def nabla_1(zz_i, sigma): ...
    
    @abstractmethod
    def nabla_2(zz_i, sigma): ...

class QuadraticCostFunction(CostFunction):
    """
    Simple quadratic cost function:

        $$ \ell_i(z_i, \sigma(\textbf{z})) = \frac{1}{2} ||z_i||^2 + \frac{1}{2} ||\sigma(\textbf{z}) - c||^2 $$
    """

    @dataclass
    class CostParams(CostFunction.CostParams):
        cc : float

    def __init__(self, cost_params: CostParams):
        super().__init__()
        self.cost_params = cost_params

    def cost_fn(self, zz_i: np.ndarray, sigma: np.ndarray) -> float:
        
        # $$ \frac{1}{2} z_i^T z_i + \frac{1}{2} (\sigma(z) - c) ^T (\sigma(z) - c) $$
        
        return 0.5 * np.dot((zz_i - self.cost_params.cc), (zz_i - self.cost_params.cc)) + 0.5 * np.dot(sigma - self.cost_params.cc, sigma - self.cost_params.cc)

    def nabla_1(self, zz_i: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return zz_i

    def nabla_2(self, zz_i: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return sigma - self.cost_params.cc

class LocalCloudTradeoffCostFunction(CostFunction):
    @dataclass
    class CostParams(CostFunction.CostParams):
        alpha : float
        beta : float
        energy_tx : float
        energy_task : float
        time_task : float
        time_rtt : float

    def __init__(self, cost_params: CostParams):
        super().__init__()
        self.cost_params = cost_params

    def cost_fn(self, zz_i, sigma):
        assert np.ndim(zz_i) == 0 or (np.ndim(zz_i) == 1 and zz_i.size == 1), \
            f"LocalCloudTradeoffCostFunction expects scalar z_i, got shape {np.shape(zz_i)}"

        """ [ cost function ]:  $$ \ell_i(z_i, \sigma(z)) := (1-z_i) \cdot \ell_i^{loc}(z_i) + z_i \cdot \ell_i^{cloud}(\sigma(\textbf{z})) $$
        """

        def _rtt(sigma): # round-trip time, increasing in sigma
            # T = time_rtt * np.eye(zz_i.shape[0])
            # T @ sigma
            return self.cost_params.time_rtt * sigma

        def _cost_function_cloud(sigma):
                """ $$ \ell_i^{cloud}(\sigma(\textbf{z})) := \alpha t_i(\sigma(\textbf{z})) + \beta e_i^{tx} $$
                    
                    with $$ t_i(\sigma(\textbf{z})) = t\cdot \sigma(\textbf{z})$$
                """
                return self.cost_params.alpha * _rtt(sigma) + self.cost_params.beta * self.cost_params.energy_tx


        def _cost_function_local():
            """ $$ \ell_i^{loc}(z_i) := e_i^{task} + t_i^{task} $$
            """
            return self.cost_params.energy_task + self.cost_params.time_task
            

        local = _cost_function_local()
        cloud = _cost_function_cloud(sigma)
        # cost = (1 - zz_i).T @ local + zz_i.T @ cloud
        cost = (1 - zz_i) * local + zz_i * cloud

        return cost

    def nabla_1(self, zz_i, sigma):
        assert np.ndim(zz_i) == 0 or (np.ndim(zz_i) == 1 and zz_i.size == 1), \
            f"LocalCloudTradeoffCostFunction expects scalar z_i, got shape {np.shape(zz_i)}"

        # \nabla_1 \ell_i (z_i, \sigma(\textbf{z}))= - (e_i^{task} + t_i^{task}) + \alpha t \sigma(\textbf{z}) + \beta e_i^{tx} + \frac{\alpha \cdot t}{N}z_i
        # $$ \nabla_1 \ell_i (z_i, \sigma(\textbf{z}))= - (e_i^{task} + t_i^{task}) + \alpha t \sigma(\textbf{z}) + \beta e_i^{tx} $$
        grad =  -(self.cost_params.energy_task + self.cost_params.time_task) \
                + self.cost_params.alpha * self.cost_params.time_rtt * sigma \
                + self.cost_params.beta * self.cost_params.energy_tx \
                # + (self.alpha * self.time_rtt * zz_i)/N             # no chain rule term
        return grad

    def nabla_2(self, zz_i, sigma):
        assert np.ndim(zz_i) == 0 or (np.ndim(zz_i) == 1 and zz_i.size == 1), \
            f"LocalCloudTradeoffCostFunction expects scalar z_i, got shape {np.shape(zz_i)}"
        
        # $$ \nabla_2 \ell_i (z_i, \sigma(\textbf{z})) = z_i \cdot \alpha \cdot t  $$
        grad = zz_i * self.cost_params.alpha * self.cost_params.time_rtt
        return grad