import numpy as np
from .cost import CostFunction
from .phi import AggregationContributionFunction
from .constraints import Constraint
from multipledispatch import dispatch


class Agent:
    def __init__(self, idx, cost_fn : CostFunction, phi : AggregationContributionFunction, init_state : np.array, local_constraint : Constraint = None):
        self.idx = idx
        self.zz = init_state
        self._phi_obj = phi
        self._cost_fn_obj = cost_fn
        self.local_constraint = local_constraint

        # print(f"[Agent-{idx}] init cond. = {self.zz}")

        # Tieni gli Agent slim e uniformi, e sposti tutta la variabilità dentro al runner, ma in modo più strutturato:
        # Agent ha un dizionario agent.buffers: dict[str, np.ndarray].
        # Ogni algoritmo, quando parte, decide quali chiavi popolare ("s", "v", "lambda", …).
        # Il runner può poi mettere un dump finale di tutti i buffers in RunResult.aux.
        
        # Agent mantiene solo lo stato corrente (e i buffer correnti se l’algoritmo ne richiede).
        # RunResult mantiene le traiettorie complete (z, s, v, costi, ecc.).
        self.buffers = {}   # algo-specific, initialized by Algorithm

    # "getitem" → agent["key"]
    def __getitem__(self, key):
        if key == "zz":
            return self.zz
        return self.buffers.get(key, None)

    # "setitem" → agent["key"] = value
    def __setitem__(self, key, value):
        if key == "zz":
            # TODO: add the prejection in [0,1] here?
            self.zz = value
        else:
            self.buffers[key] = value

    def __contains__(self, key):
        return (key == "zz") or key in self.buffers

    @dispatch(np.ndarray, np.ndarray)
    def cost_fn(self, zz, sigma):
        # $$ \ell_i(z_i, \sigma(\textbf{z})) $$
        return self._cost_fn_obj.cost_fn(zz, sigma)
    
    @dispatch(np.ndarray)
    def cost_fn(self, sigma):
        # $$ \ell_i(z_i, \sigma(\textbf{z})) $$
        return self.cost_fn(self.zz, sigma)
    
    def get_cost_params(self):
        return self._cost_fn_obj.cost_params
    
    @dispatch(np.ndarray, np.ndarray)
    def nabla_1(self, zz, sigma):
        # $$ \nabla_1 \ell_i(z_i, \sigma(\textbf{z})) $$
        return self._cost_fn_obj.nabla_1(zz, sigma) 
    
    @dispatch(np.ndarray)
    def nabla_1(self, sigma):
        # $$ \nabla_1 \ell_i(z_i, \sigma(\textbf{z})) $$
        return self.nabla_1(self.zz, sigma) 

    @dispatch(np.ndarray, np.ndarray)
    def nabla_2(self, zz, sigma):
        # $$ \nabla_2 \ell_i(z_i, \sigma(\textbf{z})) $$
        return self._cost_fn_obj.nabla_2(zz, sigma)
    
    @dispatch(np.ndarray)
    def nabla_2(self, sigma):
        # $$ \nabla_2 \ell_i(z_i, \sigma(\textbf{z})) $$
        return self.nabla_2(self.zz, sigma)

    @dispatch(np.ndarray)
    def phi(self, zz):
        return self._phi_obj.eval(zz)

    @dispatch()
    def phi(self):
        return self.phi(self.zz)
    
    @dispatch(np.ndarray)
    def nabla_phi(self, zz):
        return self._phi_obj.nabla(zz)
    
    @dispatch()
    def nabla_phi(self):
        return self.nabla_phi(self.zz)

    @dispatch(np.ndarray)
    def check_constraint(self, zz):
        if self.local_constraint is not None:
            return self.local_constraint.check(zz)
        else:
            raise ValueError(f"Local constraint of Agent-{self.idx} is None")
        
    @dispatch()
    def check_constraint(self):
        self.check_constraint(self.zz)