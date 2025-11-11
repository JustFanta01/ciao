import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Literal
from models.optimization_problem import OptimizationProblem, ConstrainedOptimizationProblem
from models.algorithm_interface import RunResult
from matplotlib.patches import Patch, Rectangle

def show_graph_and_adj_matrix(fig, axs, graph, adj_matrix=None):
    """
    Show a networkx graph and its adjacency matrix.
    """
    nx.draw(graph, with_labels=True, ax=axs[0])
    if adj_matrix is None:
        adj_matrix = nx.adjacency_matrix(graph).toarray()
    cax = axs[1].matshow(adj_matrix, cmap='plasma')
    fig.colorbar(cax)
def close(event):
    if event.key == 'q':
        plt.close()   # close figure
    elif event.key == 'n':
        pass          # future extension: skip to next figure

def show_and_wait(fig=None):
    if fig is None:
        plt.gcf().canvas.mpl_connect('key_press_event', close)
    else:
        fig.canvas.mpl_connect('key_press_event', close)
    plt.show(block=True)

@dataclass
class PlotTask:
    name: str
    kwargs: Dict[str, Any]

class BaseRunResultPlotter:
    """
    Fluent interface for creating common plots for a single RunResult.
    Example:
        BaseRunResultPlotter(run).plot_cost().plot_gradient_norm().show()
    """

    def __init__(self, problem : OptimizationProblem, result : RunResult, *, figsize: Tuple[int, int] = (12, 6)):
        self.problem = problem
        self.result = result
        self._tasks: List[PlotTask] = []
        self._figsize = figsize
        self._registry: Dict[str, Callable] = {
            "cost": self._render_cost,
            "grad_norm": self._render_grad_norm,
            "aux_traj": self._render_aux,
            "agent_traj": self._render_agent_traj,
            "sigma_traj": self._render_sigma_traj,
            "phase2d": self._render_phase2d
        }

    def plot_cost(self, semilogy: bool = True, legend: Optional[str] = None):
        self._tasks.append(PlotTask("cost", {"semilogy": semilogy, "legend": legend}))
        return self
    
    def plot_aux(self, keys: list[str], semilogy: bool = False, legend: Optional[str] = None):
        """Plot any aux variable stored in RunResult. Shape can be (K,), (K,N) or (K,d)."""
        self._tasks.append(PlotTask("aux_traj", {"keys": keys, "semilogy": semilogy, "legend": legend}))
        return self

    def plot_phase2d(self):
        args = {
            "title": "Trajectory",
        }
        self._tasks.append(PlotTask("phase2d", args))
        return self

    def plot_grad_norm(self, semilogy: bool = True, legend: Optional[str] = None):
        self._tasks.append(PlotTask("grad_norm", {"semilogy": semilogy, "legend": legend}))
        return self

    def plot_agents_trajectories(self,semilogy: bool = False, legend: Optional[str] = None):
        self._tasks.append(
            PlotTask("agent_traj", {"semilogy": semilogy, "legend": legend})
        )
        return self

    def plot_sigma_trajectory(self,semilogy: bool = False, legend: Optional[str] = None):
        self._tasks.append(
            PlotTask("sigma_traj", {"semilogy": semilogy, "legend": legend})
        )
        return self

    def clear(self):
        self._tasks.clear()
        return self

    # ---------- rendering ----------
    def render(self):
        n = len(self._tasks)
        if n == 0:
            raise ValueError("No plots requested. Call .plot_*() before render().")
        
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=self._figsize)
        axes = np.atleast_1d(axes).ravel()

        for ax, task in zip(axes, self._tasks):
            renderer = self._registry.get(task.name)
            if renderer is None:
                ax.text(0.5, 0.5, f"Missing renderer: {task.name}", ha="center")
                continue
            renderer(ax, **task.kwargs)
        for ax in axes[n:]:
            ax.axis("off")
        fig.tight_layout()
        return fig

    def show(self):
        fig = self.render()
        plt.show()
        return self

    def save(self, path: str, **kwargs):
        fig = self.render()
        fig.savefig(path, **kwargs)
        return self

    def _render_cost(self, ax, *, semilogy: bool, legend: Optional[str]):
        cost = np.asarray(self.result.cost_traj)  # (K, ?)
        if cost.ndim == 1: # (K,)
            self._timeseries(
                ax, cost, 
                legend = legend or r'$\ell(z) = \sum_{i=1}^{N} \ell_i(z_i, \sigma(z))$',
                title="Cost evolution", 
                ylabel="cost", 
                semilogy=semilogy
            )


        if cost.ndim == 2: # (K, costs)
            assert cost.shape[1] == 2, "\ell(z), Lagrangian, and ?"
            self._timeseries(
                ax, cost[:, 1], 
                legend = legend or r'$\mathcal{L}(z,\lambda)$',
                title="Cost evolution of Lagrangian", 
                ylabel="cost", 
                semilogy=semilogy
            )

            self._timeseries(
                ax, cost[:, 0], 
                legend = legend or r'$\ell(z) = \sum_{i=1}^{N} \ell_i(z_i, \sigma(z))$',
                title="Cost evolution", 
                ylabel="cost", 
                semilogy=semilogy
            )
        
    def _render_grad_norm(self, ax, *, semilogy: bool, legend: Optional[str]):
        grad = np.asarray(self.result.grad_traj)  # (K,N,d) or (K,d)
        # the norm of the whole matrix / vector present at each iteration
        # so if grad.shape = (K, N, d1, d2, d3)
        # grad_norm[k] is the norm of the (N, d1, d2, d3) matrix at grad[k]
        grad_norm = np.linalg.norm(grad, axis=tuple(range(1, grad.ndim)))
        
        assert grad_norm.ndim == 1, "grad_norm.ndim != 1, expected (K,)"
        
        self._timeseries(
            ax, grad_norm,
            legend = legend or r'$ ||\nabla \ell(z) ||^2 $', 
            title="Gradient norm evolution", 
            ylabel="gradient norm", 
            semilogy=semilogy,
            split_components=True,
            aggregate_agents="all"
        )

    def _render_agent_traj(self, ax, *, semilogy: bool, legend: Optional[str]):
        y = self.result.zz_traj
        assert y.ndim == 3, "zz_traj.ndim != 3, expected (K,N,d)"
        
        # (K, N, d)
        self._timeseries(
            ax, y,
            semilogy=semilogy,
            legend = legend or r'$z$', # inside it will be added '_0, _1, ...'
            title="Agent trajectories",
            ylabel=r'$ z_i $',
            agents = "all",
            split_components = True,
            # component = 0,
            # reduce_components = "norm",
            # aggregate_agents = "quantiles",
            # sample_agents_k = None,
            y_lim=(-0.1, 1.1)
        )

    def _render_sigma_traj(self, ax, *, semilogy: bool, legend: Optional[str]):
        if "sigma_traj" not in self.result.aux:
            ax.set_title("Sigma trajectory (missing in aux)")
            return
        sigma = self.result.aux["sigma_traj"]
        
        if len(sigma.shape) == 2: # (K, d)
            self._timeseries(
                ax, sigma,
                legend = legend or r'$\sigma(z)$',
                semilogy = semilogy,
                ylabel = r'$ \sigma(z) $',
                title ="Sigma trajectories",
                split_components = True,
                interpret_2d="components"
            )
        if len(sigma.shape) == 3: # (K, N, d)
            self._timeseries(
                ax, sigma,
                legend = legend or r'$ s $', # inside it will be added '_0, _1, ...'
                semilogy=semilogy,
                title="Sigma trajectories",
                ylabel=r'$s_i$',
                agents = "all",
                # component = None,
                split_components = True,
                # reduce_components = "norm",
                # aggregate_agents = "quantiles",
                # sample_agents_k = None,
            )

    def _render_aux(self, ax, *, keys, semilogy: bool, legend: Optional[str]):
        for key in keys:
            if key not in self.result.aux:
                ax.set_title(f"{key} trajectory (missing in aux)")
                return
            y = self.result.aux[key]
            
            # if y.ndim == 2: # (K, m)
            #     self._timeseries(
            #         ax, y,
            #         legend = legend or key,
            #         semilogy = semilogy,
            #         ylabel = key,
            #         title = f"{key} trajectory",
            #         split_components = True,
            #         interpret_2d="components"
            #     )
            
            # elif y.ndim == 3: # (K, N, m)
            key = '{' + key + '}'
            self._timeseries(
                ax, y,
                legend = legend or fr'\text{key}',
                semilogy=semilogy,
                title="Aux vars trajectories",
                ylabel = fr'$\text{key}$',
                agents = "all",
                # component = None,
                # split_components = True,
                reduce_components = "norm",
                # aggregate_agents = "quantiles",
                # sample_agents_k = None,
            )

    def _timeseries(
        self, ax, Y: np.ndarray,
        *,
        legend: str,
        title: str,
        ylabel: str,
        semilogy: bool = False,
        # agent selection / component handling
        agents: Union[str, List[int]] = "all",     # "all" or a list of agent indices
        component: Optional[int] = None,        # pick a single component if d>1
        split_components: bool = False,      # plot every component separately if d>1
        # reductions across axes
        reduce_components: Optional[str] = None, # None | "norm" | "sum" | "mean"
        aggregate_agents: Optional[str] = None,# None | "mean" | "median" | "quantiles" | "sample" | "all"
        sample_agents_k: int = 0,            # when aggregate_agents == "sample"
        x_lim = None, # tuple (x_min, x_max)
        y_lim = None, # tuple (y_min, y_max)
        interpret_2d: Optional[Literal["components","agents"]] = None,  # <— NEW

    ):
        """
        Generic time-series renderer supporting shapes:
        - (K,)       : scalar over time
        - (K, N)     : one scalar per agent over time
        - (K, N, d)  : d-dimensional vector per agent over time
        Behavior:
        - You can select agents, pick or split components, or reduce components by norm/sum/mean.
        - You can aggregate over agents (mean/median/quantiles) or plot samples/all.
        """
        
        # aggregate_agents → gestisce l’asse "N" (insieme degli agenti).
        # split_components → gestisce l’asse "d" (dimensione del vettore per agente).
        # reduce_components → riduce l’asse "d" a un singolo scalare.

        Y = np.asarray(Y)
        K = Y.shape[0]
        t = np.arange(K)

        legend = legend.replace('$', '') # remove the LaTeX math inline

        if Y.ndim == 1:
            # Y.shape: (K,)
            ax.plot(t, Y, label=fr'${legend}$')
        elif Y.ndim == 2:
            K2, D2 = Y.shape

            # Decide how to read (K, D2)
            mode = interpret_2d
            if mode is None:
                # heuristic fallback: if we’re asked to aggregate over agents,
                # it’s likely (K,N); otherwise treat as (K,d)
                mode = "agents" if aggregate_agents is not None else "components"

            if mode == "components":
                # ---- (K, d): vector-in-time logic ----
                if split_components:
                    for comp in range(D2):
                        ax.plot(t, Y[:, comp], alpha=0.9, label=fr"${legend}[{comp}]$")
                else:
                    if component is not None:
                        ax.plot(t, Y[:, component], alpha=0.9, label=fr"${legend}[{component}]$")
                    else:
                        if reduce_components == "norm":
                            yn = np.linalg.norm(Y, axis=-1)
                            ax.plot(t, yn, label=fr"$||{legend}||$")
                        
                        elif reduce_components == "sum":
                            ys = Y.sum(axis=-1)
                            ax.plot(t, ys, label=fr"$\sum {legend}$")
                        
                        elif reduce_components == "mean":
                            ym = Y.mean(axis=-1)
                            ax.plot(t, ym, label=fr"${legend}\ (mean)$")
                        else:
                            # Default: plot all components as separate lines
                            for comp in range(D2):
                                ax.plot(t, Y[:, comp], alpha=0.9, label=fr"${legend}[{comp}]$")

            elif mode == "agents":
                # ---- (K, N): per-agent time series ----
                Nsel = D2
                if aggregate_agents is None:
                    for j in range(Nsel):
                        ax.plot(t, Y[:, j], alpha=0.8, label=fr"${legend}_{{{j}}}$")
                elif aggregate_agents in ("mean", "quantiles"):
                    m = Y.mean(axis=1)
                    ax.plot(t, m, label=fr"${legend}\ (mean)$")
                    if aggregate_agents == "quantiles":
                        q1, q3 = np.quantile(Y, [0.25, 0.75], axis=1)
                        ax.fill_between(t, q1, q3, alpha=0.2)
                elif aggregate_agents == "median":
                    med = np.median(Y, axis=1)
                    ax.plot(t, med, label=fr"${legend}\ (median)$")
                elif aggregate_agents == "sample":
                    idx = np.linspace(0, Nsel - 1, num=min(sample_agents_k, Nsel), dtype=int)
                    for j in idx:
                        ax.plot(t, Y[:, j], alpha=0.8, label=fr"${legend}_{{{j}}}$")
                elif aggregate_agents == "all":
                    for j in range(Nsel):
                        ax.plot(t, Y[:, j], alpha=0.6)
                    ax.plot([], [], alpha=0.6, label=fr"${legend}\ (\mathrm{{all\ agents}})$")
                else:
                    ax.text(0.5, 0.5, f"Unsupported aggregate_agents: {aggregate_agents}", ha="center")
                    return

        elif Y.ndim == 3:
            # (K, N, d)
            K, N, d = Y.shape
            agent_idx = list(range(N)) if agents == "all" else agents
            Y = Y[:, agent_idx, :]  # (K, N_sel, d)
            
            # Component handling
            if split_components:
                # Plot each component separately, aggregated over agents
                for comp in range(d):
                    Yc = Y[:, :, comp]  # (K, N_sel)
                    if aggregate_agents in (None, "all", "sample"):
                        # If not aggregating, show samples/all for each component (can be heavy)
                        if aggregate_agents == "sample":
                            N_sel = Yc.shape[1]
                            idx = np.linspace(0, N_sel - 1, num=min(sample_agents_k, N_sel), dtype=int)
                            for j in idx:
                                ax.plot(t, Yc[:, j], alpha=0.8, label=fr"${legend}_{agent_idx[j]}[{comp}]$")
                        elif aggregate_agents == "all":
                            for j in range(Yc.shape[1]):
                                ax.plot(t, Yc[:, j], alpha=0.5)
                            ax.plot([], [], alpha=0.6, label=f"${legend}[{comp}] (all)$")
                        else:  # None
                            for j in range(Yc.shape[1]):
                                ax.plot(t, Yc[:, j], alpha=0.8, label=fr'${legend}_{agent_idx[j]}[{comp}]$')
                    else:
                        # Mean/median/quantiles across agents
                        if aggregate_agents == "mean" or aggregate_agents == "quantiles":
                            m = Yc.mean(axis=1)
                            ax.plot(t, m, label=fr'${legend}[{comp}] (mean)$')
                            
                            if aggregate_agents == "quantiles":
                                q1, q3 = np.quantile(Yc, [0.25, 0.75], axis=1)
                                ax.fill_between(t, q1, q3, alpha=0.2)
                            
                        elif aggregate_agents == "median":
                            med = np.median(Yc, axis=1)
                            ax.plot(t, med, label=f"${legend}[{comp}] (median)$")                   
            else:
                # Reduce components first (by norm / sum / mean) or pick a single component
                if component is not None:
                    Y = Y[:, :, component]  # (K, N_sel)
                else:
                    if reduce_components == "norm":
                        Y = np.linalg.norm(Y, axis=-1)  # (K, N_sel)
                        legend=fr'$||{legend}||$'

                    elif reduce_components == "sum":
                        Y = Y.sum(axis=-1)

                        legend=r'$\sum_{i=1}^{N}{}_i$'.format(legend)
                    
                    elif reduce_components == "mean":
                        Y = Y.mean(axis=-1)
                        
                        legend=r'$\frac{1}{N}\sum_{i=1}^{N}{}_i$'.format(legend)
                    else:
                        ax.text(0.5, 0.5, "No reduction requested and no component selected", ha="center")
                        return

                # Now we are in (K, N_sel): reuse 2D logic
                return self._timeseries(
                    ax, Y,
                    legend=legend,
                    title=title,
                    ylabel=ylabel,
                    semilogy=semilogy,
                    agents="all",                # already selected
                    component=None,
                    split_components=True,
                    reduce_components=None,
                    aggregate_agents=aggregate_agents,
                    sample_agents_k=sample_agents_k,
                    interpret_2d="agents",
                )
        else:
            ax.text(0.5, 0.5, f"Unsupported shape: {Y.shape}", ha="center")
            return

        if semilogy:
            ax.set_yscale("log")

        ax.grid(True, which="both", ls="--", alpha=0.6)
        
        if x_lim is not None:
            (x_min, x_max) = x_lim
            ax.set_xlim(x_min, x_max)
        
        if y_lim is not None:
            (y_min, y_max) = y_lim
            ax.set_ylim(y_min, y_max)

        ax.set_title(title)
        ax.set_xlabel("iterations")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")


    def _render_phase2d(self, ax, *,
                        show_cost_levels: bool = True,
                        level_grid: int = 400, 
                        level_count: int = 15, 
                        title: Optional[str]):
        zz = self.result.zz_traj  # (K,N,d)
        K, N, d = zz.shape
        assert N == 2 and d == 1, f"Expected N=2, d=1 but got {N}, {d}"
        
        x = zz[:, 0, 0]
        y = zz[:, 1, 0]

        agent_x = self.problem.agents[0]
        agent_y = self.problem.agents[1]
        
        ax.plot(x, y, 'b.-', label="Trajectory", lw=1.5)
        ax.plot(x[0], y[0], "go", ms=4)   # start
        ax.plot(x[-1], y[-1], "ro", ms=4) # end

        # Fixed axes limits
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0

        # Box Constraint
        rect1 = Rectangle((0.0, 0.0), 1, 1, linestyle="dashed", ec='black', fc='None', lw=2)
        ax.add_patch(rect1)
        
        # add margins
        Xg = np.linspace(x_min, x_max, level_grid)
        Yg = np.linspace(y_min, y_max, level_grid)
        XX, YY = np.meshgrid(Xg, Yg)
        
        # somma dei costi locali (vectorized)
        SS = 1/2 * (agent_x.phi(XX) + agent_y.phi(YY))
        JJ = agent_x.cost_fn(XX, SS) + agent_y.cost_fn(YY, SS)
        
        # ===== Level sets of the cost =====
        if show_cost_levels:        
            Jmin, Jmax = np.percentile(JJ, 2), np.percentile(JJ, 98)
            levels = np.linspace(Jmin, Jmax, level_count)
            
            cs = ax.contour(XX, YY, JJ, levels=levels, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=8)
            contours_proxy = Patch(facecolor='none', edgecolor='gray', label='Cost level sets')

        else:
            contours_proxy = None
        
        # ===== Plot constraints & unfeasible region =====
        if isinstance(self.problem, ConstrainedOptimizationProblem):    
            ZZ = np.stack([XX, YY], axis=-1)  # shape (H,W,2)
            violated = np.ones_like(XX, dtype=bool)
            violated ^= self.problem.check(ZZ, SS) # xor: 1^1=0, 1^0=1
            VV = violated.astype(int)
            ax.contourf(XX, YY, VV, levels=[0.5, 1.5], colors=["red"], alpha=0.18)

            self.problem.draw_constraints(ax)
        # Legend
        red_patch = Patch(facecolor='red', alpha=0.18, label='Unfeasible region')
        handles, labels = ax.get_legend_handles_labels()
        if contours_proxy is not None:
            handles.append(contours_proxy); labels.append('Cost level sets')
        handles.append(red_patch); labels.append('Unfeasible region')
        ax.legend(handles, labels, loc='best')

        ax.set_title("Trajectory in (z1, z2) plane with multiple affine constraints")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.grid(True, ls="--", alpha=0.6)

        pad = 0.1
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        
        plt.axis("equal")


class ConstrainedRunResultPlotter(BaseRunResultPlotter):
    def __init__(self, problem : OptimizationProblem, result : RunResult, *, figsize: Tuple[int, int] = (12, 6)):
        super().__init__(problem, result, figsize=figsize)
        self._registry["lambda_traj"] = self._render_lambda
        self._registry["lagr_stationarity"] = self._render_lagr_stationarity
        self._registry["kkt_cond"] = self._render_kkt_cond
        self._registry["consensus_error"] = self._render_consensus_error
            
    def plot_lambda_trajectory(self, semilogy: bool = True, legend: Optional[str] = None):
        self._tasks.append(PlotTask("lambda_traj", {"semilogy": semilogy, "legend": legend}))
        return self
    
    def plot_aux(self, keys : list[str], semilogy: bool = True, legend: Optional[str] = None):
        self._tasks.append(PlotTask("aux_traj", {"keys": keys, "semilogy": semilogy, "legend": legend}))
        return self

    def plot_lagr_stationarity(self, semilogy: bool = True):
        self._tasks.append(PlotTask("lagr_stationarity", {"semilogy": semilogy}))
        return self

    def plot_kkt_conditions(self, semilogy: bool = False):
        self._tasks.append(PlotTask("kkt_cond", {"semilogy": semilogy}))
        return self


    def plot_consensus_error(self, keys : list[str], semilogy: bool = True, legend: Optional[str] = None):
        self._tasks.append(PlotTask("consensus_error", {"keys": keys, "semilogy": semilogy, "legend": legend}))
        return self

    def _render_lambda(self, ax, *, semilogy: bool, legend: Optional[str]):
        key = "lambda_traj"
        if key not in self.result.aux:
            ax.set_title(f"{key} trajectory (missing in aux)")
            return
        y = self.result.aux[key]
        
        if y.ndim == 2: # (K, m)
            self._timeseries(
                ax, y,
                legend = legend or r'$\lambda$',
                semilogy = semilogy,
                ylabel = r'$ \lambda$',
                title ="Lambda trajectory",
                split_components = True,
                interpret_2d="components"
            )
        
        elif y.ndim == 3: # (K, N, m)
            self._timeseries(
                ax, y,
                legend = legend or r'$ \lambda $', # inside it will be added '_0, _1, ...'
                semilogy=semilogy,
                title="Lambda trajectories",
                ylabel=r'$\lambda_i$',
                agents = "all",
                # component = None,
                split_components = True,
                # reduce_components = "norm",
                # aggregate_agents = "quantiles",
                # sample_agents_k = None,
            )
    
    def _render_lagr_stationarity(self, ax, *, semilogy):
        if "grad_L_x_traj" not in self.result.aux or "grad_L_l_traj" not in self.result.aux:
            ax.set_title("Stationarity (missing in aux)")
            return self

        grad_L_x = self.result.aux["grad_L_x_traj"]  # shape (K, n_tot)
        grad_L_l = self.result.aux["grad_L_l_traj"]  # shape (K, m)

        grad_L_x_dim = len(grad_L_x.shape)
        grad_L_l_dim = len(grad_L_l.shape)
        # the norm of the whole matrix / vector present at each iteration
        # so if grad.shape = (K, N, d1, d2, d3)
        # grad_norm[k] is the norm of the (N, d1, d2, d3) matrix at grad[k]
        grad_L_x_norm = np.linalg.norm(grad_L_x, axis=tuple(range(1, grad_L_x_dim)))
        grad_L_l_norm = np.linalg.norm(grad_L_l, axis=tuple(range(1, grad_L_l_dim)))
        
        self._timeseries(
            ax, grad_L_x_norm, 
            legend = r'$||\nabla_z \mathcal{L}(z,\lambda)||$',
            title="Lagrangian Stationarity", 
            ylabel=r'$||\nabla\mathcal{L}(z,\lambda)||$',
            semilogy=semilogy
        )
        self._timeseries(
            ax, grad_L_l_norm, 
            legend = r'$||\nabla_\lambda \mathcal{L}(z,\lambda)||$',
            title="Lagrangian Stationarity", 
            ylabel=r'$||\nabla\mathcal{L}(z,\lambda)||$',
            semilogy=semilogy
        )
    
    def _render_kkt_cond(self, ax, *, semilogy: bool):
        if "kkt_traj" not in self.result.aux:
            ax.set_title("KKT conditions (missing in aux)")
            return self

        kkt = self.result.aux["kkt_traj"]  # shape (K, 3)
        legend = ["Stationarity", "Primal violation", "Complementarity"]
        for i in range(kkt.shape[1]): # 3
            legend_i = '{' + legend[i] + '}'
            y = kkt[:, i]
            self._timeseries(
                ax, y,
                legend = fr'\text{legend_i}',
                semilogy=semilogy,
                title="KKT conditions evolution",
                ylabel = "KKT conditions",
                agents = "all",
            )

    
    def _render_consensus_error(self, ax, *, keys, semilogy: bool, legend: Optional[str]):
        for key in keys:
            if key not in self.result.aux:
                ax.set_title("{key} trajectory (missing in aux)")
                return self
            val = self.result.aux[key]
            K, N, d = val.shape
            val_mean = np.mean(val, axis=1)
            
            consensus_err = val - val_mean[:, None, :]
            
            key = '{' + key + '}'
            self._timeseries(
                ax, consensus_err,
                legend = legend or fr'$\text{key}$',
                semilogy=semilogy,
                title="Consensus Error evolutions",
                ylabel = fr'$\text{key}$',
                agents = "all",
                aggregate_agents=None,
                reduce_components="norm"
            )


