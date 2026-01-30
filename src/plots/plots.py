import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Literal
from models.optimization_problem import OptimizationProblem, ConstrainedOptimizationProblem
from models.algorithm_interface import RunResult
from matplotlib.patches import Patch, Rectangle
from plots.timeseries import _timeseries
from pathlib import Path

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

from pathlib import Path

def plot_filename_from_result(
    run_result,
    tag: str,
    img_root: str = "../output",
    ext: str = "png",
) -> str:
    parts = run_result.algorithm_module.split(".")

    if parts[0] == "algorithms":
        core = parts[1:]   # family, mode, AlgorithmName
        prefix = "__".join(core)
    else:
        prefix = parts[-1]

    return str(Path(img_root) / f"{prefix}__{tag}.{ext}")

def plot_filename_comparison_from_results(
    results,
    tag: str,
    img_root="../output",
    ext="png",
) -> str:
    parts = results[0].algorithm_module.split(".")
    if parts[0] == "algorithms":
        family = parts[1]
    else:
        family = "misc"
    return str(Path(img_root) / f"{family}__comparison_{tag}.{ext}")


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
        self._width_col = 3
        self._height_row = 3
        self._registry: Dict[str, Callable] = {
            "cost": self._render_cost,
            "grad_norm": self._render_grad_norm,
            "aux_traj": self._render_aux,
            "agent_traj": self._render_agent_traj,
            "sigma_traj": self._render_sigma_traj,
            "phase2d": self._render_phase2d
        }


    # --- color management (shared across all plotters) ---
    _algorithm_color_map: Dict[str, str] = {}
    _available_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    _next_color_idx: int = 0

    @classmethod
    def _get_color_for_algorithm(cls, algorithm_name: str) -> str:
        if algorithm_name not in cls._algorithm_color_map:
            color = cls._available_colors[
                cls._next_color_idx % len(cls._available_colors)
            ]
            cls._algorithm_color_map[algorithm_name] = color
            cls._next_color_idx += 1
        return cls._algorithm_color_map[algorithm_name]

    # -------------------- public API --------------------
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

    def show(self):
        fig = self.render()
        plt.show()
        return self

    def save(self, path: str, **kwargs):
        fig = self.render()
        fig.savefig(path, **kwargs)
        return self

    def clear(self):
        self._tasks.clear()
        return self

    # -------------------- high-level renderer --------------------
    def render(self):
        n = len(self._tasks)
        if n == 0:
            raise ValueError("No plots requested. Call .plot_*() before render().")
        
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        s = self._figsize if self._figsize else (self._width_col * cols, self._height_row * rows)
        fig, axes = plt.subplots(rows, cols, figsize=s, layout="constrained")
        axes = np.atleast_1d(axes).ravel()

        for ax, task in zip(axes, self._tasks):
            renderer = self._registry.get(task.name)
            if renderer is None:
                ax.text(0.5, 0.5, f"Missing renderer: {task.name}", ha="center")
                continue
            renderer(ax, **task.kwargs)
        for ax in axes[n:]:
            ax.axis("off")
        return fig

    # -------------------- low-level renderer --------------------

    @staticmethod
    def _render_cost_from_result(ax, result: RunResult, *, semilogy: bool, legend: Optional[str]):
        cost = np.asarray(result.cost_traj)  # (K, ?)
        if cost.ndim == 1: # (K,)
            _timeseries(
                ax, cost, 
                legend = legend or r'$\ell(z) = \sum_{i=1}^{N} \ell_i(z_i, \sigma(z))$',
                title="Cost evolution", 
                ylabel="cost", 
                semilogy=semilogy
            )


        if cost.ndim == 2: # (K, costs)
            assert cost.shape[1] == 2, "\ell(z), Lagrangian, and ?"
            _timeseries(
                ax, cost[:, 1], 
                legend = legend or r'$\mathcal{L}(z,\lambda)$',
                title="Cost evolution of Lagrangian", 
                ylabel="cost", 
                semilogy=semilogy
            )

            _timeseries(
                ax, cost[:, 0], 
                legend = legend or r'$\ell(z) = \sum_{i=1}^{N} \ell_i(z_i, \sigma(z))$',
                title="Cost evolution", 
                ylabel="cost", 
                semilogy=semilogy
            )
    def _render_cost(self, ax, *, semilogy: bool, legend: Optional[str]):
        BaseRunResultPlotter._render_cost_from_result(ax, self.result, semilogy=semilogy, legend=legend)

    @staticmethod
    def _render_grad_norm_from_result(ax, result: RunResult, *, semilogy: bool, legend: Optional[str]):
        grad = np.asarray(result.grad_traj)  # (K,N,d) or (K,d)
        # the norm of the whole matrix / vector present at each iteration
        # so if grad.shape = (K, N, d1, d2, d3)
        # grad_norm[k] is the norm of the (N, d1, d2, d3) matrix at grad[k]
        grad_norm = np.linalg.norm(grad, axis=tuple(range(1, grad.ndim)))
        
        assert grad_norm.ndim == 1, "grad_norm.ndim != 1, expected (K,)"
        
        _timeseries(
            ax, grad_norm,
            legend = legend or r'$ ||\nabla \ell(z) ||^2 $', 
            title="Gradient norm evolution", 
            ylabel="gradient norm", 
            semilogy=semilogy,
            split_components=True,
            aggregate_agents="all"
        )
    def _render_grad_norm(self, ax, *, semilogy: bool, legend: Optional[str]):
        BaseRunResultPlotter._render_grad_norm_from_result(ax, self.result, semilogy=semilogy, legend=legend)

    @staticmethod
    def _render_agent_traj_from_result(ax, result: RunResult, *, semilogy: bool, legend: Optional[str]):
        y = result.zz_traj
        assert y.ndim == 3, "zz_traj.ndim != 3, expected (K,N,d)"
        
        # (K, N, d)
        _timeseries(
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
    def _render_agent_traj(self, ax, *, semilogy: bool, legend: Optional[str]):
        BaseRunResultPlotter._render_agent_traj_from_result(ax, self.result, semilogy=semilogy, legend=legend)

    @staticmethod 
    def _render_sigma_traj_from_result(ax, result, *, semilogy: bool, legend: Optional[str]):
        if "sigma_traj" not in result.aux:
            ax.set_title("Sigma trajectory (missing in aux)")
            return
        sigma = result.aux["sigma_traj"]
        
        if len(sigma.shape) == 2: # (K, d)
            _timeseries(
                ax, sigma,
                legend = legend or r'$\sigma(z)$',
                semilogy = semilogy,
                ylabel = r'$ \sigma(z) $',
                title ="Sigma trajectories",
                split_components = True,
                interpret_2d="components"
            )
        if len(sigma.shape) == 3: # (K, N, d)
            _timeseries(
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
    def _render_sigma_traj(self, ax, *, semilogy: bool, legend: Optional[str]):
        BaseRunResultPlotter._render_sigma_traj_from_result(ax, self.result, semilogy=semilogy, legend=legend)

    
    def _render_aux(self, ax, *, keys, semilogy: bool, legend: Optional[str]):
        for key in keys:
            if key not in self.result.aux:
                ax.set_title(f"{key} trajectory (missing in aux)")
                return
            y = self.result.aux[key]
            
            # if y.ndim == 2: # (K, m)
            #     _timeseries(
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
            _timeseries(
                ax, y,
                legend = legend or fr'\text{key}',
                semilogy=semilogy,
                title="Aux vars trajectories",
                ylabel = fr'$\text{key}$',
                agents = "all",
                # component = None,
                # split_components = True,
                reduce_components = "norm",
                aggregate_agents = "quantiles",
                # sample_agents_k = None,
            )
    

    @staticmethod
    def _render_phase2d_traj_from_result(ax, problem, result, *, label: str = "Trajectory"):
        """Draw only the trajectory for a single result (no background)."""
        global colors

        zz = result.zz_traj  # (K,N,d)
        K, N, d = zz.shape
        assert N == 2 and d == 1, f"Expected N=2, d=1 but got {N}, {d}"

        x = zz[:, 0, 0]
        y = zz[:, 1, 0]
        
        color = BaseRunResultPlotter._get_color_for_algorithm(result.algorithm_name)
        ax.plot(x, y, f"{color}.-", label=f"Trajectory {result.algorithm_name}", lw=1.5)
        ax.plot(x[0], y[0], "go", ms=4)   # start
        ax.plot(x[-1], y[-1], "ro", ms=4) # end
        

    @staticmethod
    def _finalize_phase2d_legend(ax, *, contours_proxy, has_constraints: bool):        
        handles, labels = ax.get_legend_handles_labels()
        if contours_proxy is not None:
            handles.append(contours_proxy); labels.append('Cost level sets')
        if has_constraints:
            red_patch = Patch(facecolor='red', alpha=0.18, label='Unfeasible region')
            handles.append(red_patch); labels.append('Unfeasible region')
        # Remove duplicate labels while preserving order
        seen = set()
        uniq_handles, uniq_labels = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            uniq_handles.append(h)
            uniq_labels.append(l)
        ax.legend(uniq_handles, uniq_labels, loc='best')

    @staticmethod
    def _render_phase2d_from_result(ax, problem, result, *,
                        show_cost_levels: bool = True,
                        level_grid: int = 400,
                        level_count: int = 15,
                        title: Optional[str]):
        """Backward-compatible: background + a single trajectory + legend."""
        meta = BaseRunResultPlotter._render_phase2d_background(
            ax, problem,
            show_cost_levels=show_cost_levels,
            level_grid=level_grid,
            level_count=level_count,
            title=title,
        )
        BaseRunResultPlotter._render_phase2d_traj_from_result(ax, problem, result, label="Trajectory")
        BaseRunResultPlotter._finalize_phase2d_legend(ax, contours_proxy=meta["contours_proxy"], has_constraints=meta["has_constraints"])


    @staticmethod
    def _render_phase2d_background(ax, problem, *,
                        show_cost_levels: bool = True,
                        level_grid: int = 400, 
                        level_count: int = 15, 
                        title: Optional[str]):
        N = problem.N
        d = problem.d
        assert N == 2 and d == 1, f"Expected N=2, d=1 but got {N}, {d}"
        
        agent_x = problem.agents[0]
        agent_y = problem.agents[1]
        
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
        has_constraints = isinstance(problem, ConstrainedOptimizationProblem)
        if has_constraints:
            ZZ = np.stack([XX, YY], axis=-1)  # shape (H,W,2)
            violated = np.ones_like(XX, dtype=bool)
            violated ^= problem.check(ZZ, SS) # xor: 1^1=0, 1^0=1
            VV = violated.astype(int)
            ax.contourf(XX, YY, VV, levels=[0.5, 1.5], colors=["red"], alpha=0.18)

            problem.draw_constraints(ax)

        ax.set_title("Trajectory in (z1, z2) plane with multiple affine constraints")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.grid(True, ls="--", alpha=0.6)

        pad = 0.1
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        
        # plt.axis("equal")
        ax.set_aspect("equal", adjustable="box")

        return {
            "contours_proxy": contours_proxy,
            "has_constraints": has_constraints
        }
    
    
    # @staticmethod
    # def _render_phase2d_from_result(ax, problem, result, *,
    #                     show_cost_levels: bool = True,
    #                     level_grid: int = 400, 
    #                     level_count: int = 15, 
    #                     title: Optional[str]):
    #     zz = result.zz_traj  # (K,N,d)
    #     K, N, d = zz.shape
    #     assert N == 2 and d == 1, f"Expected N=2, d=1 but got {N}, {d}"
        
    #     x = zz[:, 0, 0]
    #     y = zz[:, 1, 0]

    #     agent_x = problem.agents[0]
    #     agent_y = problem.agents[1]
        
    #     ax.plot(x, y, 'b.-', label="Trajectory", lw=1.5)
    #     ax.plot(x[0], y[0], "go", ms=4)   # start
    #     ax.plot(x[-1], y[-1], "ro", ms=4) # end

    #     # Fixed axes limits
    #     x_min, x_max = 0.0, 1.0
    #     y_min, y_max = 0.0, 1.0

    #     # Box Constraint
    #     rect1 = Rectangle((0.0, 0.0), 1, 1, linestyle="dashed", ec='black', fc='None', lw=2)
    #     ax.add_patch(rect1)
        
    #     # add margins
    #     Xg = np.linspace(x_min, x_max, level_grid)
    #     Yg = np.linspace(y_min, y_max, level_grid)
    #     XX, YY = np.meshgrid(Xg, Yg)
        
    #     # somma dei costi locali (vectorized)
    #     SS = 1/2 * (agent_x.phi(XX) + agent_y.phi(YY))
    #     JJ = agent_x.cost_fn(XX, SS) + agent_y.cost_fn(YY, SS)
        
    #     # ===== Level sets of the cost =====
    #     if show_cost_levels:        
    #         Jmin, Jmax = np.percentile(JJ, 2), np.percentile(JJ, 98)
    #         levels = np.linspace(Jmin, Jmax, level_count)
            
    #         cs = ax.contour(XX, YY, JJ, levels=levels, alpha=0.6)
    #         ax.clabel(cs, inline=True, fontsize=8)
    #         contours_proxy = Patch(facecolor='none', edgecolor='gray', label='Cost level sets')

    #     else:
    #         contours_proxy = None
        
    #     # ===== Plot constraints & unfeasible region =====
    #     if isinstance(problem, ConstrainedOptimizationProblem):    
    #         ZZ = np.stack([XX, YY], axis=-1)  # shape (H,W,2)
    #         violated = np.ones_like(XX, dtype=bool)
    #         violated ^= problem.check(ZZ, SS) # xor: 1^1=0, 1^0=1
    #         VV = violated.astype(int)
    #         ax.contourf(XX, YY, VV, levels=[0.5, 1.5], colors=["red"], alpha=0.18)

    #         problem.draw_constraints(ax)
    #     # Legend
    #     red_patch = Patch(facecolor='red', alpha=0.18, label='Unfeasible region')
    #     handles, labels = ax.get_legend_handles_labels()
    #     if contours_proxy is not None:
    #         handles.append(contours_proxy); labels.append('Cost level sets')
    #     handles.append(red_patch); labels.append('Unfeasible region')
    #     ax.legend(handles, labels, loc='best')

    #     ax.set_title("Trajectory in (z1, z2) plane with multiple affine constraints")
    #     ax.set_xlabel("z1")
    #     ax.set_ylabel("z2")
    #     ax.grid(True, ls="--", alpha=0.6)

    #     pad = 0.1
    #     ax.set_xlim(x_min - pad, x_max + pad)
    #     ax.set_ylim(y_min - pad, y_max + pad)
        
    #     plt.axis("equal")
    def _render_phase2d(self, ax, *,
                        show_cost_levels: bool = True,
                        level_grid: int = 400, 
                        level_count: int = 15, 
                        title: Optional[str]):
        BaseRunResultPlotter._render_phase2d_from_result(ax, self.problem, self.result, show_cost_levels=show_cost_levels, level_grid=level_grid, level_count=level_count, title=title)


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
        ConstrainedRunResultPlotter._render_lambda_from_result(ax, self.result, semilogy=semilogy, legend=legend)

    @staticmethod
    def _render_lambda_from_result(ax, result, *, semilogy: bool, legend: Optional[str]):
        key = "lambda_traj"
        if key not in result.aux:
            ax.set_title(f"{key} trajectory (missing in aux)")
            return
        y = result.aux[key]
        
        if y.ndim == 2: # (K, m)
            _timeseries(
                ax, y,
                legend = legend or r'$\lambda$',
                semilogy = semilogy,
                ylabel = r'$ \lambda$',
                title ="Lambda trajectory",
                split_components = True,
                interpret_2d="components"
            )
        
        elif y.ndim == 3: # (K, N, m)
            _timeseries(
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
        ConstrainedRunResultPlotter._render_lagr_stationarity_from_result(ax, self.result, semilogy=semilogy)

    @staticmethod
    def _render_lagr_stationarity_from_result(ax, result, *, semilogy):
        if "grad_L_x_traj" not in result.aux or "grad_L_l_traj" not in result.aux:
            ax.set_title("Stationarity (missing in aux)")
            return

        grad_L_x = result.aux["grad_L_x_traj"]  # shape (K, n_tot)
        grad_L_l = result.aux["grad_L_l_traj"]  # shape (K, m)

        grad_L_x_dim = len(grad_L_x.shape)
        grad_L_l_dim = len(grad_L_l.shape)
        # the norm of the whole matrix / vector present at each iteration
        # so if grad.shape = (K, N, d1, d2, d3)
        # grad_norm[k] is the norm of the (N, d1, d2, d3) matrix at grad[k]
        grad_L_x_norm = np.linalg.norm(grad_L_x, axis=tuple(range(1, grad_L_x_dim)))
        grad_L_l_norm = np.linalg.norm(grad_L_l, axis=tuple(range(1, grad_L_l_dim)))
        
        _timeseries(
            ax, grad_L_x_norm, 
            legend = r'$||\nabla_z \mathcal{L}(z,\lambda)||$',
            title="Lagrangian Stationarity", 
            ylabel=r'$||\nabla\mathcal{L}(z,\lambda)||$',
            semilogy=semilogy
        )
        _timeseries(
            ax, grad_L_l_norm, 
            legend = r'$||\nabla_\lambda \mathcal{L}(z,\lambda)||$',
            title="Lagrangian Stationarity", 
            ylabel=r'$||\nabla\mathcal{L}(z,\lambda)||$',
            semilogy=semilogy
        )
    

    def _render_kkt_cond(self, ax, *, semilogy: bool):
        ConstrainedRunResultPlotter._render_kkt_cond_from_result(ax, self.result, semilogy=semilogy)

    @staticmethod
    def _render_kkt_cond_from_result(ax, result, *, semilogy: bool):
        if "kkt_traj" not in result.aux:
            ax.set_title("KKT conditions (missing in aux)")
            return

        kkt = result.aux["kkt_traj"]  # shape (K, 3)
        legend = ["Stationarity", "Primal violation", "Complementarity"]
        for i in range(kkt.shape[1]): # 3
            legend_i = '{' + legend[i] + '}'
            y = kkt[:, i]
            _timeseries(
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
            _timeseries(
                ax, consensus_err,
                legend = legend or fr'$\text{key}$',
                semilogy=semilogy,
                title="Consensus Error evolutions",
                ylabel = fr'$\text{key}$',
                agents = "all",
                aggregate_agents="quantiles",
                reduce_components="norm"
            )


class ComparisonBaseRunResultPlotter:
    def __init__(self, problem: OptimizationProblem, results: list[RunResult], *, figsize: Tuple[int, int] = (12, 6), layout: str = "overlay"):
        self.problem = problem
        self.results = results
        
        assert layout in ("overlay", "vertical", "horizontal", "grid")
        self._layout = layout

        iterations = [res.zz_traj.shape[0] for res in self.results]
        assert iterations.count(iterations[0]) == len(iterations) # all equal number of iterations

        self._tasks: List[PlotTask] = []
        self._figsize = figsize
        self._width_col = 3
        self._height_row = 3
        self._registry: Dict[str, Callable] = {
            "cost": self._render_cost,
            "grad_norm": self._render_grad_norm,
            # # "aux_traj": self._render_aux,
            "agent_traj": self._render_agent_traj,
            "sigma_traj": self._render_sigma_traj,
            "phase2d": self._render_phase2d
        }

    def show(self):
        fig = self.render()
        plt.show()
        return self

    def save(self, path: str, **kwargs):
        fig = self.render()
        fig.savefig(path, **kwargs)
        return self

    def clear(self):
        self._tasks.clear()
        return self

    def plot_cost(self, semilogy: bool = True, legend: Optional[str] = None):
        self._tasks.append(PlotTask("cost", {"semilogy": semilogy, "legend": legend}))
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

    def plot_phase2d(self):
        args = {
            "title": "Trajectory",
        }
        self._tasks.append(PlotTask("phase2d", args))
        return self


    # -------------------- high-level renderer --------------------
    def render(self):
        n_tasks = len(self._tasks)
        if n_tasks == 0:
            raise ValueError("No plots requested. Call .plot_*() before render().")

        R = len(self.results)

        if self._layout == "overlay":
            cols = math.ceil(math.sqrt(n_tasks))
            rows = math.ceil(n_tasks / cols)
        elif self._layout == "vertical":
            rows = n_tasks * R
            cols = 1
        elif self._layout == "horizontal":
            rows = n_tasks
            cols = R
        elif self._layout == "grid":
            # R righe, n_tasks colonne
            rows = R
            cols = n_tasks
        else:
            raise ValueError(f"Unknown layout: {self._layout}")

        s = self._figsize if self._figsize else (self._width_col * cols, self._height_row * rows)
        fig, axes = plt.subplots(rows, cols, figsize=s, layout="constrained")
        # fig.tight_layout(pad=2)
        axes = np.atleast_1d(axes)

        # normalizzazione shape: 1D -> (rows, cols)
        if axes.ndim == 1 and (rows > 1 or cols > 1):
            axes = axes.reshape(rows, cols)

        if self._layout == "overlay":
            flat_axes = axes.ravel()
            for ax, task in zip(flat_axes, self._tasks):
                renderer = self._registry.get(task.name)
                if renderer is None:
                    ax.text(0.5, 0.5, f"Missing renderer: {task.name}", ha="center")
                    continue
                renderer(ax, **task.kwargs)
            for ax in flat_axes[n_tasks:]:
                ax.axis("off")

        elif self._layout in ("vertical", "horizontal"):
            # per-result: un asse per (task, result), ma “schiacciato” verticalmente o orizzontalmente
            for t_idx, task in enumerate(self._tasks):
                renderer = self._registry.get(task.name)
                if renderer is None:
                    for r_idx in range(R):
                        if self._layout == "vertical":
                            ax = axes[t_idx * R + r_idx, 0] if axes.ndim == 2 else axes[t_idx * R + r_idx]
                        else:  # "horizontal"
                            ax = axes[t_idx, r_idx] if axes.ndim == 2 else axes[t_idx * R + r_idx]
                        ax.text(0.5, 0.5, f"Missing renderer: {task.name}", ha="center")
                    continue

                for r_idx, res in enumerate(self.results):
                    if self._layout == "vertical":
                        ax = axes[t_idx * R + r_idx, 0] if axes.ndim == 2 else axes[t_idx * R + r_idx]
                    else:  # "horizontal"
                        ax = axes[t_idx, r_idx] if axes.ndim == 2 else axes[t_idx * R + r_idx]

                    kwargs = dict(task.kwargs)
                    kwargs["result"] = res
                    renderer(ax, **kwargs)

        elif self._layout == "grid":
            # R righe (algoritmi), n_tasks colonne (metriche)
            for r_idx, res in enumerate(self.results):
                for t_idx, task in enumerate(self._tasks):
                    renderer = self._registry.get(task.name)
                    if renderer is None:
                        ax = axes[r_idx, t_idx] if axes.ndim == 2 else axes[r_idx * n_tasks + t_idx]
                        ax.text(0.5, 0.5, f"Missing renderer: {task.name}", ha="center")
                        continue

                    ax = axes[r_idx, t_idx] if axes.ndim == 2 else axes[r_idx * n_tasks + t_idx]
                    kwargs = dict(task.kwargs)
                    kwargs["result"] = res
                    renderer(ax, **kwargs)

        return fig

    
    # def plot_aux(self, keys: list[str], semilogy: bool = False, legend: Optional[str] = None):
    #     """Plot any aux variable stored in RunResult. Shape can be (K,), (K,N) or (K,d)."""
    #     self._tasks.append(PlotTask("aux_traj", {"keys": keys, "semilogy": semilogy, "legend": legend}))
    #     return self

    def _render_cost(self, ax, *, semilogy: bool, legend: Optional[str], result: Optional[RunResult] = None):
        if result is None:
            for result in self.results:
                BaseRunResultPlotter._render_cost_from_result(ax, result, semilogy=semilogy, legend=result.algorithm_name)
        else: # used for layouts different from "overlay"
            BaseRunResultPlotter._render_cost_from_result(ax, result, semilogy=semilogy, legend=result.algorithm_name)

    def _render_grad_norm(self, ax, *, semilogy: bool, legend: Optional[str], result: Optional[RunResult] = None):
        if result is None:
            for result in self.results:
                BaseRunResultPlotter._render_grad_norm_from_result(ax, result, semilogy=semilogy, legend=result.algorithm_name)
        else: # used for layouts different from "overlay"
            BaseRunResultPlotter._render_grad_norm_from_result(ax, result, semilogy=semilogy, legend=result.algorithm_name)

    def _render_agent_traj(self, ax, *, semilogy: bool, legend: Optional[str], result: Optional[RunResult] = None):
        if result is None:
            for result in self.results:
                BaseRunResultPlotter._render_agent_traj_from_result(ax, result, semilogy=semilogy, legend=result.algorithm_name)
        else: # used for layouts different from "overlay"
            BaseRunResultPlotter._render_agent_traj_from_result(ax, result, semilogy=semilogy, legend=result.algorithm_name)

    def _render_sigma_traj(self, ax, *, semilogy: bool, legend: Optional[str], result: Optional[RunResult] = None):
        if result is None:
            for result in self.results:
                BaseRunResultPlotter._render_sigma_traj_from_result(ax, result, semilogy=semilogy, legend=result.algorithm_name)
        else: # used for layouts different from "overlay"
            BaseRunResultPlotter._render_sigma_traj_from_result(ax, result, semilogy=semilogy, legend=result.algorithm_name)

    def _render_phase2d(self, ax, *,
                        show_cost_levels: bool = True,
                        level_grid: int = 400, 
                        level_count: int = 15, 
                        title: Optional[str],
                        result: Optional[RunResult] = None):
        # Overlay case: draw background once, then all trajectories
        if result is None:
            meta = BaseRunResultPlotter._render_phase2d_background(
                ax, self.problem,
                show_cost_levels=show_cost_levels,
                level_grid=level_grid,
                level_count=level_count,
                title=title,
            )
            for res in self.results:
                BaseRunResultPlotter._render_phase2d_traj_from_result(
                    ax, self.problem, res, label=res.algorithm_name
                )
            BaseRunResultPlotter._finalize_phase2d_legend(
                ax, contours_proxy=meta["contours_proxy"], has_constraints=meta["has_constraints"]
            )
        else:
            # Single-result panel (used by non-overlay comparison layouts)
            BaseRunResultPlotter._render_phase2d_from_result(
                ax, self.problem, result,
                show_cost_levels=show_cost_levels,
                level_grid=level_grid,
                level_count=level_count,
                title=title,
            )

class ComparisonConstrainedRunResultPlotter(ComparisonBaseRunResultPlotter):
    def __init__(self, problem: OptimizationProblem, results: list[RunResult], *, figsize: Tuple[int, int] = (12, 6), layout: str = "overlay"):
        super().__init__(problem, results, figsize=figsize, layout=layout)
        self._registry["lambda_traj"] = self._render_lambda
        self._registry["lagr_stationarity"] = self._render_lagr_stationarity
        self._registry["kkt_cond"] = self._render_kkt_cond
        # self._registry["consensus_error"] = self._render_consensus_error

    def plot_lambda_trajectory(self, semilogy: bool = True, legend: Optional[str] = None):
        self._tasks.append(PlotTask("lambda_traj", {"semilogy": semilogy, "legend": legend}))
        return self

    def plot_lagr_stationarity(self, semilogy: bool = True):
        self._tasks.append(PlotTask("lagr_stationarity", {"semilogy": semilogy}))
        return self

    def plot_kkt_conditions(self, semilogy: bool = False):
        self._tasks.append(PlotTask("kkt_cond", {"semilogy": semilogy}))
        return self


    def _render_lambda(self, ax, *, semilogy: bool, legend: Optional[str], result: Optional[RunResult] = None):
        if result is None:
            for result in self.results:
                ConstrainedRunResultPlotter._render_sigma_traj_from_result(ax, result, semilogy=semilogy, legend=result.algorithm_name)
        else: # used for layouts different from "overlay"
            ConstrainedRunResultPlotter._render_sigma_traj_from_result(ax, result, semilogy=semilogy, legend=result.algorithm_name)

    def _render_kkt_cond(self, ax, *, semilogy: bool, legend: Optional[str] = None, result: Optional[RunResult] = None):
        if result is None:
            for result in self.results:
                ConstrainedRunResultPlotter._render_kkt_cond_from_result(ax, result, semilogy=semilogy)
        else: # used for layouts different from "overlay"
            ConstrainedRunResultPlotter._render_kkt_cond_from_result(ax, result, semilogy=semilogy)

    def _render_lagr_stationarity(self, ax, *, semilogy: bool, legend: Optional[str] = None, result: Optional[RunResult] = None):
        if result is None:
            for result in self.results:
                ConstrainedRunResultPlotter._render_lagr_stationarity_from_result(ax, result, semilogy=semilogy)
        else: # used for layouts different from "overlay"
            ConstrainedRunResultPlotter._render_lagr_stationarity_from_result(ax, result, semilogy=semilogy)