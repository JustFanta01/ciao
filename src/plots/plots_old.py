import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class BaseRunResultPlotter:
    def __init__(self, result):
        """
        Parameters
        ----------
        result : RunResult
            Object containing zz_traj, cost_traj, grad_traj, aux.
        semilogy : bool
            If True, use log scale for y-axis in cost and gradient plots.
        """
        self.result = result
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.tight_layout(pad=4.0)

    def plot_cost(self, semilogy=True):
        cost = self.result.cost_traj
        K = len(cost)
        ax = self.axes[0, 0]
        if semilogy:
            ax.semilogy(np.arange(K), np.abs(cost), label="Cost")
        else:
            ax.plot(np.arange(K), cost, label="Cost")
        ax.set_title("Cost evolution")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.legend()
        return self

    def plot_grad_norm(self, semilogy=True):
        grad = self.result.grad_traj
        dims = len(grad.shape)
        # the norm of the whole matrix / vector present at each iteration
        # so if grad.shape = (K, N, d1, d2, d3)
        # grad_norm[k] is the norm of the (N, d1, d2, d3) matrix at grad[k]
        grad_norm = np.linalg.norm(grad, axis=tuple(range(1, dims)))
        K = grad_norm.shape[0]
        ax = self.axes[0, 1]
        if semilogy:
            ax.semilogy(np.arange(K), grad_norm, label="||Grad||")
        else:
            ax.plot(np.arange(K), grad_norm, label="||Grad||")
        ax.set_title("Norm of total gradient")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Norm")
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.legend()
        return self

    def plot_agents_trajectories(self, semilogy=True):
        zz = self.result.zz_traj  # (K, N, d)
        K, N, d = zz.shape
        ax = self.axes[1, 0]
        for i in range(N):
            if d == 1:
                ax.plot(np.arange(K), zz[:, i, 0], label=f"Agent {i+1}")
            else:
                for j in range(d):
                    ax.plot(np.arange(K), zz[:, i, j], label=f"Agent {i+1}, dim {j}")
        ax.set_title("Agent trajectories")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("z_i")
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.legend()
        return self

    def plot_sigma_trajectory(self, semilogy=True):
        if "sigma_traj" not in self.result.aux:
            ax = self.axes[1, 1]
            ax.set_title("Sigma trajectory (missing in aux)")
            return self
        sigma = self.result.aux["sigma_traj"]
        if len(sigma.shape) == 2: # (K, d)
            K, d = sigma.shape
            ax = self.axes[1, 1]
            for j in range(d):
                if semilogy:
                    ax.semilogy(np.arange(K), sigma[:, j], label=f"dim {j}")
                else:
                    ax.plot(np.arange(K), sigma[:, j], label=f"dim {j}")
            ax.set_title("Sigma evolution")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Sigma")
            ax.grid(True, which="both", ls="--", alpha=0.6)
            ax.legend()
            return self
        elif len(sigma.shape) == 3: # (K, N, d)
            K, N, d = sigma.shape
            ax = self.axes[1, 1]
            for i in range(N):
                if d == 1:
                    ax.plot(np.arange(K), sigma[:, i, 0], label=f"Agent {i+1}")
                else:
                    for j in range(d):
                        ax.plot(np.arange(K), sigma[:, i, j], label=f"Agent {i+1}, dim {j}")
            ax.set_title("Sigma Tracker evolutions")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Sigma Tracker")
            ax.grid(True, which="both", ls="--", alpha=0.6)
            ax.legend()
            return self

    def show(self):
        plt.show()


class ConstrainedRunResultPlotter(BaseRunResultPlotter):    
    def __init__(self, result):
        # TODO: do not call init for not having 2 figs... ?!?!
        self.result = result
        self.fig, self.axes = plt.subplots(3, 3, figsize=(12, 8))
        self.fig.tight_layout(pad=4.0)

    def plot_cost(self, semilogy=True):
        cost = self.result.cost_traj
        K = cost.shape[0]
        cost_f = cost[:,0]
        cost_L = cost[:,1]
        ax = self.axes[0, 0]
        if semilogy:
            ax.semilogy(np.arange(K), np.abs(cost_f), label="Cost function ")
            ax.semilogy(np.arange(K), np.abs(cost_L), label="Cost Lagrangian")
        else:
            ax.plot(np.arange(K), cost_f, label="Cost")
            ax.plot(np.arange(K), cost_L, label="Cost")
        ax.set_title("Cost evolution")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.legend()
        return self

    def plot_lagr_stationarity(self, semilogy=True):
        """
        Plot the norms of grad L wrt x and grad L wrt lambda.
        Requires that result.aux contains 'grad_L_x_traj' and 'grad_L_l_traj'.
        """
        ax = self.axes[2, 0]
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
        K = grad_L_x_norm.shape[0]
        
        if semilogy:
            ax.semilogy(np.arange(K), grad_L_x_norm, label="||∇_x L(x,λ||")
            ax.semilogy(np.arange(K), grad_L_l_norm, label="||∇_λ L(x,λ)||")
        else:
            ax.plot(np.arange(K), grad_L_x_norm, label="||∇_x L(x,λ)||")
            ax.plot(np.arange(K), grad_L_l_norm, label="||∇_λ L(x,λ)||")

        ax.set_title("Norm of ∇1 and ∇2 of L(x,λ)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Norm")
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.legend()

        return self

    def plot_lambda(self, semilogy=True):
        ax = self.axes[0, 2]
        if "lambda_traj" not in self.result.aux:
            ax.set_title("Lambda trajectory (missing in aux)")
            return self
        lamda = self.result.aux["lambda_traj"]
        if len(lamda.shape) == 2: # (K, m)
            K, m = lamda.shape
            for j in range(m):
                if semilogy:
                    ax.semilogy(np.arange(K), lamda[:, j], label=f"dim {j}")
                else:
                    ax.plot(np.arange(K), lamda[:, j], label=f"dim {j}")
            ax.set_title("Lambda evolution")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Lambda")
            ax.grid(True, which="both", ls="--", alpha=0.6)
            ax.legend()
            return self
        elif len(lamda.shape) == 3: # (K, N, m)
            K, N, m = lamda.shape
            for i in range(N):
                if m == 1:
                    ax.plot(np.arange(K), lamda[:, i, 0], label=f"Agent {i+1}")
                else:
                    for j in range(m):
                        ax.plot(np.arange(K), lamda[:, i, j], label=f"Agent {i+1}, dim {j}")
            ax.set_title("Lambda evolutions")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Lambda Tracker")
            ax.grid(True, which="both", ls="--", alpha=0.6)
            ax.legend()
            return self

    def plot_kkt_conditions(self, semilogy=True):
        """
        Plot the three KKT residuals over iterations:
        - Stationarity: ||∇f(x) + A^T λ||
        - Primal violation: ||(Ax - b)_+||_∞
        - Complementarity: max |λ_j * r_j|
        """
        ax = self.axes[2, 2]
        if "kkt_traj" not in self.result.aux:
            ax.set_title("KKT conditions (missing in aux)")
            return self

        kkt = self.result.aux["kkt_traj"]  # shape (K, 3)
        K = kkt.shape[0]

        labels = ["Stationarity", "Primal violation", "Complementarity"]

        for i in range(3):
            if semilogy:
                ax.semilogy(np.arange(K), kkt[:, i], label=labels[i])
            else:
                ax.plot(np.arange(K), kkt[:, i], label=labels[i])

        ax.set_title("KKT conditions")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residuals")
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.legend()
        return self

    # def plot_KKT_stationarity(self, semilogy=True):
    #     ax = self.axes[1, 2]
    #     if "KKT_stationarity" not in self.result.aux:
    #         ax.set_title("KKT-stationarity trajectory (missing in aux)")
    #         return self
    #     KKT_stationarity = self.result.aux["KKT_stationarity"]
    #     K, n_tot = KKT_stationarity.shape
        
    #     K = KKT_stationarity.shape[0]
    #     norm = np.linalg.norm(KKT_stationarity, axis=1)

    #     if semilogy:
    #         ax.semilogy(np.arange(K), norm, label="")
    #     else:
    #         ax.plot(np.arange(K), norm, label="||∇f(x) + sum λ_i * ∇h_i(x)||")

    #     ax.set_title("KKT stationarity condition")
    #     ax.set_xlabel("Iteration")
    #     ax.set_ylabel("Norm")
    #     ax.grid(True, which="both", ls="--", alpha=0.6)
    #     ax.legend()

    #     return self

    def plot_aux(self, keys, semilogy=True):
        ax = self.axes[1, 2]
        for key in keys:
            if key not in self.result.aux:
                ax.set_title("{key} trajectory (missing in aux)")
                return self
            val = self.result.aux[key]
            K, N, m = val.shape
            for i in range(N):
                if m == 1:
                    ax.plot(np.arange(K), val[:, i, 0], label=f"{key} Agent {i+1}")
                else:
                    for j in range(m):
                        ax.plot(np.arange(K), val[:, i, j], label=f"{key} Agent {i+1}, dim {j}")
        ax.set_title("Aux var. evolutions")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"{keys}")
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.legend()
        return self


# def plot_feasibility(self, semilogy=True):
#     ax = self.axes[2, 1]
#     if "kkt_traj" not in self.result.aux:
#         ax.set_title("Feasibility (missing kkt_traj)")
#         return self
#     kkt = self.result.aux["kkt_traj"]  # (K,3): [stationarity, primal_violation, complementarity]
#     K = kkt.shape[0]
#     names = ["Primal violation ||(Az-b)_+||_∞", "Complementarity max|λ⊙r|"]
#     series = [kkt[:,1], kkt[:,2]]
#     for s, name in zip(series, names):
#         if semilogy: ax.semilogy(np.arange(K), s, label=name)
#         else:        ax.plot(np.arange(K), s, label=name)
#     ax.set_title("Feasibility / Complementarity")
#     ax.set_xlabel("Iteration"); ax.set_ylabel("Value")
#     ax.grid(True, which="both", ls="--", alpha=0.6); ax.legend()
#     return self


    def plot_consensus_error(self, keys, semilogy=True):
        ax = self.axes[2, 1]
        for key in keys:
            if key not in self.result.aux:
                ax.set_title("{key} trajectory (missing in aux)")
                return self
            val = self.result.aux[key]
            K, N, d = val.shape
            val_bar = np.mean(val, axis=1)
            for i in range(N):
                if semilogy:
                    ax.semilogy(np.arange(K), np.linalg.norm(val[:, i] - val_bar, axis=1), label=f"{key} agent {i+1}")
                else:
                    ax.plot(np.arange(K), np.linalg.norm(val[:, i] - val_bar, axis=1), label=f"{key} agent {i+1}")
        
        ax.set_title(f"consensus error (val_i[k] - avg(val[k]))")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"{keys}")
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.legend()
        return self

# ---------------- Graph visualization ----------------
def show_graph_and_adj_matrix(fig, axs, graph, adj_matrix=None):
    """
    Show a networkx graph and its adjacency matrix.
    """
    nx.draw(graph, with_labels=True, ax=axs[0])
    if adj_matrix is None:
        adj_matrix = nx.adjacency_matrix(graph).toarray()
    cax = axs[1].matshow(adj_matrix, cmap='plasma')
    fig.colorbar(cax)


# ---------------- Cost evolution ----------------
def show_cost_evolution(ax, result, semilogy=False, cost_opt=None, label=None):
    """
    Plot the cost trajectory from RunResult.
    """
    cost = result.cost_traj
    K = len(cost)

    title = f"Cost evolution \n {'(Logarithmic y scale)' if semilogy else '(Linear scale)'}"
    ax.set_title(title)

    if semilogy:
        ax.semilogy(np.arange(K), np.abs(cost), label=label)
        if cost_opt is not None:
            ax.semilogy(np.arange(K), np.abs(cost_opt) * np.ones(K), "r--", label="Optimal cost")
    else:
        ax.plot(np.arange(K), cost, label=label)
        if cost_opt is not None:
            ax.plot(np.arange(K), cost_opt * np.ones(K), "r--", label="Optimal cost")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.grid(True, which="both", ls="--", alpha=0.6)
    if label is not None or cost_opt is not None:
        ax.legend()


def show_optimal_cost_error(ax, result, cost_opt, semilogy=False):
    """
    Plot the difference between current cost and optimal cost.
    """
    cost = result.cost_traj
    K = len(cost)

    title = f"Optimal cost error \n {'(Logarithmic y scale)' if semilogy else '(Linear scale)'}"
    ax.set_title(title)

    if semilogy:
        ax.semilogy(np.arange(K), np.abs(cost - cost_opt))
    else:
        ax.plot(np.arange(K), cost - cost_opt)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost - Cost*")
    ax.grid(True, which="both", ls="--", alpha=0.6)


# ---------------- Consensus error ----------------
def show_consensus_error(ax, result, semilogy=False):
    """
    Plot consensus error: distance of each agent's z from the mean at each iteration.
    Works only for d=1.
    """
    zz = result.zz_traj  # shape (K, N, d)
    K, N, d = zz.shape
    assert d == 1, "Consensus error plot only supports d=1 for now"

    z = zz[:, :, 0]              # (K, N)
    z_avg = np.mean(z, axis=1)   # (K,)

    title = f"Consensus error \n {'(Logarithmic y scale)' if semilogy else '(Linear scale)'}"
    ax.set_title(title)

    for i in range(N):
        if semilogy:
            ax.semilogy(np.arange(K), np.abs(z[:, i] - z_avg), label=f"Agent {i+1}")
        else:
            ax.plot(np.arange(K), z[:, i] - z_avg, label=f"Agent {i+1}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Deviation from mean")
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.legend()


# ---------------- Gradient norm ----------------
def show_norm_of_total_gradient(ax, result, semilogy=False, label=None):
    """
    Plot the norm of the total gradient from RunResult.
    """
    grad = result.grad_traj  # shape (K, d)
    K = grad.shape[0]

    title = f"Norm of the total gradient \n {'(Logarithmic y scale)' if semilogy else '(Linear scale)'}"
    ax.set_title(title)

    grad_norm = np.linalg.norm(grad, axis=1)

    if semilogy:
        ax.semilogy(np.arange(K), grad_norm, label=label)
    else:
        ax.plot(np.arange(K), grad_norm, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("||Grad||")
    ax.grid(True, which="both", ls="--", alpha=0.6)
    if label is not None:
        ax.legend()


# ---------------- Sigma trajectory ----------------
def show_sigma_evolution(ax, result, semilogy=False, label=None):
    """
    Plot the trajectory of sigma (average offloading ratio) from RunResult. 
    Assumes result.aux["sigma_traj"] exists.
    """
    if "sigma_traj" not in result.aux:
        raise KeyError("RunResult.aux does not contain 'sigma_traj'")

    sigma = result.aux["sigma_traj"]  # shape (K, d)
    K, d = sigma.shape

    title = "Sigma evolution"
    ax.set_title(title)

    for j in range(d):
        if semilogy:
            ax.semilogy(np.arange(K), sigma[:, j], label=f"dim {j}")
        else:
            ax.plot(np.arange(K), sigma[:, j], label=f"dim {j}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sigma value")
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.legend()


# ---------------- Helpers ----------------
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


# from matplotlib.patches import Patch
# def plot_trajectory_plane_with_affine_constraints(result, constraints, tol=1e-12):
#     """
#     Plot the trajectory of (z1, z2) for N=2, d=1, with multiple affine constraints.
#     Shades in RED the UNFEASIBLE region (union of violations).

#     Parameters
#     ----------
#     result : RunResult
#         Expected zz_traj shape (K, 2, 1).
#     constraints : list[tuple[(a1, a2), b]]
#         Each item is ((a1, a2), b) representing a1*z1 + a2*z2 <= b.
#     tol : float
#         Numerical tolerance for violation mask.
#     """
#     zz = result.zz_traj  # (K, 2, 1)
#     K, N, d = zz.shape
#     assert N == 2 and d == 1, f"Expected N=2, d=1 but got {N}, {d}"

#     z1 = zz[:, 0, 0]
#     z2 = zz[:, 1, 0]

#     fig, ax = plt.subplots(figsize=(6, 6))

#     # Trajectory
#     ax.plot(z1, z2, 'b.-', label="Trajectory")
#     ax.plot(z1[0], z2[0], 'go', label="Start")
#     ax.plot(z1[-1], z2[-1], 'ro', label="End")

#     # Fixed axes limits
#     ax.set_xlim(-0.1, 1.1)
#     ax.set_ylim(-0.1, 1.1)

#     # Box Constraint
#     rect1 = matplotlib.patches.Rectangle((0.0, 0.0), 1, 1, linestyle="dashed", ec='black', fc='None', lw=2)
#     ax.add_patch(rect1)

#     # Build a grid to shade the UNFEASIBLE region (union of violations)
#     xs = np.linspace(-0.1, 1.1, 400)
#     ys = np.linspace(-0.1, 1.1, 400)
#     X, Y = np.meshgrid(xs, ys)
#     violated = np.zeros_like(X, dtype=bool)

#     # Plot each constraint line and accumulate violation mask
#     for (a1, a2), b in constraints:
#         if abs(a2) > 1e-12:
#             # explicit form for z2
#             y_line = (b - a1 * xs) / a2
#             ax.plot(xs, y_line, 'k--', label=f"{a1}·z1 + {a2}·z2 = {b}")
#         else:
#             # vertical line: a1*z1 = b
#             if abs(a1) < 1e-12:
#                 continue  # degenerate constraint, skip
#             x_line = np.full_like(ys, b / a1)
#             ax.plot(x_line, ys, 'k--', label=f"{a1}·z1 = {b}")

#         # accumulate violations: a1*X + a2*Y > b
#         violated |= (a1 * X + a2 * Y > b + tol)

#     # Shade UNFEASIBLE region (union of all violations)
#     # Use contourf with two levels (0: feasible, 1: unfeasible)
#     Z = violated.astype(int)
#     ax.contourf(X, Y, Z, levels=[0.5, 1.5], colors=["red"], alpha=0.18)

#     # Legend: add a proxy for the red region
#     red_patch = Patch(facecolor='red', alpha=0.18, label='Unfeasible region')
#     handles, labels = ax.get_legend_handles_labels()
#     handles.append(red_patch); labels.append('Unfeasible region')
#     ax.legend(handles, labels, loc='best')

#     ax.set_title("Trajectory in (z1, z2) plane with multiple affine constraints")
#     ax.set_xlabel("z1")
#     ax.set_ylabel("z2")
#     ax.grid(True, ls="--", alpha=0.6)
#     plt.axis("equal")
#     plt.show()


from matplotlib.patches import Patch

def plot_trajectory_plane_with_affine_constraints(result, constraints, c, 
                                                  show_contours=True, n_levels=15, tol=1e-12):
    """
    Plot the trajectory of (z1, z2) for N=2, d=1, with multiple affine constraints.
    Shades in RED the UNFEASIBLE region (union of violations) and (optionally)
    overlays level sets of the cost J(x,y)=0.5*x^2 + 0.5*y^2 + ((x+y)/2 - c)^2.

    Parameters
    ----------
    result : RunResult
        Expected zz_traj shape (K, 2, 1).
    constraints : list[tuple[(a1, a2), b]]
        Each item is ((a1, a2), b) representing a1*z1 + a2*z2 <= b.
    tol : float
        Numerical tolerance for violation mask.
    c : float
        Target for the average (x+y)/2 in the cost.
    show_contours : bool
        If True, draws the level sets of J(x,y).
    n_levels : int
        Number of contour levels.
    """
    zz = result.zz_traj  # (K, 2, 1)
    K, N, d = zz.shape
    assert N == 2 and d == 1, f"Expected N=2, d=1 but got {N}, {d}"

    z1 = zz[:, 0, 0]
    z2 = zz[:, 1, 0]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Trajectory
    ax.plot(z1, z2, 'b.-', label="Trajectory")
    ax.plot(z1[0], z2[0], 'go', label="Start")
    ax.plot(z1[-1], z2[-1], 'ro', label="End")

    # Fixed axes limits
    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Box Constraint
    rect1 = matplotlib.patches.Rectangle((0.0, 0.0), 1, 1, linestyle="dashed", ec='black', fc='None', lw=2)
    ax.add_patch(rect1)

    # Grid (use same extents as axes)
    xs = np.linspace(x_min, x_max, 400)
    ys = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(xs, ys)

    # ===== Level sets of the cost J(x,y) =====
    if show_contours:
        # J = 0.5*X**2 + 0.5*Y**2 + ((np.square(X) + np.square(Y))/2.0 - c)**2
        J = 0.5*X**2 + 0.5*Y**2 + ((X + Y)/2.0 - c)**2
        # Choose levels that avoid extremely small/large values for readability
        Jmin, Jmax = np.percentile(J, 2), np.percentile(J, 98)
        levels = np.linspace(Jmin, Jmax, n_levels)
        CS = ax.contour(X, Y, J, levels=levels, linewidths=0.8, linestyles='solid', alpha=0.7)
        ax.clabel(CS, inline=True, fontsize=8, fmt="%.2f")
        # Add a legend handle
        contours_proxy = Patch(facecolor='none', edgecolor='gray', label='Cost level sets')
    else:
        contours_proxy = None

    # ===== Affine constraints & unfeasible region =====
    violated = np.zeros_like(X, dtype=bool)

    for (a1, a2), b in constraints:
        if abs(a2) > 1e-12:
            y_line = (b - a1 * xs) / a2
            ax.plot(xs, y_line, 'k--', label=f"{a1}·z1 + {a2}·z2 = {b}")
        else:
            if abs(a1) < 1e-12:
                continue
            x_line = np.full_like(ys, b / a1)
            ax.plot(x_line, ys, 'k--', label=f"{a1}·z1 = {b}")

        violated |= (a1 * X + a2 * Y > b + tol)

    Z = violated.astype(int)
    ax.contourf(X, Y, Z, levels=[0.5, 1.5], colors=["red"], alpha=0.18)

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
    plt.axis("equal")
    plt.show()


def plot_trajectory_plane(result, cost_function_offset, constraint, component, show_contours=True, n_levels=15, tol=1e-12):
    """
    Plot the trajectory of (z1, z2) for N=2, d=1, with multiple affine constraints.
    Shades in RED the UNFEASIBLE region (union of violations) and (optionally)
    overlays level sets of the cost J(x,y)=0.5*x^2 + 0.5*y^2 + ((x+y)/2 - c)^2.
    """
    zz = result.zz_traj  # (K, 2, 1)
    K, N, d = zz.shape
    # assert N == 2 and i == 1, f"Expected N=2, d=1 but got {N}, {d}"

    z1 = zz[:, 0, component]
    z2 = zz[:, 1, component]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Trajectory
    ax.plot(z1, z2, 'b.-', label="Trajectory")
    ax.plot(z1[0], z2[0], 'go', label="Start")
    ax.plot(z1[-1], z2[-1], 'ro', label="End")

    # Fixed axes limits
    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Box Constraint
    rect1 = matplotlib.patches.Rectangle((0.0, 0.0), 1, 1, linestyle="dashed", ec='black', fc='None', lw=2)
    ax.add_patch(rect1)

    # Grid (use same extents as axes)
    xs = np.linspace(x_min, x_max, 400)
    ys = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(xs, ys)

    # ===== Level sets of the cost J(x,y) =====
    if show_contours:
        J = 0.5*X**2 + 0.5*Y**2 + ((np.square(X) + np.square(Y))/2.0 - cost_function_offset)**2
        # Choose levels that avoid extremely small/large values for readability
        Jmin, Jmax = np.percentile(J, 2), np.percentile(J, 98)
        levels = np.linspace(Jmin, Jmax, n_levels)
        CS = ax.contour(X, Y, J, levels=levels, linewidths=0.8, linestyles='solid', alpha=0.7)
        ax.clabel(CS, inline=True, fontsize=8, fmt="%.2f")
        # Add a legend handle
        contours_proxy = Patch(facecolor='none', edgecolor='gray', label='Cost level sets')
    else:
        contours_proxy = None


    # ===== Sigma constraints & unfeasible region =====
    violated = np.zeros_like(X, dtype=bool)
    violated |= (0.5*(X**2 + Y**2) >= constraint)
    Z = violated.astype(int)
    ax.contourf(X, Y, Z, levels=[0.5, 1.5], colors=["red"], alpha=0.18)
    
    ax.set_title("Trajectory in (z1, z2) plane with sigma constraints")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.grid(True, ls="--", alpha=0.6)
    plt.axis("equal")
    plt.show()