import numpy as np
from typing import List, Optional, Union, Literal

def _timeseries(
    ax, Y: np.ndarray,
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
            return _timeseries(
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