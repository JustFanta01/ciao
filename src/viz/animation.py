import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def animate_offloading_with_mean(result, agents, interval=100):
    """
    Animate the evolution of offloading ratios z_i (bar chart) and the mean zbar (line plot).

    Parameters
    ----------
    result : RunResult
        Contains result.zz_traj of shape (K, N, d).
    agents : list[Agent]
        List of agents, each with .params to extract constants.
    interval : int
        Delay between frames in milliseconds.
    """
    zz = result.zz_traj  # (K, N, d)
    K, N, d = zz.shape
    assert d == 1, "Animation currently supports d=1."

    # mean trajectory
    zbar = zz[:, :, 0].mean(axis=1)

    # figure with 2 subplots
    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 2]})

    # labels from agents' params
    labels = []
    for i, ag in enumerate(agents):
        p = ag.cost_params()
        labels.append(
            f"Agent {i+1}\n"
            f"E_task={p.energy_task}\n"
            f"T_task={p.time_task}\n"
            f"E_tx={p.energy_tx}\n"
            f"RTT={p.time_rtt}"
        )
    labels.append("Mean")

    x = np.arange(N + 1)

    # init bars
    values0 = np.append(zz[0, :, 0], zbar[0])
    bars = ax_bar.bar(x, values0, tick_label=labels)
    bars[-1].set_color("orange")
    annotations = [ax_bar.text(b.get_x() + b.get_width()/2., b.get_height() + 0.02,
                               f"{b.get_height():.2f}", ha='center', va='bottom', fontsize=8)
                   for b in bars]

    ax_bar.set_ylim(0, 1.2)
    ax_bar.set_ylabel("Offloading ratio z_i")
    ax_bar.set_title("Offloading ratios per agent (+ mean)")

    # init line plot
    line, = ax_line.plot([], [], lw=2, label="Mean zbar")
    current_pt, = ax_line.plot([], [], 'ro')
    ax_line.set_xlim(0, K-1)
    ax_line.set_ylim(0, 1.05)
    ax_line.set_xlabel("Iteration")
    ax_line.set_ylabel("Mean offloading ratio")
    ax_line.set_title("Mean offloading ratio evolution")
    ax_line.grid(True)
    ax_line.legend()

    # update function
    def update(frame):
        # bars
        values = np.append(zz[frame, :, 0], zbar[frame])
        for bar, val, ann in zip(bars, values, annotations):
            bar.set_height(val)
            ann.set_y(val + 0.02)
            ann.set_text(f"{val:.2f}")
        ax_bar.set_title(f"Offloading ratios — Iteration {frame}")

        # line
        line.set_data(np.arange(frame+1), zbar[:frame+1])
        current_pt.set_data([frame], [zbar[frame]])
        return list(bars) + annotations + [line, current_pt]

    ani = animation.FuncAnimation(fig, update, frames=K, interval=interval, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
