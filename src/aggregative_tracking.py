import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import graph_utils, plot_utils

seed = 42
np.random.seed(seed)

# [ global parameters ]
max_iter = 200

def cost_fn(zz_i, z_bar, alpha, beta, energy_tx, energy_task, time_task, time_rtt):
    def _rtt(z_bar): # round-trip time, increasing in z_bar
        return time_rtt * z_bar
    
    def _cost_function_cloud(z_bar):
            """ $$ J_i^{cloud}(\bar{z}) := \alpha t_i(\bar{z}) + \beta e_i^{tx} $$
                with $$ t_i(\bar{z}) = t\cdot \bar{z}$$
            """
            return alpha * _rtt(z_bar) + beta * energy_tx


    def _cost_function_local():
        """ $$ J_i^{loc}(z_i) := e_i^{task} + t_i^{task} $$
        """
        return energy_task + time_task
        

    """ [ cost function ]:  $$ J_i(z_i, \bar{z}) := (1-z_i) \cdot J_i^{loc}(z_i) + z_i \cdot J_i^{cloud}(\bar{z}) $$
    """
    local = _cost_function_local()
    cloud = _cost_function_cloud(z_bar)
    cost = (1 - zz_i) * local + zz_i * cloud
    
    return cost
    
def gradient_computation(zz_i, avg_offload_ratio, constants, N, type):
    (alpha, beta, time_rtt, time_task, energy_tx, energy_task) = constants
    if type == 'first':
        # Thus, the gradient of the cost function $$ J_i(z_i, \bar{z}) $$ with respect to $$ z_i $$ is:
        
        # $$ \frac{\partial J_i}{\partial z_i} = - (e_i^{task} + t_i^{task}) + \alpha t \bar{z} + \beta e_i^{tx} + \frac{\alpha \cdot t}{N}z_i $$
    
        grad = -(energy_task + time_task) + alpha * time_rtt * avg_offload_ratio + beta * energy_tx + (alpha * time_rtt * zz_i)/N

    elif type == 'second':
        
        # $$ \frac{\partial J_i}{\partial z_i} = z_i \cdot \alpha \cdot t  $$
        
        grad = zz_i * alpha * time_rtt
    else:
        raise ValueError("Invalid type. Use 'first' or 'second'.")
    return grad

def aggregative_variable(zz):
    avg_offload_ratio = np.mean(zz, axis=0)[0]
    print(f"avg_offload_ratio: {avg_offload_ratio}")
    return avg_offload_ratio

def centralized_aggregative_method(stepsize: float, z_init: np.ndarray, dim: tuple, cost_functions, alpha, beta, time_rtt, time_task, energy_tx, energy_task):
    (max_iter, N, d) = dim
    zz = np.zeros((max_iter, N, d))  # state: positions of the agents
    cost = np.zeros(max_iter)
    avg_offload_ratio_history = np.zeros(max_iter)
    grad = np.zeros((max_iter, d))
    
    zz[0] = z_init

    def centralized_cost_fn(zz, avg_offload_ratio):
        total_cost = 0
        for i in range(N):
            c = cost_functions[i](zz[i], avg_offload_ratio)
            total_cost += c
        return total_cost
    
    for k in range(max_iter - 1):
        avg_offload_ratio = aggregative_variable(zz[k])    # Compute the avg_offload_ratio of the agents
        avg_offload_ratio_history[k] = avg_offload_ratio
        cost[k] = centralized_cost_fn(zz[k], avg_offload_ratio)
        print(f"Centralized cost at iteration {k}: {cost[k]}")
        
        for i in range(N):
            gradient_sum = np.zeros(d)
            for j in range(N):
                constants_j = (alpha[j], beta[j], time_rtt[j], time_task[j], energy_tx[j], energy_task[j])
                gradient_sum += gradient_computation(zz[k,j], avg_offload_ratio, constants_j, N, type='second')

            constants_i = (alpha[i], beta[i], time_rtt[i], time_task[i], energy_tx[i], energy_task[i])
            nabla_1 = gradient_computation(zz[k,i], avg_offload_ratio, constants_i, N, type='first')
            nabla_phi = np.eye(d)
            grad_i = nabla_1 + 1/N * nabla_phi @ gradient_sum
            zz[k+1, i] = zz[k,i] - stepsize * grad_i
            zz[k+1, i] = np.clip(zz[k+1, i], 0, 1)   # vincolo [0,1]
            
            grad[k] += grad_i

    return cost, grad, zz, avg_offload_ratio_history

def aggregative_tracking(stepsize, z_init, dim, cost_functions, alpha, beta, time_rtt, time_task, energy_tx, energy_task, adj):
    # 3 states
    (max_iter, N, d) = dim
    zz = np.zeros((max_iter, N, d))  # positions of the agents
    
    ss = np.zeros((max_iter, N, d))  # proxy of $$\sigma(x^k)$$

    vv = np.zeros((max_iter, N, d))  # proxy of $$\frac{1}{N}\sum_{j=1}^{N}\nabla_2\ell_J(z_j^k, \sigma(z^k))$$
    
    total_cost = np.zeros(max_iter)
    total_grad = np.zeros((max_iter, d))

    # initialization
    zz[0] = z_init 
    ss[0] = z_init # \phi_{i}(z) is the identity function
    # xx[0] = z_init 
    for i in range(N):
        constants_i = (alpha[i], beta[i], time_rtt[i], time_task[i], energy_tx[i], energy_task[i])
        vv[0, i] = gradient_computation(zz[0,i], ss[0, i], constants_i, N=N, type='second')

    # run
    for k in range(max_iter-1):
        for i in range(N):
            # NOTE: usage of the tracker ss[k,i] instead of avg_offload_ratio
            cost = cost_functions[i](zz[k,i], ss[k,i])
            constants_i = (alpha[i], beta[i], time_rtt[i], time_task[i], energy_tx[i], energy_task[i])
            
            # [ zz update ]
            nabla_1 = gradient_computation(zz[k,i], ss[k,i], constants_i, N=N, type='first')
            nabla_phi = np.eye(d)
            grad = nabla_1 + nabla_phi @ vv[k,i]

            total_cost[k] += cost
            total_grad[k] += grad

            zz[k+1, i] = zz[k, i] - stepsize * grad
            zz[k+1, i] = np.clip(zz[k+1, i], 0, 1)   # vincolo [0,1]
                
            # [ ss update ]
            ss_consensus = ss[k].T @ adj[i].T
            ss_local_innovation = zz[k+1, i] - zz[k, i] # $$\phi(z_i^{k+1}) - \phi(z_i^{k})$$
            ss[k+1, i] = ss_consensus + ss_local_innovation

            # [ vv update ]
            vv_consensus = vv[k].T @ adj[i].T
            nabla2_k = gradient_computation(zz[k,i], ss[k,i], constants_i, N=N, type='second')
            nabla2_k_plus_1 = gradient_computation(zz[k+1,i], ss[k+1,i], constants_i, N=N, type='second')
            vv_local_innovation = nabla2_k_plus_1 - nabla2_k
            vv[k+1, i] = vv_consensus + vv_local_innovation

    return total_cost, total_grad, zz, ss, vv


def animate_offloading(zz, interval=100):
    """
    Crea un'animazione dei valori z_i per tutti gli agenti e la media.
    zz ha shape (max_iter, N, d).
    """
    max_iter, N, d = zz.shape
    fig, ax = plt.subplots(figsize=(8, 5))

    # Inizializzazione delle barre
    values = np.append(zz[0,:,0], np.mean(zz[0,:,0]))
    bars = ax.bar(range(N+1), values, tick_label=[f"Agente {i+1}" for i in range(N)] + ["Media"])
    bars[-1].set_color("orange")

    ax.set_ylim(0, 1.2)
    ax.set_ylabel("z_i")
    ax.set_title("Evoluzione degli offloading ratio")

    def update(frame):
        ax.clear()
        values = np.append(zz[frame,:,0], np.mean(zz[frame,:,0]))
        bars = ax.bar(range(N+1), values, tick_label=[f"Agente {i+1}" for i in range(N)] + ["Media"])
        bars[-1].set_color("orange")
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("z_i")
        ax.set_title(f"Iterazione {frame}")
        return bars

    ani = animation.FuncAnimation(fig, update, frames=max_iter, interval=interval, blit=False, repeat=False)
    plt.show()


def main():
    N = 5  # number of agents
    d = 1  # dimension of the state space

    # z_init = np.zeros((N, d))
    z_init = np.random.uniform(low=0, high=1, size=(N, d))

    args = {'edge_probability': 0.65, 'seed': seed}
    graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.ERDOS_RENYI, args)

    # define ell_i
    alpha = np.ones(N)
    beta = np.ones(N)
    energy_task = np.ones(N) * 4    # local
    time_task = np.ones(N) * 3      # local
    energy_tx = np.ones(N) * 3      # cloud
    time_rtt = np.ones(N) * 2       # cloud

    cost_functions = []
    for i in range(N):
        cost_functions.append(lambda zz_i, avg_offload_ratio, i=i: cost_fn(zz_i, avg_offload_ratio, alpha[i], beta[i], energy_tx[i], energy_task[i], time_task[i], time_rtt[i]))

    stepsize = 0.01
    dim = (max_iter, N, d)

    # -----------------------
    # |     CENTRALIZED     |
    # -----------------------
    cost_centr, grad_centr, zz_centr, avg_offload_ratio = centralized_aggregative_method(stepsize, z_init, dim, cost_functions, alpha, beta, time_rtt, time_task, energy_tx, energy_task)

    # -----------------------
    # |     DISTRIBUTED     |
    # -----------------------
    res = aggregative_tracking(stepsize, z_init, dim, cost_functions, alpha, beta, time_rtt, time_task, energy_tx, energy_task, adj)
    (total_cost_distr, total_grad_distr, zz_distr, ss_distr, vv_distr) = res

    # TODO: bias del tracker s rispetto alla vera media, puo' essere utile!!!
    sbar_bias = np.linalg.norm(np.mean(ss_distr[-1, :, 0]) - np.mean(zz_distr[-1, :, 0]))
    print(f"[CHECK] s-tracker bias on zbar: {sbar_bias:.3e}")

    fig, axs = plt.subplots(figsize=(7, 4), nrows=1, ncols=2)
    title = f"Graph and Adj Matrix"
    fig.suptitle(title)
    fig.canvas.manager.set_window_title(title)
    plot_utils.show_graph_and_adj_matrix(fig, axs, graph, adj)
    plot_utils.show_and_wait(fig)


    fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=2)

    ax = axes[0]
    plot_utils.show_cost_evolution(ax, cost_centr, max_iter, semilogy=True, label="Centralized Aggregative Tracking")
    plot_utils.show_cost_evolution(ax, total_cost_distr, max_iter, semilogy=True, label="Distributed Aggregative Tracking")

    ax = axes[1]
    plot_utils.show_norm_of_total_gradient(ax, grad_centr, max_iter, semilogy=True, label="Centralized Aggregative Tracking")
    plot_utils.show_norm_of_total_gradient(ax, total_grad_distr, max_iter, semilogy=True, label="Distributed Aggregative Tracking")

    plot_utils.show_and_wait(fig)

    animate_offloading(zz_centr)
    animate_offloading(zz_distr)


if __name__ == "__main__":
    main()
        
