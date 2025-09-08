import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

def show_graph_and_adj_matrix(fig, axs, graph, adj_matrix=None):
    nx.draw(graph, with_labels=True, ax=axs[0])

    if adj_matrix is None:
        adj_matrix = nx.adjacency_matrix(graph).toarray()
    
    cax = axs[1].matshow(adj_matrix, cmap='plasma')# , vmin=0, vmax=1)
    fig.colorbar(cax)

def show_cost_evolution(ax, cost, max_iter, semilogy=False, cost_opt=None, label=None):
    # optimal cost error - one line! we are minimizing the sum not each l_i
    title = f"Cost evolution \n {'(Logaritmic y scale)' if semilogy else '(Linear scale)'}"
    ax.set_title(title)
    if semilogy:
        ax.semilogy(np.arange(max_iter - 1), np.abs(cost[:-1]), label=label)
        if cost_opt is not None:
            ax.semilogy(np.arange(max_iter - 1), np.abs(cost_opt * np.ones((max_iter - 1))), "r--")
    else:
        ax.plot(np.arange(max_iter - 1), cost[:-1], label=label)
        if cost_opt is not None:
            ax.plot(np.arange(max_iter - 1), cost_opt * np.ones((max_iter - 1)), "r--")

    if label is not None:
        ax.legend()

def show_optimal_cost_error(ax, cost, cost_opt, max_iter, semilogy=False):
    title = f"Optimal cost error \n {'(Logaritmic y scale)' if semilogy else '(Linear scale)'}"
    ax.set_title(title)
    
    if semilogy:
        ax.semilogy(np.arange(max_iter - 1), np.abs(cost[:-1] - cost_opt))
    else:
        ax.plot(np.arange(max_iter - 1), cost[:-1] - cost_opt)
        
def show_consensus_error(ax, NN, z, max_iter, semilogy=False):
    z_avg = np.mean(z, axis=1)
    title = f"Consensus error \n {'(Logaritmic y scale)' if semilogy else '(Linear scale)'}"
    ax.set_title(title)

    for i in range(NN):
        # distance for each agent from the average of that iteration
        if semilogy:
            ax.semilogy(np.arange(max_iter), np.abs(z[:, i] - z_avg))
        else:
            ax.plot(np.arange(max_iter), z[:, i] - z_avg)

def show_norm_of_total_gradient(ax, grad, max_iter, semilogy=False, label=None):
    title = f"Norm of the total gradient \n {'(Logaritmic y scale)' if semilogy else '(Linear scale)'}"
    ax.set_title(title)

    if semilogy:    
        ax.semilogy(np.arange(max_iter - 1), np.linalg.norm(grad[:-1], axis=1), label=label)
    else:
        ax.plot(np.arange(max_iter - 1), np.linalg.norm(grad[:-1], axis=1), label=label)

    if label is not None:
        ax.legend()


def close(event):
    if event.key == 'q':    # Check if the pressed key is 'q'
        plt.close()         # Close the figure
    elif event.key == 'n':
        pass
        # TODO: is there a way to press 'n' and stop waiting for the user to close the plot?

def show_and_wait(fig=None):
    if fig == None:
        # gcf() == get_current_figure()
        plt.gcf().canvas.mpl_connect('key_press_event', close)
    else:
        fig.canvas.mpl_connect('key_press_event', close)
    
    plt.show(block=True)