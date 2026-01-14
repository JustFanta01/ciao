import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from plots import plots
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import graph_utils

N = 5
args = {'edge_probability': 0.45, 'seed': 1}

# +--------------------+
# |  ADJACENCY MATRIX  |
# +--------------------+
graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(NN=N, graph_type=graph_utils.GraphType.ERDOS_RENYI, args=args)

fig, axs = plt.subplots(figsize=(7, 4), nrows=1, ncols=2)
title = f"Graph and Adj Matrix"
fig.suptitle(title)
fig.canvas.manager.set_window_title(title)
plots.show_graph_and_adj_matrix(fig, axs, graph, adj)
plots.show_and_wait(fig)

# +---------------------+
# | Compute $$I, J$$ and $$ \mathcal L^2 $$ |
# +---------------------+
I = np.eye(N)
J = 1/N * np.ones((N,N))

L2 = 0.5 * (I - adj)

# +-----------------+
# |  SVD  to get $$\mathcal L$$ |
# +-----------------+
if False:
    U, S, V = np.linalg.svd(L2)
    print("---------[ U ]--------")
    print(U)

    print("---------[ S ]--------")
    print(S)

    print("---------[ V ]--------")
    print(V)

    print("---------[ sqrt(S) ]--------")
    print(np.sqrt(S))

    L = U @ np.diag(np.sqrt(S)) @ V
    print("---------[ L ]--------")
    print(L)

    print(f"np.linalg.eigvals(L): {np.linalg.eigvals(L)}")
    print(f"np.linalg.matrix_norm(L, ord=2): {np.linalg.matrix_norm(L, ord=2)}")


if True:
    M = (I-J)@(I-L2)
    print("---------[ M ]--------")
    print(M)
    
    print("---------[ L2 ]--------")
    print(f"np.linalg.eigvals(L2): {np.linalg.eigvals(L2)}")
    print(f"np.linalg.matrix_norm(L2, ord=2): {np.linalg.matrix_norm((L2), ord=2)}")
    
    print("---------[ I-L2 ]--------")
    print(f"np.linalg.eigvals(I-L2): {np.linalg.eigvals(I-L2)}")
    print(f"np.linalg.matrix_norm(I-L2, ord=2): {np.linalg.matrix_norm((I-L2), ord=2)}")
    

    print("---------[ M = (I-J)(I-L2) ]--------")
    print(f"np.linalg.eigvals(M): {np.linalg.eigvals(M)}")
    print(f"np.linalg.matrix_norm(M, ord=2): {np.linalg.matrix_norm(M, ord=2)}")