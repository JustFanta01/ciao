import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from enum import Enum

class GraphType(Enum):
    ERDOS_RENYI = 1
    CYCLE = 2
    PATH = 3
    STAR = 4
    COMPLETE = 5
    # RANDOM_REGULAR_EXPANDER = 6

def metropolis_hastings_weights(graph):
    """    
    Computes the Metropolis-Hastings weights for a given graph.
    
    The weights are defined as follows:
    - If there is an edge between nodes i and j, the weight W[i,j] is given by:
      W[i,j] = 1 / (1 + max(d_i, d_j))
      where d_i and d_j are the degrees of nodes i and j, respectively.
    - The diagonal elements W[i,i] are set to 1 - sum(W[i,j]) for all j != i.

    """
    N = graph.number_of_nodes()
    # A = nx.adjacency_matrix(graph).toarray()
    W = np.zeros(shape=(N, N), dtype='float64')
    for i in range(N):
        N_i = list(graph.neighbors(i))
        d_i = len(N_i)
        for j in range(N):
            N_j = list(graph.neighbors(j))
            d_j = len(N_j)
            if graph.has_edge(i,j):
                W[i,j] = 1 / (1 + max(d_i, d_j))

        W[i,i] = 1 - np.sum(W[i, :])

    return W

def create_graph_with_metropolis_hastings_weights(NN, graph_type, args={}):
    """
    Creates a graph with Metropolis-Hastings weights based on the specified graph type.
    """
    ONES = np.ones((NN, NN))
    
    if graph_type == GraphType.ERDOS_RENYI:
        p_er = args['edge_probability']
        seed = args['seed']
        while True:
            G = nx.erdos_renyi_graph(NN, p_er, seed)
            Adj = nx.adjacency_matrix(G).toarray()
            is_strongly_connected = np.all(np.linalg.matrix_power(Adj + np.eye(NN), NN) > 0)
            
            if is_strongly_connected:
                break

    elif graph_type == GraphType.CYCLE:
        G = nx.cycle_graph(NN)

    elif graph_type == GraphType.PATH:
        G = nx.path_graph(NN)
    
    elif graph_type == GraphType.STAR:
        G = nx.star_graph(NN - 1)

    elif graph_type == GraphType.COMPLETE:
        G = nx.complete_graph(NN)

    A = metropolis_hastings_weights(G)

    return G, A

def create_graph_birkhoff_von_neumann(NN, num_vertices):
    I_n = np.eye(NN)
    
    # key=hash; value=np.ndarray
    vertices = {}
    
    # Number of ways to choose k items from n items without repetition and with order.
    max = math.perm(NN, NN)
    print(f"Number of possibles vertices: {math.perm(NN, NN)}") # math.factorial(NN)
    assert num_vertices <= max, "Vertices won't be unique"

    # Ensure the presence of the Identity Matrix, self-loops
    vertices[hash(I_n.tobytes())] = I_n

    while(len(vertices) < num_vertices):
        vertix = np.random.permutation(I_n)
        vertix_hash = hash(vertix.tobytes())
        
        if vertix_hash not in vertices:
            vertices[vertix_hash] = vertix
    
    # k=class_weights, the len(k) is the number of extracted numbers
    class_weights = np.ones((num_vertices)) # equally distributed classes

    convex_coefficients = np.random.dirichlet(alpha=class_weights, size=1).squeeze()
    # print(f"convex_coefficients: {convex_coefficients}")

    doubly_stochastic_matrix = np.zeros((NN, NN))
    vertices = list(vertices.values())
    # print(f"vertices: {vertices}")
    
    for i in range(num_vertices):
        doubly_stochastic_matrix += vertices[i] * convex_coefficients[i]

    # Ensure symmetry i.e. undirected graph
    doubly_stochastic_matrix = (doubly_stochastic_matrix + doubly_stochastic_matrix.T) / 2

    # print(doubly_stochastic_matrix)
    # print(np.sum(doubly_stochastic_matrix, axis=0))
    # print(np.sum(doubly_stochastic_matrix, axis=1))

    paths_up_to_N = np.linalg.matrix_power(doubly_stochastic_matrix + np.eye(NN), NN)

    # if is full, then in strongly connected
    if np.all(paths_up_to_N > 0):
        print("The graph is strongly connected")
    else: 
        print("The graph is NOT strongly connected!")

    graph = nx.from_numpy_array(doubly_stochastic_matrix)

    return graph, doubly_stochastic_matrix
