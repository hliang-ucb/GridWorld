import numpy as np
import networkx as nx
from itertools import product

def distance(node1, node2, type="tele"):
    if type == "space":
        g = grid_graph(4, 4)
    else:
        g = grid_graph(4, 4, tele=[0, 15])
        
    return nx.shortest_path_length(g, node1, node2)

def nodes_from_string(node_str):
    return np.array([int(node) for node in node_str.split(",")])

def use_tele(node_list):
    """
    Check if there are consecutive occurrences of 0 and 15 in the given node list.

    Args:
        node_list (list): A list of nodes.

    Returns:
        bool: True if there are consecutive occurrences of 0 and 15, False otherwise.
    """
    node_list = np.array(node_list)
    if (0 in node_list and 15 in node_list):
        where_0 = np.where(node_list == 0)[0]
        where_15 = np.where(node_list == 15)[0]
        consecutive_0_15 = any(abs(i - j) <= 1 for i in where_0 for j in where_15)
        return consecutive_0_15
    else:
        return False
    
def append_use_tele(df):
    n_trials = df.shape[0]
    if_use_tele = np.array([use_tele(nodes_from_string(df.iloc[trial].nodes)) for trial in range(n_trials)])
    df["use_tele"] = if_use_tele
    return df

def xy(node, nrows, ncols):
    """
    Transforms the node identity into X, y coordinates
    :param node: (int) node ID
    :param nrows: (int) number of rows
    :param ncols: (int) number of columns
    :return: (X, Y): (tuple) pair of XY coords
    """
    X = int(node % nrows)
    Y = int(np.floor(node / ncols))
    return X, Y


def optimal_steps(G, start, target):
    """
    :param G: (nx.graph)
    :param start: (int) starting node id
    :param target: (int) target node id
    :return: (int) number of steps
    """
    return nx.shortest_path_length(G, start, target)


def optimality(G, start, target, nsteps):
    """
    Computes the percent optimality given the starting node, target node, and number of steps over a graph G
    :param start: (int) starting node ID
    :param target: (int) target node ID
    :param nsteps: (int) number of actual steps taken
    :param G: (nx.graph) input graph
    :return: (float) percent optimality of the actual steps taken
    """
    optimal = optimal_steps(G, start, target)
    if nsteps == 0:
        return 0
    return 1 - ((nsteps - optimal) / nsteps)


def grid_graph(n_rows, n_cols, tele=None):
    """
    create a n_rows x n_cols networkx undirected grid graph
    :param n_rows: (int) # of rows
    :param n_cols: (int) # of columns
    :param tele: (tuple) pair of nodes to add a teleporteer between
    :return: nx.graph
    """
    edge_list = []
    vertical_pairs = np.array(
        [[j + i * n_rows, j + i * n_rows + 1] for (i, j) in product(range(n_cols), range(n_rows - 1))])
    for i, j in zip(vertical_pairs[:, 0], vertical_pairs[:, 1]):
        edge_list.append((i, j))
        edge_list.append((j, i))

    horizontal_pairs = np.array([[col, col+n_rows] for col in range((n_cols-1)*n_rows)])
    for i, j in zip(horizontal_pairs[:, 0], horizontal_pairs[:, 1]):
        edge_list.append((i, j))
        edge_list.append((j, i))

    if tele is not None:
        edge_list.append(tele)

    G = nx.Graph()
    G.add_edges_from(edge_list)
    return G


def neighbors(G, node):
    """
    return a list of neighbors of node on graph G
    :param G: (nx.graph)
    :param node: (int)
    :return:
    """
    return np.array(list(G.neighbors(node)))


def non_neighbors(G, node):
    """
    Return a list of nodes that are not neighbors of node from graph G.
    :param G: (nx.graph)
    :param node: (int)
    :return:
    """
    nodes = list(G.nodes).copy()
    for nei in neighbors(G, node):
        nodes.remove(nei)
    nodes.remove(node)
    return np.array(nodes)


def value_similarity(G, targets):
    nodes = np.array(G.nodes)
    n_nodes = len(nodes)
    n_targets = len(targets)
    value_by_target = np.zeros([n_nodes, n_targets])
    for t_i in range(len(targets)):
        for n_i in range(len(nodes)):
            value_by_target[n_i, t_i] = 5 - nx.shortest_path_length(G=G, source=nodes[n_i], target=targets[t_i])
    
    cm = np.zeros([n_nodes, n_nodes])
    for i in range(n_nodes):
        for j in range(n_nodes):
            cm[i, j] = sum(value_by_target[i, :] == value_by_target[j, :])
    
    return cm


def space_similarity(G):
    nodes = np.array(G.nodes)
    n_nodes = len(nodes)

    cm = np.zeros([n_nodes, n_nodes])
    for i in range(n_nodes):
        for j in range(n_nodes):
            cm[i, j] = nx.shortest_path_length(G=G, source=i, target=j)
    cm = np.max(cm) - cm
    return cm
    
def generate_policies(G):
    """
    generate optimal behavioral policies for spatial grid world
    :param G: (nx.graph) input graph to generate the policy over
    :return: [n x n x n] np.array where n := #nodes
        - dim 0: target location
        - dim 1: current location
        - dim 2: optimal next step
    """
    nnodes = len(G.nodes)
    policies = np.zeros([nnodes, nnodes, nnodes])
    for target in range(nnodes):
        for start in range(nnodes):
            shortest_paths = list(nx.all_shortest_paths(G, start, target))
            for path in shortest_paths:
                if len(path) > 1:
                    next_node = path[1]
                    policies[target, start, next_node] = 1
    return policies


