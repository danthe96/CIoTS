import networkx as nx
import numpy as np
from pcalg import estimate_cpdag
from CIoTS.stat_tools import partial_corr_test
from CIoTS.tools import transform_ts
from itertools import combinations, permutations


def pc_chen(indep_test_func, ts_data, p, alpha):
    dim = ts_data.shape[1]
    node_mapping, data_matrix = transform_ts(ts_data, p)
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)

    adj_matrix = np.ones((data_matrix.shape[1], data_matrix.shape[1]))
    np.fill_diagonal(adj_matrix, 0)
    G = nx.from_numpy_matrix(adj_matrix)
    
    G, sep_sets = _estimate_skeleton(G, partial_corr_test, data_matrix,
                                     alpha, corr_matrix=corr_matrix)

    DG = G.to_directed()
    DG.remove_edges_from([(u, v) for (u, v) in DG.edges()
                         if v >= dim])
    DAG = estimate_cpdag(DG, sep_sets)
    return nx.relabel_nodes(DAG, node_mapping)


def pc_chen_modified(indep_test_func, ts_data, p, alpha):
    dim = ts_data.shape[1]
    node_mapping, data_matrix = transform_ts(ts_data, p)
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)

    adj_matrix = np.zeros((data_matrix.shape[1], data_matrix.shape[1]))
    adj_matrix[dim:, :dim] = 1
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    G = nx.from_numpy_matrix(adj_matrix)
    G, _ = _estimate_skeleton(G, partial_corr_test, data_matrix,
                              alpha, corr_matrix=corr_matrix)

    DAG = G.to_directed()
    DAG.remove_edges_from([(u, v) for (u, v) in DAG.edges()
                          if v >= u])
    return nx.relabel_nodes(DAG, node_mapping)


def _estimate_skeleton(g, indep_test_func, data_matrix, alpha, **kwargs):
    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"

    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]

    l = 0
    while True:
        cont = False
        remove_edges = []
        for (i, j) in permutations(node_ids, 2):
            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
                pass
            if len(adj_i) >= l:
                if len(adj_i) < l:
                    continue
                for k in combinations(adj_i, l):
                    p_val = indep_test_func(data_matrix, i, j, set(k),
                                            **kwargs)
                    if p_val > alpha:
                        if g.has_edge(i, j):
                            if method_stable(kwargs):
                                remove_edges.append((i, j))
                            else:
                                g.remove_edge(i, j)
                        sep_set[i][j] |= set(k)
                        sep_set[j][i] |= set(k)
                        break
                cont = True
        l += 1
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    return (g, sep_set)