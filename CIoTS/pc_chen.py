import networkx as nx
import numpy as np
from pcalg import estimate_skeleton, estimate_cpdag
from stat_tools import partial_corr_test
from tools import transform_ts


def pc_chen(indep_test_func, ts_data, p, alpha):
    dim = ts_data.shape[1]
    node_mapping, data_matrix = transform_ts(ts_data, p)
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)
    G, sep_sets = estimate_skeleton(partial_corr_test, data_matrix,
                                    alpha, corr_matrix=corr_matrix)
    DG = G.to_directed
    DG.remove_edges_from([(u, v) for (u, v) in G.edges()
                          if v >= dim])
    DAG = estimate_cpdag(DG, sep_sets)
    return nx.relabel_nodes(DAG, node_mapping)
