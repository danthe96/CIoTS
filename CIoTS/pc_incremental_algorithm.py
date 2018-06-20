import networkx as nx
import numpy as np
from CIoTS.pc_chen_algorithm import pc_chen_modified
from CIoTS.tools import transform_ts
from CIoTS.simple_var import VAR
from itertools import product
from time import time


def _graph_ic(p, dim, data_matrix, graph, ic='bic'):
    free_params = len(graph.edges()) + len(graph.nodes())
    model = VAR(p)
    model.fit_from_graph(dim, data_matrix, graph)
    return model.information_criterion(ic, free_params=free_params)


def pc_incremental(indep_test, ts, alpha=0.05, max_p=20, start=0, steps=1,
                   ic='bic', patiency=1, verbose=False, **kwargs):
    # precalculated information
    dim = ts.shape[1]
    node_mapping, data_matrix = transform_ts(ts, max_p)
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)

    # verbose information
    graphs = {}
    times = {}
    bics = {}

    # initial graph
    present_nodes = range(dim)
    if start > 0:
        start_time = time()
        G = pc_chen_modified(indep_test, ts, start, alpha)
        times[start] = time() - start_time
        graphs[start] = nx.relabel_nodes(G.copy(), node_mapping)
        bics[start] = _graph_ic(start, dim, data_matrix, G, ic)
        best_bic = bics[start]
        best_p = start
    else:
        G = nx.DiGraph()
        G.add_nodes_from(present_nodes)
        best_bic = np.inf
        best_p = 0

    no_imp = 0

    # iteration step
    for p in range(start+steps, max_p+1, steps):
        start_time = time()
        new_nodes = range(p*dim, min(p+steps, max_p)*dim)

        # step 1
        G.add_nodes_from(new_nodes)

        # step 2
        for x_t, x in product(present_nodes, new_nodes):
            p_value = indep_test(data_matrix, x_t, x, set(),
                                 corr_matrix=corr_matrix)
            if p_value <= alpha:
                G.add_edge(x, x_t)

        # step 3
        for x_t in present_nodes:
            in_set = set(G.predecessors(x_t))
            for x in in_set:
                cond = in_set - set([x])
                p_value = indep_test(data_matrix, x_t, x, cond,
                                     corr_matrix=corr_matrix)
                if p_value > alpha:
                    G.remove_edge(x, x_t)

        # verbose information
        graphs[p] = nx.relabel_nodes(G.copy(), node_mapping)
        times[p] = time() - start_time
        bics[p] = _graph_ic(p, dim, data_matrix, G, ic)

        # early stopping
        if bics[p] < best_bic:
            best_bic = bics[p]
            best_p = p
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patiency:
                break

    if verbose:
        return nx.relabel_nodes(graphs[best_p], node_mapping), graphs, times, bics
    else:
        return nx.relabel_nodes(graphs[best_p], node_mapping)
