import networkx as nx
import numpy as np
from CIoTS.pc_chen_algorithm import pc_chen_modified
from CIoTS.tools import transform_ts
from itertools import product
from time import time


def pc_incremental(indep_test, ts, alpha=0.05, max_p=20,
                   start=0, steps=1, stopping=None, verbose=False, **kwargs):
    # precalculated information
    dim = ts.shape[1]
    node_mapping, data_matrix = transform_ts(ts, max_p)
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)

    # verbose information
    graphs = {}
    times = {}

    # initial graph
    present_nodes = range(dim)
    if start > 0:
        start_time = time()
        G = pc_chen_modified(indep_test, ts, start, alpha)
        times[start] = time() - start_time
        graphs[start] = nx.relabel_nodes(G.copy(), node_mapping)
    else:
        G = nx.DiGraph()
        G.add_nodes_from(present_nodes)

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
                    ins_set = in_set - set([x])

        # check stopping criterion if exists
        if stopping is not None:
            stop = stopping(G, data_matrix, **kwargs)
        else:
            stop = False

        # verbose information
        graphs[p] = nx.relabel_nodes(G.copy(), node_mapping)
        times[p] = time() - start_time

        if stop:
            break

    if verbose:
        return nx.relabel_nodes(G, node_mapping), graphs, times
    else:
        return nx.relabel_nodes(G, node_mapping)