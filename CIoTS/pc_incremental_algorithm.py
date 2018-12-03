from collections import defaultdict

import networkx as nx
import numpy as np
from CIoTS.pc_chen_algorithm import pc_chen_modified
from CIoTS.tools import transform_ts
from CIoTS.simple_var import VAR
from itertools import product, combinations
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

    # verbose information
    graphs = {}
    times = {}
    bics = {}
    sepsets = {}

    # initial graph
    present_nodes = range(dim)
    if start > 0:
        node_mapping, data_matrix = transform_ts(ts, start)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)
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
        node_mapping, data_matrix = transform_ts(ts, p)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)
        new_nodes = list(range((p-steps+1)*dim, min(p+1, max_p+1)*dim))

        # step 1
        G.add_nodes_from(new_nodes)

        # step 2
        for x_t, x in product(present_nodes, new_nodes):
            p_value, statistic = indep_test(data_matrix, x_t, x, set(),
                                            corr_matrix=corr_matrix)
            if p_value <= alpha:
                G.add_edge(x, x_t)

        # step 3
        for x_t in present_nodes:
            in_set = set(G.predecessors(x_t))
            for x in in_set:
                cond = in_set - set([x])
                p_value, statistic = indep_test(data_matrix, x_t, x, cond,
                                                corr_matrix=corr_matrix)
                if p_value > alpha:
                    G.remove_edge(x, x_t)
                    sepsets[(node_mapping[x], node_mapping[x_t])] = [node_mapping[n] for n in cond]

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
        return nx.relabel_nodes(graphs[best_p], node_mapping), graphs, times, bics, sepsets
    else:
        return nx.relabel_nodes(graphs[best_p], node_mapping)


def pc_incremental_pc1(indep_test, ts, alpha=0.05, max_p=20, start=0, steps=1, ic='bic',
                       patiency=1, verbose=False, max_cond=float('inf'), **kwargs):
    # precalculated information
    dim = ts.shape[1]

    # verbose information
    graphs = {}
    times = {}
    bics = {}
    sepsets = {}

    # initial graph
    present_nodes = range(dim)
    if start > 0:
        node_mapping, data_matrix = transform_ts(ts, start)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)
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

    condition_sizes = []
    # iteration step
    for p in range(start+steps, max_p+1, steps):
        start_time = time()
        node_mapping, data_matrix = transform_ts(ts, p)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)
        new_nodes = list(range((p-steps+1)*dim, min(p+1, max_p+1)*dim))

        # step 1: Add new nodes
        G.add_nodes_from(new_nodes)

        # step 2: Connect new nodes if not unconditionally independent
        for x_t, x in product(present_nodes, new_nodes):
            p_value, statistic = indep_test(data_matrix, x_t, x, set(),
                                            corr_matrix=corr_matrix)
            if p_value <= alpha:
                G.add_edge(x, x_t)

        # step 3: Check all connected nodes
        cur_condition_sizes = []
        for x_t in present_nodes:
            parents = list(set(G.predecessors(x_t)))
            # Goes up to full neighborhood, perhaps limit this
            max_cond_size = float('inf') if no_imp >= patiency - 1 or p == max_p else max_cond
            condition_size = 0
            # PC_1
            while condition_size < max_cond_size and condition_size < len(parents) - 1:
                parent_stats = defaultdict(lambda: float('inf'))
                for x in parents:
                    other_parents = [e for e in parents if e != x]
                    condition = other_parents[:condition_size]

                    p_value, statistic = indep_test(data_matrix, x_t, x, condition,
                                                    corr_matrix=corr_matrix)
                    parent_stats[x] = min(parent_stats[x], np.abs(statistic))

                    if p_value > alpha:
                        G.remove_edge(x, x_t)
                        sepsets[(node_mapping[x], node_mapping[x_t])] = [node_mapping[n] for n in condition]
                        del parent_stats[x]

                parents = [k for k, v in sorted(parent_stats.items(), key=lambda v:v[1], reverse=True)]
                condition_size += 1
            cur_condition_sizes.append(condition_size)
        condition_sizes.append(cur_condition_sizes)

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
        return nx.relabel_nodes(graphs[best_p], node_mapping), graphs, times, bics, sepsets, condition_sizes
    else:
        return nx.relabel_nodes(graphs[best_p], node_mapping)


def pc_incremental_extensive(indep_test, ts, alpha=0.05, max_p=20, start=0,
                             steps=1, ic='bic', patiency=1, verbose=False):
    # precalculated information
    dim = ts.shape[1]

    # verbose information
    graphs = {}
    times = {}
    bics = {}
    sepsets = {}

    # initial graph
    present_nodes = range(dim)
    if start > 0:
        node_mapping, data_matrix = transform_ts(ts, start)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)
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
        node_mapping, data_matrix = transform_ts(ts, p)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)
        new_nodes = list(range((p-steps+1)*dim, min(p+1, max_p+1)*dim))

        # step 1
        G.add_nodes_from(new_nodes)

        # step 2
        for x_t, x in product(present_nodes, new_nodes):
            p_value, statistic = indep_test(data_matrix, x_t, x, set(),
                                            corr_matrix=corr_matrix)
            if p_value <= alpha:
                G.add_edge(x, x_t)

        # step 3 till number of edges converges
        num_edges = np.inf
        new_num_edges = len(G.edges())
        while num_edges > new_num_edges:
            for x_t in present_nodes:
                in_set = set(G.predecessors(x_t))
                for x in in_set:
                    cond = in_set - set([x])
                    p_value, statistic = indep_test(data_matrix, x_t, x, cond,
                                                    corr_matrix=corr_matrix)
                    if p_value > alpha:
                        G.remove_edge(x, x_t)
                        sepsets[(node_mapping[x], node_mapping[x_t])] = [node_mapping[n] for n in cond]
            num_edges = new_num_edges
            new_num_edges = len(G.edges())

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
        return nx.relabel_nodes(graphs[best_p], node_mapping), graphs, times, bics, sepsets
    else:
        return nx.relabel_nodes(graphs[best_p], node_mapping)


def pc_incremental_subsets(indep_test, ts, alpha=0.05, max_p=20, start=0,
                           steps=1, ic='bic', patiency=1, verbose=False):
    # precalculated information
    dim = ts.shape[1]

    # verbose information
    graphs = {}
    times = {}
    bics = {}
    sepsets = {}

    # initial graph
    present_nodes = range(dim)
    if start > 0:
        node_mapping, data_matrix = transform_ts(ts, start)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)
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
        node_mapping, data_matrix = transform_ts(ts, p)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)
        new_nodes = list(range((p-steps+1)*dim, min(p+1, max_p+1)*dim))

        # step 1
        G.add_nodes_from(new_nodes)

        # step 2
        for x_t, x in product(present_nodes, new_nodes):
            p_value, statistic = indep_test(data_matrix, x_t, x, set(),
                                            corr_matrix=corr_matrix)
            if p_value <= alpha:
                G.add_edge(x, x_t)

        # step 3 for each subset
        for subset_size in range(1, len(G.nodes())):
            for x_t in present_nodes:
                in_set = set(G.predecessors(x_t))
                if len(in_set) <= subset_size:
                    continue
                for x in in_set:
                    cond_max = in_set - set([x])
                    for cond in set(combinations(cond_max, subset_size)):
                        p_value, statistic = indep_test(data_matrix, x_t, x, cond,
                                                        corr_matrix=corr_matrix)
                        if p_value > alpha:
                            G.remove_edge(x, x_t)
                            sepsets[(node_mapping[x], node_mapping[x_t])] = [node_mapping[n] for n in cond]
                            break

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
        return nx.relabel_nodes(graphs[best_p], node_mapping), graphs, times, bics, sepsets
    else:
        return nx.relabel_nodes(graphs[best_p], node_mapping)
