from collections import defaultdict

import networkx as nx
import numpy as np
from CIoTS.pc_chen_algorithm import pc_chen_modified
from CIoTS.stoppers import ICStopper
from CIoTS.tools import transform_ts
from itertools import product, combinations
from time import time


def _base_incremental(indep_test, ts, step3, alpha=0.05, max_tau=20, start=0,
                      steps=1, use_stopper=True, stopper=None, verbose=False, **kwargs):
    # precalculated information
    dim = ts.shape[1]

    # verbose information
    graphs = {}
    times = {}
    sepsets = {}

    if stopper is None:
        ic = 'bic'
        patiency = 2
        stopper = ICStopper(dim, patiency, ic)

    # initial graph
    present_nodes = range(dim)
    if start > 0:
        node_mapping, data_matrix = transform_ts(ts, 2*start)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)
        start_time = time()
        G = pc_chen_modified(indep_test, ts, start, alpha)
        times[start] = time() - start_time
        graphs[start] = nx.relabel_nodes(G.copy(), node_mapping)

        stopper.check_stop(G, start, data_matrix)
    else:
        G = nx.DiGraph()
        G.add_nodes_from(present_nodes)

    # if not use_stopper:
        # print(f'Stopper deactivated. Running until {list(range(start+steps, max_p+1, steps))[-1:]}')

    # iteration step
    for tau in range(start+steps, max_tau+1, steps):
        start_time = time()
        node_mapping, data_matrix = transform_ts(ts, 2*tau)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)
        new_nodes = list(range((tau-steps+1)*dim, min(tau+1, max_tau+1)*dim))

        # step 1
        G.add_nodes_from(new_nodes)

        # step 2
        for x_t, x in product(present_nodes, new_nodes):
            p_value, statistic = indep_test(data_matrix, x_t, x, set(),
                                            corr_matrix=corr_matrix)
            if p_value <= alpha:
                G.add_edge(x, x_t)

        # step 3
        step3(G, present_nodes, data_matrix, corr_matrix)

        # verbose information
        graphs[tau] = nx.relabel_nodes(G.copy(), node_mapping)
        if verbose:
            times[tau] = time() - start_time

        # early stopping
        # Even if stopper disabled, we keep running it to have scores for verbose output
        should_stop = stopper.check_stop(G, tau, data_matrix)
        if use_stopper and should_stop:
            break

    if verbose:
        return (nx.relabel_nodes(graphs[stopper.best_tau if use_stopper else tau], node_mapping),
                graphs, times, stopper, sepsets)
    else:
        return nx.relabel_nodes(graphs[stopper.best_tau if use_stopper else tau], node_mapping)


def pc_incremental(indep_test, ts, alpha=0.05, max_p=20, start=0, steps=1,
                   use_stopper=True, stopper=None, verbose=False, **kwargs):
    def step3(G, present_nodes, data_matrix, corr_matrix):
        for x_t in present_nodes:
            in_set = set(G.predecessors(x_t))
            for x in in_set:
                cond = in_set - set([x])
                p_value, statistic = indep_test(data_matrix, x_t, x, cond,
                                                corr_matrix=corr_matrix)
                if p_value > alpha:
                    G.remove_edge(x, x_t)
                    if verbose:
                        # sepsets[(node_mapping[x], node_mapping[x_t])] = [node_mapping[n] for n in cond]
                        pass

    return _base_incremental(indep_test, ts, step3, alpha=alpha, max_tau=max_p, start=start, steps=steps,
                             use_stopper=use_stopper, stopper=stopper, verbose=verbose, **kwargs)


def pc_incremental_extensive(indep_test, ts, alpha=0.05, max_p=20, start=0, steps=1,
                             use_stopper=True, stopper=None, verbose=False, **kwargs):
    def step3(G, present_nodes, data_matrix, corr_matrix):
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
                        if verbose:
                            # sepsets[(node_mapping[x], node_mapping[x_t])] = [node_mapping[n] for n in cond]
                            pass
            num_edges = new_num_edges
            new_num_edges = len(G.edges())

    return _base_incremental(indep_test, ts, step3, alpha=alpha, max_tau=max_p, start=start, steps=steps,
                             use_stopper=use_stopper, stopper=stopper, verbose=verbose, **kwargs)


def pc_incremental_subsets(indep_test, ts, alpha=0.05, max_p=20, start=0, steps=1,
                           use_stopper=True, stopper=None, verbose=False, **kwargs):
    def step3(G, present_nodes, data_matrix, corr_matrix):
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
                            if verbose:
                                # sepsets[(node_mapping[x], node_mapping[x_t])] = [node_mapping[n] for n in cond]
                                pass
                            break

    return _base_incremental(indep_test, ts, step3, alpha=alpha, max_tau=max_p, start=start, steps=steps,
                             use_stopper=use_stopper, stopper=stopper, verbose=verbose, **kwargs)


def pc1_step3(G, present_nodes, data_matrix, corr_matrix,
              indep_test, max_cond, alpha, verbose):
    for x_t in present_nodes:
        parents = list(set(G.predecessors(x_t)))
        # Goes up to full neighborhood, perhaps limit this
        max_cond_size = max_cond
        condition_size = 0
        # PC_1
        while condition_size < max_cond_size and condition_size <= len(parents) - 1:
            parent_stats = defaultdict(lambda: float('inf'))
            for x in parents:
                other_parents = [e for e in parents if e != x]
                condition = other_parents[:condition_size]

                p_value, statistic = indep_test(data_matrix, x_t, x, condition,
                                                corr_matrix=corr_matrix)
                parent_stats[x] = min(parent_stats[x], np.abs(statistic))

                if p_value > alpha:
                    G.remove_edge(x, x_t)
                    if verbose:
                        # sepsets[(node_mapping[x], node_mapping[x_t])] = [node_mapping[n] for n in condition]
                        pass
                    del parent_stats[x]

            parents = [k for k, v in sorted(parent_stats.items(), key=lambda v:v[1], reverse=True)]
            condition_size += 1


def pc_incremental_pc1(indep_test, ts, alpha=0.05, max_p=20, start=0, steps=1,
                       use_stopper=True, stopper=None, verbose=False, max_cond=float('inf'), **kwargs):
    def step3(G, present_nodes, data_matrix, corr_matrix):
        pc1_step3(G, present_nodes, data_matrix, corr_matrix,
                  indep_test, max_cond, alpha, verbose)

    return _base_incremental(indep_test, ts, step3, alpha=alpha, max_tau=max_p, start=start, steps=steps,
                             use_stopper=use_stopper, stopper=stopper, verbose=verbose, **kwargs)


def pc_incremental_pc1mci(indep_test, ts, alpha=0.05, max_p=20, start=0, steps=1,
                          use_stopper=True, stopper=None, verbose=False, max_cond=float('inf'), **kwargs):
    def step3(G, present_nodes, data_matrix, corr_matrix):
        # PC1
        pc1_step3(G, present_nodes, data_matrix, corr_matrix,
                  indep_test, max_cond, alpha, verbose)
        # MCI
        dim = len(present_nodes)
        for x_t in present_nodes:
            conds_y = list(G.predecessors(x_t))
            for cond_y in conds_y:
                cond_y_dim, cond_y_tau = cond_y % dim, cond_y // dim
                # import IPython; IPython.embed()
                conds_x = [c + cond_y_tau * dim for c in list(G.predecessors(cond_y_dim))]
                # if G.has_node(c + cond_y_tau * dim)]
                other_conds_y = [e for e in conds_y if e != cond_y]

                condition = set(other_conds_y + conds_x)

                p_value, statistic = indep_test(data_matrix, x_t, cond_y, condition,
                                                corr_matrix=corr_matrix)
                if p_value > alpha:
                    G.remove_edge(cond_y, x_t)

    return _base_incremental(indep_test, ts, step3, alpha=alpha, max_tau=max_p, start=start, steps=steps,
                             use_stopper=use_stopper, stopper=stopper, verbose=verbose, **kwargs)
