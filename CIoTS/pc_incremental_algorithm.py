from abc import ABC, abstractmethod
from collections import defaultdict

import networkx as nx
import numpy as np
from CIoTS.pc_chen_algorithm import pc_chen_modified
from CIoTS.tools import transform_ts
from CIoTS.simple_var import VAR
from itertools import product, combinations
from time import time


class Stopper(ABC):

    @abstractmethod
    def scores(self):
        pass

    @abstractmethod
    def best_p(self):
        pass

    @abstractmethod
    def check_stop(self):
        pass


class ICStopper():

    def __init__(self, dim, patiency=1, ic='bic'):
        self.dim = dim
        self.ic = ic
        self.patiency = patiency

        self.bics = {}
        self.no_imp = 0
        self.best_p = 0
        self.best_bic = np.inf

    def scores(self):
        return self.bics

    def best_p(self):
        return self.best_p

    def check_stop(self, graph, p, data_matrix):
        self.bics[p] = self._graph_ic(graph, p, data_matrix)

        if self.bics[p] < self.best_bic:
            self.best_bic = self.bics[p]
            self.best_p = p
            self.no_imp = 0
        else:
            self.no_imp += 1
            if self.no_imp >= self.patiency:
                return True
        return False

    def _graph_ic(self, graph, p, data_matrix):
        free_params = len(graph.edges()) + len(graph.nodes())
        model = VAR(p)
        model.fit_from_graph(self.dim, data_matrix, graph)
        return model.information_criterion(self.ic, free_params=free_params)


def _base_incremental(indep_test, ts, step3, alpha=0.05, max_p=20, start=0, steps=1,
                      ic='bic', patiency=1, verbose=False, **kwargs):
    # precalculated information
    dim = ts.shape[1]

    # verbose information
    graphs = {}
    times = {}
    sepsets = {}

    stopper = ICStopper(dim, patiency, ic)

    # initial graph
    present_nodes = range(dim)
    if start > 0:
        node_mapping, data_matrix = transform_ts(ts, start)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False)
        start_time = time()
        G = pc_chen_modified(indep_test, ts, start, alpha)
        times[start] = time() - start_time
        graphs[start] = nx.relabel_nodes(G.copy(), node_mapping)

        stopper.check_stop(G, start, data_matrix)
    else:
        G = nx.DiGraph()
        G.add_nodes_from(present_nodes)

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
        step3(G, present_nodes, data_matrix, corr_matrix)

        # verbose information
        graphs[p] = nx.relabel_nodes(G.copy(), node_mapping)
        times[p] = time() - start_time

        # early stopping
        if stopper.check_stop(G, p, data_matrix):
            break

    if verbose:
        return nx.relabel_nodes(graphs[stopper.best_p], node_mapping), graphs, times, stopper.scores(), sepsets
    else:
        return nx.relabel_nodes(graphs[stopper.best_p], node_mapping)


def pc_incremental(indep_test, ts, alpha=0.05, max_p=20, start=0, steps=1,
                   ic='bic', patiency=1, verbose=False, **kwargs):
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

    return _base_incremental(indep_test, ts, step3, alpha, max_p,
                             start, steps, ic, patiency, verbose, **kwargs)


def pc_incremental_pc1(indep_test, ts, alpha=0.05, max_p=20, start=0, steps=1, ic='bic',
                       patiency=1, verbose=False, max_cond=float('inf'), **kwargs):
    def step3(G, present_nodes, data_matrix, corr_matrix):
        for x_t in present_nodes:
            parents = list(set(G.predecessors(x_t)))
            # Goes up to full neighborhood, perhaps limit this
            max_cond_size = max_cond
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
                        if verbose:
                            # sepsets[(node_mapping[x], node_mapping[x_t])] = [node_mapping[n] for n in condition]
                            pass
                        del parent_stats[x]

                parents = [k for k, v in sorted(parent_stats.items(), key=lambda v:v[1], reverse=True)]
                condition_size += 1

    return _base_incremental(indep_test, ts, step3, alpha, max_p,
                             start, steps, ic, patiency, verbose, **kwargs)


def pc_incremental_extensive(indep_test, ts, alpha=0.05, max_p=20, start=0,
                             steps=1, ic='bic', patiency=1, verbose=False, **kwargs):
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

    return _base_incremental(indep_test, ts, step3, alpha, max_p,
                             start, steps, ic, patiency, verbose, **kwargs)


def pc_incremental_subsets(indep_test, ts, alpha=0.05, max_p=20, start=0,
                           steps=1, ic='bic', patiency=1, verbose=False, **kwargs):
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

    return _base_incremental(indep_test, ts, step3, alpha, max_p,
                             start, steps, ic, patiency, verbose, **kwargs)
