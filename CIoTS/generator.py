import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import is_stable
from random import sample, seed
from itertools import product, permutations


def node_name(dim, tau):
    return f'X{dim}_t-{tau}' if tau > 0 else f'X{dim}_t'


def node_id(name):
    regex = re.search(r'X(\d+)_t(-(\d+))?', name)
    return int(regex.group(1)), int(regex.group(3) or 0)


def draw_graph(graph, dimensions, max_p, positions=None):
    if positions is None:
        positions = {}
        for node in graph.nodes():
            i, j = node_id(node)
            positions[node] = (max_p-j, dimensions-i)
    nx.draw(graph, positions, with_labels=True, font_size=10, node_size=1000)
    rounded_labels = dict([(k, round(v, 2)) for k, v in nx.get_edge_attributes(graph, 'weight').items()])
    nx.draw_networkx_edge_labels(
        graph, positions, edge_labels=rounded_labels,
        font_size=10, node_size=1000, label_pos=0.75
    )
    plt.show()


class CausalTSGenerator:

    def __init__(self, dimensions, max_p, data_length=1000, graph_density=0.1,
                 random_state=None, autocorrelation=True, coupling_coeff=None,
                 draw_coupling=None, noise_std=1):
        self.dimensions = dimensions
        self.max_p = max_p
        self.length = max_p + 1
        self.data_length = data_length
        self.graph_density = graph_density
        self.autocorrelation = autocorrelation
        self.noise_std = noise_std
        self.coupling_coeff = coupling_coeff

        if coupling_coeff is not None:
            self.draw_coupling = (lambda: np.random.choice([-coupling_coeff, +coupling_coeff]))
        elif draw_coupling is not None:
            self.draw_coupling = draw_coupling
        else:
            raise Exception('Either coupling_coeff or min_coupling and max_coupling has to be set')

        min_edges = 1
        max_edges = dimensions * (dimensions - (0 if self.autocorrelation else 1)) * max_p
        self.edge_count = round(graph_density * dimensions * dimensions * max_p)
        if self.edge_count > max_edges or self.edge_count < min_edges:
            raise Exception(f'Edge count {self.edge_count} not in valid interval of ({min_edges}, {max_edges})')

        self.graph = None
        self.ts = None

        np.random.seed(random_state)
        seed(random_state)

    def generate(self):
        if self.graph is None:
            self.generate_stable_graph()

        start_sample = np.pad(
            np.random.normal(scale=1e-8, size=(self.dimensions, 1)),
            [(0, 0), (self.max_p-1, 0)], 'constant'
        )
        X = start_sample

        for _ in range(self.data_length):
            X_t = np.sum([np.dot(self.VAR_exog[-i], X[:, -i]) for i in range(1, self.max_p+1)], axis=0) + \
                  np.random.normal(scale=self.noise_std)
            X = np.append(X, np.expand_dims(X_t, axis=1), axis=1)

        X = X[:, self.max_p:]
        self.ts = pd.DataFrame(X.T, columns=[f'X{i}' for i in range(self.dimensions)])
        return self.ts

    def generate_stable_graph(self):
        stable_var = False
        while not stable_var:
            print('Not stable')
            VAR_exog = self._generate_fully_random_graph()
            stable_var = is_stable(VAR_exog[::-1])
        self.VAR_exog = VAR_exog

    # Deprecated
    def _generate_graph(self):
        c = self.coupling_coeff

        self.graph = nx.DiGraph()
        node_ids = list(product([d for d in range(self.dimensions)],
                                [l for l in reversed(range(self.max_p+1))]))
        self.graph.add_nodes_from([node_name(*n) for n in node_ids])

        # Ensure max_p is utilized
        d_t = np.random.choice(self.dimensions)
        d_p = np.random.choice([d for d in range(self.dimensions) if d != d_t])

        for d in range(self.dimensions):
            # Generate random edges to previous nodes
            candidates = [n for n in node_ids if n[1] > 0 and n[0] != d]
            remaining_edges = self.incoming_edges

            if d == d_t:
                remaining_edges -= 1
                candidates.remove((d_p, self.max_p))
                weight = np.random.choice([-c, c])
                # np.random.multivariate_normal(
                # mean=[-0.75, 0.75], cov=0.1*np.ones((2, 2)))[random.randint(0, 1)]
                self.graph.add_edge(node_name(d_p, self.max_p), node_name(d, 0),
                                    weight=weight)

            # autocorrelation
            if self.autocorrelation:
                # candidates.remove((d, 1))
                weight = np.random.choice([-c, c])
                # np.random.multivariate_normal(
                # mean=[-0.75, 0.75], cov=0.1*np.ones((2, 2)))[random.randint(0, 1)]
                self.graph.add_edge(node_name(d, 1), node_name(d, 0),
                                    weight=weight)

            picks = sample(candidates, remaining_edges)
            for candidate in picks:
                weight = np.random.choice([-c, c])
                # np.random.multivariate_normal(
                # mean=[-0.75, 0.75], cov=0.1*np.ones((2, 2)))[random.randint(0, 1)]
                self.graph.add_edge(node_name(*candidate), node_name(d, 0), weight=weight)

        adjacency = np.array(nx.adjacency_matrix(self.graph).todense())
        VAR_exog = []
        target = [self.max_p + d * self.length for d in range(self.dimensions)]
        for i in range(self.max_p):
            source = [i + d * self.length for d in range(self.dimensions)]
            VAR_exog.append(adjacency[:, target][source].T)
        return np.array(VAR_exog)

    # Deprecated
    def _generate_random_graph(self):
        c = self.coupling_coeff

        self.graph = nx.DiGraph()
        node_ids = list(product([d for d in range(self.dimensions)],
                                [l for l in reversed(range(self.max_p+1))]))
        self.graph.add_nodes_from([node_name(*n) for n in node_ids])

        candidates = list(product(permutations(np.arange(self.dimensions), 2), np.arange(self.max_p) + 1))
        # Ensure max_p is utilized
        d_t = np.random.choice(self.dimensions)
        d_p = np.random.choice([d for d in range(self.dimensions) if d != d_t])
        candidates.remove(((d_p, d_t), self.max_p))
        self.graph.add_edge(node_name(d_p, self.max_p), node_name(d_t, 0), weight=np.random.choice([-c, c]))

        edge_idxs = np.random.choice(len(candidates), size=(self.incoming_edges*self.dimensions - 1,), replace=False)
        for edge_idx in edge_idxs:
            (from_d, to_d), tau = candidates[edge_idx]
            self.graph.add_edge(node_name(from_d, tau), node_name(to_d, 0), weight=np.random.choice([-c, c]))

        if self.autocorrelation:
            for d in range(self.dimensions):
                weight = np.random.choice([-c, c])
                self.graph.add_edge(node_name(d, 1), node_name(d, 0), weight=weight)

        adjacency = np.array(nx.adjacency_matrix(self.graph).todense())
        VAR_exog = []
        target = [self.max_p + d * self.length for d in range(self.dimensions)]
        for i in range(self.max_p):
            source = [i + d * self.length for d in range(self.dimensions)]
            VAR_exog.append(adjacency[:, target][source].T)
        return np.array(VAR_exog)

    def _generate_dense_random_graph(self):
        c = self.coupling_coeff

        self.graph = nx.DiGraph()
        node_ids = list(product([d for d in range(self.dimensions)],
                                [l for l in reversed(range(self.max_p+1))]))
        self.graph.add_nodes_from([node_name(*n) for n in node_ids])

        candidates = list(product(permutations(np.arange(self.dimensions), 2), np.arange(self.max_p) + 1))

        cur_edge_count = 0

        if self.autocorrelation:
            for d in range(self.dimensions):
                weight = np.random.choice([-c, c])
                time_lag = np.random.choice(self.max_p) + 1
                self.graph.add_edge(node_name(d, time_lag), node_name(d, 0), weight=weight)
                cur_edge_count += 1

        # Ensure max_p is utilized
        if cur_edge_count < self.edge_count:
            d_t = np.random.choice(self.dimensions)
            d_p = np.random.choice([d for d in range(self.dimensions) if d != d_t])
            candidates.remove(((d_p, d_t), self.max_p))
            cur_edge_count += 1
            self.graph.add_edge(node_name(d_p, self.max_p), node_name(d_t, 0), weight=np.random.choice([-c, c]))

        edge_idxs = np.random.choice(len(candidates), size=(self.edge_count - cur_edge_count,), replace=False)
        for edge_idx in edge_idxs:
            (from_d, to_d), tau = candidates[edge_idx]
            self.graph.add_edge(node_name(from_d, tau), node_name(to_d, 0), weight=np.random.choice([-c, c]))
            cur_edge_count += 1
        assert cur_edge_count == self.edge_count

        adjacency = np.array(nx.adjacency_matrix(self.graph).todense())
        VAR_exog = []
        target = [self.max_p + d * self.length for d in range(self.dimensions)]
        for i in range(self.max_p):
            source = [i + d * self.length for d in range(self.dimensions)]
            VAR_exog.append(adjacency[:, target][source].T)
        return np.array(VAR_exog)

    # Ignores autocorrelation
    def _generate_fully_random_graph(self):
        self.graph = nx.DiGraph()
        node_ids = list(product([d for d in range(self.dimensions)],
                                [l for l in reversed(range(self.max_p+1))]))
        self.graph.add_nodes_from([node_name(*n) for n in node_ids])

        dim_prod = product(np.arange(self.dimensions), np.arange(self.dimensions)) if self.autocorrelation \
            else permutations(np.arange(self.dimensions), 2)
        candidates = list(product(dim_prod, np.arange(self.max_p) + 1))
        cur_edge_count = 0

        # Ensure max_p is utilized
        d_t, d_p = np.random.choice(self.dimensions, size=2)
        candidates.remove(((d_p, d_t), self.max_p))
        self.graph.add_edge(node_name(d_p, self.max_p), node_name(d_t, 0), weight=self.draw_coupling())
        cur_edge_count += 1

        # guarantee parent for each dim
        for to_d in range(self.dimensions):
            if to_d != d_t and len(candidates) > 0 and cur_edge_count < self.edge_count:
                from_d, tau = np.random.choice(self.dimensions), np.random.choice(self.max_p + 1)
                self.graph.add_edge(node_name(from_d, tau), node_name(to_d, 0), weight=self.draw_coupling())
                cur_edge_count += 1

        if len(candidates) > 0 and cur_edge_count < self.edge_count:
            edge_idxs = np.random.choice(len(candidates), size=(self.edge_count - cur_edge_count,), replace=False)
            for edge_idx in edge_idxs:
                (from_d, to_d), tau = candidates[edge_idx]
                self.graph.add_edge(node_name(from_d, tau), node_name(to_d, 0), weight=self.draw_coupling())
                cur_edge_count += 1

        adjacency = np.array(nx.adjacency_matrix(self.graph).todense())
        VAR_exog = []
        target = [self.max_p + d * self.length for d in range(self.dimensions)]
        for i in range(self.max_p):
            source = [i + d * self.length for d in range(self.dimensions)]
            VAR_exog.append(adjacency[:, target][source].T)
        return np.array(VAR_exog)

    def draw_graph(self):
        draw_graph(self.graph, self.dimensions, self.max_p)

    def fulltime_graph(self, fulltime_length=None):
        if fulltime_length is None:
            fulltime_length = 2 * self.max_p

        fulltime_graph = nx.DiGraph()
        node_ids = list(product([d for d in range(self.dimensions)],
                                [l for l in reversed(range(fulltime_length + 1))]))
        fulltime_graph.add_nodes_from([node_name(*n) for n in node_ids])
        for from_node, to_node, edge_data in self.graph.edges(data=True):
            dim_from, tau = node_id(from_node)
            dim_to, _ = node_id(to_node)
            for i in range(fulltime_length - tau + 1):
                fulltime_graph.add_edge(node_name(dim_from, tau + i), node_name(dim_to, i),
                                        weight=edge_data['weight'])
        return fulltime_graph
