import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import is_stable
from random import sample, seed
from itertools import product


def node_name(time_series, l):
    return f'X{time_series}_t-{l}' if l > 0 else f'X{time_series}_t'


def node_id(name):
    regex = re.search(r'X(\d+)_t(-(\d+))?', name)
    return int(regex.group(1)), int(regex.group(3) or 0)


def draw_graph(graph, dimensions, max_p):
    positions = {}
    for node in graph.nodes():
        i, j = node_id(node)
        positions[node] = (max_p-j, dimensions-i)
    nx.draw(graph, pos=positions, with_labels=True, font_size=10, node_size=1000)
    plt.show()


class CausalTSGenerator:

    def __init__(self, dimensions, max_p, data_length=1000, incoming_edges=4,
                 random_state=None, autocorrelation=0):
        self.dimensions = dimensions
        self.max_p = max_p
        self.length = max_p + 1
        self.data_length = data_length
        self.incoming_edges = incoming_edges
        self.graph = None
        self.autocorrelation = autocorrelation

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
                  np.random.normal()
            X = np.append(X, np.expand_dims(X_t, axis=1), axis=1)

        X = X[:, self.max_p:]
        return pd.DataFrame(X.T, columns=[f'X{i}' for i in range(self.dimensions)])

    def generate_stable_graph(self):
        stable_var = False
        while not stable_var:
            VAR_exog = self._generate_graph()
            stable_var = is_stable(VAR_exog[::-1])
        self.VAR_exog = VAR_exog

    def _generate_graph(self):
        self.graph = nx.DiGraph()
        node_ids = list(product([d for d in range(self.dimensions)],
                                [l for l in reversed(range(self.max_p+1))]))
        self.graph.add_nodes_from([node_name(*n) for n in node_ids])

        # Ensure max_p is utilized
        d_t = np.random.choice(self.dimensions)
        d_p = np.random.choice(self.dimensions)

        for d in range(self.dimensions):
            # Generate random edges to previous nodes
            candidates = [n for n in node_ids if n[1] > 0]
            remaining_edges = self.incoming_edges

            if d == d_t:
                remaining_edges -= 1
                candidates.remove((d_p, self.max_p))
                self.graph.add_edge(node_name(d_p, self.max_p), node_name(d, 0),
                                    weight=np.random.normal())

            # autocorrelation
            if self.autocorrelation != 0:
                candidates.remove((d, 1))
                self.graph.add_edge(node_name(d, 1), node_name(d, 0),
                                    weight=self.autocorrelation)

            picks = sample(candidates, remaining_edges)
            for candidate in picks:
                self.graph.add_edge(node_name(*candidate), node_name(d, 0), weight=np.random.normal())

        adjacency = np.array(nx.adjacency_matrix(self.graph).todense())
        VAR_exog = []
        target = [self.max_p + d * self.length for d in range(self.dimensions)]
        for i in range(self.max_p):
            source = [i + d * self.length for d in range(self.dimensions)]
            VAR_exog.append(adjacency[:, target][source].T)
        return np.array(VAR_exog)

    def draw_graph(self):
        draw_graph(self.graph, self.dimensions, self.max_p)
