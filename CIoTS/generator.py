from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import is_stable


def node_name(time_series, l):
        return f'X{time_series}_t-{l}' if l > 0 else f'X{time_series}_t'


class CausalTSGenerator:

    def __init__(self, dimensions, max_p, data_length=1000, incoming_edges=4, random_state=None):
        self.dimensions = dimensions
        self.max_p = max_p
        self.length = max_p + 1
        self.data_length = data_length
        self.incoming_edges = incoming_edges

        np.random.seed(random_state)

    def generate(self):
        stable_var = False
        while not stable_var:
            VAR_exog = self._generate_graph()
            stable_var = is_stable(VAR_exog[::-1])
        self.VAR_exog = VAR_exog

        start_sample = np.pad(
            np.random.normal(scale=1e-8, size=(self.dimensions, 1)),
            [(0, 0), (self.max_p-1, 0)], 'constant'
        )
        X = start_sample  # TODO: Find better method to initialize

        for _ in range(self.data_length):
            X_t = np.sum([np.dot(VAR_exog[-i], X[:, -i]) for i in range(1, self.max_p+1)], axis=0) + \
                  np.random.normal()
            X = np.append(X, np.expand_dims(X_t, axis=1), axis=1)

        X = X[:, self.max_p:]
        return pd.DataFrame(X.T, columns=[f'X{i}' for i in range(self.dimensions)])

    def _generate_graph(self):
        self.graph = nx.DiGraph()
        node_ids = np.array([[(d, l) for l in range(self.max_p, -1, -1)] for d in range(self.dimensions)])
        self.graph.add_nodes_from([node_name(d, l) for d, l in
                                   np.reshape(node_ids, (self.dimensions * self.length, 2))])

        # Ensure max_p is utilized
        d_t = np.random.choice(self.dimensions)
        d_p = np.random.choice(self.dimensions)
        self.graph.add_edge(node_name(d_p, self.max_p), node_name(d_t, 0))

        for d in range(self.dimensions):
            # Generate random edges to previous nodes
            candidates = np.delete(np.reshape(node_ids[:, :-1], (-1, 2)), [d_p * self.max_p], axis=0)
            remaining_edges = self.incoming_edges - (1 if d_t == d else 0)
            picks = candidates[np.random.choice(candidates.shape[0], remaining_edges, replace=False)]
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
        positions = []
        for i, j in product(range(self.dimensions), range(self.length)):
            positions.append((node_name(i, j), (self.max_p-j, self.dimensions-i)))
        nx.draw(self.graph, pos=dict(positions), with_labels=True)
        plt.show()
