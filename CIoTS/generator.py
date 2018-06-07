import networkx as nx
import numpy as np
import pandas as pd


def node_name(time_series, l):
        return f'X{time_series}_t-{l}' if l > 0 else f'X{time_series}_t'


class CausalTSGenerator:

    def __init__(self, dimensions, max_p, length=1000, incoming_edges=4, random_state=None):
        self.dimensions = dimensions
        self.max_p = max_p
        self.length = length
        self.incoming_edges = incoming_edges

        np.random.seed(random_state)

    def generate(self):
        self.graph = nx.DiGraph()
        node_ids = np.array([[(d, l) for l in range(self.max_p - 1, -1, -1)] for d in range(self.dimensions)])
        self.graph.add_nodes_from([node_name(d, l) for d, l in
                                   np.reshape(node_ids, (self.dimensions * self.max_p, 2))])

        # Sequential edges
        for i in range(self.dimensions):
            for j in range(self.max_p - 1, 0, -1):
                self.graph.add_edge(node_name(i, j), node_name(i, j-1), weight=0.3)

        for d in range(self.dimensions):
            # Generate random edges to previous nodes
            candidates = np.delete(np.reshape(node_ids[:, :-1], (-1, 2)), [(d + 1) * (self.max_p - 1) - 1], axis=0)
            picks = candidates[np.random.choice(candidates.shape[0], self.incoming_edges, replace=False)]
            for candidate in picks:
                self.graph.add_edge(node_name(*candidate), node_name(d, 0), weight=0.6)

        # Assume we have a covariance matrix
        cov = None
        start_sample = np.random.multivariate_normal(np.random.rand(cov.shape[0]), cov)
        X = np.reshape(start_sample, (self.dimensions, self.max_p))

        for _ in range(self.length):
            X_t = []
            for d in range(self.dimensions):
                covar = np.reshape(cov[(self.max_p - 1) + d * self.max_p], (self.dimensions, self.max_p))
                X_t.append([np.sum(start_sample[:, -self.max_p + 1:] * covar[:, :-1])])
            X = np.append(X, X_t, axis=1)

        X = X[:, self.max_p:]
        return pd.DataFrame(X.T, columns=[f'X{i}' for i in range(self.dimensions)])
