from itertools import product

import networkx as nx
import numpy as np

from .tools import transform_ts
from .generator import node_name


class VAR():

    def __init__(self, p):
        self.p = p
        self.is_fitted = False

    def fit(self, ts_data):
        _, data_matrix = transform_ts(ts_data, self.p)
        self.dim = ts_data.shape[1]
        self.length = data_matrix.shape[0]

        y = data_matrix[:, :self.dim]
        X = data_matrix[:, self.dim:]
        X_ = np.insert(X, 0, 1, axis=1)

        self.params = np.linalg.lstsq(X_, y, rcond=1e-15)[0]
        self.free_params = self.params.size
        self.residuals = y - np.dot(X_, self.params)
        self.sse = np.dot(self.residuals.T, self.residuals)
        # self.sigma_u = self.sse/(self.length - self.dim*self.p - 1)
        self.sigma_u = self.sse/self.length
        self.is_fitted = True

    def fit_from_graph(self, dim, data_matrix, graph, mapping=None):
        self.dim = dim
        self.length = data_matrix.shape[0]
        self.inputs = []
        self.params = np.zeros((self.dim, self.dim * self.p + 1))
        self.free_params = 0

        if mapping is None:
            mapping = {n: n for n in graph.nodes()}
        inverted_mapping = {v: k for k, v in mapping.items()}

        for x_t in range(self.dim):
            input_nodes = list(graph.predecessors(mapping[x_t]))

            inputs = np.array([inverted_mapping[x] for x in input_nodes])
            y = data_matrix[:, x_t]
            if input_nodes:
                X = data_matrix[:, inputs]
                positions = np.insert(inputs - self.dim + 1, 0, 0)
            else:
                X = np.array([[]]*len(data_matrix))
                positions = np.array([0])
            X_ = np.insert(X, 0, 1, axis=1)
            params = np.linalg.lstsq(X_, y, rcond=1e-15)[0]

            self.params[x_t, positions] = params
            self.free_params += params.size

        self.params = self.params.T
        y = data_matrix[:, :self.dim]
        X = data_matrix[:, self.dim: self.dim*(self.p+1)]
        X_ = np.insert(X, 0, 1, axis=1)
        self.residuals = y - np.dot(X_, self.params)
        self.sse = np.dot(self.residuals.T, self.residuals)
        # not sure whether this is ok
        self.sigma_u = self.sse/self.length
        self.is_fitted = True

    def to_graph(self, threshold=0.1):
        end_nodes = ['X{}_t'.format(i) for i in range(self.dim)]
        start_nodes = ['X{}_t-{}'.format(j, i) for i, j in product(range(self.p, 0, -1), range(self.dim))]

        A = self.params[1:]
        assert A.shape == (len(start_nodes), len(end_nodes))

        estimated_graph = nx.DiGraph()
        node_ids = np.array([[(d, l) for l in range(self.p, -1, -1)] for d in range(self.dim)])
        estimated_graph.add_nodes_from([node_name(d, l) for d, l in
                                       np.reshape(node_ids, (self.dim * (self.p+1), 2))])

        for i in range(len(start_nodes)):
            for j in range(len(end_nodes)):
                if np.abs(A[i][j]) > threshold:
                    estimated_graph.add_edge(start_nodes[i], end_nodes[j], weight=A[i][j])
        return estimated_graph

    def information_criterion(self, ic, offset=0, free_params=None):
        if not self.is_fitted:
            raise Exception('model is not fitted')
        ll = self._log_likelihood()
        if free_params is None:
            free_params = self.free_params
        nobs = self.length-offset

        if ic == 'bic':
            return self._bic(ll, free_params, nobs)
        elif ic == 'aic':
            return self._aic(ll, free_params, nobs)
        elif ic == 'hqic':
            return self._hqic(ll, free_params, nobs)
        else:
            raise Exception('unknown information criterion')

    def _bic(self, ll, free_params, nobs):
        return ll + (np.log(nobs) / nobs) * free_params

    def _aic(self, ll, free_params, nobs):
        return ll + (2. / nobs) * free_params

    def _hqic(self, ll, free_params, nobs):
        return ll + (2. * np.log(np.log(nobs)) / nobs) * free_params

    def _log_likelihood(self):
        _, logdet = np.linalg.slogdet(self.sigma_u)
        return logdet
