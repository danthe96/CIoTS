from abc import ABC, abstractmethod
import networkx as nx
import numpy as np


from CIoTS.simple_var import VAR


class Stopper(ABC):

    def __init__(self):
        self.best_tau = 0

    @abstractmethod
    def scores(self):
        pass

    @abstractmethod
    def check_stop(self):
        pass


class ICStopper():

    def __init__(self, dim, patiency=1, ic='bic'):
        super().__init__()
        self.dim = dim
        self.patiency = patiency
        self.ic = ic

        self.bics = {}
        self.no_imp = 0
        self.best_bic = np.inf

    def scores(self):
        return self.bics

    def check_stop(self, graph, tau, data_matrix):
        self.bics[tau] = self._graph_ic(graph, tau, data_matrix)

        if self.bics[tau] < self.best_bic:
            self.best_bic = self.bics[tau]
            self.best_tau = tau
            self.no_imp = 0
        else:
            self.no_imp += 1
            if self.no_imp >= self.patiency:
                return True
        return False

    def _graph_ic(self, graph, tau, data_matrix):
        free_params = len(graph.edges()) + len(graph.nodes())
        model = VAR(tau)
        model.fit_from_graph(self.dim, data_matrix, graph)
        return model.information_criterion(self.ic, free_params=free_params)


class CorrStopper():

    def __init__(self, dim, patiency=1, max_tau=10):
        super().__init__()
        self.dim = dim
        self.patiency = patiency

        self.corr_tests = {}
        self.no_imp = 0
        self.best_tau = max_tau

    def scores(self):
        return self.corr_tests

    def check_stop(self, graph, tau, data_matrix):
        # if corr_test is not set, returns 0/1 adj matrix
        # can be used for more complex decisions
        adj_matrix = nx.to_numpy_matrix(graph, weight='corr_test')
        self.corr_tests[tau] = np.sum(adj_matrix[:-(self.dim+1):-1, :self.dim])

        if self.corr_tests[tau] > 0:
            self.no_imp = 0
        else:
            self.no_imp += 1
            if self.no_imp >= self.patiency:
                self.best_tau = tau
                return True
        return False
