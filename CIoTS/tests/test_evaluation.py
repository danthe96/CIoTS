import unittest

import networkx as nx

from CIoTS.evaluation import evaluate_edges


class TestEvaluation(unittest.TestCase):

    def test_evaluate_edges(self):
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([(1, 0), (2, 0), (3, 1)])

        pred_graph = nx.DiGraph()
        pred_graph.add_edges_from([(0, 1), (2, 0), (3, 1), (4, 0)])

        scores_d = {'accuracy': 0.85, 'f1-score': 0.5714285714285715,
                    'matthews_corrcoef': 0.4900980294098034,
                    'FDR': 0.5, 'FPR': 0.11764705882352941,
                    'TPR': 0.6666666666666666, 'precision': 0.5}
        result_d = evaluate_edges(true_graph, pred_graph, directed=True)
        self.assertDictEqual(result_d, scores_d)

        scores_u = {'accuracy': 0.9, 'f1-score': 0.8571428571428571,
                    'matthews_corrcoef': 0.8017837257372732,
                    'FDR': 0.25, 'FPR': 0.14285714285714285, 'TPR': 1.0,
                    'precision': 0.75}
        result_u = evaluate_edges(true_graph, pred_graph, directed=False)
        self.assertDictEqual(result_u, scores_u)
