import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import networkx as nx

from CIoTS.pc_chen import pc_chen


class TestPcChen(unittest.TestCase):

    @patch('CIoTS.pc_chen.estimate_skeleton')
    def test_pc_chen(self, mock_estimate_skeleton):
        ts = pd.DataFrame({'a': np.arange(4), 'b': np.arange(4)})
        p = 1
        alpha = 0.005

        true_graph = nx.DiGraph()
        true_graph.add_edges_from([('b_0', 'a_0'), ('a_1', 'a_0'),
                                   ('b_1', 'b_0')])

        mocked_graph = nx.Graph()
        mocked_graph.add_edges_from([(0, 1), (0, 2), (1, 3)])
        mocked_sep_sets = [[set() for i in range(4)]
                           for j in range(4)]
        mocked_sep_sets[0][3] = set([1])
        mocked_sep_sets[3][0] = set([1])
        mock_estimate_skeleton.return_value = (mocked_graph, mocked_sep_sets)

        result_graph = pc_chen(None, ts, p, alpha)
        self.assertSetEqual(set(result_graph.nodes()),
                            set(true_graph.nodes()))
        self.assertSetEqual(set(result_graph.edges()),
                            set(true_graph.edges()))
