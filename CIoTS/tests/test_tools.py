import unittest

import numpy as np
import pandas as pd

from CIoTS.tools import transform_ts


class TestTool(unittest.TestCase):

    def test_transform_test(self):
        p = 3
        df = pd.DataFrame({'a': np.arange(100), 'b': np.arange(100, 200)})
        expected_mapping = {0: 'a_t', 1: 'b_t', 2: 'a_t-1', 3: 'b_t-1',
                            4: 'a_t-2', 5: 'b_t-2', 6: 'a_t-3', 7: 'b_t-3'}
        expected_matrix = np.array([np.arange(3, 100), np.arange(103, 200),
                                    np.arange(2, 99), np.arange(102, 199),
                                    np.arange(1, 98), np.arange(101, 198),
                                    np.arange(0, 97), np.arange(100, 197)]).T
        result_mapping, result_matrix = transform_ts(df, p)
        self.assertDictEqual(expected_mapping, result_mapping)
        self.assertTrue(np.all(expected_matrix == result_matrix))
