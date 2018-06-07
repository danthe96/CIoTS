import unittest

import numpy as np

from CIoTS.stat_tools import partial_corr


class TestStatTools(unittest.TestCase):

    def test_partial_corr(self):
        # example from wikipedia
        y = np.array([1, 2, 3, 4])
        x = np.array([5, 10, 15, 20])
        z = np.array([0, 0, 1, 1])
        data = np.array([x, y, z]).reshape(4, 3)
        corr = np.corrcoef(data, rowvar=False)
        corr_xy = 1
        corr_xy_z = 0.91
        result_xy = partial_corr(0, 1, set([]), corr)
        result_xy_z = partial_corr(0, 1, set([2]), corr)
        self.assertAlmostEqual(corr_xy, result_xy, delta=0.01)
        self.assertAlmostEqual(corr_xy_z, result_xy_z)
