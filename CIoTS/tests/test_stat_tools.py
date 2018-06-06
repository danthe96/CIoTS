import numpy as np
import unittest
from CIoTS.stat_tools import partial_corr


class TestStatTools(unittest.TestCase):

    def test_partial_corr(self):
        # example from wikipedia
        x = np.array([5, 10, 15, 20])
        y = np.array([1, 2, 3, 4])
        z = np.array([0, 0, 1, 1])
        data = np.array([x, y, z]).T
        corr = np.corrcoef(data)
        corr_xy = 1
        corr_xy_z = 0.91
        result_xy = partial_corr(0, 1, set([]), corr)
        result_xy_z = partial_corr(0, 1, set([2]), corr)
        self.assertAlmostEqual(corr_xy, result_xy)
        self.assertAlmostEqual(corr_xy_z, result_xy_z)
