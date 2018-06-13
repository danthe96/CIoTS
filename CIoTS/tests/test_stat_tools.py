import unittest

import numpy as np

from CIoTS.stat_tools import partial_corr


class TestStatTools(unittest.TestCase):

    def test_partial_corr(self):
        # example from wikipedia
        x = np.array([2, 4, 15, 20])
        y = np.array([1, 2, 3, 4])
        z = np.array([0, 0, 1, 1])
        data = np.array([x, y, z]).T
        corr = np.corrcoef(data, rowvar=False)

        corr_xy = 0.97
        corr_xy_z = 0.92
        result_xy = partial_corr(0, 1, {}, corr)
        result_xy_z = partial_corr(0, 1, {2}, corr)
        self.assertAlmostEqual(corr_xy, result_xy, delta=0.01)
        self.assertAlmostEqual(corr_xy_z, result_xy_z, delta=0.01)
