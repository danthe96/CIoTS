from math import sqrt, log

import numpy as np
from scipy.stats import norm
from scipy.linalg import pinv


def partial_corr(i, j, S, corr_matrix):
    S = list(S)
    if len(S) < 1:
        return corr_matrix[i, j]
    indices = [i, j]+S
    sub_corr_matrix = corr_matrix[indices, :][:, indices]
    # pseudo inverse matrix
    p_matrix = pinv(sub_corr_matrix)
    return -p_matrix[0, 1]/sqrt(p_matrix[0, 0]*p_matrix[1, 1])


def partial_corr_test(data_matrix, i, j, S, **kwargs):
    corr_matrix = kwargs.get('corr_matrix',
                             np.corrcoef(data_matrix, rowvar=False))
    S = list(S)
    n = data_matrix.shape[0]
    r = partial_corr(i, j, S, corr_matrix)
    # fisher transform
    z = sqrt(n - 3) * (1 / 2) * log((1 + r) / (1 - r))
    # p-test
    return 2 * norm.sf(abs(z))
