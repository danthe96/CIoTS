import numpy as np
import pandas as pd
from math import sqrt, log
from scipy.stats.norm import sf


def partial_corr(i, j, S, corr_matrix):
    S = list(S)
    if len(S) < 1:
        return corr_matrix[i, j]
    indices = [i, j]+S
    # pseudo inverse matrix
    p_matrix, _ = np.linalg.pinv(corr_matrix[indices, indices])
    return p_matrix[0, 1]/sqrt(p_matrix[0, 0]*p_matrix[1, 1])


def partial_corr_test(data_matrix, i, j, S, **kwargs):
    if 'corr_matrix' in kwargs:
        corr_matrix = kwargs['corr_matrix']
    else:
        corr_matrix = np.corrcoef(data_matrix.T)
    n = data_matrix.shape[0]
    r = partial_corr(i, j, S, corr_matrix)
    # fisher transform
    z = sqrt(n - 3)*(1/2)*log((1+r)/(1-r))
    # p-test
    return 2*sf(abs(z))
