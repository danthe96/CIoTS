from math import sqrt, log
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import pinv
from scipy.signal import correlate
from itertools import combinations_with_replacement


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
    # clip r for numerical reasons
    if r >= 1:
        r = 1 - sys.float_info.epsilon
    elif r <= -1:
        r = sys.float_info.epsilon - 1
    # fisher transform
    z = sqrt(n - len(S) - 3) * (1 / 2) * log(1 + 2 * r / (1 - r))
    # p-test
    return 2 * norm.sf(abs(z)), abs(z)


def cross_correlation(ts, include_autocorr=True, return_df=False):
    dims = [(col1, col2) for col1, col2 in combinations_with_replacement(ts.columns, 2)
            if include_autocorr or col1 != col2]
    dims_str = []
    corrs = []
    for col1, col2 in dims:
        corr = correlate(ts[col1], ts[col2])
        corr /= corr[len(ts)-1]
        corrs.append(corr[len(ts)-1::-1])
        dims_str.append(f'{col1} vs {col2}')
        if col1 != col2:
            corrs.append(corr[len(ts)-1::])
            dims_str.append(f'{col2} vs {col1}')
    if return_df:
        return pd.DataFrame(np.array(corrs).T, columns=dims_str)
    else:
        return np.array(corrs)
