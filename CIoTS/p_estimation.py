import numpy as np
from CIoTS.simple_var import VAR
from CIoTS.stat_tools import cross_correlation


def var_order_select(ts, max_p=20, ics=["aic", "bic", "hqic"], drop=True):
    ic_scores = {ic: [] for ic in ics}
    p_rankings = {ic: np.arange(1, max_p+1) for ic in ics}

    for p in range(1, max_p+1):
        model = VAR(p)
        model.fit(ts)

        offset = max_p - p
        for ic in ics:
            ic_scores[ic].append(model.information_criterion(ic,
                                                             offset=offset))

    for ic in ics:
        ic_scores[ic] = np.array(ic_scores[ic])
        if drop:
            valid = ~np.isinf(ic_scores[ic])
            p_rankings[ic] = p_rankings[ic][valid]
            ic_scores[ic] = ic_scores[ic][valid]
        ordering = np.argsort(ic_scores[ic])
        p_rankings[ic] = p_rankings[ic][ordering]
        ic_scores[ic] = ic_scores[ic][ordering]
    return p_rankings, ic_scores


def cross_corr_peaks(ts, include_autocorr=True, n_peaks=1):
    corr_matrix = cross_correlation(ts, include_autocorr=True)
    corr_matrix = corr_matrix[::, 1:]
    peak_matrix = np.fabs(corr_matrix).argsort(axis=1)[::, ::-1] + 1
    peaks = peak_matrix[::, 0]
    # get next maximum (higher lag with max cross corr)
    for c in range(len(peaks)):
        current_peak = 1
        current_index = 1
        while current_peak < n_peaks and current_index < len(ts):
            if peak_matrix[c, current_index] > peaks[c]:
                current_peak += 1
                peaks[c] = peak_matrix[c, current_index]
            current_index += 1
    return peaks.max()
