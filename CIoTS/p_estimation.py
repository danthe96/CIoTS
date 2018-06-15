import numpy as np
from CIoTS.simple_var import VAR


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
