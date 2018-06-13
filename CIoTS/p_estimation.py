import numpy as np
from statsmodels.tsa.api import VAR


def var_order_select(ts, max_p=20, ics=["aic", "bic", "hqic", "fpe"]):
    data_matrix = ts.as_matrix()
    model = VAR(data_matrix)
    p_estimation = model.select_order(max_p)
    tested_p = np.arange(len(p_estimation.ics['bic']))
    # sort tested lags by score for each ic
    p_rankings = {ic: tested_p[np.argsort(scores)]
                  for ic, scores in p_estimation.ics.items()
                  if ic in ics}
    # sort scores according value
    ic_scores = {ic: np.sort(scores)
                 for ic, scores in p_estimation.ics.items()
                 if ic in ics}
    return p_rankings, ic_scores
