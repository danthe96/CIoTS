import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def var_order_select(ts, max_p=20, ics=["aic", "bic", "hqic", "fpe"]):
    data_matrix = ts.as_matrix()
    model = VAR(data_matrix)
    p_estimation = model.select_order(max_p)
    tested_p = np.arange(1, max_p+1)
    # sort tested lags by score for each ic
    p_rankings = {ic: test_p[np.argsort(scores)] 
                  for ic, scores in p_estimation.ics.items()
                  if ic in ics}
    # sort scores according value
    ic_scores = {ic: np.sort(scores)
                 for ic, scores in p_estimation.ics.items()
                 if ic in ics}
    return p_rankings, ic_scores
