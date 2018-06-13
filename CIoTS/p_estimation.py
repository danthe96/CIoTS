import numpy as np
from statsmodels.tsa.api import VAR


def _log_likelihood(var_res):
    _, logdet = np.linalg.slogdet(var_res.sigma_u)
    return logdet


def _free_parameters(var_res, p):
    return p * var_res.neqs**2 + var_res.neqs


def bic(ll, params, nobs):
    return ll + (np.log(nobs) / nobs) * params


def aic(ll, params, nobs): 
    return ll + (2. / nobs) * params


def hqic(ll, params, nobs):
    return ll + (2. * np.log(np.log(nobs)) / nobs) * params


def fpe(ll, nobs, model, resid, neqs):
    return ((nobs + model) / resid) ** neqs * np.exp(ll)


def var_order_select(ts, max_p=20, ics=["aic", "bic", "hqic", "fpe"], drop=True):
    data_matrix = ts.as_matrix()
    model = VAR(data_matrix)

    ic_scores = {ic: [] for ic in ics}
    p_rankings = {ic: np.arange(1, max_p+1) for ic in ics}

    for p in range(1, max_p+1):
        res = model.fit(p)
        ll = _log_likelihood(res)
        params = _free_parameters(res, p)
        nobs = res.nobs - (max_p - p)

        if "aic" in ics:
            ic_scores["aic"].append(aic(ll, params, nobs))
        if "bic" in ics:
            ic_scores["bic"].append(bic(ll, params, nobs))
        if "hqic" in ics:
            ic_scores["hqic"].append(hqic(ll, params, nobs))
        if "fpe" in ics:
            m = res.df_model
            resid = res.df_resid
            neqs = res.neqs
            ic_scores["fpe"].append(fpe(ll, nobs, m, resid, neqs))

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
