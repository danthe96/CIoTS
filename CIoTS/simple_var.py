import numpy as np
from CIoTS.tools import transform_ts


class VAR():

    def __init__(self, p):
        self.p = p
        self.is_fitted = False

    def fit(self, ts_data):
        _, data_matrix = transform_ts(ts_data, self.p)
        self.dim = ts_data.shape[1]
        self.length = data_matrix.shape[0]

        y = data_matrix[:, :self.dim]
        X = data_matrix[:, self.dim:]
        X_ = np.insert(X, 0, 1, axis=1)

        self.params = np.linalg.lstsq(X_, y, rcond=1e-15)[0]
        self.residuals = y - np.dot(X_, self.params)
        self.sse = np.dot(self.residuals.T, self.residuals)
        self.sigma_u = self.sse/(self.length - self.dim*self.p - 1)
        self.is_fitted = True

    def information_criterion(self, ic, offset=0, free_params=None):
        if not self.is_fitted:
            raise Exception('model is not fitted')
        ll = self._log_likelihood()
        if free_params is None:
            free_params = self._free_params()
        nobs = self.length-offset

        if ic == 'bic':
            return self._bic(ll, free_params, nobs)
        elif ic == 'aic':
            return self._aic(ll, free_params, nobs)
        elif ic == 'hqic':
            return self._hqic(ll, free_params, nobs)
        else:
            raise Exception('unknown information criterion')

    def _bic(self, ll, free_params, nobs):
        return ll + (np.log(nobs) / nobs) * free_params

    def _aic(self, ll, free_params, nobs):
        return ll + (2. / nobs) * free_params

    def _hqic(self, ll, free_params, nobs):
        return ll + (2. * np.log(np.log(nobs)) / nobs) * free_params

    def _log_likelihood(self):
        _, logdet = np.linalg.slogdet(self.sigma_u)
        return logdet

    def _free_params(self):
        return self.p * self.dim**2 + self.dim
