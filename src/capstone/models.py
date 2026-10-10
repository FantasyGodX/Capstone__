"""Model wrappers sharing one interface: fit(X, y), predict_score(X), threshold."""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class AlwaysUp:
    """Benchmark: predict 'up' every day (equivalent to buy-and-hold)."""
    threshold = 0.5

    def fit(self, X, y):
        return self

    def predict_score(self, X):
        return np.ones(len(X))


class LogisticModel:
    """Classical baseline. Scaler is fitted on the training window only."""
    threshold = 0.5

    def __init__(self, C: float = 0.1, seed: int = 42):
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000, random_state=seed))

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def predict_score(self, X):
        return self.pipe.predict_proba(X)[:, 1]


class ARIMAModel:
    """ARIMA(p,0,q) on daily returns; order chosen by AIC on the training window.

    One-step-ahead forecasts through the test period use the fitted parameters with the
    filter rolled forward over observed returns (no re-estimation, no look-ahead).
    Score = forecast of next-day return; predicts 'up' if the forecast is > 0.
    """
    threshold = 0.0

    def __init__(self, max_p: int = 2, max_q: int = 2):
        self.orders = [(p, 0, q) for p in range(max_p + 1) for q in range(max_q + 1)]

    @staticmethod
    def _returns(X: pd.DataFrame) -> pd.Series:
        return X["ret_lag1"]  # today's return, known at close t

    def fit(self, X, y):
        from statsmodels.tsa.arima.model import ARIMA
        r = self._returns(X).reset_index(drop=True) * 100  # percent scale helps the optimiser
        best = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in self.orders:
                try:
                    res = ARIMA(r, order=order, trend="c").fit()
                except Exception:
                    continue
                if best is None or res.aic < best[0]:
                    best = (res.aic, order, res)
        self.order, self.res = best[1], best[2]
        return self

    def predict_score(self, X):
        r = self._returns(X).reset_index(drop=True) * 100
        # Append one placeholder so we can read the forecast made at the final test row.
        # A one-step prediction at position k only uses observations up to k-1.
        ext = pd.concat([r, pd.Series([0.0])], ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self.res.apply(ext, refit=False)
        pred = applied.predict(start=1, end=len(ext) - 1)  # position k+1 forecast made at row k
        return np.asarray(pred) / 100
