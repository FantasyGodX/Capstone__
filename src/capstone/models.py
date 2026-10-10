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


class RandomForestModel:
    """Shallow, heavily-regularised forest (min_samples_leaf large) - financial data is very noisy."""
    threshold = 0.5

    def __init__(self, seed: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self.m = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20,
                                        max_features="sqrt", n_jobs=-1, random_state=seed)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_score(self, X):
        return self.m.predict_proba(X)[:, 1]


class XGBoostModel:
    threshold = 0.5

    def __init__(self, seed: int = 42):
        from xgboost import XGBClassifier
        self.m = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8,
                               colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0,
                               eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_score(self, X):
        return self.m.predict_proba(X)[:, 1]


class LSTMModel:
    """LSTM over a rolling window of the last `seq_len` days of standardised features.

    - scaler fitted on the training window only
    - chronological 15% tail of the training window used for early stopping
    - the last seq_len-1 training rows are kept so the first test windows are complete
      (past features only, no leakage)
    """
    threshold = 0.5

    def __init__(self, seq_len: int = 20, hidden: int = 32, dropout: float = 0.3, lr: float = 1e-3,
                 max_epochs: int = 40, patience: int = 5, batch_size: int = 64, seed: int = 42):
        self.seq_len, self.hidden, self.dropout, self.lr = seq_len, hidden, dropout, lr
        self.max_epochs, self.patience, self.batch_size, self.seed = max_epochs, patience, batch_size, seed

    def _windows(self, Z: np.ndarray) -> np.ndarray:
        L = self.seq_len
        return np.stack([Z[i - L + 1: i + 1] for i in range(L - 1, len(Z))])

    def _build(self, n_feat: int):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s, n_in, hidden, dropout):
                super().__init__()
                s.lstm = nn.LSTM(n_in, hidden, batch_first=True)
                s.drop = nn.Dropout(dropout)
                s.out = nn.Linear(hidden, 1)

            def forward(s, x):
                h, _ = s.lstm(x)
                return s.out(s.drop(h[:, -1])).squeeze(-1)

        return Net(n_feat, self.hidden, self.dropout)

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.scaler = StandardScaler().fit(X)
        Z = np.clip(self.scaler.transform(X), -5, 5).astype(np.float32)
        self.tail = Z[-(self.seq_len - 1):]
        W = torch.from_numpy(self._windows(Z))
        t = torch.from_numpy(np.asarray(y)[self.seq_len - 1:].astype(np.float32))
        n_val = int(0.15 * len(W))
        Wtr, ttr, Wva, tva = W[:-n_val], t[:-n_val], W[-n_val:], t[-n_val:]

        self.net = self._build(Z.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(len(Wtr), generator=g)  # shuffling windows inside the train set only
            for i in range(0, len(perm), self.batch_size):
                b = perm[i: i + self.batch_size]
                opt.zero_grad()
                loss_fn(self.net(Wtr[b]), ttr[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = loss_fn(self.net(Wva), tva).item()
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: p.clone() for k, p in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict_score(self, X):
        import torch
        Z = np.clip(self.scaler.transform(X), -5, 5).astype(np.float32)
        W = torch.from_numpy(self._windows(np.vstack([self.tail, Z])))
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(W)).numpy()
