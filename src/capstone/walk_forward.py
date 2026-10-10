"""Rolling walk-forward evaluation, run separately per ticker."""
from typing import Callable

import pandas as pd

from .features import feature_columns


def make_folds(dates: pd.DatetimeIndex, first_test: int, last_test: int, train_years: int):
    """Yield (test_year, train_mask, test_mask) over a date index (boolean numpy arrays)."""
    yr = dates.year
    for y in range(first_test, last_test + 1):
        train = (yr >= y - train_years) & (yr < y)
        test = yr == y
        idx = train.nonzero()[0]
        train[idx[-1]] = False  # purge: last train label uses the first test-year close
        yield y, train, test


def run_walk_forward(ds: pd.DataFrame, model_factories: dict[str, Callable], cfg: dict,
                     tickers: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Return a long frame of out-of-sample predictions.

    Each factory builds a fresh model with `.fit(X, y)` and `.predict_score(X) -> P(up)`-like
    score (higher = more likely up) for every fold/ticker, so no state leaks across folds.
    """
    wf = cfg["walk_forward"]
    feats = feature_columns(ds)
    rows = []
    for t in tickers or sorted(ds.index.get_level_values("ticker").unique()):
        d = ds.xs(t, level="ticker")
        for year, tr, te in make_folds(d.index, wf["first_test_year"], wf["last_test_year"], wf["train_years"]):
            Xtr, ytr, Xte = d.loc[tr, feats], d.loc[tr, "y"], d.loc[te, feats]
            for name, factory in model_factories.items():
                m = factory()
                m.fit(Xtr, ytr)
                score = m.predict_score(Xte)
                rows.append(pd.DataFrame({
                    "model": name, "ticker": t, "test_year": year, "date": Xte.index,
                    "y_true": d.loc[te, "y"].values, "ret_next": d.loc[te, "ret_next"].values,
                    "score": score, "pred": (score >= m.threshold).astype(int),
                }))
        if verbose:
            print(f"done {t}")
    return pd.concat(rows, ignore_index=True)
