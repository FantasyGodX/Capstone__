import numpy as np
import pandas as pd

from capstone.models import ARIMAModel
from capstone.walk_forward import make_folds


def test_folds_are_chronological_and_purged():
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    folds = list(make_folds(dates, 2020, 2024, 5))
    assert [f[0] for f in folds] == [2020, 2021, 2022, 2023, 2024]
    for year, tr, te in folds:
        assert dates[tr].max() < dates[te].min()
        assert dates[te].year.unique().tolist() == [year]
        assert dates[tr].min().year >= year - 5
        # purge: the final pre-test trading day is excluded from training
        assert dates[tr].max() < dates[dates.year == year - 1].max()


def test_arima_forecast_uses_only_past_returns():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    X = pd.DataFrame({"ret_lag1": rng.normal(0, 0.01, 400)}, index=idx)
    m = ARIMAModel().fit(X.iloc[:300], None)
    base = m.predict_score(X.iloc[300:])
    X2 = X.copy()
    X2.iloc[350:, 0] = 0.5  # perturb the future: earlier forecasts must not move
    pert = m.predict_score(X2.iloc[300:])
    np.testing.assert_allclose(base[:50], pert[:50])
    assert len(base) == 100
