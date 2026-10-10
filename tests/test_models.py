import numpy as np
import pandas as pd

from capstone.models import LogisticModel, RandomForestModel, XGBoostModel


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = (X["a"] + 0.5 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_models_learn_a_simple_signal_and_score_in_unit_interval():
    X, y = _toy()
    for m in (LogisticModel(), RandomForestModel(), XGBoostModel()):
        m.fit(X.iloc[:300], y.iloc[:300])
        s = m.predict_score(X.iloc[300:])
        assert s.shape == (100,) and ((s >= 0) & (s <= 1)).all()
        assert ((s >= 0.5) == y.iloc[300:]).mean() > 0.7


def test_lstm_output_alignment_and_signal():
    from capstone.models import LSTMModel
    X, y = _toy(n=500)
    m = LSTMModel(max_epochs=15, seed=0).fit(X.iloc[:350], y.iloc[:350])
    s = m.predict_score(X.iloc[350:])
    assert s.shape == (150,)  # one score per test row, even for the first rows of the test window
    assert ((s >= 0.5) == y.iloc[350:]).mean() > 0.65
