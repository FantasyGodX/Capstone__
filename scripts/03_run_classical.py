"""Step 3: walk-forward predictions for benchmarks + classical models."""
from capstone.config import DATA_PROCESSED, RESULTS, load_config
from capstone.models import AlwaysUp, ARIMAModel, LogisticModel
import pandas as pd

from capstone.walk_forward import run_walk_forward


def main():
    cfg = load_config()
    ds = pd.read_parquet(DATA_PROCESSED / "dataset.parquet")
    factories = {
        "AlwaysUp": AlwaysUp,
        "Logistic": lambda: LogisticModel(seed=cfg["seed"]),
        "ARIMA": ARIMAModel,
    }
    preds = run_walk_forward(ds, factories, cfg)
    out = RESULTS / "predictions" / "classical.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out)
    print(preds.groupby("model").apply(lambda g: (g.pred == g.y_true).mean(), include_groups=False).round(4))


if __name__ == "__main__":
    main()
