"""Step 4: walk-forward predictions for Random Forest and XGBoost."""
import pandas as pd

from capstone.config import DATA_PROCESSED, RESULTS, load_config
from capstone.models import RandomForestModel, XGBoostModel
from capstone.walk_forward import run_walk_forward


def main():
    cfg = load_config()
    ds = pd.read_parquet(DATA_PROCESSED / "dataset.parquet")
    s = cfg["seed"]
    preds = run_walk_forward(ds, {"RandomForest": lambda: RandomForestModel(s),
                                  "XGBoost": lambda: XGBoostModel(s)}, cfg)
    out = RESULTS / "predictions" / "ml.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out)
    print(preds.groupby("model").apply(lambda g: (g.pred == g.y_true).mean(), include_groups=False).round(4))


if __name__ == "__main__":
    main()
