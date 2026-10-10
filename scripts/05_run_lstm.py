"""Step 5: walk-forward predictions for the LSTM."""
import pandas as pd

from capstone.config import DATA_PROCESSED, RESULTS, load_config
from capstone.models import LSTMModel
from capstone.walk_forward import run_walk_forward


def main():
    cfg = load_config()
    ds = pd.read_parquet(DATA_PROCESSED / "dataset.parquet")
    preds = run_walk_forward(ds, {"LSTM": lambda: LSTMModel(seed=cfg["seed"])}, cfg)
    out = RESULTS / "predictions" / "lstm.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out)
    print("accuracy:", round((preds.pred == preds.y_true).mean(), 4), " mean score:", round(preds.score.mean(), 3))


if __name__ == "__main__":
    main()
