"""Step 10: export results to static JSON for the website in site/ (hosted on Netlify).

The site is plain HTML + JavaScript. It never runs Python; it reads the files written here:
  site/data/meta.json         config values and model/ticker lists
  site/data/predictions.json  every out-of-sample prediction (columnar, compact)
  site/data/tables.json       all results/tables/*.csv
  site/data/prices.json       daily OHLCV for the traded tickers, SPY and ^VIX
  site/data/features.json     the modelling dataset (26 features + next-day return + label)
  site/data/logic.md          the written project explanation (Logic.md)
Re-run this script after the pipeline, then commit site/data and push.
"""
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from capstone.config import DATA_PROCESSED, DATA_RAW, RESULTS, ROOT, load_config  # noqa: E402
from capstone.data import load_prices  # noqa: E402
from capstone.features import feature_columns  # noqa: E402

OUT = ROOT / "site" / "data"
ORDER = ["AlwaysUp", "Logistic", "ARIMA", "RandomForest", "XGBoost", "LSTM"]


def dump(name: str, obj) -> None:
    path = OUT / name
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"  {name:18s} {path.stat().st_size / 1024:8.0f} KB")


def r(a, nd):
    return [None if pd.isna(x) else round(float(x), nd) for x in a]


def export_predictions(cfg):
    p = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(RESULTS / "predictions" / "*.parquet")))])
    p = p.sort_values(["model", "ticker", "date"])
    tickers = sorted(p["ticker"].unique())
    dates = sorted(p["date"].unique())
    base = p[p["model"] == ORDER[0]]
    n_t, n_d = len(tickers), len(dates)
    # The site indexes predictions as [ticker * n_dates + date]; that needs a full, identical grid.
    assert len(base) == n_t * n_d, "predictions are not a full ticker x date grid"
    for m in ORDER:
        g = p[p["model"] == m]
        assert len(g) == len(base) and (g["ticker"].values == base["ticker"].values).all() \
            and (g["date"].values == base["date"].values).all(), f"{m}: rows differ from {ORDER[0]}"
    date_str = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
    out = {
        "tickers": tickers,
        "dates": date_str,
        "year": [int(pd.Timestamp(d).year) for d in dates],
        "y": base["y_true"].astype(int).tolist(),
        "ret": r(base["ret_next"], 6),
        "models": {},
    }
    for m in ORDER:
        g = p[p["model"] == m]
        out["models"][m] = {"score": r(g["score"], 6), "pred": g["pred"].astype(int).tolist()}
    dump("predictions.json", out)
    return tickers


def export_tables():
    tables = {}
    for f in sorted(glob.glob(str(RESULTS / "tables" / "*.csv"))):
        df = pd.read_csv(f)
        cols = list(df.columns)
        # Index columns saved without a name come back as "Unnamed: 0" (and "Unnamed: 1").
        fixed = []
        for i, c in enumerate(cols):
            fixed.append(("model", "ticker")[i] if str(c).startswith("Unnamed") else c)
        df.columns = fixed
        df = df.astype(object).where(df.notna(), None)
        tables[Path(f).name] = {"columns": fixed, "rows": df.values.tolist()}
    dump("tables.json", tables)


def export_prices(cfg, tickers):
    px = load_prices(DATA_RAW / "prices.db")
    out = {}
    for t in tickers + [cfg["market_ticker"], cfg["vix_ticker"]]:
        d = px[t]
        out[t] = {"d": [x.strftime("%Y-%m-%d") for x in d.index], "o": r(d["Open"], 3), "h": r(d["High"], 3),
                  "l": r(d["Low"], 3), "c": r(d["Close"], 3), "v": [int(x) for x in d["Volume"].fillna(0)]}
    dump("prices.json", out)


def export_features():
    ds = pd.read_parquet(DATA_PROCESSED / "dataset.parquet")
    cols = feature_columns(ds) + ["ret_next", "y"]
    out = {"columns": cols, "tickers": {}}
    for t, g in ds.groupby(level="ticker"):
        g = g.droplevel("ticker")
        out["tickers"][t] = {"d": [x.strftime("%Y-%m-%d") for x in g.index],
                             "v": [r(row, 5) for row in g[cols].to_numpy()]}
    dump("features.json", out)


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    print("Exporting site data ->", OUT)
    tickers = export_predictions(cfg)
    export_tables()
    export_prices(cfg, tickers)
    export_features()
    shutil.copyfile(ROOT / "Logic.md", OUT / "logic.md")
    meta = {
        "order": ORDER,
        "tickers": tickers,
        "market": cfg["market_ticker"],
        "vix": cfg["vix_ticker"],
        "start_date": cfg["start_date"], "end_date": cfg["end_date"], "seed": cfg["seed"],
        "walk_forward": cfg["walk_forward"], "backtest": cfg["backtest"], "stats": cfg["stats"],
    }
    dump("meta.json", meta)


if __name__ == "__main__":
    main()
