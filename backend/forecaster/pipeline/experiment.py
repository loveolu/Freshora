"""Accuracy experiments on a fixed harness: train once on data through 2024-01-19 and score the
identical real-series rows of the selection window (2024-01-20 → 2024-05-05). Levers are added one
at a time on top of the kept ones; a lever is kept only if global WAPE drops by ≥ 0.003 and no major
category gets more than 5% worse.

Three-way split: train ≤ 2024-01-19 | select 2024-01-20 → 2024-05-05 | test 2024-05-06 → 2024-06-02.
Every design choice (this file, architecture_study, policy_study, the synthetic noise level) is
scored on the selection window only; the test window train_production reports is never used to choose.

    python -m forecaster.pipeline.experiment
"""
from __future__ import annotations

import json
import sys
import time

import pandas as pd

from forecaster.config import settings
from forecaster.data import synthetic
from forecaster.data.prepare import processed_dir
from forecaster.db.schema import get_engine
from forecaster.features.build import DISCOUNT_TYPES
from forecaster.models import demand
from forecaster.models.metrics import summarize
from forecaster.pipeline import lifecycle
from forecaster.seed import excluded_days

CUTOFF = pd.Timestamp("2024-01-19")  # harness models train on data through this day
SELECT_END = pd.Timestamp("2024-05-05")  # last day any design choice is scored on
START = pd.Timestamp("2022-01-01")
NEW_PRICE = (*DISCOUNT_TYPES, "discount_max_lag_7")
MIN_GAIN = 0.003
MAX_CAT_DEGRADATION = 0.05

LEVERS = [  # (name, config change, include the fs_v5 discount features?)
    ("1_discount_types", {}, True),
    ("2_censor_stockouts", {"censor_stockouts": True}, True),
    ("3_produce_specialist", {"produce_specialist": True}, True),
    ("4_store_calibration", {"store_calibration": True}, True),
    ("5_volatility_weight", {"volatility_weight": True}, True),
]


def selection_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Real-series rows in (CUTOFF, SELECT_END] — the only rows design choices are scored on."""
    real = ~rows["series_id"].astype(str).str.startswith(synthetic.SYNTHETIC_PREFIX)
    return rows[(rows["date"] > CUTOFF) & (rows["date"] <= SELECT_END) & real]


def score(pred, df) -> dict:
    out = {"model": summarize(pred, df["sales"])}
    base = demand.baseline_predictions(df)
    out["last_week"] = summarize(base["seasonal_naive_7"], df["sales"])
    out["roll_28"] = summarize(base["rolling_mean_28"], df["sales"])
    cat = df["category"].astype(str)
    for name, m in [("produce", cat == "Fruit and vegetable"), ("bakery", cat == "Bakery"),
                    ("meat", cat == "Meat and fish"), ("promo", df["discount_max"] > 0),
                    ("stockout", df["availability"] < 0.9)]:
        out[name] = summarize(pred[m.to_numpy()], df.loc[m, "sales"])["wape"]
    return out


def line(name: str, r: dict) -> str:
    m = r["model"]
    return (f"{name:24s} WAPE {m['wape']:.4f} bias {m['bias']:+.4f} | last-week {r['last_week']['wape']:.4f} "
            f"roll-28 {r['roll_28']['wape']:.4f} | produce {r['produce']:.4f} bakery {r['bakery']:.4f} "
            f"meat {r['meat']:.4f} promo {r['promo']:.4f} stockout {r['stockout']:.4f}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    feat = pd.read_parquet(processed_dir() / "features_h1.parquet")
    rows = demand.training_rows(feat, excluded_days(get_engine()))
    eval_rows = selection_rows(rows)  # identical scoring rows for every experiment

    def run(cfg, with_new):
        cand = lifecycle.train_candidate(rows, CUTOFF, START, exclude=() if with_new else NEW_PRICE, config=cfg)
        return score(cand.predict(eval_rows), eval_rows)

    results = {}
    best_cfg, best_new = {}, False
    best = run(best_cfg, best_new)
    results["0_baseline_fs_v4"] = {"result": best, "kept": True}
    print(f"[{time.time() - t0:5.0f}s] " + line("0_baseline_fs_v4", best), flush=True)
    for name, change, with_new in LEVERS:
        cfg = {**best_cfg, **change}
        r = run(cfg, with_new)
        gain = best["model"]["wape"] - r["model"]["wape"]
        worst = max((r[c] - best[c]) / best[c] for c in ("produce", "bakery", "meat"))
        kept = gain >= MIN_GAIN and worst <= MAX_CAT_DEGRADATION
        results[name] = {"result": r, "gain": gain, "worst_category_change": worst, "kept": kept, "config": cfg}
        print(f"[{time.time() - t0:5.0f}s] " + line(name, r) + f" | gain {gain:+.4f} worst-cat {worst:+.1%} "
              f"→ {'KEEP' if kept else 'reject'}", flush=True)
        if kept:
            best, best_cfg, best_new = r, cfg, with_new
    summary = {"kept_config": best_cfg, "discount_features": best_new, "final": best["model"], "results": results}
    (settings.artifacts_dir / "accuracy_experiments.json").write_text(json.dumps(summary, indent=2, default=str))
    print("kept:", best_cfg, "discount features:", best_new, "final WAPE", round(best["model"]["wape"], 4))


if __name__ == "__main__":
    main()
