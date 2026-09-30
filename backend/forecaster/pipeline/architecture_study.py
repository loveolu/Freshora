"""Can the system be one model? Same fixed harness as experiment.py (train through 2024-01-19,
score the selection window 2024-01-20 → 2024-05-05, real series), four variants:

    A  current: Tweedie demand model + separate P80 model + stage-1 traffic forecast input
    B  one model: a single multi-quantile XGBoost (median + 80th percentile), + traffic forecast
    C  current without the stage-1 traffic forecast (recent customer counts stay as inputs)
    D  one model without the stage-1 traffic forecast — the simplest system

Rule (stated before running): a simpler variant is adopted if its WAPE is no more than 0.001 worse
than A, no major category is > 5% worse, and P80 coverage stays in 0.75–0.85.

    USE_TRAFFIC_FORECAST=true python -m forecaster.pipeline.architecture_study

The flag makes the stage-1 traffic forecast a model input, so A and B really include it (production
runs with it off, the outcome of this study); the feature table is rebuilt in memory with it.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from forecaster.config import settings
from forecaster.data.prepare import prepare, processed_dir
from forecaster.db.schema import get_engine
from forecaster.models import demand
from forecaster.models.metrics import summarize
from forecaster.pipeline import lifecycle
from forecaster.pipeline.dataset import feature_table
from forecaster.pipeline.experiment import CUTOFF, START, selection_rows
from forecaster.seed import excluded_days

VARIANTS = {
    "A_current": ({}, ()),
    "B_one_model": ({"single_model": True}, ()),
    "C_no_traffic_forecast": ({}, ("expected_customer_count",)),
    "D_one_model_no_traffic_forecast": ({"single_model": True}, ("expected_customer_count",)),
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    if not settings.use_traffic_forecast:
        raise SystemExit("run with USE_TRAFFIC_FORECAST=true so variants A and B include the traffic forecast")
    feat = pd.read_parquet(processed_dir() / "features_h1.parquet")
    if "expected_customer_count" not in feat:
        feat, _ = feature_table(prepare())
    rows = demand.training_rows(feat, excluded_days(get_engine()))
    ev = selection_rows(rows)
    cat = ev["category"].astype(str)
    cats = [c for c in lifecycle.MAJOR_CATEGORIES if (cat == c).any()]  # real categories in the window
    out = {}
    for name, (cfg, exclude) in VARIANTS.items():
        cand = lifecycle.train_candidate(rows, CUTOFF, START, exclude=exclude, config=cfg)
        p50 = cand.predict(ev)
        p80 = cand.predict_p80(ev, p50)
        r = {"model": summarize(p50, ev["sales"]), "p80_coverage": float(np.mean(ev["sales"].to_numpy() <= p80)),
             "models_to_train": 1 if cfg.get("single_model") else 2,
             "uses_traffic_forecast": "expected_customer_count" not in exclude}
        for c in cats:
            m = (cat == c).to_numpy()
            r[c] = summarize(p50[m], ev.loc[m, "sales"])["wape"]
        out[name] = r
        print(f"[{time.time() - t0:5.0f}s] {name:34s} WAPE {r['model']['wape']:.4f} bias {r['model']['bias']:+.4f} "
              f"P80 cov {r['p80_coverage']:.3f} | produce {r['Fruit and vegetable']:.4f} bakery {r['Bakery']:.4f} "
              f"meat {r['Meat and fish']:.4f}", flush=True)
    base = out["A_current"]
    for name, r in out.items():
        if name == "A_current":
            continue
        worst = max((r[c] - base[c]) / base[c] for c in cats)
        r["adopt"] = (r["model"]["wape"] <= base["model"]["wape"] + 0.001 and worst <= 0.05
                      and 0.75 <= r["p80_coverage"] <= 0.85)
        r["worst_category_change"] = worst
        print(f"{name}: WAPE change {r['model']['wape'] - base['model']['wape']:+.4f}, worst category {worst:+.1%}, "
              f"adopt={r['adopt']}")
    (settings.artifacts_dir / "architecture_study.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
