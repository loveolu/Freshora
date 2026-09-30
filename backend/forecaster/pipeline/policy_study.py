"""Two measurements the pitch depends on, on the selection window (2024-01-20 → 2024-05-05):

1. Weather ablation on the fixed harness (the production holdout said weather hurts; confirm or
   refute on 4.5 months instead of 4 weeks).
2. Waste vs. lost-sales trade-off of model-driven vs. naive ordering at several cost settings,
   same ordering rule for both (forecast skill only).

    python -m forecaster.pipeline.policy_study
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from forecaster.config import settings
from forecaster.data.prepare import processed_dir
from forecaster.db.schema import get_engine
from forecaster.decisions.policy import compare_policies, critical_ratio
from forecaster.features.build import WEATHER
from forecaster.models import demand
from forecaster.models.metrics import summarize
from forecaster.pipeline import lifecycle
from forecaster.pipeline.experiment import CUTOFF, START, selection_rows
from forecaster.seed import excluded_days


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    feat = pd.read_parquet(processed_dir() / "features_h1.parquet")
    rows = demand.training_rows(feat, excluded_days(get_engine()))
    ev = selection_rows(rows).copy()
    out = {"window": [str((CUTOFF + pd.Timedelta(days=1)).date()), str(ev["date"].max().date())]}

    with_w = lifecycle.train_candidate(rows, CUTOFF, START)
    no_w = lifecycle.train_candidate(rows, CUTOFF, START, exclude=tuple(WEATHER))
    ev["pred"] = with_w.predict(ev)
    ev["p80"] = with_w.predict_p80(ev, ev["pred"].to_numpy())
    out["weather_ablation"] = {"with_weather": summarize(ev["pred"], ev["sales"]),
                               "without_weather": summarize(no_w.predict(ev), ev["sales"])}
    print("weather:", {k: round(v["wape"], 4) for k, v in out["weather_ablation"].items()}, flush=True)

    ev["naive"] = demand.baseline_predictions(ev)["seasonal_naive_7"]
    total_demand = float(ev["sales"].sum())
    curve = []
    for wcr in (0.1, 0.25, 0.5, 1.0):
        sim = compare_policies(ev, "pred", "p80", "naive", waste_cost_ratio=wcr)
        t = sim[["model_waste", "baseline_waste", "model_lost", "baseline_lost"]].sum()
        row = {"waste_cost_ratio": wcr, "critical_ratio": critical_ratio(wcr),
               "model_waste_pct": t["model_waste"] / total_demand, "naive_waste_pct": t["baseline_waste"] / total_demand,
               "model_lost_pct": t["model_lost"] / total_demand, "naive_lost_pct": t["baseline_lost"] / total_demand}
        curve.append(row)
        print({k: round(v, 4) for k, v in row.items()}, flush=True)
    out["tradeoff"] = curve
    out["total_demand_units"] = total_demand
    out["label"] = "SIMULATED waste and lost sales (FIFO shelf-life simulator), % of actual demand"
    (settings.artifacts_dir / "policy_study.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
