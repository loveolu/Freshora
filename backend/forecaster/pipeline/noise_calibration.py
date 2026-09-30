"""Measure the day-to-day noise of real staples, which sets synthetic.DISPERSION_K.

The harness model (trained through experiment.CUTOFF) forecasts the first 28 days of the selection
window; its residuals on real Bakery give the volume-weighted overdispersion
1/k = Σ((y − μ)² − μ) / Σμ², an upper bound on the true noise since residuals also contain model
error. Only selection-window rows are used — never the test window train_production reports.

    python -m forecaster.pipeline.noise_calibration
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from forecaster.config import settings
from forecaster.data import synthetic
from forecaster.data.prepare import processed_dir
from forecaster.db.schema import get_engine
from forecaster.models import demand
from forecaster.pipeline import lifecycle
from forecaster.pipeline.experiment import CUTOFF, START, selection_rows
from forecaster.seed import excluded_days

WINDOW_DAYS = 28


def overdispersion(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(mu, 1e-6)
    return max(float(((y - mu) ** 2 - mu).sum() / (mu ** 2).sum()), 0.0)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    feat = pd.read_parquet(processed_dir() / "features_h1.parquet")
    rows = demand.training_rows(feat, excluded_days(get_engine()))
    ev = selection_rows(rows)
    ev = ev[ev["date"] <= CUTOFF + pd.Timedelta(days=WINDOW_DAYS)].copy()
    ev["pred"] = lifecycle.train_candidate(rows, CUTOFF, START).predict(ev)
    out = {"window": [str((CUTOFF + pd.Timedelta(days=1)).date()), str(ev["date"].max().date())],
           "current_k": synthetic.DISPERSION_K, "inv_k": {}}
    for cat, g in ev.groupby(ev["category"].astype(str), observed=True):
        out["inv_k"][cat] = overdispersion(g["sales"].to_numpy(), g["pred"].to_numpy())
        print(f"{cat:22s} 1/k <= {out['inv_k'][cat]:.4f}", flush=True)
    inv = out["inv_k"]["Bakery"]
    out["suggested_k"] = round(1 / inv) if inv else None
    print(f"Bakery → k = {out['suggested_k']} (generator uses {synthetic.DISPERSION_K})")
    (settings.artifacts_dir / "noise_calibration.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
