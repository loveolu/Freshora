"""Train the production (served) champion and precompute what the dashboard needs.

    python -m forecaster.pipeline.train_production

1. Train on everything except a 28-day holdout; report the holdout (test) metrics honestly.
2. Register it as the production champion (immutable version).
3. Build next-day rows (as of the last data day), predict P50/P80, simulate inventory state,
   and write the predictions to the ledger (outcomes arrive later).
4. Simulated waste: model-driven ordering vs naive ordering over the holdout (labeled simulated).
5. Weather cross-check: dataset precipitation vs Open-Meteo reanalysis (kept side by side).
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import delete, insert, select, update

from forecaster.config import STORE_LOCATIONS, settings
from forecaster.data import rohlik, synthetic
from forecaster.data.prepare import processed_dir
from forecaster.db import schema as S
from forecaster.db.schema import create_all, get_engine
from forecaster.decisions.policy import compare_policies, simulate_policy
from forecaster.features.build import CATEGORICAL, FEATURE_SCHEMA_VERSION, WEATHER, build_features
from forecaster.models import demand, registry
from forecaster.models import traffic as stage1
from forecaster.models.metrics import by_segment, summarize
from forecaster.pipeline import lifecycle
from forecaster.pipeline.dataset import store_calendar
from forecaster.pipeline.experiment import SELECT_END
from forecaster.pipeline.store_learning import shelf_cohorts
from forecaster.seed import excluded_days, seed_anomaly_days, seed_stores
from forecaster.weather import open_meteo as om

HOLDOUT_DAYS = 28
LEGACY_SERVICE_LEVEL = 0.9  # assumed legacy ordering target
MARKDOWN_LEVELS = (0.2, 0.3, 0.35, 0.45, 0.5, 0.6)  # every depth markdown_suggestion can return
TRAIN_START = pd.Timestamp("2022-01-01")


def _now():
    return datetime.now(timezone.utc)


def next_day_frame(panel: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per active series for as_of + 1 with planned price (last known) and no promotion
    (unless a planned promotion is supplied). Sales unknown."""
    last = panel[panel["date"] == as_of]
    nxt = last.copy()
    nxt["date"] = as_of + pd.Timedelta(days=1)
    for c in ["sales", "availability", "customer_count"]:
        nxt[c] = np.nan
    for c in [c for c in nxt.columns if c.endswith("_discount")] + ["discount_max"]:
        nxt[c] = 0.0
    for c in ["school_holidays", "winter_school_holidays"]:
        nxt[c] = last[c].values  # school terms persist day to day
    nxt["holiday"] = 0.0  # filled from the public holiday calendar inside build_features
    nxt["shops_closed"] = 0.0  # planned closures would come from the store; none known
    return nxt


def replace_ungraded(conn, store_ids: list[str], forecast_date) -> int:
    """A re-plan for the same day (new champion or corrected inputs) supersedes that day's forecasts
    that have no outcome yet. Graded forecasts are never touched."""
    graded = select(S.outcomes.c.prediction_id)
    res = conn.execute(delete(S.predictions).where(
        (S.predictions.c.context == "production") & S.predictions.c.store_id.in_(store_ids)
        & (S.predictions.c.forecast_date == forecast_date) & S.predictions.c.prediction_id.not_in(graded)))
    return res.rowcount or 0


def with_uploaded_history(engine, rows: pd.DataFrame, cutoff: pd.Timestamp, end: pd.Timestamp,
                          log=print) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Onboarding: every uploading store's uploaded days join training, so the global model knows the
    stores it serves (features built exactly as at serving time). Days inside the warehouses' holdout
    window (cutoff, end] are left out so the headline test stays clean; days a store uploads later
    are unseen by construction. Returns (rows, onboarding rows, onboarded stores)."""
    from forecaster.pipeline.store_learning import store_features, uploading_stores
    extra = []
    for store in uploading_stores(engine):
        f = store_features(engine, store, with_next_day=False)
        f = demand.training_rows(f[(f["date"] <= cutoff) | (f["date"] > end)]) if len(f) else f
        if len(f):
            extra.append(f)
    if not extra:
        return rows, None, []
    ex = pd.concat(extra, ignore_index=True)
    for c in CATEGORICAL:
        rows[c], ex[c] = rows[c].astype(str), ex[c].astype(str)
    log(f"onboarding: +{len(ex)} uploaded rows from {sorted(ex['store_id'].unique())} "
        f"({ex['date'].min().date()} → {ex['date'].max().date()}, holdout window excluded)")
    return rows, ex, sorted(ex["store_id"].unique())


def plan_existing_stores(engine, cand, version: str, log=print) -> pd.DataFrame:
    """Tomorrow's plan for the built-in stores with the given model: forecasts, the discount
    counterfactuals, simulated legacy stock, the plan file, and the ledger rows. Re-run whenever
    a new global champion is promoted so every store is served by the same model."""
    pdir = processed_dir()
    feat = pd.read_parquet(pdir / "features_h1.parquet")
    panel = pd.read_parquet(pdir / "panel.parquet")
    weather = pd.read_parquet(pdir / "weather.parquet")
    traffic = pd.read_parquet(pdir / "traffic.parquet")
    s1 = pd.read_parquet(pdir / "stage1_h1.parquet")
    rows = demand.training_rows(feat, excluded_days(engine))
    end = rows["date"].max()
    hold = rows[rows["date"] > end - pd.Timedelta(days=HOLDOUT_DAYS)].copy()
    for k, v in demand.baseline_predictions(hold).items():
        hold[k] = v
    # Next-day recommendations as of the last data day.
    as_of = end
    tomorrow = as_of + pd.Timedelta(days=1)
    wx = []
    for store_id, loc in STORE_LOCATIONS.items():
        w = om.previous_runs_d1(loc["lat"], loc["lon"], tomorrow.date(), tomorrow.date(), loc["tz"])
        w["store_id"] = store_id
        wx.append(w)
    weather_next = pd.concat([weather, pd.concat(wx)], ignore_index=True).drop_duplicates(["store_id", "date"], keep="last")
    recent = panel[panel["date"] > as_of - pd.Timedelta(days=70)]
    ext = pd.concat([recent, next_day_frame(panel, as_of)], ignore_index=True)
    # Stage-1 expected traffic for tomorrow from a model trained on everything up to as_of.
    stores = sorted(panel["store_id"].unique())
    cal_next = store_calendar(ext)
    s1_next = stage1.store_day_frame(pd.concat([traffic, pd.DataFrame({"store_id": stores, "date": tomorrow, "customer_count": np.nan})]),
                                     cal_next, weather_next, 1, stores)
    exp_traffic = None
    if settings.use_traffic_forecast:
        s1_model = stage1.fit(s1_next, as_of)
        s1_next = s1_next[s1_next["date"] == tomorrow].copy()
        s1_next["expected_customer_count"] = stage1.predict(s1_model, s1_next)
        exp_traffic = pd.concat([s1[["store_id", "date", "expected_customer_count"]],
                                 s1_next[["store_id", "date", "expected_customer_count"]]])
    nf = build_features(ext, weather_next, traffic, 1, expected_traffic=exp_traffic)
    nf = nf[nf["date"] == tomorrow].copy()
    nf["p50"] = cand.predict(nf)
    nf["p80"] = cand.predict_p80(nf, nf["p50"].to_numpy())
    # Model-estimated demand at each markdown depth the policy can suggest (counterfactual on the
    # discount features, learned from Rohlik's real historical discount variation).
    for d in MARKDOWN_LEVELS:
        cf = nf.copy()
        cf["discount_max"], cf["any_discount"], cf["n_discount_types"] = d, 1, 1
        nf[f"p50_if_markdown_{int(d * 100)}"] = cand.predict(cf)

    # Inventory entering tomorrow = what the store's LEGACY practice leaves on the shelf (the
    # situation on install day): order same-weekday-last-week sales plus a 90% service-level
    # buffer (z90 x trailing 7-day std), through the same FIFO shelf. Labeled in the UI.
    z80 = 0.8416
    inv_rows = []
    for (st, pid), g in hold.sort_values("date").groupby(["store_id", "product_id"], observed=True):
        life = settings.shelf_life_days.get(str(g["category"].iloc[0]), 3)
        legacy50 = g["seasonal_naive_7"].fillna(g["sales_roll_28_mean"]).fillna(0).to_numpy()
        legacy80 = legacy50 + z80 * g["sales_roll_7_std"].fillna(0).to_numpy()
        _, shelf = simulate_policy(g["sales"].to_numpy(), legacy50, legacy80, LEGACY_SERVICE_LEVEL, life)
        cohorts = shelf_cohorts(shelf)
        inv_rows.append({"store_id": str(st), "product_id": str(pid), "on_hand": sum(u for u, _ in cohorts),
                         "expiring_tomorrow": sum(u for u, d in cohorts if d == 1), "shelf_life": life,
                         "cohorts": json.dumps(cohorts), "stock_source": "simulated legacy practice"})
    inv = pd.DataFrame(inv_rows)
    rec = nf[["store_id", "product_id", "name", "category", "sell_price_main", "p50", "p80", *[f"p50_if_markdown_{int(d * 100)}" for d in MARKDOWN_LEVELS],
              "customer_count_rolling_7_mean", *om.WEATHER_FEATURES]].copy()
    for c in ["store_id", "product_id", "category"]:
        rec[c] = rec[c].astype(str)
    rec = rec.merge(inv, on=["store_id", "product_id"], how="left").fillna({"on_hand": 0.0, "expiring_tomorrow": 0.0})
    rec["forecast_date"] = str(tomorrow.date())
    rec["as_of"] = str(as_of.date())
    rec["model_version"] = version
    rec.to_parquet(settings.artifacts_dir / "recommendations.parquet", index=False)

    # Ledger: the predictions exist before their outcomes do.
    created = _now()
    preds = []
    for _, r in rec.iterrows():
        preds.append(dict(prediction_id=uuid.uuid4().hex, context="production", store_id=r["store_id"],
                          product_id=r["product_id"], category=r["category"], prediction_created_at=created,
                          forecast_date=tomorrow.date(), horizon=1, predicted_units=float(r["p50"]), p80_units=float(r["p80"]),
                          weather_forecast={k: (None if pd.isna(r[k]) else float(r[k])) for k in om.WEATHER_FEATURES} | {"weather_source": "previous_runs_d1"},
                          expected_customers=None,
                          model_version=version, feature_schema_version=FEATURE_SCHEMA_VERSION))
    with engine.begin() as conn:
        replace_ungraded(conn, rec["store_id"].unique().tolist(), tomorrow.date())
        conn.execute(insert(S.predictions), preds)

    log(f"planned {len(rec)} products for {tomorrow.date()} with {version}")
    return rec


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    pdir = processed_dir()
    from forecaster.pipeline.run_replay import features_schema
    if features_schema() != FEATURE_SCHEMA_VERSION:
        raise SystemExit(f"feature table was built with schema {features_schema()}, code expects "
                         f"{FEATURE_SCHEMA_VERSION}; run forecaster.pipeline.run_replay --force first")
    feat = pd.read_parquet(pdir / "features_h1.parquet")

    engine = get_engine()
    create_all(engine)
    seed_stores(engine)
    seed_anomaly_days(engine)
    rows = demand.training_rows(feat, excluded_days(engine))
    end = rows["date"].max()
    cutoff = end - pd.Timedelta(days=HOLDOUT_DAYS)
    if cutoff < SELECT_END:  # the test must start after the last day any design choice was scored on
        raise SystemExit(f"test window starts {(cutoff + pd.Timedelta(days=1)).date()}, inside the "
                         f"design-selection window ending {SELECT_END.date()}")
    rows, onboard, onboarded = with_uploaded_history(engine, rows, cutoff, end, log)
    cand = lifecycle.train_candidate(rows, cutoff, TRAIN_START, extra_train=onboard)
    hold = rows[rows["date"] > cutoff].copy()
    hold["pred"] = cand.predict(hold)
    hold["p80"] = cand.predict_p80(hold, hold["pred"].to_numpy())
    hold["pred_no_traffic"] = cand.predict(hold, drop_traffic=True)
    for k, v in demand.baseline_predictions(hold).items():
        hold[k] = v
    # Headline numbers use real (observed) series only; the synthetic Dairy/Eggs series are reported
    # separately and labeled — accuracy on generated data proves nothing about real stores.
    synth = hold["series_id"].astype(str).str.startswith(synthetic.SYNTHETIC_PREFIX)
    hold_all, hold = hold, hold[~synth]
    test = {c: summarize(hold[c], hold["sales"]) for c in ["pred", "pred_no_traffic", "seasonal_naive_7", "rolling_mean_28"]}
    test["by_category"] = by_segment(hold, "pred", "sales", "category").to_dict("records")
    test["by_store"] = by_segment(hold, "pred", "sales", "store_id").to_dict("records")
    test["by_weekday"] = by_segment(hold.assign(weekday=hold["date"].dt.day_name()), "pred", "sales", "weekday").to_dict("records")
    test["synthetic_categories"] = {
        "label": "SYNTHETIC — generated Dairy/Eggs series, not observed data",
        "by_category": by_segment(hold_all[synth], "pred", "sales", "category").to_dict("records"),
        "last_week_baseline": summarize(hold_all.loc[synth, "seasonal_naive_7"], hold_all.loc[synth, "sales"])}
    p80_cov = float((hold["sales"] <= hold["p80"]).mean())
    log(f"production candidate: test WAPE {test['pred']['wape']:.3f} (no traffic {test['pred_no_traffic']['wape']:.3f}, "
        f"naive {test['seasonal_naive_7']['wape']:.3f}); P80 coverage {p80_cov:.2f}")

    # Weather ablation — measured, reported whichever way it goes.
    if "--skip-weather-ablation" in sys.argv:  # measured on every earlier run: weather gives no gain
        test["pred_no_weather"] = None
    else:
        cand_nw = lifecycle.train_candidate(rows, cutoff, TRAIN_START, exclude=tuple(WEATHER))
        test["pred_no_weather"] = summarize(cand_nw.predict(hold), hold["sales"])  # real series only
        log(f"weather ablation: with {test['pred']['wape']:.4f} vs without {test['pred_no_weather']['wape']:.4f}")

    version = registry.next_version()
    registry.save(version, cand.booster, {**cand.meta, "context": "production", "train_metrics": cand.train_metrics,
                                          "trained_with_uploads_from": onboarded or None,
                                          "test_metrics": test["pred"], "holdout": [str((cutoff + pd.Timedelta(days=1)).date()), str(end.date())],
                                          "p80_coverage": p80_cov}, lifecycle.extra_models(cand))
    with engine.begin() as conn:
        conn.execute(update(S.model_versions).where((S.model_versions.c.context == "production") &
                                                    (S.model_versions.c.promotion_status == "champion"))
                     .values(promotion_status="retired"))
        conn.execute(insert(S.model_versions).values(
            version=version, stage="demand", training_start=TRAIN_START.date(), training_end=cutoff.date(),
            feature_schema_version=FEATURE_SCHEMA_VERSION, params=cand.meta["params"],
            metrics={"train": cand.train_metrics, "test": test["pred"], "p80_coverage": p80_cov},
            artifact_uri=str(registry.models_dir() / version), created_at=_now(), promotion_status="champion",
            decision_reason="Bootstrap model: Rohlik history + synthetic Dairy/Eggs + Open-Meteo archived forecasts; "
                            "28-day holdout test on real series",
            context="production"))
        conn.execute(insert(S.champion_history).values(context="production", version=version, effective_at=_now(),
                                                       as_of_date=end.date(), reason="bootstrap"))

    # Simulated waste over the holdout (labeled simulated everywhere it is shown).
    sim = compare_policies(hold, "pred", "p80", "seasonal_naive_7")
    sim_summary = {
        "label": "SIMULATED — no public dataset records waste; FIFO shelf-life simulator over the 28-day holdout",
        "shelf_life_days": settings.shelf_life_days,
        "waste_cost_ratio": settings.default_waste_cost_ratio,
        "totals": {k: float(sim[k].sum()) for k in ["model_waste", "baseline_waste", "model_lost", "baseline_lost", "model_sold", "baseline_sold"]},
        "by_category": sim.groupby("category")[["model_waste", "baseline_waste", "model_lost", "baseline_lost"]].sum().reset_index().to_dict("records"),
    }
    t = sim_summary["totals"]
    log(f"simulated waste: model {t['model_waste']:.0f} vs naive {t['baseline_waste']:.0f}; lost sales {t['model_lost']:.0f} vs {t['baseline_lost']:.0f}")

    rec = plan_existing_stores(engine, cand, version, log)
    as_of, tomorrow = end, end + pd.Timedelta(days=1)

    # Weather cross-check: dataset's own precipitation vs Open-Meteo reanalysis (never merged).
    orders = rohlik.load_orders()
    xcheck = {}
    for store_id, loc in STORE_LOCATIONS.items():
        ds = orders[orders["store_id"] == store_id][["date", "dataset_precipitation", "dataset_snow"]]
        if ds.empty:
            continue
        api = om.archive_observed(loc["lat"], loc["lon"], ds["date"].min().date(), ds["date"].max().date(), loc["tz"])
        xcheck[store_id] = om.cross_check(ds, api)

    exp_path = settings.artifacts_dir / "accuracy_experiments.json"
    headline = {
        "metric": "WAPE = sum|pred - actual| / sum(actual); accuracy, where quoted, = 1 - WAPE",
        "holdout": [str((cutoff + pd.Timedelta(days=1)).date()), str(end.date())],
        "model_wape": test["pred"]["wape"], "model_bias": test["pred"]["bias"],
        "last_week_wape": test["seasonal_naive_7"]["wape"], "rolling_28_wape": test["rolling_mean_28"]["wape"],
        "relative_to_last_week": 1 - test["pred"]["wape"] / test["seasonal_naive_7"]["wape"],
        "p80_coverage": p80_cov,
        "simulated_waste_units": {"model": t["model_waste"], "last_week_baseline": t["baseline_waste"]},
        "simulated_lost_sales_units": {"model": t["model_lost"], "last_week_baseline": t["baseline_lost"]},
    }
    report = {"headline": headline,
              "accuracy_experiments": json.loads(exp_path.read_text()) if exp_path.exists() else None,
              "version": version, "as_of": str(as_of.date()), "forecast_date": str(tomorrow.date()),
              "holdout": [str((cutoff + pd.Timedelta(days=1)).date()), str(end.date())],
              "train_metrics": cand.train_metrics, "test": test, "p80_coverage": p80_cov,
              "simulation": sim_summary, "weather_cross_check": xcheck,
              "feature_importance": _importance(cand.booster)}
    (settings.artifacts_dir / "production_report.json").write_text(json.dumps(report, indent=2, default=str))
    from forecaster.db import hooks
    hooks.after_everything(engine)
    log(f"done: {version}, {len(rec)} next-day recommendations for {tomorrow.date()}")


def _importance(booster) -> list[dict]:
    gain = booster.get_score(importance_type="total_gain")
    total = sum(gain.values()) or 1.0
    return [{"feature": k, "share": v / total} for k, v in sorted(gain.items(), key=lambda kv: -kv[1])[:25]]


if __name__ == "__main__":
    main()
