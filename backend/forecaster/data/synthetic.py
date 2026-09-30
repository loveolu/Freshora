"""Synthetic Dairy and Eggs series for the training warehouses (the Rohlik data has neither).

Generated, not observed — every row carries series_id "SYN-…" and is labeled synthetic wherever it
is reported. Each product's daily demand is driven by the warehouse's REAL daily customer count and
calendar (holidays, closures), with a product-specific base rate, slow drift, occasional price
promotions with a product-specific response, and gamma-Poisson noise:

    demand_t = base_j * (customers_t / mean customers) * drift_t * (1 + response_j * discount_t)
               * (1.15 the day before a holiday) * (0 when closed),   sales_t ~ Poisson(Gamma(k, demand_t / k))

    ensure(panel) → panel with the synthetic rows (re-generated deterministically, idempotent)
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

SYNTHETIC_PREFIX = "SYN-"
CATEGORIES = {  # category → (products, price relative to the store's median price, subcategories)
    "Dairy products": (10, 0.7, 3),
    "Eggs": (4, 1.4, 1),
}
# Day-to-day noise (gamma shape; smaller = noisier). Dairy and eggs are staples, so their noise is set to
# the measured noise of the real staple category: the volume-weighted overdispersion of the harness
# model's residuals on real Bakery over the first 28 days of the selection window, 1/k <= 0.016 — an
# upper bound, since residuals also contain model error (pipeline/noise_calibration.py; never the test
# window). The first version used k = 12, five times noisier than real staples.
DISPERSION_K = 62.0
PROMO_DAY_SHARE = 0.06


def is_synthetic(series_id) -> bool:
    return str(series_id).startswith(SYNTHETIC_PREFIX)


def _series(store: pd.DataFrame, store_id: str, category: str, j: int, n_sub: int, price_scale: float,
            base_units: float, rng: np.random.Generator) -> pd.DataFrame:
    days = store.sort_values("date").reset_index(drop=True)
    n = len(days)
    cust = days["customer_count"].ffill().bfill().to_numpy()
    rel_cust = cust / np.nanmean(cust)
    base = base_units * rng.lognormal(0.0, 0.6)
    drift = np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    drift /= drift.mean()
    response = rng.uniform(0.8, 2.2)  # +8–22% sales per 10 points of discount
    promo = np.zeros(n)
    t = 0
    while t < n:
        if rng.random() < PROMO_DAY_SHARE / 2:
            run = int(rng.integers(1, 4))
            promo[t:t + run] = rng.choice([0.1, 0.15, 0.2, 0.25, 0.3])
            t += run
        t += 1
    holiday_next = np.r_[days["holiday"].to_numpy()[1:], 0.0] > 0
    closed = days["shops_closed"].to_numpy() > 0
    mean = base * rel_cust * drift * (1 + response * promo) * np.where(holiday_next, 1.15, 1.0) * np.where(closed, 0.0, 1.0)
    sales = rng.poisson(rng.gamma(DISPERSION_K, np.maximum(mean, 1e-9) / DISPERSION_K)).astype(float)
    price = round(float(price_scale * rng.uniform(0.6, 1.6)), 2)
    code = "D" if category.startswith("Dairy") else "E"
    sub = j % n_sub + 1
    short = "Dairy" if code == "D" else "Eggs"
    out = pd.DataFrame({
        "date": days["date"], "series_id": f"{SYNTHETIC_PREFIX}{code}{j}-{store_id}", "name": f"{short}_{j}",
        "category": category, "category_l2": f"{category}_L2_{sub}", "category_l3": f"{category}_L3_{sub}",
        "category_l4": f"{category}_L4_{j}", "store_id": store_id, "customer_count": days["customer_count"],
        "sales": sales, "sell_price_main": price, "availability": 1.0,
        "holiday": days["holiday"], "shops_closed": days["shops_closed"],
        "winter_school_holidays": days["winter_school_holidays"], "school_holidays": days["school_holidays"],
        "product_id": f"{code}{100 + j}", "discount_max": promo,
    })
    for k in range(7):
        out[f"type_{k}_discount"] = promo if k == 1 else 0.0
    return out


def ensure(panel: pd.DataFrame, seed: int = 20260926) -> pd.DataFrame:
    """Drop any previous synthetic rows and regenerate them for every store (deterministic)."""
    real = panel[~panel["series_id"].astype(str).str.startswith(SYNTHETIC_PREFIX)]
    frames = [real]
    for store_id, g in real.groupby("store_id", sort=True):
        rng = np.random.default_rng([seed, zlib.crc32(store_id.encode())])
        days = g.groupby("date", as_index=False).agg(
            customer_count=("customer_count", "first"), holiday=("holiday", "max"), shops_closed=("shops_closed", "max"),
            winter_school_holidays=("winter_school_holidays", "max"), school_holidays=("school_holidays", "max"))
        per_product_day = float(g.groupby("series_id")["sales"].mean().median())  # a typical product, not the average
        median_price = float(g["sell_price_main"].median())
        for category, (count, rel_price, n_sub) in CATEGORIES.items():
            for j in range(count):
                frames.append(_series(days, store_id, category, j, n_sub, median_price * rel_price,
                                      per_product_day, rng))
    out = pd.concat(frames, ignore_index=True)
    return out[panel.columns]
