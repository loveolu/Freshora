import numpy as np
import pandas as pd
import pytest

from forecaster.data.validation import validate_observations, validate_traffic
from forecaster.decisions.policy import critical_ratio, order_quantity, simulate_inventory
from forecaster.features.build import build_features, complete_daily_index, demand_feature_names
from forecaster.models.metrics import summarize
from forecaster.pipeline.lifecycle import MAJOR_CATEGORIES, decide


def _panel(days=90, products=("1", "2"), stores=("Prague_1",)):
    rng = np.random.default_rng(0)
    rows = []
    for s in stores:
        for p in products:
            for d in pd.date_range("2023-01-01", periods=days):
                rows.append(dict(store_id=s, product_id=p, series_id=f"{s}{p}", name=f"x_{p}",
                                 category="Bakery", category_l2="b2", category_l3="b3", category_l4="b4",
                                 date=d, sales=float(rng.poisson(20)), sell_price_main=10.0, availability=1.0,
                                 customer_count=1000.0, holiday=0.0, shops_closed=0.0,
                                 winter_school_holidays=0.0, school_holidays=0.0,
                                 **{f"type_{i}_discount": 0.0 for i in range(7)}))
    return pd.DataFrame(rows)


def test_features_do_not_see_the_future():
    panel = complete_daily_index(_panel())
    traffic = panel.groupby(["store_id", "date"], as_index=False)["customer_count"].first()
    cut = pd.Timestamp("2023-02-15")
    a = build_features(panel, None, traffic, horizon=1)
    tampered = panel.copy()
    tampered.loc[tampered["date"] >= cut, "sales"] = 9999.0
    t2 = traffic.copy()
    t2.loc[t2["date"] >= cut, "customer_count"] = 1e9
    b = build_features(tampered, None, t2, horizon=1)
    names = [n for n in demand_feature_names(1) if n not in ("store_id", "product_id", "category", "category_l2")]
    ra = a[a["date"] == cut].sort_values("product_id")[names].reset_index(drop=True)
    rb = b[b["date"] == cut].sort_values("product_id")[names].reset_index(drop=True)
    pd.testing.assert_frame_equal(ra, rb)  # the row for date `cut` cannot see `cut`'s own sales


def test_tomorrow_gets_traffic_features_from_observed_days():
    """Serving row (tomorrow) has no observed traffic yet; its lags/rolling means must still be filled."""
    panel = complete_daily_index(_panel())
    traffic = panel.groupby(["store_id", "date"], as_index=False)["customer_count"].first()
    last = panel["date"].max()
    nxt = panel[panel["date"] == last].assign(date=last + pd.Timedelta(days=1), sales=np.nan, customer_count=np.nan)
    feat = build_features(pd.concat([panel, nxt], ignore_index=True), None, traffic, horizon=1)
    row = feat[feat["date"] == last + pd.Timedelta(days=1)]
    assert row["customer_count_lag_h"].eq(1000.0).all()
    assert row["customer_count_rolling_7_mean"].eq(1000.0).all()


def test_validation_quarantines_with_reasons():
    df = _panel(days=5)
    df.loc[0, "sales"] = -3
    df = pd.concat([df, df.iloc[[1]]])
    clean, bad = validate_observations(df)
    assert set(bad["reason"]) == {"negative_sales", "duplicate_record"}
    assert len(clean) + len(bad) == len(df)


def test_traffic_validation():
    df = pd.DataFrame({"store_id": ["a", "a", None], "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                       "customers_entered": [10, 12, -1]})
    clean, bad = validate_traffic(df)
    assert len(clean) == 1
    assert "duplicate_record" in bad["reason"].iloc[0] or "duplicate_record" in bad["reason"].iloc[1]


def test_simulator_counts_waste_and_lost_sales():
    # 10 units/day ordered, demand 6/day, shelf life 2 → 4 left each day expire next day after one more chance
    r = simulate_inventory(np.array([6.0] * 10), np.array([10.0] * 10), shelf_life=2)
    assert r.lost_sales_units == 0
    assert r.waste_units > 0
    r2 = simulate_inventory(np.array([10.0] * 5), np.array([5.0] * 5), shelf_life=3)
    assert r2.waste_units == 0 and r2.lost_sales_units == 25


def test_newsvendor_orders_more_when_waste_is_cheap():
    p50, p80 = np.array([100.0]), np.array([130.0])
    cheap = order_quantity(p50, p80, critical_ratio(0.1))
    costly = order_quantity(p50, p80, critical_ratio(2.0))
    assert cheap[0] > 100 > costly[0]


def _eval(g, cats):
    return {"global": {"wape": g}, "by_category": {c: {"wape": v} for c, v in zip(MAJOR_CATEGORIES, cats)}}


def test_promotion_rule_requires_improvement_and_no_category_regression():
    base = {"naive": _eval(0.3, [0.3] * 3)}
    champ = _eval(0.20, [0.2, 0.2, 0.2])
    d, _, _ = decide(champ, _eval(0.199, [0.2] * 3), base, {"wape": 0.15})
    assert d == "rejected"  # 0.5% better is not enough
    d, _, _ = decide(champ, _eval(0.18, [0.17, 0.17, 0.23]), base, {"wape": 0.15})
    assert d == "rejected"  # a major category degraded 15%
    d, _, _ = decide(champ, _eval(0.18, [0.18, 0.18, 0.2]), base, {"wape": 0.15})
    assert d == "promoted"
    d, _, _ = decide(champ, _eval(0.18, [0.18, 0.18, 0.2]), base, {"wape": 0.05})
    assert d == "rejected"  # overfit guard


def test_metrics_sign_convention():
    m = summarize(np.array([100.0]), np.array([80.0]))
    assert m["bias"] == pytest.approx(0.25) and m["wape"] == pytest.approx(0.25)


def test_holiday_proximity_sees_easter_coming_from_the_public_calendar():
    from forecaster.features.calendar import calendar_frame, holiday_features
    flags = pd.DataFrame({"store_id": [], "date": pd.to_datetime([]), "holiday": [], "shops_closed": []})
    hf = holiday_features(calendar_frame(flags, pd.Timestamp("2024-03-20"), pd.Timestamp("2024-04-05")))
    row = hf[(hf["store_id"] == "Prague_1") & (hf["date"] == "2024-03-27")].iloc[0]
    assert row["days_to_next_holiday"] == 2  # Good Friday, 2024-03-29 (CZ)
    assert row["next_holiday_type"] == 2  # Easter
    gf = hf[(hf["store_id"] == "Prague_1") & (hf["date"] == "2024-03-29")].iloc[0]
    assert gf["is_holiday"] and gf["days_to_next_holiday"] == 0


def test_policy_orders_net_of_stock_still_on_the_shelf():
    from forecaster.decisions.policy import simulate_policy
    # forecast 10 then 8; only 2 sell on day 1, so 8 are still sellable on day 2 → order ~0
    res, shelf = simulate_policy(np.array([2.0, 8.0]), np.array([10.0, 8.0]), np.array([10.0, 8.0]),
                                 ratio=0.5, shelf_life=3)
    assert res.ordered_units == pytest.approx(10.0)
    assert res.sold_units == pytest.approx(10.0) and res.waste_units == 0 and res.lost_sales_units == 0


def test_model_and_baseline_use_the_same_netting_rule():
    from forecaster.decisions.policy import compare_policies
    df = pd.DataFrame({"store_id": "s", "product_id": "p", "category": "Bakery",
                       "date": pd.date_range("2024-01-01", periods=6), "sales": [5.0] * 6,
                       "pred": [5.0] * 6, "p80": [5.0] * 6, "naive": [5.0] * 6,
                       "sales_roll_7_std": [0.0] * 6, "sell_price_main": [1.0] * 6})
    r = compare_policies(df, "pred", "p80", "naive").iloc[0]
    # identical forecasts + identical rule → identical outcomes; perfect forecast → no waste
    assert r.model_waste == r.baseline_waste == 0 and r.model_ordered == r.baseline_ordered == 30


def test_promotion_rejected_when_one_week_is_lost():
    base = {"naive": _eval(0.3, [0.3] * 3)}
    champ = _eval(0.20, [0.2, 0.2, 0.2])
    good = _eval(0.18, [0.18, 0.18, 0.2])
    d, reason, _ = decide(champ, good, base, {"wape": 0.15}, halves=[(0.21, 0.16), (0.19, 0.20)])
    assert d == "rejected" and "week" in reason
    d, _, _ = decide(champ, good, base, {"wape": 0.15}, halves=[(0.21, 0.19), (0.19, 0.17)])
    assert d == "promoted"


def test_surplus_ladder_prefers_selling_then_markdown_then_donation():
    from forecaster.decisions.impact import ladder
    assert ladder(0, 20, None, 0)["action"] == "sell"
    assert ladder(5, 20, 30, 0.3) == {"action": "markdown", "donate_units": 0.0}  # +10 demand clears 5
    r = ladder(15, 20, 30, 0.3)
    assert r["action"] == "markdown_then_donate" and r["donate_units"] == 5
    assert ladder(4, 20, None, 0)["action"] == "donate"


def test_impact_uses_cited_factors():
    from forecaster.decisions.impact import CO2E_KG_PER_KG, KG_PER_MEAL, units_to_impact
    assert 2.6 < CO2E_KG_PER_KG < 2.7 and 0.5 < KG_PER_MEAL < 0.6  # WRAP 16/6.0; ReFED ≈1.22 lb/meal
    out = units_to_impact({"Bakery": 100.0})  # 100 × 0.1 kg assumed
    assert out["kg_assumed"] == pytest.approx(10.0) and out["co2e_kg"] == pytest.approx(10 * CO2E_KG_PER_KG)


def test_discount_chosen_by_money_not_just_clearance():
    from forecaster.decisions.markdown_plan import plan
    # 10/day demand at price 1, 30 units that expire after 2 days → 10 would spoil.
    # 20% off lifts demand 50%: sells all 30 (revenue 24) vs. full price 20 revenue − disposal → discount wins.
    p = plan([(30, 2)], 10, price=1.0, lift_at={0.2: 0.5})
    assert [(s.discount, s.day) for s in p.steps] == [(0.2, 0)] and p.donate_units == 0
    assert p.money_plan > p.money_no_action
    # Same surplus but a weak response (20% off → only +5%): the discount would give away more on the
    # units that sell anyway than it saves → no discount, donate the surplus.
    p = plan([(30, 2)], 10, price=1.0, lift_at={0.2: 0.05})
    assert p.steps == [] and round(p.donate_units) == 10


def test_discount_starts_as_late_as_pays():
    from forecaster.decisions.markdown_plan import plan
    # 25 units over 2 days at 10/day → 5 spare. A 1-day discount on the last day is enough, so it
    # shouldn't discount tomorrow's full-price sales too.
    p = plan([(25, 2)], 10, price=1.0, lift_at={0.2: 0.5})
    assert [(s.discount, s.day, s.days) for s in p.steps] == [(0.2, 1, 1)]


def test_lift_is_made_monotone():
    from forecaster.decisions.markdown_plan import monotone
    assert monotone({0.2: 0.3, 0.3: -0.1, 0.6: 0.2})[0.3] == 0.3  # a deeper discount never sells less


def test_older_cohorts_sell_first():
    from forecaster.decisions.markdown_plan import project_unsold
    # 10/day: the 1-day cohort (12) leaves 2 unsold; the 3-day cohort gets 30 capacity minus 10 used
    assert project_unsold([(12, 1), (15, 3)], 10) == [(2, 1)]


def test_stockout_days_detected_from_owner_sheets():
    from forecaster.pipeline.store_learning import stock_availability
    panel = pd.DataFrame({"product_id": ["a"] * 3, "date": pd.date_range("2026-09-20", periods=3),
                          "units_received": [20, 10, 10], "sales": [15, 15, 8], "stock_end": [5, 0, 2]})
    # day 2: 5 left + 10 delivered, all 15 sold, shelf empty → stockout (sales capped)
    assert stock_availability(panel).tolist() == [1.0, 0.5, 1.0]


def test_chat_flags_numbers_that_are_not_in_the_data():
    from forecaster.chat import unverified_numbers
    pack = "WAPE 20.4%, bias +13.7%. graded_forecasts=3340\nproduct,p50\nGrape_10,1423\nshare,70%\n"
    assert unverified_numbers("Error 20.4% over 3,340 forecasts; Grape_10 sells 1,423 units; 70% sold", pack) == []
    assert unverified_numbers("MAE is 53.32 and RMSE 279.32 across 3 products", pack) == ["53.32", "279.32"]
    csv_pack = "product,received,sold,wasted\nGrape_10,46728,32501,14227\n"  # CSV commas are separators
    assert unverified_numbers("Grape_10: 14,227 units wasted of 46728 received", csv_pack) == []


def test_lowest_clearing_discount_is_the_smallest_level_that_sells_everything():
    from forecaster.decisions import markdown_plan as mp
    lift = {0.2: 0.10, 0.3: 0.25, 0.35: 0.30, 0.45: 0.40, 0.5: 0.50, 0.6: 0.60}
    # 125 units, 2 days left, demand 50/day: 25 would spoil; 30% off sells 62.5/day → clears all.
    c = mp.lowest_clearing_discount([(125.0, 2)], 50.0, 1.0, lift)
    assert c.discount == 0.3 and c.money_vs_no_action is not None
    assert mp.lowest_clearing_discount([(80.0, 2)], 50.0, 1.0, lift) is None  # nothing would spoil
    none = mp.lowest_clearing_discount([(400.0, 2)], 50.0, 1.0, lift)  # even 60% off sells only 160
    assert none.discount is None and round(none.waste_at_deepest) == 240


def test_synthetic_dairy_and_eggs_follow_real_traffic_and_are_reproducible():
    from forecaster.data import synthetic
    panel = _panel(days=120)
    panel["series_id"] = panel["series_id"].astype(str)
    a, b = synthetic.ensure(panel), synthetic.ensure(panel)
    assert a.equals(b) and synthetic.ensure(a).equals(a)  # deterministic and idempotent
    syn = a[a["series_id"].str.startswith(synthetic.SYNTHETIC_PREFIX)]
    assert set(syn["category"]) == {"Dairy products", "Eggs"} and syn["product_id"].nunique() == 14
    assert (syn["sales"] >= 0).all() and syn["sales"].mean() > 0
    assert len(a) - len(panel) == len(syn)  # real rows untouched


def test_briefing_template_is_four_bullets():
    from forecaster.briefing import template
    f = {"total_forecast_units": 1234, "forecast_date": "2024-06-03", "items_at_high_waste_risk": 2,
         "waste_risk_items": [{"name": "Eggs_1", "suggested_markdown_pct": 20}], "largest_orders": [{"name": "Dairy_2", "order_qty": 90}],
         "recent_model_error": {"days": 14, "wape": 0.12, "bias": -0.02}}
    lines = template(f).splitlines()
    assert len(lines) == 4 and all(l.startswith("- ") for l in lines) and "20% off" in lines[1]


def test_design_choices_never_see_the_test_window():
    """Three-way split: experiments score (CUTOFF, SELECT_END] on real series; the production test
    window (the last 28 days, 2024-05-06 → 2024-06-02) starts after it."""
    from forecaster.pipeline.experiment import CUTOFF, SELECT_END, selection_rows
    from forecaster.pipeline.train_production import HOLDOUT_DAYS
    days = pd.date_range("2024-01-01", "2024-06-02")
    rows = pd.DataFrame({"date": np.repeat(days, 2), "series_id": ["Prague_1|1", "SYN-Prague_1|Eggs|0"] * len(days)})
    sel = selection_rows(rows)
    assert sel["date"].min() > CUTOFF and sel["date"].max() == SELECT_END
    assert not sel["series_id"].str.startswith("SYN-").any()
    assert days.max() - pd.Timedelta(days=HOLDOUT_DAYS) >= SELECT_END
