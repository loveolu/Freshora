# Devpost write-up (draft)

**Main track: A Marina's Mission (Social Good).** MLH: Tiger Data (TimescaleDB), Gemini, Vultr, .Tech.
Hosted demo (freshora.tech) is offline since the hackathon; screenshots are in the README.

## Tagline
Keep fresh food out of the bin: forecasts that act on themselves — and prove they were right.

## Inspiration
Roughly a third of the food supply goes unsold or uneaten, and wasted food carries its whole
life-cycle footprint to the landfill. Fresh departments still order on gut feel or "same as last
week". Forecasting vendors sell predictions, but a store can't see whether yesterday's prediction
was right — or whether this week's "improved" model actually is. We wanted a forecaster that
prevents waste first, earns trust by showing its track record, and routes what can't be sold to
people instead of the bin.

## What it does
- **Prevents waste at the source.** Tomorrow's forecast per fresh product (bakery, produce, meat and
  fish, dairy, eggs) drives a shelf-life-aware order quantity, net of stock already on the shelf. A
  slider sets how much the store fears waste vs. empty shelves.
- **Plans what's left** along the EPA Wasted Food Scale: for each product at risk it shows the
  discount that keeps the most money, the lowest discount at which everything still sells, and what
  to donate.
- **Answers questions.** Owners ask in plain words ("do I need any promotions today?", "what did I
  waste most of?") and Gemini answers only from the store's own plan, sheets and forecast record;
  every number in a reply is checked against that data before it's shown. A four-point morning
  briefing is written the same way.
- **Logs every forecast before the day's sales arrive** and attaches what actually sold (the
  Ledger), so accuracy is a track record, not a claim.
- **Learns under governance.** Owners upload daily sheets (one row per product per day, or a
  delivery sheet with expiry dates); once a store has enough new days the model retrains on its own.
  A new model is promoted only if it beats the current one by ≥2% on days neither has seen, doesn't
  lose in either week of that window, doesn't make any major category >5% worse, and beats two naive
  baselines — otherwise it's rejected with the reason recorded. Every version is kept.

## Results
Headline test: the served model on the last four weeks of real sales it never trained on
(2024-05-06 → 2024-06-02, 7 warehouses, real products only). Three-way time split: every design
choice was made on 2024-01-20 → 2024-05-05, so the test weeks were never used to choose anything.
- **13.7% forecast error (WAPE), i.e. 86.3% accurate**, vs 22.5% for "same weekday last week" —
  39% less error. Bakery 88.8%, produce 83.6%, meat and fish 77.0%.
- **Demo stores, live:** four weeks of day-by-day forecasts, each saved before that day's sales:
  Atlanta 86.6%, Augusta 85.1%, Macon 80.7% accurate.
- **Simulated waste vs. naive ordering at a realistic service level (≈5% lost sales):** ≈87% less
  waste *and* ≈34% fewer lost sales — waste isn't cut by simply ordering less. (FIFO shelf-life
  simulator; no public dataset records waste.)
- An earlier model generation, replayed Jan–Jun 2024 with weekly governed retraining: 14.4% error vs
  24.2% for "same weekday last week"; 10 challengers trained, 6 promoted, 4 rejected.
- Holiday-proximity features cut pre-Easter error from 22.0% to 18.3%; seven discount types cut
  error 15.45% → 15.11% (on the selection window).
- Measured and **rejected**: dropping stockout days as targets, a produce specialist, per-store
  calibration, spike weighting, one multi-quantile model. Weather didn't help one day ahead.

## How we built it
- **Data:** real Rohlik e-grocer sales (Kaggle; 7 warehouses; fresh products; 2021–2024) with prices,
  seven discount types, availability, and each warehouse's daily order count as customer traffic.
- **Weather done right:** Open-Meteo archived forecasts for training (not observed weather, so no
  leakage) and the live forecast for each store's city when serving.
- **Model:** one global XGBoost demand model (Tweedie) across all stores and products, plus an
  80th-percentile model for order sizing; one feature function shared by training and serving,
  guarded by a feature-schema version.
- **Stack:** Python/FastAPI; Tiger Cloud (Postgres + TimescaleDB hypertables and a continuous
  aggregate of forecast error); React + TypeScript + Recharts; Gemini for the chat and briefing; one
  Docker image on Vultr behind freshora.tech with automatic HTTPS.

## Impact factors (cited)
- 2.67 kg CO2e per kg of wasted food — WRAP UK 2021-22 (≈16 Mt CO2e / 6.0 Mt).
- ≈0.55 kg (1.22 lb) of food per meal — derived from ReFED (29% of 240 M tons ≈ 114 B meals).
- Unit weights per category are **assumptions** (Rohlik records pieces or kg by product).

## Challenges we ran into
(Full log with numbers: docs/LESSONS.md.)
- **Every challenger was rejected in our first replay.** Early stopping held out the newest seven
  weeks, so new data was never learned from. Refitting on the full snapshot fixed it.
- **Easter broke the model.** A 0/1 holiday flag can't say "Easter is in two days". Holiday-proximity
  features cut pre-Easter error from 22.0% to 18.3%.
- **Our first waste numbers were flattering.** We moved to a realistic service level and now report
  waste and lost sales together.
- **The cheapest-looking markdown wasn't the cheapest.** A discount also applies to units that would
  have sold anyway, so markdowns are chosen by money using the model's predicted sales at each depth.
- **Dairy and eggs were stuck at 74%.** We measured the ceiling: even a forecaster that knew the true
  expected demand scored ~75%, because our synthetic data was five times noisier than real staples.
  Calibrating the noise to measured bakery noise (not to a target) gave 84%.
- **Uploaded stores scored lower than the warehouses they replay.** The upload path gave the model
  different inputs (no subcategory, closures or discount types); onboarding each store's history into
  training closed part of the gap.

## What's next
Connect a real store's point of sale, real food-recovery partners, and replace the model's
discount-response estimate with each store's measured response as its promotion experiments add up.

## Honesty notes
Waste, inventory and impact are simulated/derived. Rohlik is an online grocer, so warehouses stand in
for stores. The Atlanta, Macon and Augusta demo stores replay the three largest warehouses' real sales
at Georgia locations, 121 weeks later (so tomorrow's plan is for 27 September 2026). Dairy and eggs
are synthetic. AI tools: Gemini (store chat and briefing text) and Claude (development assistance);
the forecasting, governance and data pipeline are ours.

## Built with
python, fastapi, xgboost, pandas, timescaledb, tiger-data, postgresql, react, typescript, recharts,
open-meteo, gemini, vultr, docker, caddy, .tech

> Add Cursor / SpaceXAI or Notability only if the team actually used them (Notability needs at least
> two screenshots on Devpost).
