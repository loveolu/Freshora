# Struggles and lessons (HackGT 13 build log)

What went wrong, how we found it, what we changed, and the measured result. Numbers come from the
generated files in `artifacts/` unless noted.

## Data

| Struggle | What we learned / did |
|---|---|
| The Kaggle Rohlik data needs a logged-in account and accepted competition rules. | Found the same data as Parquet in the fev-bench mirror on HuggingFace (`autogluon/fev_datasets`) — no credentials, reproducible download. |
| Two "customer" series disagreed: the sales file's `total_orders` vs the orders file's `orders` (correlation 0.95, not identical). | Used the sales file's series (covers every day we forecast) as the traffic feed; kept the other as a cross-check and documented the difference. |
| No public grocery dataset records waste or expiry. | Built a transparent first-in-first-out shelf-life simulator and labeled every waste number "simulated". Real waste only comes from owner uploads. |
| Rohlik records sales in pieces *or* kg depending on the product. | Weight-based impact (kg, CO2e, meals) needs assumed unit weights per category — shown as assumptions next to the numbers. |
| The sample store is built from a warehouse the model already knows. | Its dramatic retrain gain (25.3% → 11.3% error) is partly familiarity; we say so wherever it's shown. |

## Weather

| Struggle | What we learned / did |
|---|---|
| Training on observed weather but serving on forecasts would make offline accuracy look better than real life. | Trained on archived forecasts; for the 2024 replay used the forecast actually issued the day before (Open-Meteo Previous Runs). |
| Previous Runs data starts 2024-01-20, not Jan 1, and has no daily variables. | Replay window starts 2024-01-20; daily values are aggregated from hourly `_previous_day1` data. |
| Weather didn't help — measured three times (16.17% vs 16.06%; 15.24% vs 15.09%; 14.20% vs 14.14% without). | For an online grocer, rain doesn't stop anyone ordering. We report it rather than claim a benefit; the pipeline keeps weather for stores where it might matter. |

## Training and evaluation

| Struggle | What we learned / did |
|---|---|
| **First replay: every challenger was rejected**, even with more data. | Early stopping held out the newest 7 weeks, so the new data was only ever used for validation, never learned from. Fix: pick the tree count on that window, then refit on the whole snapshot. Result: 6 of 10 challengers promoted, error falling from ~16% to ~13% on recent windows. |
| **Easter:** error spiked on the days *before* the holiday (Mar 27–28: 24–25% error, −20% under-forecast). | A 0/1 "is it a holiday" flag can't say "Easter is in two days", and serving set tomorrow's holiday flag to 0 even though holidays are known in advance. Added holiday-proximity features from the public calendar: pre-Easter error 22.0% → 18.3%. |
| Most "obvious" accuracy ideas didn't work. | Measured on a fixed harness, kept only if ≥0.3 points better with no category >5% worse: seven discount types **kept** (15.45% → 15.11%); dropping stockout days as targets, a produce specialist, per-store calibration (+0.28, just under the bar) and spike weighting all **rejected**. |
| A "simplify to one model" idea looked obviously good. | One model giving both the median and the 80th percentile was 2.2 points worse (bakery +27%) — rejected. Dropping the separate traffic-forecast model was simpler *and* better (15.26% → 15.11%) — adopted. |
| **Our design experiments peeked at the test.** The studies above were first scored on 2024-01-20 → 06-02, which overlapped the final four-week test (05-06 → 06-02), and the synthetic dairy/eggs noise level was measured on the test weeks. That's selection bias: the headline would be slightly optimistic. | Three-way split, enforced in code and by a test: train ≤ 01-19, **select** 01-20 → 05-05, **test** 05-06 → 06-02 (production training refuses a test window that overlaps selection). Reran every study on the selection window only: every decision came out the same and the noise level was unchanged (k = 62), so the served model and its 13.7% test error stand — with no selection bias. The numbers in the two rows above are from the clean rerun. |
| A single lucky fortnight could promote a model. | Promotion now also requires the challenger not to lose in either week of its evaluation window. |
| A model trained on one feature list could be served on another. | Every model records its feature-schema version; serving and retraining refuse a mismatch. |

## Decisions (orders, markdowns, waste)

| Struggle | What we learned / did |
|---|---|
| The waste simulator ordered a full day's amount on top of stock already on the shelf. | Waste was inflated. Orders are now net of stock that is still sellable tomorrow, with the same rule for our model and the baseline, so the comparison measures forecast skill only. |
| **Our waste numbers were flattering.** The default cost setting ran the simulated store at ~13% stockouts, so waste looked tiny. | Moved the default to a realistic ~5% lost-sales service level and report waste *and* lost sales together: −87% waste and −22% lost sales vs "same as last week". |
| The demo plan showed no waste risk at all. | Starting stock came from our own conservative policy, so shelves were nearly empty. Install-day stock now comes from the store's legacy habit (last week + buffer), labeled. |
| "Smallest discount that clears the surplus" was wrong — a teammate's insight. | A discount applies to every unit sold, including ones that would have sold at full price. Now every discount level and start day is simulated with the trained model's predicted sales, and the option that keeps the most money wins — sometimes "no discount pays for itself, donate". |
| The model's estimated lift was sometimes zero or negative at a deeper discount. | Lift is made monotone (a deeper discount never sells less) before planning. |
| "Unsold today = spoiled" would be wrong: deliveries arrive on different days with different shelf lives. | Stock is tracked per delivery (cohorts): owner batch sheet when available, otherwise first-in-first-out from the daily sheet with per-product shelf lives. |

## Learning from real stores

| Struggle | What we learned / did |
|---|---|
| New stores start badly (cold start: 21–25% error). | Owner uploads are graded against the forecasts made for them, then retrain the model. |
| Store-specific models vs one model. | Switched to **one global model**: a store's uploads retrain it, and it is promoted only if it is better for that store **and** no worse for the existing stores. |
| A later retrain would have dropped earlier stores' uploads. | Every retrain now includes every store's uploaded sheets. |
| "Learned from this store" showed yes before any retraining. | It compared dates; it now reads the model's actual training record. |
| **Tomorrow's forecast was served without its traffic inputs.** Found while wiring the new dashboard's "customers per day" card, which showed a dash. | Traffic features were built only up to the last *observed* day, so tomorrow's row (whose lags and rolling means are already known) got blanks. Training and backtests were unaffected; only served plans were. The model had been trained with traffic dropout, so the damage was small (≈1% per store), but it was real. Fixed at the source, with a regression test, then tomorrow was re-planned with the same champion. |
| Every retrain re-planned tomorrow on top of the old forecasts: 4 model versions' forecasts for the same day. | A re-plan now replaces that day's forecasts that have no outcome yet; graded ones are never touched. |
| Stockout days make sales look lower than demand. | Detected from the sheets (shelf ended empty after selling everything on hand) and fed to the model's in-stock inputs. |

## Engineering

| Struggle | What we learned / did |
|---|---|
| Windows console crashed the pipeline on a "≥" in a log line. | Force UTF-8 output in every CLI. |
| Removing a feature crashed the replay at its first ledger write. | Grep for every reference before removing a column; the fix took one line, the rerun 45 minutes. |
| An interrupted command left partial uploads in the real database. | Cleaned them up and moved all experiments to isolated database copies. |
| Docker image was 3.3 GB. | XGBoost's Linux wheel pulled CUDA libraries a CPU server never uses → `xgboost-cpu`; the image dropped to 1.1 GB. Then learned uploads need the model files on the server → models and training data go on a persistent volume. |
| TimescaleDB refused to turn tables with ID keys into hypertables. | Hypertables need the time column in every unique key: observations use natural keys (store, product, date); graded forecasts are copied into a dedicated `forecast_errors` hypertable with a continuous aggregate. |
| Re-copying data into Postgres failed on foreign keys. | Clear child tables before parents, then copy parents before children. |
| **First live test on Tiger Cloud: the briefing timed out (>120 s).** | The database answered in 0.08 s; the time was the transfer (the free tier measured ~170 KB/s from the laptop) of all ~102k backtest forecasts, downloaded to compute one 14-day error. Every ledger read now filters, limits and sums in SQL; results were checked identical to the old pandas path (to 1e-13). |
| After copying into Postgres, a rollback said "done" but the dashboard still showed the old champion. | Copied rows kept their ids but Postgres' id sequences didn't move, so the new row got id 1 and "latest by id" picked the old champion (and later inserts would have collided). The copy now advances every sequence. |
| An owner chat can quietly invent figures, and prompts alone don't stop it. | Every number in a chat reply is matched against the data it was given (rounding and % allowed); unmatched figures trigger one rewrite, then a visible warning. Building the check also caught our own bug: CSV commas were being read as thousands separators. |
| Gemini 3.5 Flash took 12–22 s per briefing and sometimes returned 503 "high demand". | The briefing only rewords computed numbers: switched to 3.5 Flash-Lite (1.5–2 s), with 2.5 Flash as a fallback and the template as the last resort. |
| The copy script read its *source* from `DATABASE_URL`, which by then points at Postgres; tests would have picked up real keys from `.env`. | The copy always reads the local SQLite file; tests set `FORECASTER_NO_DOTENV=1` and never read `.env`. |

## Research and strategy

| Struggle | What we learned / did |
|---|---|
| The name "WasteLess" belongs to a funded company selling AI markdowns to grocers. | Rename before submission. |
| Most ideas we researched already existed — six of eleven had been shipped or built at hackathons within months. | Novelty comes from the combination and the rigor (a visible, governed learning loop on real data), not the category. |
| Confused sponsor challenges with main tracks. | One main track (A Marina's Mission); sponsor challenges are separate and only worth tagging if actually built. |
| Web-search budget ran out mid-research. | Fell back to direct page fetches, Devpost/GitHub/arXiv/HN — and said so in the report. |

## Round 3: demo chain, new categories, UI rebuild

| Struggle | What we learned / did |
|---|---|
| The team wanted Dairy and Eggs; the dataset has neither. | Generated them (driven by each warehouse's real customer counts and calendar, with promotions and noise), labeled them synthetic, and kept them **out of the headline accuracy** — accuracy on data you generated proves nothing. Real-series error after retraining: 14.4% vs 22.5% last-week. |
| Dairy/eggs stuck at 74% accurate after retraining. | Measured the ceiling: a forecaster that knew the true expected demand scored only ~75%, so the model was already at the limit set by our generator's noise, which was 5× noisier than real staples (overdispersion 1/12 vs ≤1/62 measured on real bakery residuals). Calibrated the noise to the measured bakery level (not to a target): 83.9% dairy, 84.4% eggs. Measure the ceiling before tuning a model. |
| Demo stores scored 78–84% live while the model scores 85.8% overall. | For the same products and days, the upload path gave the model different inputs than training did (no subcategory, closures, discount types; a different stockout measure) — worth ~6 points together. Onboarding (store history in training) added ~1 point; per-store accuracy is also capped by the store: the model scores 83.4% on Prague_2's own data. The chain now replays the three largest warehouses, disclosed on the About page. |
| Georgia stores with no Georgia data. | A demo chain that replays real warehouses' sales at Georgia locations through the normal upload path, one day at a time, so every forecast in the Ledger was saved before its day's sales. Holidays follow the source warehouse's calendar, because that is what the sales followed. |
| Two screens restyled each other. | The new dashboard reused the `.kpis` class of Today's plan (and `.notes` of the manager-notes list); the overrides also caused the "extra gap" the team noticed. Unique class names fixed both. |
| One upload box for two kinds of sheet. | Detect the sheet by its header (expiry/batch columns → delivery sheet). The first version read the file with the daily-sheet parser, which fails on delivery columns and silently routed them to the wrong importer; a test caught it. |
| Owners type categories their own way. | "Dairy", "Fruits and vegetables", "Meat & fish" are mapped to the model's names before validation instead of being rejected. |

## What we'd tell the next team
1. Measure everything against a simple baseline; most clever ideas don't beat it.
2. Promotion rules matter more than the model: a new model must *earn* its place on data it hasn't seen.
3. Watch for flattering numbers — check the other side of every trade-off (waste vs lost sales).
4. Label what's simulated, every time; it's the first thing a good judge asks.
5. Test risky operations on a copy of the data.
