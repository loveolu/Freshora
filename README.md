# Freshora

**Live: https://freshora.tech**

Eliminating food waste, one supermarket at a time


## Results

Tested on the last four weeks of real sales the model never saw (2024-05-06 → 2024-06-02; Rohlik
e-grocer, 7 warehouses, real products only):

- **13.7% forecast error (WAPE)**, i.e. 86.3% accurate by volume, vs 22.5% for "same weekday last week" (39% less error).
- **Three-way time split:** train ≤ 2024-01-19, every design choice made on 2024-01-20 → 05-05, test 05-06 → 06-02.
  The test weeks were never used to choose anything; features only use data available the day before (both enforced by tests).

WAPE = Σ|forecast − actual| / Σ actual. Details: [docs/DEVPOST.md](docs/DEVPOST.md) ·
struggles and what we learned: [docs/LESSONS.md](docs/LESSONS.md) ·
reports: [`artifacts/production_report.json`](artifacts/production_report.json).
