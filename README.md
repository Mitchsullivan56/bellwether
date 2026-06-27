# Bellwether 🏈

**Model-driven fantasy football rankings that publish weekly and grade themselves in public.**

A from-scratch projection model, backtested out-of-sample against expert consensus — and
honest about exactly where it wins and loses.

> **4% lower error** than a trailing-average baseline (19% better than last-game),
> measured on **21,000+ player-weeks the model never trained on** (2021–2025) — and
> **calibrated**: a 15-point projection really does average ~15 actual points.

🔗 **Live dashboard:** _add your Streamlit URL here_ · 📈 Landing page in [`docs/`](docs/)

---

## What it does

An interactive [Streamlit](https://streamlit.io) dashboard with eight views:

| Page | |
|---|---|
| **2026 draft board** | PPR rankings (expert consensus) enriched with Bellwether's expected-production lens — tiers, value-vs-ADP, model-vs-situational labels, updated daily |
| **Draft track record** | A 4-season holdout test of the draft method — published honestly, including where the value lean *doesn't* help |
| **Weekly rankings** | In-season, the model's projected points for the upcoming week (lights up once 2026 games are played) |
| **Accuracy vs experts** | Rank-correlation and MAE vs FantasyPros consensus and two baselines, plus a calibration plot |
| **Projections vs actual** | Browse any historical week: projection vs expert rank vs what actually happened |
| **Why trust it / Why it works** | Methodology in plain English, and permutation importance explaining every prediction |

## Why it's credible

- **Walk-forward, leakage-free.** Every season is predicted using only earlier seasons;
  every feature for week *w* uses only data available before kickoff. No hindsight.
- **Benchmarked, not asserted.** Scored head-to-head against FantasyPros expert consensus
  (100+ analysts) on the same weeks, plus deliberately-hard naive baselines.
- **Honest about limits.** It beats consensus at RB, is close at WR/TE, and **loses at QB** —
  and the app shows you exactly where. The draft "value" signal backtests to ρ≈0.07
  (a weak tilt), and that's stated plainly rather than dressed up.
- **Explainable.** Accuracy is driven by *opportunity* signals — snap share, target share,
  expected points (TD luck removed) — and the Vegas game environment, shown via permutation
  importance.

## How it's built

`nflverse` (weekly stats, snaps, injuries, Vegas lines, Next Gen Stats, expected points) +
FantasyPros consensus → leakage-free feature engineering → a walk-forward
`HistGradientBoostingRegressor` with nested-CV tuning → backtested against experts.

```
build_dataset.py       pull data, build the leakage-free feature table
build_draft_board.py   2026 PPR draft board (consensus + expected-production lens)
backtest_draft.py      4-season holdout test of the draft method
ff_model.py            walk-forward training + tuning
ff_eval.py / ff_cache.py   metrics, calibration, cached compute
app.py                 the Streamlit dashboard
```

A scheduled GitHub Action ([`.github/workflows/update.yml`](.github/workflows/update.yml))
rebuilds the rankings daily and commits them, so the live app stays current.

## Run it locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py      # or double-click run.cmd on Windows
```

Python 3.13. First launch trains the model once (~15s) and caches it; later launches are instant.

## Tech

Python · pandas · scikit-learn · Streamlit · Plotly · nflreadpy · joblib

---

*Built by Mitchell Sullivan. Data via [nflverse](https://github.com/nflverse) and FantasyPros.
Not affiliated with the NFL.*
