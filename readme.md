# PSX ExpertAdvisor

A **regime-aware, explainable stock advisor for the Pakistan Stock Exchange (PSX)**. It figures out what kind of market we're in, estimates the probability that each stock goes up over the next 5 trading days, turns that into a BUY / HOLD / SELL call, builds a portfolio under real constraints, and tells you *why* in plain language.

> AI course project, NUST SEECS. <!-- EDIT: add team member names + course code -->

**Live demo:** <!-- EDIT: Render URL -->

---

## 1. The problem

Most retail investors on PSX get one of two things:

- raw charts and indicators with no interpretation, or
- "tips" and black-box predictions with no reasoning and no honest estimate of how sure the system is.

PSX is also a volatile, thinly-traded, news-driven market, so a signal that works in a calm uptrend can fail badly in a crash. A single fixed set of rules (or a single model) treats every day the same, and that's the gap this project goes after.

## 2. What's novel here

1. **Regime-aware decisions.** A Hidden Markov Model infers the hidden market state (3 states) from observable features, so the advisor behaves differently in different market conditions instead of using one rule set for everything.
2. **Calibrated probabilities, not just labels.** A naive Bayes classifier outputs P(up), and a calibration step maps it to the *observed* frequency. If the system says 70%, it should have been right about 70% of the time historically.
3. **Per-stock decision thresholds with a HOLD band.** Each stock gets its own threshold θ, and a band around it makes the system say HOLD when it isn't confident, instead of forcing a BUY or SELL on every stock every day.
4. **Constraint-based portfolio building.** A CSP (backtracking) decides which stocks are *eligible* (sector limits, risk profile, etc.), then Linear Programming computes the actual weights under caps. This replaces hand-written "pick 5 / 4 / 3" rules.
5. **Explainability by construction.** Every recommendation comes with its regime, its probability, and the reasons behind it, because the models are transparent probabilistic models (no neural-net black box).
6. **Honest evaluation.** Walk-forward backtesting (no look-ahead), compared against the original rule-based engine and buy-and-hold, including transaction costs. Causal features are unit-tested for future leakage.

## 3. State of the art and where we sit

| Approach | What it does | Weakness for this use case |
|---|---|---|
| Technical-indicator rule systems (RSI, MACD, moving-average crosses) | Simple, easy to read | Same rules in every market condition; no probability estimate |
| Deep learning price predictors (LSTM / Transformer) | Can fit complex patterns | Needs lots of data, overfits easily on short and noisy PSX histories, unexplainable |
| Reinforcement-learning traders | Learn a trading policy directly | Unstable, data-hungry, hard to trust or explain |
| News / sentiment models | Add information the price doesn't have yet | Noisy; usually used as one input, not a full decision system |
| Regime-switching models (Markov-switching / HMM, classic in econometrics) | Model changing market states | Usually stop at regime detection and don't produce actionable portfolio decisions |

**Our position:** we take the classical, well-studied regime-switching idea (HMM) and connect it end to end: regime → calibrated probability → thresholded decision → CSP eligibility → LP allocation → explanation. The contribution is the integrated, explainable pipeline, evaluated honestly on PSX data, not a bigger model. <!-- EDIT: add 2-3 real citations from your proposal's references (e.g. Hamilton 1989 on regime switching, a naive Bayes/HMM trading paper) -->

## 4. How it compares to existing tools

<!-- EDIT: make sure each claim below is true of the specific tools you name; remove any you can't back up -->

| | Charting platforms / broker apps | Generic ML "predictors" | **PSX ExpertAdvisor** |
|---|---|---|---|
| Understands market regime | No | Rarely | **Yes (HMM)** |
| Gives a calibrated probability | No | Sometimes, uncalibrated | **Yes** |
| Says "not sure" (HOLD band) | No | Rarely | **Yes** |
| Explains each decision | Partly | No | **Yes** |
| Builds a constrained portfolio | No | No | **Yes (CSP + LP)** |
| Validated without look-ahead | N/A | Often not | **Yes (walk-forward)** |
| Built for PSX specifics (sectors, costs) | Partly | No | **Yes** |

## 5. Results

<!-- EDIT: fill from reports/results/ after running backtest.py. Do not invent numbers. -->

| Strategy | Return | Sharpe | Max drawdown | Win rate |
|---|---|---|---|---|
| Buy-and-hold | | | | |
| Rule engine (baseline) | | | | |
| **AI engine (HMM + Bayes + CSP/LP)** | | | | |

Test period: <!-- EDIT: dates --> · Transaction costs included: <!-- EDIT: value from config.py -->

## 6. Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3 |
| Backend API | Flask, served by gunicorn |
| Numerics / ML | NumPy, pandas, SciPy (`linprog` for the LP) <!-- EDIT: confirm against requirements.txt --> |
| Models | HMM (Baum-Welch, forward filter, Viterbi), naive Bayes with calibration, **written from scratch** in `backend/core/models/` |
| Optimisation | CSP with backtracking, Linear Programming |
| Data | PSX price data via `data_fetcher.py` <!-- EDIT: name the data source -->, news via RSS (`news_fetcher.py`) |
| Frontend | HTML, CSS, vanilla JavaScript <!-- EDIT: name the charting library you use --> |
| Deployment | Render (`render.yaml`) |
| Testing | pytest <!-- EDIT: confirm --> |

## 7. Workflow

### Offline (training and evaluation, never called by the API)

```
raw prices ──► features.py (causal features, 5-day target)
           ──► train_models.py ──► HMM params, Bayes tables, calibration, θ per stock
           ──► backend/artifacts/*.json   (small, committed to git)
           ──► backtest.py     (walk-forward: rules vs AI vs buy-and-hold)
           ──► band_report.py  (P(up) histogram, BUY/HOLD/SELL coverage)
```

### Online (what happens on each request)

```
price data ──► features ──► HMM filter ──► current regime
                                  │
                                  ▼
                  naive Bayes ──► P(up) ──► calibration
                                  │
                                  ▼
                   per-stock θ + HOLD band ──► BUY / HOLD / SELL
                                  │
                                  ▼
                   CSP: which stocks are eligible?
                                  │
                                  ▼
                   LP: portfolio weights under caps
                                  │
                                  ▼
                   explanation ("why") ──► JSON ──► frontend
```

The original rule-based engine (`rule_engine.py`) is kept as the baseline, and the UI has an engine toggle so you can compare both side by side.

## 8. Project structure

```
backend/
  api/app.py            Flask app: engine switch, endpoints, cache
  core/                 runtime code the API imports
    config.py           tickers, sectors, costs, risk profiles, band
    features.py         causal features, buckets, 5-day target
    thresholds.py       per-stock θ, decide(), band coverage
    prob_engine.py      glue: features → HMM → Bayes → decision
    rule_engine.py      baseline (never delete)
    csp.py              eligibility + backtracking
    lp_allocator.py     LP weights
    portfolio_ai.py     portfolio builder using CSP + LP
    models/             hmm.py, bayes.py (code only, no data)
  artifacts/            trained outputs (small JSON)
  scripts/              offline jobs: train, backtest, reports
  tests/                leakage, HMM sanity, thresholds, CSP, LP
frontend/               index.html, app.js, styles.css
reports/                backtest results, final report, slides
docs/                   architecture, design decisions, pitfalls
```

## 9. Design decisions and scope

- **3 regimes, 5-day horizon:** chosen to balance enough resolution against the amount of PSX data available (reasoning in `docs/decisions.md`).
- **Techniques are limited to those in the course proposal.** Fuzzy logic and genetic algorithms are deliberately **not** used.
- **No look-ahead:** features are causal, and tests check that nothing leaks from the future.
- **Not financial advice.** This is an academic project; its output should not be used to make real investment decisions.

## 10. Running it locally

```bash
# 1. create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate      # Git Bash on Windows (use venv/bin/activate on Linux/Mac)

# 2. install dependencies
pip install -r backend/requirements.txt

# 3. (optional) retrain models and regenerate artifacts
python -m backend.scripts.train_models

# 4. run the backtest
python -m backend.scripts.backtest

# 5. start the API
gunicorn backend.api.app:app      # on Windows without WSL: python -m flask --app backend.api.app run
```

Then open `frontend/index.html`, or the deployed URL. <!-- EDIT: confirm how the frontend is served in your seed (static from Flask, or opened separately) -->

## 11. Limitations

- PSX histories are short and noisy, so results are sensitive to the train/test split.
- Naive Bayes assumes features are independent, which isn't strictly true.
- Past regime behaviour doesn't guarantee future behaviour.
- Free data sources can have gaps or delays.

## 12. state of Art
Competitors
- FinHisaab: a full-screen charting workspace for PSX stocks with candlesticks, technical indicators and drawing tools, plus stock detail pages with financials and valuation 

- Investify: live prices, financials, alerts, news, demo trading and a portfolio tracker. It’s also described as used by hundreds of thousands of investors.

- FinPocket: an app to trade, analyze and follow the PSX in real time 

- iSCSTrade PRO: technical and fundamental analysis plus a virtual portfolio 

- SmartPSX: portfolio tracking with auto-synced trades and dividends, capital gains calculation and a zakat purification calculator 

- Broker platforms (KTrade, Finalysis by Chase Securities, AZEE Tick): mainly for placing trades. For example, Finalysis offers live quotes, order execution, charts and watchlists. 

- Munir Khanani’s Online Analytics: a financial analytics portal with statements, economy indicators, a market heat map and technical charting, exclusive to their customers 

## How we’re better

Honestly, we’re a different category. They show you data and you decide. We output a decision with reasoning. These are the gaps you can defend:

- They show indicators, we interpret them. Charting tools leave the reading to you. We turn indicators into a probability and a BUY/HOLD/SELL call.
- Regime awareness. Nothing in their listings mentions detecting the market state. Our HMM does, so the same indicator value can mean different things in a crash and in a calm uptrend.
- Calibrated confidence. We say “70%” and check that it was right about 70% of the time. We also say HOLD when unsure.
- Portfolio construction. The trackers record the portfolio you already have. We build one under sector and risk constraints (CSP + LP).
- Explainability and honest testing. Every call has a “why”, and we backtest walk-forward against buy-and-hold and our own baseline.

## 13. Team

- Muhammad Abdur Rehman
   Project Work:

- Noor-Ul-Huda
   Project Work:

- Ahmed Hussain
   Project Work:
