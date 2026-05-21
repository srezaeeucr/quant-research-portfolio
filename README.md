# Quant Research Portfolio — Intraday Equity Strategy Validation

A Python research framework for designing, backtesting, and statistically
validating intraday equity trading strategies. The repository documents
**methodology over results** — emphasizing out-of-sample validation,
multiple-testing correction, and the explicit recognition of overfitting
across a sequence of independent studies.

> The unique contribution is not the strategies themselves (which are textbook)
> but the validation chain applied to them and the willingness to reject
> promising-looking configurations that fail rigorous tests.

---

## Overview

- **14 strategy archetypes** spanning breakout, trend, mean-reversion, momentum, volume, pattern, and composite (ensemble / hybrid filter) families.
- **A four-stage validation chain (V4)** combining walk-forward optimization, Monte Carlo permutation testing, slippage stress testing, and recent-data out-of-sample validation.
- **Multiple-testing correction** (Benjamini-Hochberg FDR and Bonferroni) applied to 1,596 simultaneous configuration tests.
- **Anti-overfitting discipline** documented through case studies of rejected configurations.
- **180-test test suite** covering strategies, risk manager, broker interface, and data layer.

---

## Methodology

### The V4 validation chain

Given 19 symbols × 3 strategies × 4 stop-loss levels × 7 reward-risk ratios = **1,596 configurations**, each is evaluated through:

1. **Quick backtest** on `full_2yr` (Jan 2023 – Jan 2025). Configurations with fewer than 10 trades are excluded.
2. **Walk-forward optimization** with inner parameter sweep on each train window. 10 rolling windows: 6-month train, 3-month test, rolled by 3 months.
3. **Monte Carlo permutation test** comparing the configuration's profit factor against 200 random-entry baselines. Yields a p-value for the entry signal's statistical significance.
4. **Slippage stress test** at 5 levels (0%, 0.02%, 0.05%, 0.10%, 0.20%) across 3 periods (`full_2yr`, `holdout_2022`, `bear_2025`).

A configuration is **V4-validated** when all three pass:
- WF wins ≥ 6 of 10 windows
- MC p ≤ 0.05
- Slippage-adjusted PF at 0.05% ≥ 1.0

**Result**: 100 of 1,596 configurations passed. Across 19 symbols, only 5 had any validated configuration — AMD (24), TSLA (38), META (22), GOOGL (15), MSFT (1).

### Triple and quadruple validation

V4 used data from 2022–2024. To probe regime sensitivity, an **independent recent-data test** (January–April 2026) was added.

- **Triple-validated** (V4 + recent profitable): 30 configurations.
- **Quadruple-validated** (also stable across 12 quarterly windows): 18 configurations.

A striking finding: **META and TSLA had 21 and 22 quarterly-stable configurations through 2024 but ZERO survived 2026 data.** This is direct evidence of regime change. Historical robustness does not guarantee forward performance.

### Multiple-testing correction

With 1,596 simultaneous tests at α=0.05, ~80 false-positive "passers" are expected by chance alone. Two corrections were applied:

- **Benjamini-Hochberg FDR** at q=0.05
- **Bonferroni** at α/N = 3.1 × 10⁻⁵

All 100 V4 passers survived **Bonferroni** (the strictest test), indicating the MC results were robust to multiple comparisons.

### Anti-overfitting discipline

This is the most important methodological commitment of the project. Over a 13-day period, **four different "best AMD configurations"** emerged from successive studies:

| Date | Source | Configuration |
|---|---|---|
| Apr 13 | DEFINITIVE chain | ORB / SL=0.75% / RR=5.0 |
| Apr 15 | Recent-data study | Momentum / SL=0.3% / RR=1.5 |
| Apr 21 | Quad-pass | ORB / SL=0.3% / RR=5.0 |
| Apr 26 | Extended recent | Momentum / SL=1.0% / RR=5.0 |

When the "best" answer changes with every test, **the test framework itself is being tuned, not the strategy**. The discipline applied was to revert to a mid-range, period-agnostic configuration (`SL=0.75% / RR=3.0`) — accepting lower expected backtest performance in exchange for parameter stability across regimes.

See [`docs/case-studies/anti-overfit.md`](docs/case-studies/anti-overfit.md) for the full narrative.

### Documented negative results

Equally important are the studies that did not produce a deployable finding:

| Study | Verdict |
|---|---|
| ATR dynamic stops | Rejected — lost 5/7 combos vs fixed stops |
| Trailing stops | Rejected — lost on holdout |
| Partial exits | Rejected — lost on holdout |
| Time-based exits (COIN) | Rejected — hold-to-EOD outperformed all alternatives by 50–90% on 918 trades |
| Symmetric ensemble (N-of-N agreement) | Rejected — does not replicate at bar level the config-level finding from prior research |
| ML config-selection (XGBoost) | Rejected — AUC 0.55 (barely above random) |

---

## Repository structure

```
src/
  engine.py              Backtest engine (signal → risk → broker → log)
  strategy/              14 strategy implementations
  risk/                  Position sizing, ATR, exit logic
  broker/
    base.py              Abstract broker interface
    simulator.py         Backtest broker (perfect fills)
    alpaca.py            Paper/live broker (real API)
  data/fetcher.py        Historical + live bar fetching
  logger/                Trade and run logging (SQLite)

backtest/                Research scripts (20+ studies)
  full_validation_v4.py       The main V4 validation chain
  fdr_correction.py           Benjamini-Hochberg FDR + Bonferroni
  validate_recent.py          Out-of-sample (2026) validation
  wf_window_stability.py      Quarterly robustness
  correlation_study_full.py   Cross-strategy correlation matrix
  portfolio_backtest.py       Capital-allocation policy comparison
  rr_optimization.py          RR sweep + MFE distribution
  ensemble_validation.py      N-of-N agreement test
  coin_time_exit_study.py     Time-exit hypothesis (rejected)
  ...

tests/                  180 tests covering strategies, engine, risk, broker
docs/
  case-studies/         Detailed write-ups
```

---

## Strategy archetypes

| Family | Strategy | Edge mechanism |
|---|---|---|
| Breakout | ORB | First-15-minute range, then breakout above |
| Breakout | Donchian | Rolling N-bar high breakout |
| Trend | EMA Crossover | Fast/slow MA cross |
| Trend | MACD | Signal-line crossover |
| Momentum | Momentum | Price velocity threshold |
| Mean Reversion | RSI Reversion | Oversold → recovery |
| Mean Reversion | Bollinger Reversal | Touch lower band + recovery |
| Mean Reversion | VWAP Reversion | Mean reversion to VWAP |
| Volatility | VWAP Bands | Breakout of VWAP ± Nσ |
| Pattern | Inside Bar Breakout | Compression then breakout |
| Event-driven | Gap Fill | Gap reversion to prior close |
| Oscillator | Stochastic | %K/%D crossover in oversold |
| Composite | Ensemble | Trade when N strategies agree |
| Composite | Hybrid Filter | Primary + filter agreement |

All strategies implement a common `BaseStrategy` interface (`reset_session`, `generate_signal`, `name`) and are registered in the engine via a simple dict map. Adding a new strategy requires only implementing the interface and registering it.

---

## Case studies

- **[Anti-overfit discipline](docs/case-studies/anti-overfit.md)** — How four "best configurations" emerged in two weeks and why none became the final answer.
- **[Regime shift detection](docs/case-studies/regime-shift.md)** — META and TSLA's historical edges decayed in 2026 despite strong historical validation.
- **[Multiple testing correction](docs/case-studies/multiple-testing.md)** — Bonferroni and FDR applied to 1,596 simultaneous tests.

---

## Limitations

The framework optimizes for intraday strategies with end-of-day position closure. The following are known limitations:

- **Walk-forward configuration**: 6-month train / 3-month test / 3-month roll. Alternative configurations (3 / 9 / 12-month trains) tested but not exhaustively.
- **Monte Carlo permutation count**: 200 per configuration. Smallest measurable p-value is 1/200 = 0.005; "p=0.000" results should be read as "p < 0.005".
- **Symbol universe**: 19 names selected for liquidity and prior interest. A broader universe scan would reduce selection bias.
- **Overnight risk**: Not modeled — all positions close end-of-day.
- **Transaction cost modeling**: Slippage tested up to 0.20% but commissions are not modeled (broker is commission-free for our universe).
- **Live execution effects**: Adverse selection at fill time, latency, and order routing costs are not fully reflected in backtests.

---

## Reproducibility

Each backtest script in `backtest/` is self-contained and reproducible.
Example — reproducing the V4 validation:

```bash
pip install -r requirements.txt
export ALPACA_API_KEY=...    # or use yfinance fallback
python backtest/full_validation_v4.py --workers 19
```

To reproduce the multiple-testing correction analysis (requires V4 results JSON):

```bash
python backtest/fdr_correction.py
```

Bar cache files (parquet/pkl) are not included due to size. Scripts will
fetch from Alpaca on first run.

---

## License

MIT — see [LICENSE](LICENSE).
