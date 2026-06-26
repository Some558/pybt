# pybt

**One-command backtesting harness for reproducible TradingView strategy verification.**

pybt runs the full verification loop that traders usually repeat manually in TradingView — baseline metrics, cost sensitivity, direction tests, parameter sweeps, and out-of-sample splits — in a single CLI command. Results are saved as Markdown reports ready for maintainer review (e.g. Codex piston).

## Why pybt?

- **Pine Script → reproducible evidence**: Bridge chart ideas and statistical claims with lookahead-safe defaults.
- **Cost realism**: Presets for zero (TV default), index CFD, FX major, and Japanese equities — plus 0–3× cost sweeps.
- **Standard verification suite**: Baseline → cost sensitivity → direction → RR neighborhood → OOS period splits.
- **Dual data sources**: Daily bars via yfinance; intraday FX/CFD/index via Dukascopy (2003+).
- **Markdown output**: Machine-generated reports replace copy-paste from TV tables.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Daily swing — Nikkei 225 ORB-style strategy (yfinance)
python run.py orb --market daily --symbol "^N225" --years 10 --cost zero

# Intraday NAS100 15m ORB (Dukascopy, index CFD costs)
python run.py orb --market intraday --symbol NAS100 --tf 15m \
  --start 2018-01-01 --end 2024-01-01 --cost cfd_index
```

Reports are written to `reports/<strategy>_<symbol>_<market>_<date>.md`.

## Built-in strategies

| CLI name | Description |
|----------|-------------|
| `orb` | Opening Range Breakout (Pine-faithful ORB) |
| `ma25` | MA25 pullback swing |
| `vwap` | VWAP reversion |

## Verification suite

Each run executes five checks:

1. **Baseline** — PF, max drawdown, win rate, trade count
2. **Cost sensitivity** — 0× to 3× base cost (does the edge survive?)
3. **Direction** — Long-only, Short-only, inverted (noise detection)
4. **Parameter neighborhood** — RR sweep (sharp peak vs robust plateau)
5. **OOS splits** — first/second half and 3-way period splits

A heuristic **durable edge** verdict flags obvious failures before human or AI review.

## Cost presets

| Preset | Use case |
|--------|----------|
| `zero` | TradingView default (optimistic upper bound) |
| `cfd_index` | Index CFD (~1 bp spread) |
| `fx_major` | FX majors |
| `jp_stock` | Japanese equities |

## Data sources

| Market | Source | Symbols |
|--------|--------|---------|
| Daily | yfinance | `^N225`, `AAPL`, `7203.T`, etc. |
| Intraday | dukascopy-python | `NAS100`, `SP500`, `N225`, `EURUSD` |

Local pickle cache under `pybt/_cache/` avoids re-fetching heavy intraday data.

## Project structure

```
pybt/
  pybt/           # Core package (data, costs, suite, strategies)
  run.py          # Main CLI
  reports/        # Generated Markdown reports
  requirements.txt
```

## Lookahead safety

`trade_on_close=False` by default: signals evaluated on bar close are filled on the **next** bar open. Same-bar close fills are avoided to prevent lookahead bias.

## License

MIT — see [LICENSE](LICENSE).

## Maintainer

Primary maintainer: [@Some558](https://github.com/Some558). Used in open market-myth verification research (chart patterns, trading anomalies, Japanese + global audiences).
