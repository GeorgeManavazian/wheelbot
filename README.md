# Wheel Bot

A rules-based cash-secured put and covered-call strategy (the "wheel") tested across 531 US large-cap stocks with five years of option chains, filled at the bid and ask a real order would get. Paper trading only.

![Bot vs SPY buy-and-hold by year, 2024-08 to 2026-07](results/equity_by_year.png)

Yearly return of the current config against SPY buy-and-hold, 2024-08-01 to 2026-07-01 (2024 is Aug to Dec, 2026 is Jan to Jun). From the scorecard in `bench/scorecards/exit-impact-model/`, dated 2026-09-05.

## Thesis

Selling a cash-secured put on a company I would be happy to own is getting paid to wait for a price I already like. If the stock holds, I keep the premium and repeat; if it falls through the strike, I own it at a discount and sell calls against it until it is called away. For this to beat holding the same names, the premium has to exceed the cost of the tail, meaning the assignments that keep falling, and the names I pick have to matter, because the option market prices that tail fairly on average. A broad mechanical test separates the two: sell puts blind on everything to see what premium alone is worth, then see whether a selection rule adds anything.

## What I tested and what I learned

- Selling puts blind loses. A 0.40-delta put sold on every eligible day across 153 names, 2024-01-16 to 2026-07-01 (42,226 ticker-days), lost 11.8 bps per trade at an 81.8% win rate. The 19% tail eats the wins, so selection is the whole game.
- Selection is real. Ranking candidates by volatility percentile and filling five slots from the top returned +22.92% on the verdict window; filling the same slots in arbitrary order returned +1.77%, with cycles falling from 142 to 79 and the assignment rate rising from 19.7% to 32.9% (2026-08-18, before the exit-cost assumption below). The ranking carries the signal; the mechanics alone do not.
- The current config trails SPY. Over 2024-08-01 to 2026-07-01, 531 names, five slots, it returns +8.16% (Sharpe 0.34, max drawdown -21.6%, 104 cycles) against +37.34% for SPY (Sharpe 1.07, -19.0%), with the exit-side cost model below treated as an assumption. I tested roughly ninety variations of exits, entry gates, and slot counts against the same window; none reached SPY, and the best came in at +31.83%. Premium plus selection covers the tail, not the opportunity cost of a strong bull market.

Mechanics I measured and set aside (2024-08 to 2026-07 unless dated otherwise, before the exit-cost assumption):

| mechanic | number | date | what I learned |
|---|---|---|---|
| IV-rank as the ranking key | +10.05% vs +22.92% | 2026-08-22 | High implied vol marks names the market already fears |
| Put stop-loss at 30% | -10.8% vs +22.9% | 2026-08-17 | Stops sell dips that mostly recover; the risk is correlated, so a per-position stop does not remove it |
| Rolling tested puts | 143 of 143 candidate rolls on a nine-year SPY wheel were net-debit | 2026-07-13 | A roll for credit is not on offer when you need it |
| Entry "weather" gates, ten definitions | none cut the assignment rate; deleting the gate did best, +36.4% vs -4.4% | 2026-08-16 | Put delta sets the assignment probability; a regime label on top does not |
| Macro regime entry gate | GDX single-name wheel +17.5k to +4.7k on 100k | 2026-07-15 | Waiting for a good regime mostly means missing the premium |
| Out-of-the-money covered calls (0.25 / 0.35 delta) | -24.6% / -30.4% vs +22.9%; all five slots jammed with stock | 2026-08-17 | The at-the-money call is what frees the slot |

## How it was tested

Data. ThetaData end-of-day option chains: 290.6M rows, 530 of 531 tickers (NVR lists no expirations), 2021-08-09 to 2026-08-07, DTE 60 or less, about 7 GB. Open interest bought separately: 534 tickers, 61 months, 205M rows. The stores do not ship here; the scorecards do.

Fills. Sell at the bid, buy at the ask, Schwab quotes. Commission $0.65 plus $0.05 pass-through per contract per side, $1.40 round trip. Assignment fee $0 (not yet checked against a statement). Covered-call writes and put buybacks larger than 10% of open interest or 25% of volume pay one extra spread, a declared lower bound on impact rather than a measured one.

Config (frozen 2026-09-05). Put delta 0.30, call delta 0.50, target DTE 11 (band 9 to 15), put take-profit at 60% of premium, calls held to expiry, five slots, open-interest floor 10 and volume floor 4, 8% annualised yield floor on collateral, earnings blackout through expiry plus one day.

Validation. One fixed verdict window, 2024-08-01 to 2026-07-01, 531 names. Each variant is a TOML file in `variants/` that declares its pass bar before it runs; the bench scores it against the frozen config and SPY and writes a JSON scorecard. Open-interest data is complete from 2024-08 (533 of 534 tickers, against 86 of 534 before), so the verdict window starts there.

Known gaps:

- No walk-forward and no held-out period; every variation was scored on the same 23 months, so the search is in-sample. Re-scoring each candidate at 20 slots is the only defence against a lucky cell.
- The window contains no sustained bear market, which is where the wheel's tail lives.
- Slippage is the quoted spread plus the exit-side assumption; no partial fills.

## Where it stands

Paused. Finished: the engine, the fill model, the bench, the frozen config, and a verdict on the 2024-08 to 2026-07 window. Paper trading against live quotes has run on a VPS since 2026-07-18; research paused in 2026-09 when a gap-fill question on the same universe looked more promising. Not finished: a held-out test and a bear-market window. The open question is whether a selection rule can close a 29-point gap to SPY, or whether the wheel is a yield strategy that should be judged on income and drawdown rather than total return. Next: a 2022-style window, and a scorecard that reports premium income and drawdown alongside total return.

## What this is not

Paper trading only, with simulated fills against real quotes and no order-placement code. Not investment advice. Does not claim a live edge; on the window tested, there is none over SPY.

## How to run it

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q                                  # engine and bench tests on bundled fixtures
.venv/bin/python results/plot_equity_by_year.py      # rebuilds the chart from bench/scorecards/
```

Reproducing a scorecard needs the ThetaData stores (about 8 GB, licensed, not included) in `data/options/`; then `scripts/wheelbench run <variant>`. Accepted mechanics are in `bench/artifact/blocks.toml`; fill rules in `src/engine_v2/options/fills.py`.

## Built with

Python 3.12, pandas, pyarrow, pytest. Option chains and open interest from ThetaData; quotes for paper fills from the Schwab API. Built with AI-assisted development (Claude Code); the research questions, hypotheses, validation choices, and conclusions are mine.
