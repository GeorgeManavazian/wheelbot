# The live wheel bot -- what changed, and why

GENERATED from `variants/frozen.lock.json` by `bench registry`.
Do not edit by hand. Every entry is one `bench promote`.

## 2026-09-05 -- `exit-impact-model`

Charging extra slippage on oversized SELL_CALL/CLOSE_PUT fills (the two legs that cannot refuse) instead of pricing them at the clean quote -- does it materially change P&L, or just make the fill log honest?

| knob | from | to |
|---|---|---|
| `exit_impact_max_spreads` | `None` | `1.0` |

**Verdict.** Adopted 2026-09-05: covered-call writes and put buybacks now pay for their own size instead of pricing at the clean quote -- the two legs that can never refuse a fill. Live tier, clean tree: total return +8.16% against +22.92% for the base arm (-14.76pp), Sharpe 0.34 against 0.72, campaigns 104 against 142. Same lower-bound cap (1 spread) the promotion note declared; entry-side decisions unchanged on a flat-state replay of the one captured live day (0 of 0 differed), and the entry-warning-count drift in the full backtest follows from different campaign timing.

**Evidence.** scorecard `live/f92ee3aba856bf77` · engine `094d658cc678` · data `b3cbc28b26c62418` · sharpe_delta `-0.383` · ret_per_maxdd_delta `-0.651`

## 2026-08-17 -- `liq-junk-floor`

Set the absolute floor to exactly what the per-account size cap already implies -- open interest 10, volume 4 -- so the floor stops rationing the menu and goes back to being a junk filter. Does it match `liq-floor-off` without inheriting its two defects?

| knob | from | to |
|---|---|---|
| `liq_min_open_interest` | `250` | `10.0` |
| `liq_min_volume` | `25` | `4.0` |

**Verdict.** Owner decision 2026-08-17: ship it. The old bar was blocking too much for no reason, the other rule already covers it, and I am not buying it for the profit number. The floor goes from 250/25 open interest and volume to 10/4 -- the exact points at which the per-account size cap (10% of open interest, 25% of volume) reaches one contract, so the bot was already refusing everything below them. Proof it is not a loophole: liq-junk-floor trades a book IDENTICAL to liq-floor-off -- 142 entries, same date, ticker, strike and size on every one, same P&L to the cent -- because names below OI 10 / volume 4 were never tradeable anyway. Lowering the floor therefore buys everything that removing it buys, while keeping two behaviours that removal destroys: liquidity_ok returns early when all three thresholds are None and silently disables its own no_two_sided_market check, and an unset threshold stops refusing missing/NaN fields. Measured live tier, n_slots=5, engine 431c75f634ed clean: menu depth 3.5 -> 11.7, starved sessions 23.5% -> 5.0%, campaigns 104 -> 142, days_uncovered 1,037 -> 641, idle slot-days 47 -> 7. Every realism bar declared before the run was met exactly -- minimum entry open interest 10, minimum volume 4, maximum order 25.0% of the day's volume and 10.0% of open interest, zero entries into a zero-volume strike. NOT PROMOTED FOR THE RETURN. ret_per_maxdd went 0.44 -> 1.03 but the ladder 250/50/10-4 is non-monotone (0.44 / 0.22 / 1.03), the middle rung worse than both neighbours, which is one assigned name holding a slot for months rather than a dial; the mechanism and realism columns are monotone and this rests on those. Declared cost: earnings per slot-day on clean campaigns fall $42.70 -> $35.93, and campaigns touching an implausible fill rise 10% -> 17% -- though every such fill in every arm is a SELL_CALL or CLOSE_PUT, both ungated, and not one is an entry. That is the next piece of work, not a defect of this one.

**Evidence.** scorecard `live/76c904270c347994` · engine `431c75f634ed` · data `194898688a1b3ca0` · sharpe_delta `+0.387` · ret_per_maxdd_delta `+0.587`

## 2026-08-16 -- `calls-to-expiry`

Does holding covered calls to expiry beat buying them back at 60%, holding the live delta, DTE and put take-profit fixed?

| knob | from | to |
|---|---|---|
| `call_take_profit_pct` | `None` | `1.0` |

**Verdict.** Owner decision 2026-08-16: ship it, and ship ONLY this. Covered calls now run to expiry; the short put keeps its 60% take-profit. Measured at n_slots=5 on both evidence tiers with the write-credit floor in place. Full tier (2022-06..2026-07, 529 names, 45 assignments): ret/DD -0.24 -> 1.61, return -6.7% -> +33.9%, max drawdown -27.8% -> -21.1%, and uncovered slot-days 2,511 -> 1,965. Better in five years out of five, which is what separates this from the -40% stop that fired three times. The declared counter-argument was refuted by the run: campaigns ROSE 158 -> 198, because holding the call to expiry is what gets the shares called away and frees the slot, while the take-profit bought the hedge back and kept the position stuck. Live tier agrees (ret/DD -0.16 -> 0.44, uncovered 1,146 -> 1,037). The bar was declared before the run - days_uncovered must fall AND ret/DD must not worsen, with P&L alone explicitly not sufficient - and both halves were met. NOT shipped alongside: call-otm-25 failed its own bar on the long window at exactly 2 of 5 years, and exit-pair (both knobs) is worse than either single on both windows, so the two are substitutes rather than complements. Call-write refusals go 792 -> 749.

**Evidence.** scorecard `full/366f1c919ec5761a` · engine `daf481722377` · data `b702e8f0548b8fcb` · sharpe_delta `+0.488` · ret_per_maxdd_delta `+1.854`

## 2026-08-14 -- migration

Extracted verbatim from the FROZEN dict literal of the live runner at commit b7030cf. Values only -- the prose moved to variants/frozen.toml with nothing summarised or dropped.
