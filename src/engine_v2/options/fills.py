"""ONE shared take-profit fill rule for the four engines (A18).

Given a quote (and, for the hourly engines, a day of trade prints) and an open
short, decide: does the take-profit fill, at what price, at what cash cost,
stamped with what date. The four call sites and the mode each uses:

  portfolio.step_one_day   (EOD decision engine)   quote only (mark.ask)
  live.intraday            (live exit manager)     quote only, wall-clock stamp
  wheel.run_wheel          (solo backtest)         print-next-bar, then quote
  regime_router            (router backtest)       print-next-bar, then quote

This is a SEAM, not a model choice: every path reproduces the rule its engine
already had, verbatim. The divergence it finally names instead of hiding: the
print rule fires on 81.5% of campaigns and the quote rule on 76.7% -- 159
campaigns (5.3%) the backtest takes profit on that live can never close
(finding A18). WHICH rule is right is the deferred strategy question -- do not
resolve it here, and do not add a new mode without an owner decision.

Admission (which quotes are valid at all) deliberately stays UPSTREAM in the
chain builders (`chain.py`, `live/data.py`, `live/held_legs.py`,
`live/marks.py`): folding those rules together changes the population the
backtest sees, which is exactly what A20 exists to size first.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FillDecision:
    filled: bool
    price: float = 0.0     # per-share, as booked on Trade.price_per_contract
    cost: float = 0.0      # UNSIGNED cash out (buy_cost convention; caller does cash -=)
    stamp: object = None   # what goes on Trade.date (day, bar timestamp, or wall clock)
    via: str = ""          # filled: "print" | "quote"; not filled: the refusal reason
    # A17: how many contracts actually filled. Both current modes are
    # instant-and-whole, so they always set this to the full size; a future
    # model that partially fills has somewhere to say so, and the close
    # bookkeeping (portfolio.close_short_fill) already honors it. Trailing
    # with a default so refusal-path equality and positional construction
    # are unchanged (A18 test pins).
    filled_contracts: int = 0


# Smallest quotable increment. Market-structure fact for penny-pilot classes;
# nickel-tick classes exist and the bot has no per-class tick model -- $0.01
# is the PERMISSIVE choice (smallest floor, fewest refusals), so the guard
# can under-refuse on nickel names but never over-refuse. A comment, not a
# knob, on purpose (A3).
MIN_TICK = 0.01


def tp_exit_floor(cfg):
    """A3: the minimum entry credit at which this config's take-profit exit is
    both REACHABLE and NOT A GUARANTEED LOSS. None when no TP exit exists
    (tp None, >= 1.0 hold-to-expiry, or degenerate <= 0).

    Two parameter-free zero-crossings, derived from try_take_profit's own
    arithmetic (thresh = (1-tp)*credit; fill at ask <= thresh; cost
    ask*mult*n + comm*n):
      reachability:  (1-tp)*credit >= MIN_TICK  -- admitted quotes have
                     ask > 0 on the tick grid, so a thresh below one tick can
                     never be satisfied (the WBD 25P case: thresh $0.004).
      net-positive:  tp*credit*mult > 2*comm -- the worst fill the rule
                     accepts is thresh itself; below this the "winning" exit
                     loses money after both commissions, by construction.
    Guard order mirrors try_take_profit (tp tested before any arithmetic
    touches it -- the A6 amendment lesson)."""
    tp = cfg.take_profit_pct
    if tp is None or tp >= 1.0:
        return None                     # hold-to-expiry: no TP exit exists
    if tp <= 0.0:
        # A3 skeptic F1: try_take_profit treats tp=0 as a LIVE rule that fires
        # at any ask <= credit, where the worst accepted fill nets
        # 0*credit*mult - 2*comm < 0 -- EVERY entry's TP exit is a guaranteed
        # loss. Infinite floor (refuse all), never "inert": inert was the one
        # gap that waved the WBD class through under a degenerate config.
        return float("inf")
    # NOTE (skeptic F2): the friction conjunct evaluates the worst fill at
    # continuous thresh; the engine's actual worst fill is the penny at/below
    # it, so for tp <= ~0.43 this over-refuses a thin band of survivable
    # entries. Conservative-only; exact at every config in the repo (tp 0.50,
    # 0.60). Revisit only if a small-tp config ever appears. A13: friction
    # (commission + fees), never raw commission -- the guard must price
    # exactly what the fills charge.
    return max(MIN_TICK / (1.0 - tp),
               2.0 * cfg.friction_per_contract / (tp * cfg.contract_multiplier))


def tp_exit_feasible(credit, cfg):
    """(True, "") when an entry at `credit` has a satisfiable, non-guaranteed-
    loss TP exit under cfg -- or when no TP exit exists at all (nothing to be
    infeasible). Else (False, "tp_unreachable" | "tp_net_negative")."""
    floor = tp_exit_floor(cfg)
    if floor is None or credit >= floor:
        return True, ""
    tp = cfg.take_profit_pct
    if (1.0 - tp) * credit < MIN_TICK:
        return False, "tp_unreachable"
    return False, "tp_net_negative"


def write_credit_floor(cfg) -> float:
    """The minimum credit at which OPENING a short is worth doing at all --
    independent of take_profit_pct, and so of whether a TP exit exists.

    `tp_exit_floor` answers "can this position's take-profit exit be reached
    without losing money", and correctly answers None for a hold-to-expiry
    config (tp None or >= 1.0): there is no exit to be unreachable. But
    `tp_exit_feasible` then admits EVERY credit, $0.00 included -- and on the
    covered-call leg A3b is the only minimum-credit rule there is, because
    calls are deliberately exempt from the A2 liquidity gate (refusing a call
    leaves shares naked). So `call_take_profit_pct = 1.0` does not relax that
    rule, it DELETES it, and the engine writes $0.00 calls at a guaranteed
    -friction per contract. This floor is the half of A3b that was never
    really about the take-profit.

    Two parameter-free reasons, and the max of them, so neither can be slipped
    under by a config this repo does not currently hold:

      not a quote:   credit >= MIN_TICK. Below the smallest quotable
                     increment there is no order to send.
      net-positive:  credit * mult > friction. `sell_proceeds` charges the
                     one-sided friction on the way in, so a credit that does
                     not clear it books a loss the moment it fills, whatever
                     the underlying does next.

    At the live config the tick binds ($0.01 vs $0.007). At every take-profit
    in the repo (0.50, 0.60) `tp_exit_floor` is already $0.02-$0.025 -- above
    this -- so ANDing it in changes no decision and the live arm needs no
    re-measurement. A13: friction (commission + fees), never raw commission.
    """
    return max(MIN_TICK, cfg.friction_per_contract / cfg.contract_multiplier)


def write_credit_ok(credit, cfg) -> tuple[bool, str]:
    """(True, "") when `credit` clears the write floor, else (False,
    "no_credit").

    Its own reason string, deliberately NOT one of tp_exit_feasible's. The
    2026-08-14 sweep read `call_gated_unclosable` falling 1,333 -> 0 as the
    jam being fixed, when the guard that raises that counter had simply been
    switched off. A second rule filed under the same name would leave the same
    misreading available.
    """
    if credit >= write_credit_floor(cfg):
        return True, ""
    return False, "no_credit"


def try_take_profit(*, mark, credit, contracts, cfg, day, expiry,
                    bars=None, day_stamp=None) -> FillDecision:
    """The one answer to "does this short's take-profit fill today, and how".

    mark      Mark or None, already admitted upstream; never a chain.
    bars      hourly trade prints for THIS contract (or None): mode selector --
              None = quote-only engines, a frame = the print-next-bar cascade.
    day_stamp overrides the quote-path Trade stamp (live manager stamps the
              wall clock, not the normalized day it compares expiry against).

    Guard order is load-bearing (the A6 amendment lesson): take_profit_pct is
    tested before the threshold arithmetic touches it, and mark presence
    before mark.ask.
    """
    if cfg.take_profit_pct is None or cfg.take_profit_pct >= 1.0:
        return FillDecision(False, via="tp_disabled")   # >= 1.0 = hold to expiry
    if not (day < expiry):
        return FillDecision(False, via="expiry_day")    # expiry is settlement's job
    thresh = (1 - cfg.take_profit_pct) * credit
    if bars is not None:
        # close > 0 only: hourly bars are trade prints, and hours with no trade
        # arrive as close=0 -- not a price. Treating a 0 as a price lets any
        # losing put "TP" at a phantom fill (XOP 2020: a phantom +2,582%).
        # Trigger and fill both use valid prints only; decide on bar i, fill at
        # bar i+1's close (no same-bar fills). A cross on the day's LAST bar
        # has no next bar -> fall through to the quote check below.
        day_bars = (bars[(bars["timestamp"].dt.normalize() == day)
                         & (bars["close"] > 0)]
                    .sort_values("timestamp").reset_index(drop=True))
        for i in range(len(day_bars) - 1):
            if day_bars.iloc[i]["close"] <= thresh:
                fill = day_bars.iloc[i + 1]
                # arithmetic mirrors wheel.buy_cost term-for-term; fill["close"]
                # stays the numpy scalar it always was (cash dtype contamination
                # is pre-existing, pinned by test, and NOT changed by this seam)
                cost = (fill["close"] * cfg.contract_multiplier * contracts
                        + cfg.friction_per_contract * contracts)
                return FillDecision(True, float(fill["close"]), cost,
                                    fill["timestamp"], "print", contracts)
    if mark is not None and mark.ask <= thresh:
        cost = (mark.ask * cfg.contract_multiplier * contracts
                + cfg.friction_per_contract * contracts)
        return FillDecision(True, mark.ask, cost,
                            day if day_stamp is None else day_stamp, "quote",
                            contracts)
    return FillDecision(False, via="no_fill")


def credit_ok(credit, strike, cfg):
    """(allowed, reason) for a NEW short-put entry under the intrinsic filter.

    A put trading at a large fraction of its own strike is deep in the money:
    the credit is mostly intrinsic value, so selling it is a stock purchase
    dressed as premium. The wheel is supposed to be paid for taking assignment
    RISK; here assignment is near-certain and already priced in.

    Mirrors tp_exit_feasible's shape: cap unset (None) -> always allowed, so
    the default path is byte-identical. A non-positive strike cannot be
    measured and is refused rather than divided by.
    """
    cap = getattr(cfg, "max_credit_pct_of_strike", None)
    if cap is None:
        return True, ""
    if strike <= 0:
        return False, "bad_strike"
    if credit > cap * strike:
        return False, "credit_is_intrinsic"
    return True, ""


def ann_yield_on_collateral(credit, strike, dte):
    """The one-number annualized yield on collateral: `(credit / strike) *
    (365 / dte)`. None if unmeasurable (non-positive strike or dte).

    Shared by the collateral-yield floor (`yield_ok`, below) and
    `rank_by="vrp_viable"`'s tiering (portfolio.py) -- a second formula
    computing the same number would be a bug even if it agreed today.
    """
    if strike <= 0 or dte <= 0:
        return None
    return (credit / strike) * (365.0 / dte)


def yield_ok(credit, strike, dte, cfg):
    """(allowed, reason) for a NEW short-put entry under the collateral-yield
    floor.

    A put that pays a negligible fraction of the cash its strike locks up is
    not a premium trade -- the owner's case is $10 of credit against $100,000
    of collateral for 11 days, which annualises to 0.33% against a measured
    median of ~31% for an ordinary 0.30-delta 11-day put.

    The pre-existing minimum (tp_exit_floor) is ABSOLUTE and strike-blind, so
    it cannot see this. Mirrors credit_ok's shape: floor unset (None) ->
    always allowed, so the default path is byte-identical. A non-positive
    strike or dte cannot be measured and is refused rather than divided by.
    """
    floor = getattr(cfg, "min_ann_yield_on_collateral", None)
    if floor is None:
        return True, ""
    if strike <= 0:
        return False, "bad_strike"
    if dte <= 0:
        return False, "bad_dte"
    if ann_yield_on_collateral(credit, strike, dte) < floor:
        return False, "yield_below_floor"
    return True, ""
