"""Pure strike-selection and marking primitives over an OptionsChain frame.
No I/O, no engine state — consumed by the Wheel engine (sub-project 2)."""
from __future__ import annotations
import pandas as pd
from .chain import Contract, Mark

def derived_band(target_dte: int) -> tuple[int, int]:
    """DTE guard rail derived from the target. Floor rejects expiry stubs;
    ceiling rejects a monthly when the weekly is absent. Display-only upstream.

    The span is SEVEN days for a reason (widened from six, 2026-08-09). Most
    names list Friday expiries only, so for any weekday the available DTEs are
    one residue class mod 7 -- and seven consecutive integers always contain
    exactly one, while six can miss. Six missed: at target 11 no Thursday
    resolved (8 and 15 both out of (9,14)), at target 7 no Monday did (4 and 11
    both out of (5,10)). The bot could not enter weekly-only names one day in
    five, at every target, and nothing logged it. See test_select.py."""
    return max(5, target_dte - 2), target_dte + 4

def liquidity_ok(chain, date, contract, cfg):
    """A2 liquidity gate: is this contract liquid enough to OPEN a short in?
    Pure veto predicate -> (True, "") or (False, reason). Evaluated AFTER
    selection (a row-filter before selection silently moves the sold delta --
    measured 0.28 -> 0.40 on the GDX fixture -- instead of refusing the trade)
    and only on entry/roll-destination paths: never on closes, expiry, marks,
    held-only rows, or covered calls (owner-provisional 2026-08-01).

    A threshold left None disables that leg -- backtest chains carry no
    OI/volume columns, so backtest configs run rel-spread only (a declared
    one-sentence divergence, like the A18 print-vs-quote modes). With a
    threshold SET, a missing/NaN value FAILS: a contract whose liquidity
    cannot be measured is not one to sell.

    Rel-spread denominator is the computed midpoint (ask-bid)/((bid+ask)/2),
    never the `mid` column -- live's `mid` is Schwab's mark, and using the
    column would score the same contract differently in the two engines."""
    if (cfg.liq_max_rel_spread is None and cfg.liq_min_open_interest is None
            and cfg.liq_min_volume is None):
        return True, ""
    rows = chain[(chain["date"] == date) & (chain["expiry"] == contract.expiry)
                 & (chain["strike"] == contract.strike)
                 & (chain["right"] == contract.right)]
    # mirror select_contract: a spliced mark-only row must never answer for a
    # tradeable contract (duplicate-key order sensitivity, A2 skeptic F7)
    if "held_only" in rows.columns:
        rows = rows[~rows["held_only"].fillna(False).astype(bool)]
    if rows.empty:
        return False, "row_missing"
    row = rows.iloc[0]
    bid, ask = float(row["bid"]), float(row["ask"])
    den = (bid + ask) / 2.0
    if not (den > 0) or ask < bid:
        return False, "no_two_sided_market"
    if (cfg.liq_max_rel_spread is not None
            and (ask - bid) / den > cfg.liq_max_rel_spread):
        return False, "rel_spread"
    for field, thresh in (("open_interest", cfg.liq_min_open_interest),
                          ("volume", cfg.liq_min_volume)):
        if thresh is None:
            continue
        if field not in chain.columns:
            return False, field
        try:
            v = float(row[field])
        except (TypeError, ValueError):
            return False, field
        if v != v or v < thresh:          # NaN or below the floor
            return False, field
    return True, ""


def liquidity_size_cap(chain, date, contract, cfg):
    """A2b: the most contracts this order may take of the listed market.

    None -> uncapped (both knobs off, so every existing result is
    byte-identical). 0 -> refuse: a single contract is already too large a
    share, or a field the caller asked to measure is missing/NaN. Anything
    else is a ceiling the caller applies to its own sizing.

    This is the ratio the A2 floor could not express. Sizing happens per
    ACCOUNT -- run_daily builds a WheelConfig per capital level -- so the same
    contract caps a 500k account and leaves a 5k one untouched, which is what
    "757 contracts against 84/day traded" was always about.

    Mirrors liquidity_ok's row lookup exactly, held_only exclusion included: a
    spliced mark-only row must never answer for a tradeable contract.
    """
    pct_oi = getattr(cfg, "liq_max_pct_of_open_interest", None)
    pct_vol = getattr(cfg, "liq_max_pct_of_volume", None)
    if pct_oi is None and pct_vol is None:
        return None
    rows = chain[(chain["date"] == date) & (chain["expiry"] == contract.expiry)
                 & (chain["strike"] == contract.strike)
                 & (chain["right"] == contract.right)]
    if "held_only" in rows.columns:
        rows = rows[~rows["held_only"].fillna(False).astype(bool)]
    if rows.empty:
        return 0
    row = rows.iloc[0]
    cap = None
    for field, pct in (("open_interest", pct_oi), ("volume", pct_vol)):
        if pct is None:
            continue
        if field not in chain.columns:
            return 0
        try:
            v = float(row[field])
        except (TypeError, ValueError):
            return 0
        if v != v:                       # NaN: set but unmeasurable -> refuse
            return 0
        # epsilon: 200 * 0.05 lands at 10.000000000000002 in binary float and
        # a bare int() would be fine there, but 0.1-family products can land a
        # hair BELOW the integer and silently cost a contract.
        lim = int(v * pct + 1e-9)
        cap = lim if cap is None else min(cap, lim)
    return cap


def exit_impact_slippage(chain, date, contract, contracts, cfg) -> float:
    """B-exit: price-impact estimate for SELL_CALL and CLOSE_PUT, the two
    legs that cannot refuse (a covered call is exempt from every liquidity
    gate; a close leg is a position already held, not a new one to decline).
    Returns extra cost in price units (same scale as bid/ask), always >= 0.

    None, or both ratio knobs None -> 0.0, byte-identical to every prior run.

    Reuses `liq_max_pct_of_open_interest`/`liq_max_pct_of_volume` -- the SAME
    ratio already promoted for entry-side sizing (`liquidity_size_cap`) --
    rather than a new fitted threshold. An order at or under that ratio of
    the day's open interest or volume costs nothing extra; each full
    multiple beyond it costs one more bid-ask spread, capped at
    `cfg.exit_impact_max_spreads` spread-widths so an unmeasurable or
    zero-volume day prices as expensive rather than undefined.

    Mirrors liquidity_ok's row lookup and held_only exclusion exactly."""
    cap = getattr(cfg, "exit_impact_max_spreads", None)
    pct_oi = getattr(cfg, "liq_max_pct_of_open_interest", None)
    pct_vol = getattr(cfg, "liq_max_pct_of_volume", None)
    if cap is None or (pct_oi is None and pct_vol is None):
        return 0.0
    rows = chain[(chain["date"] == date) & (chain["expiry"] == contract.expiry)
                 & (chain["strike"] == contract.strike)
                 & (chain["right"] == contract.right)]
    if "held_only" in rows.columns:
        rows = rows[~rows["held_only"].fillna(False).astype(bool)]
    if rows.empty:
        return 0.0
    row = rows.iloc[0]
    bid, ask = float(row["bid"]), float(row["ask"])
    spread = ask - bid
    if not (spread > 0):
        return 0.0
    excess = 0.0
    for field_, pct in (("open_interest", pct_oi), ("volume", pct_vol)):
        if pct is None or field_ not in chain.columns:
            continue
        try:
            v = float(row[field_])
        except (TypeError, ValueError):
            v = float("nan")
        ratio = float("inf") if (v != v or v <= 0) else contracts / (v * pct)
        excess = max(excess, ratio - 1.0)
    spreads = min(max(excess, 0.0), cap)
    return spreads * spread


def at_risky_window_edge(chain, date, contract, target_delta) -> bool:
    """B11: True when the chosen PUT sits at the BOTTOM strike of its
    surveyed expiry AND carries a higher |delta| than target -- the signature
    of a clipped 12-strike window (the true target strike lies below what was
    pulled), which makes the entry riskier than configured. Strict
    riskier-than: an edge row at exactly the target delta is not a clip
    signal (and keeps single-strike synthetic chains quiet). Put side only:
    post-A4 an at-top call is usually the exchange's real extreme, not a
    window artifact."""
    rows = chain[(chain["date"] == date) & (chain["right"] == "P")
                 & (chain["expiry"] == contract.expiry)]
    if "held_only" in rows.columns:
        rows = rows[~rows["held_only"].fillna(False).astype(bool)]
    if rows.empty or contract.strike != float(rows["strike"].min()):
        return False
    sel = rows[rows["strike"] == contract.strike]
    return abs(float(sel["delta"].iloc[0])) > abs(target_delta)


def select_contract(chain, date, right, target_delta, target_dte, root, min_strike=None):
    """Expiry FIRST (nearest target_dte within derived_band, from expiries visible
    on `date` only), THEN strike (nearest |delta| within that one expiry).
    With `min_strike`, expiries are tried in nearest-DTE order (tie -> longer-
    dated) and the first one containing a strike >= min_strike is used; only
    when no in-band expiry qualifies -> None. Deterministic; None -> sit in cash."""
    lo, hi = derived_band(target_dte)
    cand = chain[(chain["date"] == date) & (chain["right"] == right)]
    # Mark-only rows are not tradeable. live/held_legs.py splices in the legs the
    # bounded chain cannot see so the take-profit can still act on them; those
    # rows were pulled by OCC symbol for a position already held, and are not
    # part of the chain the bot actually surveyed. Selecting one enters a
    # contract the bot never saw — and because run_daily builds ONE market for
    # all 25 accounts, the leg one account holds would otherwise appear in every
    # other account's candidate set. Column absent on every pre-2026-07-31 chain,
    # where absence correctly means "tradeable".
    if "held_only" in cand.columns:
        cand = cand[~cand["held_only"].fillna(False).astype(bool)]
    if cand.empty:
        return None
    dtes = cand.groupby("expiry")["dte"].first()
    dtes = dtes[(dtes >= lo) & (dtes <= hi)]
    if dtes.empty:
        return None
    err = (dtes - target_dte).abs()
    for exp in sorted(dtes.index, key=lambda e: (err[e], -dtes[e])):
        e = cand[cand["expiry"] == exp]
        if min_strike is not None:
            e = e[e["strike"] >= min_strike]
            if e.empty:
                continue
        row = e.loc[(e["delta"].abs() - abs(target_delta)).abs().idxmin()]
        return Contract(root, row["expiry"], float(row["strike"]), right)
    return None

def select_roll_contract(chain, date, right, strike, current_expiry, target_dte, root):
    """Roll destination (repair spec amendment 2026-07-13b): the canonical
    credit roll is SAME STRIKE, out in time. Candidates are expiries STRICTLY
    beyond the held leg's expiry that carry the held strike, nearest to
    (current_expiry + target_dte) — one config cycle further out; tie ->
    longer-dated. Deterministic; None -> no roll today."""
    cand = chain[(chain["date"] == date) & (chain["right"] == right)
                 & (chain["strike"] == strike)]
    if cand.empty:
        return None
    exps = cand.groupby("expiry")["dte"].first()
    # tenor guard: the extension (new expiry - held expiry) must sit within the
    # same derived band the entry uses — without it, a sparse strike grid could
    # silently roll a 7-DTE campaign months out (no such expiry -> no roll).
    lo, hi = derived_band(target_dte)
    cur = pd.Timestamp(current_expiry)
    ext = (exps.index - cur).days
    exps = exps[(ext >= max(1, lo)) & (ext <= hi)]
    if exps.empty:
        return None
    anchor = cur + pd.Timedelta(days=target_dte)
    best = sorted(exps.index, key=lambda e: (abs((e - anchor).days), -exps[e]))[0]
    return Contract(root, best, float(strike), right)

def option_mark(chain, date, contract):
    # INVARIANT: `chain` is single-root. Both callers guarantee it — BatchMarket
    # groups by ticker, LiveMarket pulls one ticker's chain_frame at a time — so
    # matching on (date,expiry,strike,right) can't cross to another underlying's
    # like-struck option. If a multi-root chain is ever passed here, add a root
    # filter (needs a `root` column on the chain frame). Noted 2026-07-17 (I9).
    m = ((chain["date"] == date) & (chain["expiry"] == contract.expiry)
         & (chain["strike"] == contract.strike) & (chain["right"] == contract.right))
    r = chain[m]
    if r.empty:
        return None
    row = r.iloc[0]
    # C1: quote_time only when the frame carries it (live path); backtest
    # frames have no column -> None -> Marks byte-identical to before.
    qt = row["quote_time"] if "quote_time" in chain.columns else None
    return Mark(float(row["bid"]), float(row["ask"]), float(row["mid"]),
                None if qt is None or pd.isna(qt) else float(qt))

def intrinsic_value(right, strike, underlying):
    return max(strike - underlying, 0.0) if right == "P" else max(underlying - strike, 0.0)

def expiry_underlying(chain, contract):
    r = chain[chain["date"] == contract.expiry]
    if r.empty:
        return None
    return float(r.iloc[0]["underlying"])
