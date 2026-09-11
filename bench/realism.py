"""Could this fill have happened? A per-trade check against the chain the
engine actually filled from.

WHY THIS EXISTS. A backtest fills at a stored quote whether or not a human
could have got that quote. Every liquidity rule in this project is an attempt
to keep that honest, and every one of them has been argued about with anecdotes
-- "757 DOW contracts against 84 a day traded" is the founding example, and it
was one name on one day. This module answers the question for EVERY fill in a
run, so a liquidity change is judged on its whole trade log rather than on the
worst case someone happened to look up.

WHAT IT IS NOT. It cannot prove a fill was possible. Nothing can, from EOD
data: a resting quote is an invitation, not a contract, and size at the touch
is not in this store. It measures the four things that make a fill IMPLAUSIBLE
and reports them per trade, so the reader judges the distribution instead of a
headline:

  * ORDER vs the day's VOLUME. The order asks the market to absorb this many
    contracts on a day it traded that many in total. Above ~25% this is the
    dominant realism risk, and it is the number the A2b size cap bounds.
  * ORDER vs OPEN INTEREST. How large the position is against everything
    outstanding in that strike.
  * ZERO-VOLUME strikes. A contract that did not trade at all that session.
    The quote may be a market maker's standing two-sided obligation -- which is
    genuinely tradeable -- or a stale artifact. It is the single loudest flag
    and it is counted separately rather than merged into a ratio.
  * THE SPREAD, in cents and as a fraction of the mid. The wheel round-trips
    every position, so the spread is paid twice, and a wide one can exceed the
    whole edge on an 11-day trade.

Entry legs (SELL_PUT / SELL_CALL) and exit legs (CLOSE_*) are both checked: a
position you can open and cannot close is the failure mode that matters most,
and it lives entirely in the exit rows.
"""
from __future__ import annotations

import pandas as pd

#: Trades that consume liquidity. Expiry and assignment are settlement events,
#: not fills -- nobody has to be on the other side of an expiry, so checking
#: them would dilute the very statistic this module exists to compute.
FILL_ACTIONS = ("SELL_PUT", "SELL_CALL", "CLOSE_PUT", "CLOSE_CALL",
                "ROLL_OPEN", "ROLL_CLOSE")

#: Share legs (A-4, batch 2 pre-work, 2026-08-17): exported as ROWS, tagged by
#: `leg`, and excluded from every realism statistic. Batch 1's per-stop
#: attribution -- which fire realized what, against which assignment -- had to
#: be inferred from option rows and cash arithmetic; with settlements and
#: stock sales in the log (each row carrying `cash_after`), it is arithmetic.
SETTLEMENT_ACTIONS = ("ASSIGNED", "CALLED_AWAY", "PUT_EXPIRED", "CALL_EXPIRED")
STOCK_ACTIONS = ("SOLD_SHARES",)

#: A trade taking more than this share of the session's volume is flagged. Not
#: a threshold the engine enforces -- `liq_max_pct_of_volume` does that -- but
#: the line this report counts against, held at the same value so the report
#: and the rule cannot drift apart.
FLAG_PCT_OF_VOLUME = 0.25
FLAG_PCT_OF_OI = 0.10
#: Above this the round trip costs more than a tenth of the mid on each side.
FLAG_REL_SPREAD = 0.20


def check_fills(trades, chains: dict) -> pd.DataFrame:
    """One row per fill, with the market it was filled into.

    `chains` is the same dict the run was given, so this reads exactly the rows
    the engine read -- not a re-pull that may have been revised since.
    """
    rows = []
    for t in trades or []:
        action = str(getattr(t, "action", ""))
        if action not in FILL_ACTIONS and action not in SETTLEMENT_ACTIONS \
                and action not in STOCK_ACTIONS:
            continue
        c = getattr(t, "contract", None)
        d = pd.Timestamp(getattr(t, "date"))
        n = int(getattr(t, "contracts", 0) or 0)
        cash_after = getattr(t, "cash_after", None)
        if action in STOCK_ACTIONS:
            # A stock leg: `contract` is the bare ticker string, `contracts`
            # is the SHARE count and the price is the spot it sold at. No
            # chain row exists for it and none is looked up.
            rows.append({"date": d, "ticker": str(c or ""), "action": action,
                         "leg": "stock", "strike": None, "expiry": None,
                         "right": None, "contracts": n,
                         "fill_price": float(getattr(t, "price_per_contract",
                                                     float("nan"))),
                         "campaign": int(getattr(t, "campaign_id", 0) or 0),
                         "cash_after": cash_after, "dte": None,
                         "bid": None, "ask": None, "open_interest": None,
                         "volume": None, "row_found": None})
            continue
        tk = str(getattr(c, "root", "") or "")
        ch = chains.get(tk)
        rec = {"date": d, "ticker": tk, "action": action,
               "leg": "settlement" if action in SETTLEMENT_ACTIONS else "option",
               "strike": float(getattr(c, "strike", float("nan"))),
               "expiry": pd.Timestamp(getattr(c, "expiry")),
               "right": str(getattr(c, "right", "")),
               "contracts": n,
               "fill_price": float(getattr(t, "price_per_contract", float("nan"))),
               "campaign": int(getattr(t, "campaign_id", 0) or 0),
               "cash_after": cash_after}
        rec["dte"] = int((rec["expiry"] - d).days)
        if action in SETTLEMENT_ACTIONS:
            # Settlement events are recorded, never checked: no liquidity was
            # consumed, so no chain row is looked up and every market column
            # stays empty. row_found=None keeps them out of rows_not_found.
            rec.update(bid=None, ask=None, open_interest=None, volume=None,
                       row_found=None)
            rows.append(rec)
            continue
        row = _chain_row(ch, d, rec)
        if row is None:
            # The engine filled from a row this lookup cannot find. That is a
            # finding, not a blank: it means the check and the fill disagree
            # about what was on the screen.
            rec.update(bid=None, ask=None, open_interest=None, volume=None,
                       row_found=False)
        else:
            bid, ask = _num(row, "bid"), _num(row, "ask")
            rec.update(bid=bid, ask=ask, row_found=True,
                       open_interest=_num(row, "open_interest"),
                       volume=_num(row, "volume"))
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
                rec["spread"] = ask - bid
                rec["rel_spread"] = (ask - bid) / mid if mid > 0 else None
        oi, vol = rec.get("open_interest"), rec.get("volume")
        rec["pct_of_oi"] = (n / oi) if oi else None
        rec["pct_of_volume"] = (n / vol) if vol else None
        rec["zero_volume"] = (vol == 0) if vol is not None else None
        rows.append(rec)
    return pd.DataFrame(rows)


def _chain_row(ch, d, rec):
    if ch is None:
        return None
    m = ((ch["date"] == d) & (ch["expiry"] == rec["expiry"])
         & (ch["strike"] == rec["strike"]) & (ch["right"] == rec["right"]))
    sel = ch[m]
    if "held_only" in sel.columns:
        # Same exclusion as liquidity_ok: a spliced mark-only row must never
        # answer for a tradeable contract.
        sel = sel[~sel["held_only"].fillna(False).astype(bool)]
    return None if sel.empty else sel.iloc[0]


def _num(row, key):
    try:
        v = float(row[key])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None if v != v else v


def summarise(df: pd.DataFrame) -> dict:
    """The distribution, not the worst case. Percentiles rather than a mean,
    because one 700-contract order among 300 sane ones is exactly the shape
    this is looking for and a mean would hide it."""
    if df is None or df.empty:
        return {"fills": 0}
    # A-4: share legs ride in the same frame but are NOT fills -- every
    # statistic below runs on the option legs only, exactly as before they
    # existed. Frames from older exports have no `leg` column; treat all
    # their rows as option legs, which is what they are.
    legs = df["leg"] if "leg" in df.columns else pd.Series("option", index=df.index)
    opt = df[legs == "option"]
    out = {"fills": int(len(opt)),
           "settlements": int((legs == "settlement").sum()),
           "stock_fills": int((legs == "stock").sum()),
           # .astype(bool) matters: with share legs in the frame the column
           # is object-dtyped and `~` would do integer bitwise-NOT (-2 per
           # True) instead of boolean negation.
           "rows_not_found": int((~opt["row_found"].fillna(False)
                                  .astype(bool)).sum())
           if not opt.empty else 0}
    if opt.empty:
        return out
    df = opt
    entries = df[df["action"].isin(("SELL_PUT", "SELL_CALL", "ROLL_OPEN"))]
    exits = df[df["action"].isin(("CLOSE_PUT", "CLOSE_CALL", "ROLL_CLOSE"))]
    out["entries"] = int(len(entries))
    out["exits"] = int(len(exits))
    for label, part in (("all", df), ("entry", entries), ("exit", exits)):
        if part.empty:
            continue
        for field in ("pct_of_volume", "pct_of_oi", "rel_spread",
                      "open_interest", "volume"):
            s = part[field].dropna() if field in part.columns else pd.Series([], dtype=float)
            if s.empty:
                continue
            out[f"{label}_{field}_median"] = float(s.median())
            out[f"{label}_{field}_p90"] = float(s.quantile(0.90))
            out[f"{label}_{field}_max"] = float(s.max())
    out["zero_volume_fills"] = int(df["zero_volume"].fillna(False).sum())
    out["over_volume_cap"] = int(
        (df["pct_of_volume"].fillna(0) > FLAG_PCT_OF_VOLUME).sum())
    out["over_oi_cap"] = int((df["pct_of_oi"].fillna(0) > FLAG_PCT_OF_OI).sum())
    out["wide_spread"] = int(
        (df["rel_spread"].fillna(0) > FLAG_REL_SPREAD).sum()) \
        if "rel_spread" in df.columns else 0
    return out


def render(summary: dict, title: str = "") -> str:
    if not summary or not summary.get("fills"):
        return "  no fills to check"
    L = [f"  fill realism{(' -- ' + title) if title else ''}",
         f"    {summary['fills']} fills "
         f"({summary.get('entries', 0)} entries, {summary.get('exits', 0)} exits)"]
    if summary.get("rows_not_found"):
        L.append(f"    !! {summary['rows_not_found']} fills whose chain row "
                 f"could not be found -- check and engine disagree")
    def line(label, key, pct=False, cur=False):
        med, p90, mx = (summary.get(f"{key}_median"), summary.get(f"{key}_p90"),
                        summary.get(f"{key}_max"))
        if med is None:
            return
        f = (lambda v: f"{v:6.1%}") if pct else (lambda v: f"{v:8.1f}")
        L.append(f"    {label:<28} median {f(med)}  p90 {f(p90)}  max {f(mx)}")
    line("order as % of day volume", "entry_pct_of_volume", pct=True)
    line("order as % of open interest", "entry_pct_of_oi", pct=True)
    line("open interest at entry", "entry_open_interest")
    line("volume at entry", "entry_volume")
    line("bid-ask, % of mid (entry)", "entry_rel_spread", pct=True)
    line("bid-ask, % of mid (exit)", "exit_rel_spread", pct=True)
    L += [f"    fills into a strike that traded 0 today: "
          f"{summary.get('zero_volume_fills', 0)}",
          f"    fills above {int(FLAG_PCT_OF_VOLUME*100)}% of the day's volume: "
          f"{summary.get('over_volume_cap', 0)}",
          f"    fills above {int(FLAG_PCT_OF_OI*100)}% of open interest: "
          f"{summary.get('over_oi_cap', 0)}",
          f"    fills with a spread over {int(FLAG_REL_SPREAD*100)}% of mid: "
          f"{summary.get('wide_spread', 0)}"]
    return "\n".join(L)
