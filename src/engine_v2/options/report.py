"""Reporting for the wheel backtest: headline + recent + year-by-year metrics,
buy-hold benchmarks (the chain's own underlying, plus real SPY when its data is
on disk), and trade-log stats. Reuses the frequency-aware metrics_simple.
No verdict/gate — diagnostics only."""
from __future__ import annotations
import os
from dataclasses import dataclass
import pandas as pd
from .wheel import underlying_series
from ..backtest import metrics_simple as m

@dataclass
class WheelReport:
    metrics: dict
    recent: dict
    yearly_return: pd.Series
    yearly_sharpe: pd.Series
    benchmark_underlying: dict
    benchmark_underlying_recent: dict
    benchmark_spy: dict | None
    stats: dict
    periods_per_year: float
    ticker: str = "SPY"
    recent_is_full: bool = False
    defense: dict | None = None   # campaign-level defense stats; None on plain runs
    gates: dict | None = None     # regime-gate counts; None when no gate armed

@dataclass
class RouterReport:
    metrics: dict
    posture: dict
    fills: dict
    benchmark_underlying: dict
    yearly_return: pd.Series
    blotter: pd.DataFrame
    ticker: str = "SPY"

def buy_hold_curve(chain, starting_capital) -> pd.Series:
    """Buy-hold the chain's OWN underlying (was spy_buy_hold, which silently
    benchmarked GDX against GDX on non-SPY chains)."""
    und = underlying_series(chain)
    return (starting_capital * und / und.iloc[0]).rename("buy_hold")

def spy_curve(starting_capital, index, path="data/options/spy_greeks_eod_all.parquet"):
    """Buy-hold real SPY, reindex-ffilled to `index` — the honest cross-ticker
    benchmark. None if the SPY file is missing or the dates don't overlap."""
    if not os.path.exists(path):
        return None
    und = pd.read_parquet(path, columns=["date", "underlying"]).groupby("date")["underlying"].first()
    und = und.reindex(index).ffill().dropna()
    if und.empty:
        return None
    return (starting_capital * und / und.iloc[0]).rename("spy_buy_hold")

def _perf(equity, ppy) -> dict:
    rets = equity.pct_change().fillna(0.0)
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else float("nan"),
        "cagr": m.cagr(equity, ppy),
        "sharpe": m.sharpe(rets, ppy),
        "max_drawdown": m.max_drawdown(equity),
    }

def wheel_stats(result, cfg) -> dict:
    trades = result.trades
    mult = cfg.contract_multiplier
    def of(a): return [t for t in trades if t.action == a]
    # rolled legs are real entries/exits; stops are real exits. Plain runs never
    # produce these actions, so plain net_premium is unchanged.
    sells = of("SELL_PUT") + of("SELL_CALL") + of("ROLL_OPEN")
    closes = of("CLOSE_PUT") + of("CLOSE_CALL") + of("ROLL_CLOSE") + of("STOP_CLOSE")
    prem_in = sum(t.price_per_contract * mult * t.contracts for t in sells)
    prem_out = sum(t.price_per_contract * mult * t.contracts for t in closes)
    commission = cfg.friction_per_contract * sum(t.contracts for t in sells + closes)
    n_puts = len(of("SELL_PUT"))
    return {
        "n_puts_sold": n_puts, "n_calls_sold": len(of("SELL_CALL")),
        "n_assignments": len(of("ASSIGNED")), "n_called_away": len(of("CALLED_AWAY")),
        "n_take_profits": len(of("CLOSE_PUT")) + len(of("CLOSE_CALL")),  # TP only, not rolls/stops
        "n_expired": len(of("PUT_EXPIRED")) + len(of("CALL_EXPIRED")),
        "premium_collected": prem_in, "premium_paid_to_close": prem_out,
        "commission_paid": commission,
        "net_premium": prem_in - prem_out - commission,
        "assignment_rate": (len(of("ASSIGNED")) / n_puts) if n_puts else 0.0,
        # entry legs only: the realized-DTE chart proves ENTRY selection
        # determinism against the derived band; rolled-in legs sit deliberately
        # beyond it (held expiry + target) and would pollute the proof.
        "realized_dte": pd.Series(
            [(pd.Timestamp(t.contract.expiry) - pd.Timestamp(t.date)).days
             for t in of("SELL_PUT") + of("SELL_CALL")],
            dtype=int).value_counts().sort_index(),
        "n_days_flat": result.days_flat,
        "pct_days_flat": (result.days_flat / len(result.equity)) if len(result.equity) else 0.0,
        "n_rolls": len(of("ROLL_CLOSE")),
        "n_stops": len(of("STOP_CLOSE")),
        "n_campaigns": len({t.campaign_id for t in trades if t.campaign_id}),
        "n_warnings": len(getattr(result, "warnings", []) or []),
        # warnings by type — 'skipped check' and 'late expiry resolution' are
        # different data-quality signals; never conflate them in a report.
        "n_no_mark_days": sum(1 for w in (getattr(result, "warnings", []) or [])
                              if w[1] in ("roll_check_no_mark", "stop_check_no_mark")),
        "n_late_expiries": sum(1 for w in (getattr(result, "warnings", []) or [])
                               if w[1] == "expiry_resolved_late"),
        "days_shares_uncovered": getattr(result, "days_shares_uncovered", 0),
    }

def _trade_cash_flow(t, cfg) -> float:
    """Cash flow of one trade from its own economics — NOT cash_after deltas,
    which would fold cash-yield interest (earned on the whole balance, campaign
    or not) into per-campaign P&L."""
    mult, comm = cfg.contract_multiplier, cfg.friction_per_contract
    px, n = t.price_per_contract, t.contracts
    if t.action in ("SELL_PUT", "SELL_CALL", "ROLL_OPEN"):
        return px * mult * n - comm * n
    if t.action in ("CLOSE_PUT", "CLOSE_CALL", "ROLL_CLOSE", "STOP_CLOSE"):
        return -(px * mult * n + comm * n)
    if t.action == "ASSIGNED":
        return -t.contract.strike * mult * n
    if t.action == "CALLED_AWAY":
        return t.contract.strike * mult * n
    if t.action == "LIQUIDATE":
        return px * mult * n
    return 0.0   # PUT_EXPIRED / CALL_EXPIRED

def campaign_table(result, cfg) -> pd.DataFrame:
    """One row per campaign: P&L summed from each trade's own cash flow (exact
    for campaigns that ended flat; the final campaign may still be open ->
    flagged, unrealized excluded)."""
    rows, cur = [], None
    for t in result.trades:
        if cur is None or t.campaign_id != cur["campaign_id"]:
            if cur is not None:
                rows.append(cur)
            cur = dict(campaign_id=t.campaign_id, opened=t.date, closed=t.date,
                       n_trades=0, n_rolls=0, pnl=0.0, open_at_end=False)
        cur["n_trades"] += 1
        cur["closed"] = t.date
        cur["pnl"] += _trade_cash_flow(t, cfg)
        if t.action == "ROLL_CLOSE":
            cur["n_rolls"] += 1
    if cur is not None:
        # the last campaign is open iff something is still on the book
        cur["open_at_end"] = (result.final_shares > 0) or result.residual_settled
        rows.append(cur)
    return pd.DataFrame(rows, columns=["campaign_id","opened","closed","n_trades",
                                       "n_rolls","pnl","open_at_end"])

def roll_counterfactuals(result, chain, cfg) -> pd.DataFrame:
    """Short-leg-only counterfactual for every leg closed by a roll: what the
    close actually realized vs what holding THAT leg to its own expiry would
    have settled at (credit - intrinsic). Does NOT model the post-assignment
    path — labeled approximation, per the repair spec."""
    mult, comm = cfg.contract_multiplier, cfg.friction_per_contract
    und = underlying_series(chain)
    opens, rows = {}, []
    skipped = 0
    for t in result.trades:
        if t.action in ("SELL_PUT", "SELL_CALL", "ROLL_OPEN"):
            opens[(t.contract, t.campaign_id)] = t
        elif t.action == "ROLL_CLOSE":
            o = opens.get((t.contract, t.campaign_id))
            exp = pd.Timestamp(t.contract.expiry)
            if exp not in und.index:
                # expiry day absent from the chain (data gap) — settle at the
                # LAST close at/before expiry, matching the engine's own
                # late-resolution moneyness rule (wheel.py).
                pre = und.index[und.index <= exp]
                exp = pre[-1] if len(pre) else None
            if o is None or exp is None:
                skipped += 1
                continue
            n = t.contracts
            intrinsic = max(t.contract.strike - float(und[exp]), 0.0)
            actual = (o.price_per_contract - t.price_per_contract) * mult * n - 2 * comm * n
            held = (o.price_per_contract - intrinsic) * mult * n - comm * n
            rows.append(dict(campaign_id=t.campaign_id, closed=t.date,
                             strike=float(t.contract.strike),
                             actual_close_pnl=actual, held_to_expiry_pnl=held,
                             roll_advantage=actual - held))
    df = pd.DataFrame(rows, columns=["campaign_id","closed","strike",
                                     "actual_close_pnl","held_to_expiry_pnl",
                                     "roll_advantage"])
    df.attrs["n_skipped"] = skipped   # legs dropped (expiry beyond data window)
    return df

def wheel_report(result, chain, cfg, recent_start="2021-07-01",
                 spy_path="data/options/spy_greeks_eod_all.parquet") -> WheelReport:
    eq = result.equity
    ppy = m.infer_periods_per_year(eq.index)
    rs = max(pd.Timestamp(recent_start), eq.index.min())
    eq_recent = eq[eq.index >= rs]
    bh = buy_hold_curve(chain, cfg.starting_capital).reindex(eq.index).ffill()
    spy = spy_curve(cfg.starting_capital, eq.index, path=spy_path)
    recent_is_full = rs <= eq.index.min()
    bh_recent = bh[bh.index >= rs]
    benchmark_recent = _perf(bh_recent, ppy) if len(bh_recent) > 1 else _perf(bh, ppy)
    stats = wheel_stats(result, cfg)
    defense = None
    # gate on the ENABLED flags, not on fired-event counts: a defense that ran
    # but never triggered (or was data-blocked) must still show its block —
    # that absence-of-fire is itself the finding.
    if (cfg.roll_tested_puts or cfg.put_stop_mult is not None
            or cfg.liquidate_assignment or cfg.call_min_strike):
        ct = campaign_table(result, cfg)
        cf = roll_counterfactuals(result, chain, cfg)
        closed = ct[~ct["open_at_end"]]
        defense = {
            "n_campaigns": stats["n_campaigns"],
            "campaign_win_rate": float((closed["pnl"] > 0).mean()) if len(closed) else float("nan"),
            "rolls_per_campaign": ct["n_rolls"].value_counts().sort_index().to_dict(),
            "n_stops": stats["n_stops"],
            "roll_advantage_total": float(cf["roll_advantage"].sum()) if len(cf) else 0.0,
            "roll_advantage_positive": int((cf["roll_advantage"] > 0).sum()) if len(cf) else 0,
            "roll_advantage_negative": int((cf["roll_advantage"] < 0).sum()) if len(cf) else 0,
            "roll_counterfactual_skipped": int(cf.attrs.get("n_skipped", 0)),
            "days_shares_uncovered": stats["days_shares_uncovered"],
            "n_warnings": stats["n_warnings"],
            "n_no_mark_days": stats["n_no_mark_days"],
            "n_late_expiries": stats["n_late_expiries"],
            "liquidate_fill_note": bool(cfg.liquidate_assignment),
        }
    gates = None
    # same visibility rule as the defense block: an ARMED gate that never fired
    # must still show its block — the absence of fires is itself the finding.
    if cfg.any_regime_gate:
        ev = getattr(result, "gate_events", []) or []
        wn = getattr(result, "warnings", []) or []
        gates = {
            "days_entry_gated": getattr(result, "days_entry_gated", 0),
            "n_rolls_denied": sum(1 for e in ev if e[1] == "roll_denied_by_gate"),
            "n_stops_suppressed": sum(1 for e in ev if e[1] == "stop_suppressed_by_gate"),
            "n_state_unknown": sum(1 for w in wn if w[1] == "gate_state_unknown"),
        }
    return WheelReport(
        metrics=_perf(eq, ppy),
        recent=_perf(eq_recent, ppy) if len(eq_recent) > 1 else _perf(eq, ppy),
        yearly_return=m.yearly_returns(eq),
        yearly_sharpe=m.yearly_sharpe(eq.pct_change().fillna(0.0), ppy),
        benchmark_underlying=_perf(bh, ppy),
        benchmark_underlying_recent=benchmark_recent,
        benchmark_spy=_perf(spy, ppy) if spy is not None else None,
        recent_is_full=recent_is_full,
        stats=stats,
        periods_per_year=ppy,
        ticker=cfg.ticker,
        defense=defense,
        gates=gates,
    )

def router_report(result, chain, cfg) -> RouterReport:
    """Report for a RouterResult. Decoupled from wheel_report, which reads
    result.days_flat (absent on RouterResult). Metrics use the same
    metrics_simple functions as scripts/run_regime_router.py, so the page and
    the CLI agree by construction."""
    eq = result.equity
    ppy = m.infer_periods_per_year(eq.index)
    rets = eq.pct_change().fillna(0.0)
    metrics = {
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "cagr": m.cagr(eq, ppy),
        "sharpe": m.sharpe(rets, ppy),
        "max_drawdown": m.max_drawdown(eq),
    }
    rlog = result.route_log or []
    transitions = sum(1 for a, b in zip(rlog, rlog[1:]) if a[3] != b[3])
    unknown = sum(1 for w in (result.warnings or []) if w[1] == "route_state_unknown")
    posture = {
        "days": result.days_in_posture,
        "transitions": transitions,
        "whipsaws": result.whipsaw_pairs,
        "unknown": unknown,
    }
    fills = {"intraday_tp": result.intraday_tp_fills, "eod_tp": result.eod_tp_fills}
    bh = buy_hold_curve(chain, cfg.starting_capital)
    benchmark_underlying = {"total_return": float(bh.iloc[-1] / bh.iloc[0] - 1)}
    return RouterReport(
        metrics=metrics, posture=posture, fills=fills,
        benchmark_underlying=benchmark_underlying,
        yearly_return=m.yearly_returns(eq),
        blotter=position_log(result, cfg),
        ticker=cfg.ticker,
    )

def _pct(x): return "n/a" if pd.isna(x) else f"{x:+.2%}"
def _num(x): return "n/a" if pd.isna(x) else f"{x:.2f}"

def format_report(rep) -> str:
    L = []
    L.append("WHEEL BACKTEST REPORT")
    L.append("=" * 40)
    def block(title, d):
        L.append(f"\n{title}")
        L.append(f"  CAGR {_pct(d['cagr'])}   Sharpe {_num(d['sharpe'])}   "
                 f"Max drawdown {_pct(d['max_drawdown'])}   Total {_pct(d['total_return'])}")
    block("Headline (recent window)", rep.recent)
    block(f"  vs buy-hold {rep.ticker} (recent window)", rep.benchmark_underlying_recent)
    if rep.recent_is_full:
        L.append("  (recent window = full history — span too short to separate)")
    block("Full history", rep.metrics)
    block(f"Benchmark — buy-hold {rep.ticker}", rep.benchmark_underlying)
    if rep.benchmark_spy is not None:
        block("Benchmark — buy-hold SPY", rep.benchmark_spy)
    L.append("\nYear-by-year (return / Sharpe)")
    for y in rep.yearly_return.index:
        L.append(f"  {y}: {_pct(rep.yearly_return[y])}  /  Sharpe {_num(rep.yearly_sharpe.get(y, float('nan')))}")
    s = rep.stats
    L.append("\nWheel stats")
    L.append(f"  puts sold {s['n_puts_sold']}  calls sold {s['n_calls_sold']}  "
             f"assignments {s['n_assignments']} (rate {s['assignment_rate']:.0%})  "
             f"called away {s['n_called_away']}  take-profits {s['n_take_profits']}  "
             f"flat {s['pct_days_flat']:.0%} of days")
    L.append(f"  premium collected {s['premium_collected']:.0f}  "
             f"paid to close {s['premium_paid_to_close']:.0f}  "
             f"commission {s['commission_paid']:.0f}  net {s['net_premium']:.0f}")
    dtes = s["realized_dte"]
    if len(dtes):
        exp = dtes.index.repeat(dtes.values).to_series()
        L.append(f"  realized DTE: {dtes.index.min()}..{dtes.index.max()}  "
                 f"(median {exp.median():.0f})")
    if rep.defense:
        dd = rep.defense
        L.append("\nDefense stats (campaign-level)")
        wr = dd['campaign_win_rate']
        wr_s = "n/a" if pd.isna(wr) else f"{wr:.0%}"
        L.append(f"  campaigns {dd['n_campaigns']}  win rate {wr_s}  "
                 f"stops {dd['n_stops']}  rolls/campaign {dd['rolls_per_campaign']}")
        skip_s = (f"  [{dd['roll_counterfactual_skipped']} leg(s) beyond data window]"
                  if dd.get('roll_counterfactual_skipped') else "")
        L.append(f"  roll counterfactual (short-leg approx): total advantage "
                 f"{dd['roll_advantage_total']:+.0f}  "
                 f"(helped {dd['roll_advantage_positive']}, hurt {dd['roll_advantage_negative']})"
                 + skip_s)
        L.append(f"  days shares uncovered {dd['days_shares_uncovered']}  "
                 f"no-mark days {dd['n_no_mark_days']}  "
                 f"late expiry resolutions {dd['n_late_expiries']}")
        if dd["liquidate_fill_note"]:
            L.append("  note: liquidation fills at EOD spot — no stock spread/slippage modeled "
                     "(options pay full spread)")
    if rep.gates is not None:
        g = rep.gates
        L.append("\nRegime gates (ticker state, strictly-prior-day)")
        L.append(f"  entry-gated days {g['days_entry_gated']}  "
                 f"rolls denied {g['n_rolls_denied']}  "
                 f"stops suppressed {g['n_stops_suppressed']}  "
                 f"state-unknown at would-act moments {g['n_state_unknown']}")
        L.append("  note: gated-entry counterfactual is not modeled — the paired "
                 "A/B run is the measurement")
    return "\n".join(L)

_RIGHT_WORD = {"P": "PUT", "C": "CALL"}
_TERM = {"CLOSE_PUT", "CLOSE_CALL", "PUT_EXPIRED", "CALL_EXPIRED", "ASSIGNED",
         "CALLED_AWAY", "ROLL_CLOSE", "STOP_CLOSE"}

def position_log(result, cfg) -> pd.DataFrame:
    mult, comm = cfg.contract_multiplier, cfg.friction_per_contract
    rows, open_opt, assign = [], None, None
    for t in result.trades:
        if t.action in ("SELL_PUT", "SELL_CALL", "ROLL_OPEN"):
            open_opt = t
        elif t.action in _TERM and open_opt is not None:
            oc, n = open_opt.contract, open_opt.contracts
            credit = open_opt.price_per_contract * mult * n - comm * n
            if t.action in ("CLOSE_PUT", "CLOSE_CALL"):
                cost, outcome = t.price_per_contract * mult * n + comm * n, "Took profit"
            elif t.action == "ROLL_CLOSE":
                cost, outcome = t.price_per_contract * mult * n + comm * n, "Rolled"
            elif t.action == "STOP_CLOSE":
                cost, outcome = t.price_per_contract * mult * n + comm * n, "Stopped"
            elif t.action in ("PUT_EXPIRED", "CALL_EXPIRED"):
                cost, outcome = 0.0, "Expired worthless"
            elif t.action == "ASSIGNED":
                cost, outcome = 0.0, "Assigned"
                assign = (oc.strike, n, t.date, t.campaign_id)
            else:  # CALLED_AWAY
                cost, outcome = 0.0, "Called away"
            realized = credit - cost
            rows.append(dict(opened=open_opt.date, closed=t.date,
                instrument=_RIGHT_WORD[oc.right], strike=oc.strike, expiry=oc.expiry,
                qty=n, credit=credit, outcome=outcome, cost_to_close=cost,
                realized_pnl=realized,
                pct_of_credit=(realized / credit if credit else 0.0),
                days_held=(t.date - open_opt.date).days,
                campaign_id=t.campaign_id))
            if t.action == "CALLED_AWAY" and assign is not None:
                astrike, aqty, adate, _acid = assign
                rows.append(dict(opened=adate, closed=t.date, instrument="SHARES",
                    strike=astrike, expiry=pd.NaT, qty=aqty,
                    credit=-astrike * mult * aqty, outcome="Called away",
                    cost_to_close=oc.strike * mult * aqty,
                    realized_pnl=(oc.strike - astrike) * mult * aqty,
                    pct_of_credit=float("nan"), days_held=(t.date - adate).days,
                    campaign_id=t.campaign_id))
                assign = None
            open_opt = None
        elif t.action == "LIQUIDATE" and assign is not None:
            # shares dumped at spot the moment assignment fired (LIQUIDATE is a
            # shares closure, not an option terminal — the put row was already
            # written by the ASSIGNED trade).
            astrike, aqty, adate, _acid = assign
            spot = t.price_per_contract
            rows.append(dict(opened=adate, closed=t.date, instrument="SHARES",
                strike=astrike, expiry=pd.NaT, qty=aqty,
                credit=-astrike * mult * aqty, outcome="Liquidated",
                cost_to_close=spot * mult * aqty,
                realized_pnl=(spot - astrike) * mult * aqty,
                pct_of_credit=float("nan"), days_held=(t.date - adate).days,
                campaign_id=t.campaign_id))
            assign = None
    if open_opt is not None:
        oc, n = open_opt.contract, open_opt.contracts
        credit = open_opt.price_per_contract * mult * n - comm * n
        prior = sum(r["realized_pnl"] for r in rows if pd.notna(r["realized_pnl"]))
        realized = (result.final_cash - cfg.starting_capital) - prior
        rows.append(dict(opened=open_opt.date, closed=pd.NaT,
            instrument=_RIGHT_WORD[oc.right], strike=oc.strike, expiry=oc.expiry, qty=n,
            credit=credit, outcome="Settled at mark", cost_to_close=credit - realized,
            realized_pnl=realized, pct_of_credit=(realized / credit if credit else 0.0),
            days_held=float("nan"), campaign_id=open_opt.campaign_id))
        open_opt = None
    if assign is not None and getattr(result, "final_shares", 0) > 0:
        astrike, aqty, adate, acid = assign
        rows.append(dict(opened=adate, closed=pd.NaT, instrument="SHARES",
            strike=astrike, expiry=pd.NaT, qty=aqty, credit=-astrike * mult * aqty,
            outcome="Open", cost_to_close=float("nan"), realized_pnl=float("nan"),
            pct_of_credit=float("nan"), days_held=float("nan"), campaign_id=acid))
    cols = ["opened","closed","instrument","strike","expiry","qty","credit","outcome",
            "cost_to_close","realized_pnl","pct_of_credit","days_held","campaign_id"]
    return pd.DataFrame(rows, columns=cols)

_PORTFOLIO_OPENS = {"SELL_PUT", "SELL_CALL", "ROLL_OPEN"}
_PORTFOLIO_TERMINAL = {"CLOSE_PUT", "CLOSE_CALL", "PUT_EXPIRED", "CALL_EXPIRED",
                       "ASSIGNED", "CALLED_AWAY", "ROLL_CLOSE", "STOP_CLOSE"}

def portfolio_campaign_table(result, cfg, last_spots) -> pd.DataFrame:
    """One row per campaign for a PortfolioResult. campaign_table can't be
    reused: portfolio campaigns interleave across tickers by date (not
    contiguous) and final_shares is a dict. Groups by campaign_id. pnl_realized
    is exact cash flow; pnl_mtm marks any still-held shares at last_spots AND
    any still-open short option (opens outnumber terminal closes) to intrinsic
    value at the last spot — an open short is a liability, not a finished win."""
    mult, comm = cfg.contract_multiplier, cfg.friction_per_contract
    agg = {}
    for t in result.trades:
        cid = t.campaign_id
        c = agg.get(cid)
        if c is None:
            c = agg[cid] = dict(campaign_id=cid, ticker=t.contract.root,
                                opened=t.date, closed=t.date, n_trades=0,
                                pnl_realized=0.0, shares_held=0, collateral=0.0,
                                opens_count=0, terminal_count=0, last_open_contract=None,
                                last_open_n=0)
        c["n_trades"] += 1
        c["closed"] = t.date
        a, n, px = t.action, t.contracts, t.price_per_contract
        if a in _PORTFOLIO_OPENS:
            c["opens_count"] += 1
            c["last_open_contract"] = t.contract
            c["last_open_n"] = n
        if a in _PORTFOLIO_TERMINAL:
            c["terminal_count"] += 1
        if a in ("SELL_PUT", "SELL_CALL"):
            c["pnl_realized"] += px * mult * n - comm * n
            if a == "SELL_PUT" and c["collateral"] == 0.0:
                c["collateral"] = t.contract.strike * mult * n
        elif a in ("CLOSE_PUT", "CLOSE_CALL"):
            c["pnl_realized"] -= px * mult * n + comm * n
        elif a == "ASSIGNED":
            c["pnl_realized"] -= t.contract.strike * mult * n
            c["shares_held"] += mult * n
        elif a == "CALLED_AWAY":
            c["pnl_realized"] += t.contract.strike * mult * n
            c["shares_held"] -= mult * n
    rows = []
    for c in agg.values():
        held = c["shares_held"]
        has_open_short = c["opens_count"] > c["terminal_count"]
        c["open_at_end"] = (held > 0) or has_open_short
        last_spot = last_spots.get(c["ticker"], 0.0)
        liability = 0.0
        oc = c["last_open_contract"]
        if has_open_short and oc is not None:
            n = c["last_open_n"]
            if oc.right == "P":
                liability = max(oc.strike - last_spot, 0.0) * mult * n
            else:
                liability = max(last_spot - oc.strike, 0.0) * mult * n
        c["pnl_mtm"] = c["pnl_realized"] + held * last_spot - liability
        c["pct_return"] = c["pnl_realized"] / c["collateral"] if c["collateral"] else 0.0
        del c["opens_count"], c["terminal_count"], c["last_open_contract"], c["last_open_n"]
        rows.append(c)
    return pd.DataFrame(rows, columns=["campaign_id", "ticker", "opened", "closed",
        "n_trades", "pnl_realized", "shares_held", "collateral", "open_at_end",
        "pnl_mtm", "pct_return"])

def portfolio_summary_stats(campaigns) -> dict:
    """Twin win-rate + tile numbers from a portfolio_campaign_table frame.
    finished_win_rate hides nothing dishonestly (closed campaigns only);
    soldtoday_win_rate marks open positions to market so hidden losses surface.
    The gap between them is the honesty signal."""
    if len(campaigns) == 0:
        return {"finished_win_rate": float("nan"), "soldtoday_win_rate": float("nan"),
                "avg_pct_per_win": float("nan"), "n_open": 0, "n_campaigns": 0}
    closed = campaigns[~campaigns["open_at_end"]]
    winners = closed[closed["pnl_realized"] > 0]
    return {
        "finished_win_rate": float((closed["pnl_realized"] > 0).mean()) if len(closed) else float("nan"),
        "soldtoday_win_rate": float((campaigns["pnl_mtm"] > 0).mean()) if len(campaigns) else float("nan"),
        "avg_pct_per_win": float(winners["pct_return"].mean()) if len(winners) else float("nan"),
        "n_open": int(campaigns["open_at_end"].sum()),
        "n_campaigns": int(len(campaigns)),
    }
