"""Regime router (Regime Bot v1, spec 2026-07-14-regime-router-design): a
per-ticker strategy router. Uptrend -> hold shares; chop -> wheel + basis
floor; downtrend+stressed -> wheel; downtrend+quiet -> cash; unknown -> wheel.
Approach A borders: no forced exits on regime flips except trend shares on a
direct flip to downtrend; wheel-assigned shares are never regime-sold
(siege-exit falsification is binding precedent); covered calls open only in
wheel cells. Zero knobs, EOD fills. The WHEEL posture is a faithful transplant
of the solo plain+basis path — all-chop states must reproduce run_wheel
byte-identically (the anchor regression)."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .select import (select_contract, option_mark, liquidity_ok,
                     at_risky_window_edge)
from .fills import try_take_profit, tp_exit_feasible, write_credit_ok
from .wheel import (Trade, WheelConfig, is_unpaid_decline, _state_before,
                    sell_proceeds)

TRIM_FRACTION = 0.5   # conviction trim (spec 2026-07-15): stressed-vol trend
                      # HOLDs are bought at half size — tail control only. Round
                      # and deliberately unfished.


@dataclass
class RouterResult:
    equity: pd.Series
    trades: list
    final_cash: float
    final_shares: int
    residual_settled: bool = False
    warnings: list = None
    route_log: list = None        # (date, trend, vol, posture) per day
    days_in_posture: dict = None  # {"CASH": n, "TREND": n, "WHEEL": n}
    days_shares_uncovered: int = 0
    whipsaw_pairs: int = 0        # SELL_SHARES <= 10 trading days after BUY_SHARES
    n_trimmed_entries: int = 0    # half-size trend HOLD entries (conviction trim)
    days_half_size: int = 0       # days holding a trimmed trend position
    intraday_tp_fills: int = 0    # TP closes filled at a next-valid hourly bar
    eod_tp_fills: int = 0         # TP closes filled at EOD ask (no intraday hit)


def _cell(states: pd.DataFrame, d: pd.Timestamp):
    """(cell, trend, vol, unknown) from the strictly-prior-day state.
    Cells: TREND (uptrend), WHEEL (chop, downtrend+stressed, unknown),
    CASH (downtrend + calm/normal)."""
    trend, vol = _state_before(states, d)
    if (trend, vol) == ("unknown", "unknown"):
        return "WHEEL", trend, vol, True
    if trend == "uptrend":
        return "TREND", trend, vol, False
    if is_unpaid_decline(trend, vol):
        return "CASH", trend, vol, False
    return "WHEEL", trend, vol, False   # chop (any vol) or downtrend+stressed


def run_regime_router(chain: pd.DataFrame, cfg: WheelConfig,
                      regime_states: pd.DataFrame, intraday=None) -> RouterResult:
    if regime_states is None:
        raise ValueError("regime_states is required — a router without weather "
                         "is a bug, not a run")
    if cfg.roll_tested_puts or cfg.put_stop_mult is not None or \
            cfg.liquidate_assignment or cfg.any_regime_gate:
        raise ValueError("router v1 runs the plain+basis wheel only — "
                         "roll/stop/gates/liquidate are solo mechanics")

    dates = sorted(pd.to_datetime(chain["date"]).unique())
    und = chain.groupby("date")["underlying"].first()
    by_date = {pd.Timestamp(k): g for k, g in chain.groupby("date")}
    mult = cfg.contract_multiplier

    cash, campaign = cfg.starting_capital, 0
    # wheel sub-state, verbatim solo fields
    short, shares, phase, basis = None, 0, "PUT", None
    campaign_premium = 0.0
    # trend sub-state
    trend_shares, trend_buy_idx = 0, None   # idx into dates of the BUY fill
    trend_trimmed = False                   # was the current trend hold half-sized?
    whipsaw_pairs = 0
    n_trimmed_entries, days_half_size = 0, 0
    warnings, route_log, trades, equity = [], [], [], {}
    days_in_posture = {"CASH": 0, "TREND": 0, "WHEEL": 0}
    prev_d, days_shares_uncovered = None, 0
    unknown_logged_days = set()
    intraday_tp_fills, eod_tp_fills = 0, 0

    for i, d in enumerate(dates):
        d = pd.Timestamp(d)
        spot = float(und.get(d))
        day_chain = by_date.get(d)
        assigned_today = False   # A9: assignment defers the call one session
        if prev_d is not None and cfg.cash_yield > 0:
            cash *= (1 + cfg.cash_yield / 365) ** (d - prev_d).days
        prev_d = d
        cell, g_trend, g_vol, unknown = _cell(regime_states, d)
        if unknown and d not in unknown_logged_days:
            # every routed unknown day is logged, whatever the posture — the
            # report's unknown count must expose blind stretches, not just
            # blind entries (review 2026-07-14).
            warnings.append((d, "route_state_unknown", cfg.ticker))
            unknown_logged_days.add(d)

        # 1) manage an open short by SOLO rules (never consults the cell)
        closed_today = None
        if short is not None:
            c, n = short["contract"], short["contracts"]
            mark = option_mark(day_chain, d, c)
            # take-profit via the shared fill seam (fills.py, A18) -- print-
            # next-bar mode when bars exist for this contract, else the EOD
            # quote at the ask. Rules (close>0 prints only, fill at bar i+1,
            # last-bar cross falls through) live in try_take_profit.
            key = (pd.Timestamp(c.expiry), float(c.strike), c.right)
            dec = try_take_profit(mark=mark, credit=short["credit"], contracts=n,
                                  cfg=cfg, day=d, expiry=c.expiry,
                                  bars=intraday.get(key) if intraday is not None else None)
            if dec.filled:
                cash -= dec.cost; campaign_premium -= dec.cost
                trades.append(Trade(dec.stamp,
                                    "CLOSE_PUT" if c.right == "P" else "CLOSE_CALL",
                                    c, dec.filled_contracts, dec.price, cash, campaign))
                # A17: decrement the live size; both current modes fill whole,
                # so this reaches zero exactly as the old `short = None` did
                short["contracts"] -= dec.filled_contracts
                if short["contracts"] <= 0:
                    short = None; closed_today = c
                if dec.via == "print":
                    intraday_tp_fills += 1
                else:
                    eod_tp_fills += 1
            if short is not None:
                n = short["contracts"]   # re-read after a (possibly partial) TP fill (A17/I3)
            if short is not None and d >= c.expiry:
                settle_spot = spot
                if d > c.expiry:
                    warnings.append((d, "expiry_resolved_late", c))
                    pre = und[und.index <= c.expiry]
                    if len(pre):
                        settle_spot = float(pre.iloc[-1])
                if c.right == "P":
                    if settle_spot < c.strike:
                        # A13: per-event assignment fee ($0 at Schwab, VERIFY)
                        cash -= c.strike * mult * n + cfg.fee_per_assignment
                        shares += mult * n
                        phase = "CALL"; basis = c.strike
                        assigned_today = True   # A9: no covered call this session
                        trades.append(Trade(d, "ASSIGNED", c, n, c.strike, cash, campaign))
                    else:
                        trades.append(Trade(d, "PUT_EXPIRED", c, n, 0.0, cash, campaign))
                else:
                    if settle_spot > c.strike:
                        cash += c.strike * mult * n - cfg.fee_per_assignment
                        shares -= mult * n
                        phase = "PUT"; basis = None
                        trades.append(Trade(d, "CALLED_AWAY", c, n, c.strike, cash, campaign))
                    else:
                        trades.append(Trade(d, "CALL_EXPIRED", c, n, 0.0, cash, campaign))
                short = None

        # 2) transitions BEFORE entries (spec sequencing rule)
        if trend_shares and g_trend == "downtrend":
            # the single forced exit: the trend was the only thesis, and the
            # state says it is dead. Fill at EOD spot, no stock spread modeled.
            cash += trend_shares * spot
            trades.append(Trade(d, "SELL_SHARES", None, trend_shares // mult,
                                spot, cash, campaign))
            if trend_buy_idx is not None and (i - trend_buy_idx) <= 10:
                whipsaw_pairs += 1
            trend_shares, trend_buy_idx, trend_trimmed = 0, None, False
        elif trend_shares and g_trend == "chop":
            # hand trend shares to the wheel: rent them under the basis floor,
            # basis = purchase price (carried in `basis` at buy time).
            shares += trend_shares
            phase = "CALL"
            trend_shares, trend_buy_idx, trend_trimmed = 0, None, False

        # 3) entries / posture actions
        if short is None and shares == 0 and trend_shares == 0:
            # totally flat: route fresh
            if cell == "TREND":
                lots = int(cash // (spot * mult))
                # conviction trim (spec 2026-07-15): stressed-vol holds carry the
                # same expected return but a fatter crash tail — buy half size.
                trimmed = cfg.conviction_trim and g_vol == "stressed"
                if trimmed:
                    lots = int(lots * TRIM_FRACTION)
                if lots > 0:
                    campaign += 1
                    campaign_premium = 0.0
                    cost = lots * mult * spot
                    cash -= cost
                    trend_shares = lots * mult
                    trend_buy_idx = i
                    trend_trimmed = trimmed
                    basis = spot   # purchase price; used if later handed to the wheel
                    if trimmed:
                        n_trimmed_entries += 1
                    trades.append(Trade(d, "BUY_SHARES", None, lots, spot, cash, campaign))
            elif cell == "WHEEL":
                c = select_contract(day_chain, d, "P", cfg.put_delta,
                                    cfg.target_dte, cfg.ticker)
                mark = option_mark(day_chain, d, c) if c is not None else None
                liq = (c is None or c == closed_today or mark is None
                       or liquidity_ok(day_chain, d, c, cfg)[0])
                if not liq:
                    # A2 skeptic F2: never a silent veto
                    warnings.append((d, "entry_gated_illiquid", cfg.ticker))
                if liq and c is not None and c != closed_today and mark is not None \
                        and not tp_exit_feasible(mark.bid, cfg)[0]:
                    # A3: the entry's own TP exit is unsatisfiable/net-negative
                    warnings.append((d, "entry_gated_unclosable", cfg.ticker))
                    liq = False
                if liq and c is not None and c != closed_today and mark is not None \
                        and not write_credit_ok(mark.bid, cfg)[0]:
                    # tp-INDEPENDENT floor -- see fills.write_credit_floor.
                    warnings.append((d, "entry_gated_no_credit", cfg.ticker))
                    liq = False
                if c is not None and c != closed_today and mark is not None and liq:
                    n = int(cash // (c.strike * mult))
                    if n > 0:
                        campaign += 1
                        campaign_premium = 0.0
                        proceeds = sell_proceeds(mark, n, cfg)
                        cash += proceeds; campaign_premium += proceeds
                        short = {"contract": c, "contracts": n,
                                 "credit": mark.bid, "last_mid": mark.mid}
                        trades.append(Trade(d, "SELL_PUT", c, n, mark.bid, cash, campaign))
                        if at_risky_window_edge(day_chain, d, c, cfg.put_delta):
                            # B11: clipped window -- riskier than configured
                            warnings.append((d, "strike_window_edge", cfg.ticker))
            # cell CASH: nothing — counted below
        elif (short is None and shares >= mult and phase == "CALL"
              and cell == "WHEEL" and day_chain is not None
              and not assigned_today):
            # covered-call entry, solo rules; ONLY in wheel cells (spec rule 6).
            floor = None
            if cfg.call_min_strike == "basis" and basis is not None:
                floor = basis - campaign_premium / shares
            c = select_contract(day_chain, d, "C", cfg.call_delta,
                                cfg.target_dte, cfg.ticker, min_strike=floor)
            mark = option_mark(day_chain, d, c) if c is not None else None
            tp_ok = mark is None or tp_exit_feasible(mark.bid, cfg)[0]
            feasible = tp_ok and (mark is None
                                  or write_credit_ok(mark.bid, cfg)[0])
            if mark is not None and not feasible:
                # A3b: covered call with an unsatisfiable TP exit -- refused.
                # A3b first when both refuse (see portfolio.py for why).
                warnings.append((d, "call_gated_unclosable" if not tp_ok
                                 else "call_gated_no_credit", cfg.ticker))
            if c is not None and c != closed_today and mark is not None and feasible:
                n = shares // mult
                proceeds = sell_proceeds(mark, n, cfg)
                cash += proceeds; campaign_premium += proceeds
                short = {"contract": c, "contracts": n,
                         "credit": mark.bid, "last_mid": mark.mid}
                trades.append(Trade(d, "SELL_CALL", c, n, mark.bid, cash, campaign))

        # 4) posture accounting + equity mark
        if trend_shares:
            posture = "TREND"
        elif short is not None or shares:
            posture = "WHEEL"
        else:
            posture = "CASH"
        days_in_posture[posture] += 1
        if posture == "TREND" and trend_trimmed:
            days_half_size += 1
        route_log.append((d, g_trend, g_vol, posture))
        # uncovered = the wheel FAILED to cover; TREND/CASH-cell days are
        # exempt (calls are forbidden there, not failed — review 2026-07-14).
        if short is None and shares >= mult and phase == "CALL" and cell == "WHEEL":
            days_shares_uncovered += 1
        liab = 0.0
        if short is not None:
            mk = option_mark(day_chain, d, short["contract"])
            if mk is not None:
                short["last_mid"] = mk.mid
            liab = short["last_mid"] * mult * short["contracts"]
        equity[d] = cash + (shares + trend_shares) * spot - liab

    residual_settled = False
    if short is not None:
        last = pd.Timestamp(dates[-1])
        mk = option_mark(by_date.get(last), last, short["contract"])
        mid = mk.mid if mk is not None else short["last_mid"]
        cash -= mid * mult * short["contracts"]
        residual_settled = True
    return RouterResult(pd.Series(equity), trades, cash, shares + trend_shares,
                        residual_settled, warnings=warnings, route_log=route_log,
                        days_in_posture=days_in_posture,
                        days_shares_uncovered=days_shares_uncovered,
                        whipsaw_pairs=whipsaw_pairs,
                        n_trimmed_entries=n_trimmed_entries,
                        days_half_size=days_half_size,
                        intraday_tp_fills=intraday_tp_fills,
                        eod_tp_fills=eod_tp_fills)
