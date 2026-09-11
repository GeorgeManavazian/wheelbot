"""Portfolio rotation (spec 2026-07-14-portfolio-rotation-design): one shared
cash pool over the seen universe, one campaign at a time, entries routed to the
eligible ticker with the richest premium (vol percentile desc, fixed tie order).
Zero knobs, EOD fills (v1), basis floor per spec. Solo-only mechanics
(roll/stop/gates/liquidate) are refused — this engine reuses the solo rules for
TP/expiry/covered-calls and adds ONLY the routing layer."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import pandas as pd
from .select import (select_contract, option_mark, liquidity_ok,
                     liquidity_size_cap, exit_impact_slippage,
                     at_risky_window_edge)
from .fills import (try_take_profit, tp_exit_feasible, credit_ok, yield_ok,
                    write_credit_ok, ann_yield_on_collateral)
from .earnings import in_blackout
from .iv_rank import NEUTRAL_IV_RANK, iv_rank_ok
from .vrp import (sort_key as vrp_sort_key, is_viable as vrp_is_viable,
                  viable_sort_key as vrp_viable_sort_key)
from .wheel import (Trade, WheelConfig, is_unpaid_decline, sell_proceeds,
                    buy_cost, GATE_STALENESS_DAYS)
from ..regime.state import (is_good_renting_weather, weather_kwargs,
                            chop_half_of, CHOP_HALVES, FAST_PAIRS)
from .market import BatchMarket

ROTATION_TIE_ORDER = ("SPY", "GDX", "SLV", "XOP",
                      "AAPL", "AMZN", "NVDA", "META", "FB")
RESERVED_TICKERS = ("XBI", "EEM", "EWZ", "TLT", "ARKK")
# A ticker is ineligible before its clean start: its stored chain carries an
# UNADJUSTED corporate action before that date (underlying jumps by the split
# ratio, the whole strike ladder re-grids) and everything computed across the
# boundary is fiction. XOP was the original (unadjusted 1:4 reverse split
# 2020-03-31); the A10e sweep (2026-08-02) found the rest of the in-sample
# set: AAPL 4:1 2020-08-31, AMZN 20:1 2022-06-06, META is symbol REUSE (rows
# before 2022-06-09 are the Roundhill Metaverse ETF at ~$12, not Facebook),
# NVDA has TWO fossils (4:1 2021-07-20, 10:1 2024-06-10) so its fence loses
# 2010-2024 history — re-pull adjusted history to restore it (declared).
# Fence = fossil date + ~2-month regime flush (XOP precedent). The expansion
# 341 map lives in the A10e verdicts, applied before any expansion run.
DEFAULT_CLEAN_START = {"XOP": pd.Timestamp("2020-07-01"),
                       "AAPL": pd.Timestamp("2020-11-01"),
                       "AMZN": pd.Timestamp("2022-08-01"),
                       "META": pd.Timestamp("2022-09-01"),
                       "NVDA": pd.Timestamp("2024-08-01")}


@dataclass
class PortfolioResult:
    equity: pd.Series
    trades: list
    final_cash: float
    final_shares: dict
    residual_settled: bool = False
    days_flat: int = 0
    warnings: list = None
    days_shares_uncovered: int = 0
    route_events: list = None   # (date, ranked [(ticker, pctile)], chosen)
    n_campaigns_opened: int = 0
    # CAPACITY counters (2026-08-16). The scorecard could say how many
    # campaigns opened and never how much of the machine was BUSY, so
    # "gates cost turnover" was an argument rather than a number. On the
    # exported best_run, 20% of campaigns (the assigned ones) held 77% of all
    # slot-time and earned -$1.02 a slot-day against +$25.11 for the other
    # 80% -- which is the whole economics of the bot and was invisible here.
    #
    # DIAGNOSTIC ONLY. Nothing reads these to make a decision, so every run
    # stays byte-identical (tests/engine_v2/options/test_slot_capacity.py
    # pins that against the plain regression).
    idle_slot_days: int = 0     # slots still empty AFTER routing, summed
    pool_depth_sum: int = 0     # candidates that passed everything, summed
    pool_depth_sessions: int = 0    # sessions that had a slot to fill
    pool_starved_days: int = 0  # sessions the pool was thinner than free slots
    # The FUNNEL above the instrumented gates: how many names the scan looked
    # at on a routing session, and where they left. Without these the refusal
    # counts read as the whole story when they are its tail.
    scan_names: int = 0         # names considered (not already held), summed
    drop_no_chain: int = 0      # no chain that day, or market.eligible() false
    drop_weather: int = 0       # refused by the weather gate
    drop_no_contract: int = 0   # no strike at the target delta inside the band
    # Batch 5 Arm 2 (2026-08-22): how populated the viable tier is under
    # rank_by="vrp_viable" -- the mechanism clause ("non-trivially populated,
    # REPORTED") depends on this being on the card, same DIAGNOSTIC-ONLY
    # stance as the capacity counters above. Zero for every other rank_by.
    rank_viable_sum: int = 0        # tier-1 candidates, summed (first pass)
    rank_viable_sessions: int = 0   # sessions counted, same gate as pool_depth


# A10: corporate-action detect-and-refuse (provisional owner defaults,
# 2026-08-01: confirm band 0.80/1.25 · gap backstop 0.25 · manual clearing).
# A split RESTATES history: the fresh series' close at prev_d stops matching
# the stored last_spot, off by exactly the ratio, no market noise. A real
# crash never rewrites yesterday. Fires only through a market providing
# prior_close() -- LiveMarket; BatchMarket never grows it (hasattr-pinned),
# so batch runs are byte-identical by construction.
# C11: which side of the book prices the short-leg liability in equity --
# owner decision B, 2026-07-31 (mid -> ask changeover). Single source of
# truth: live/snapshots.py stamps THIS constant into every snapshot row, so a
# future basis change confesses in the data instead of silently rebasing the
# curve. A free string (an A21 RTH move can become "ask@rth" schema-free).
MARK_BASIS = "ask"
CA_RESTATE_LO, CA_RESTATE_HI = 0.80, 1.25
CA_GAP_BACKSTOP = 0.25
CA_WATCH_CAL_DAYS = 7          # ~5 sessions of re-checks for a lagged adjust


def _ca_guard(pos, market, tk, d, prev_d, warnings):
    """Returns 'frozen' (skip the position entirely), 'defer' (mark and TP
    normally, refuse settlement today), or None. Runs BEFORE last_spot is
    overwritten -- updating it on a hit would erase the detector's own
    evidence (mutation target)."""
    subj = pos["short"]["contract"] if pos.get("short") else tk
    if pos.get("ca_frozen"):
        warnings.append((d, "ca_confirmed_frozen", subj))
        return "frozen"
    prior = getattr(market, "prior_close", None)
    if prior is None:
        return None                       # batch: no capability, no guard
    last = pos.get("last_spot")
    if not last or last <= 0:
        return None    # unjudgeable stored spot (C17 class) -- declared hole

    def _freeze(ratio, ref_date):
        pos["ca_frozen"] = {"date": str(pd.Timestamp(d).date()),
                            "stored_spot": float(last),
                            "ratio": round(float(ratio), 6),
                            "restated_close_of": str(ref_date)}
        pos.pop("ca_watch", None)
        warnings.append((d, "ca_confirmed_frozen", subj))

    # a pending watch: re-check the WATCHED date first (lagged adjustment)
    watch = pos.get("ca_watch")
    if watch is not None:
        wd = pd.Timestamp(watch["date"])
        fp = prior(tk, wd)
        if fp is not None and fp > 0 and watch["spot"] > 0:
            wr = fp / watch["spot"]
            if wr <= CA_RESTATE_LO or wr >= CA_RESTATE_HI:
                _freeze(wr, watch["date"])
                return "frozen"
        if (pd.Timestamp(d) - wd).days > CA_WATCH_CAL_DAYS:
            pos.pop("ca_watch", None)     # bound reached, nothing restated

    if prev_d is None:
        return None
    fresh_prev = prior(tk, prev_d)
    if fresh_prev is not None and fresh_prev > 0:
        ratio = fresh_prev / last
        if ratio <= CA_RESTATE_LO or ratio >= CA_RESTATE_HI:
            _freeze(ratio, str(pd.Timestamp(prev_d).date()))
            return "frozen"
        restate_clean = True
    else:
        restate_clean = False             # unjudgeable -> suspect via gap
    spot_today = market.spot(tk, d, last)
    gap = abs(spot_today / last - 1.0) if spot_today else 0.0
    if gap >= CA_GAP_BACKSTOP:
        if pos.get("ca_watch") is None:   # one-shot: keep the ORIGINAL pair
            pos["ca_watch"] = {"date": str(pd.Timestamp(prev_d).date()),
                               "spot": float(last)}
        warnings.append((d, "ca_suspect_settlement_deferred", subj))
        return "defer"
    if not restate_clean:
        return None    # no reference close AND no gap: a data hole, not a CA
    return None


@dataclass
class PortfolioState:
    cash: float
    positions: list
    campaign: int = 0
    days_flat: int = 0
    days_shares_uncovered: int = 0
    prev_d: object = None
    # A5: contracts the INTRADAY manager closed, date-stamped
    # [{"date": Timestamp, "contract": Contract}]. Seeds step_one_day's
    # closed_today so the 17:00 run cannot re-sell a contract bought back at
    # 10:00 -- the guard used to be rebuilt empty every step while
    # manage_intraday persisted nothing, and A6/A16 exist to increase
    # intraday closes. Stale (non-today) entries are pruned on append.
    intraday_closed: list = field(default_factory=list)
    # Ticker -> the date its post-exit cooldown expires (spec 2026-08-14).
    # Portfolio-level, not per-position: a tier-3 exit compacts the position
    # away, so a per-position ban would vanish with the thing it was banning.
    exit_cooldown: dict = field(default_factory=dict)
    # Capacity counters -- see PortfolioResult for what they are for. Carried
    # on the state so the LIVE bot accumulates them across days too; it calls
    # step_one_day directly and never touches run_portfolio_wheel.
    idle_slot_days: int = 0
    pool_depth_sum: int = 0
    pool_depth_sessions: int = 0
    pool_starved_days: int = 0
    scan_names: int = 0
    drop_no_chain: int = 0
    drop_weather: int = 0
    drop_no_contract: int = 0
    rank_viable_sum: int = 0
    rank_viable_sessions: int = 0


@dataclass
class StepResult:
    trades: list
    equity: float
    warnings: list
    route_events: list


def _row_before(states: pd.DataFrame, d: pd.Timestamp):
    """Full state row strictly before d, staleness-bounded (same information
    rule as the gates/autopsy). None -> unknown."""
    idx = states.index
    pos = idx.searchsorted(pd.Timestamp(d)) - 1
    if pos < 0 or (pd.Timestamp(d) - idx[pos]).days > GATE_STALENESS_DAYS:
        return None
    return states.iloc[pos]


def close_short_fill(pos, dec, cash, trades):
    """Book a short close from a FillDecision -- THE one place that knows how
    a (possibly partial) buy-back mutates a position (A17). Decrements the
    live size by dec.filled_contracts; zero remaining normalizes `short` to
    None, which is what every downstream `short is None` test (compaction,
    covered-call gate, equity mark, settlement) keys on. Returns
    (cash, fully_closed). Both current fill modes are instant-and-whole
    (filled_contracts == contracts), so they decrement straight to zero --
    byte-identical to the pre-A17 behavior; the capacity exists for a fill
    model that is not. Shared with live/intraday.py -- one rule, two call
    sites, same as the A18 seam."""
    short = pos["short"]
    c = short["contract"]
    k = dec.filled_contracts
    # A17 skeptic F4: a filled decision carrying zero (or negative) contracts
    # would book a 0-lot ghost Trade, bleed dec.cost from cash, and leave the
    # leg open. Unreachable from the current seam (both modes fill whole);
    # refuse it loudly so a future fill model cannot emit it silently.
    if k <= 0:
        raise ValueError(f"filled decision with filled_contracts={k}")
    if k < short["contracts"] and "opened_contracts" not in short:
        # first partial on this leg: record the original size, or the stored
        # "6 remain" reads as 6-of-6 when it was 6-of-10 (skeptic F5)
        short["opened_contracts"] = short["contracts"]
    cash -= dec.cost
    pos["premium"] -= dec.cost
    # held legs load as Contract dataclasses but old raw dicts must not crash
    right = getattr(c, "right", None) if hasattr(c, "right") else c["right"]
    trades.append(Trade(dec.stamp, "CLOSE_PUT" if right == "P" else "CLOSE_CALL",
                        c, k, dec.price, cash, pos["campaign"]))
    short["contracts"] -= k
    fully = short["contracts"] <= 0
    if fully:
        pos["short"] = None
    return cash, fully


def _net_basis(pos):
    """Assignment strike minus premium banked this campaign -- true breakeven.
    The same quantity the basis floor uses; never the gross strike."""
    if pos.get("basis") is None or not pos.get("shares"):
        return None
    return pos["basis"] - pos["premium"] / pos["shares"]


def tier2_should_fire(pos, tier1_failed, spot, cfg) -> bool:
    """Should the basis floor be abandoned for an ITM exit call today?"""
    trig = cfg.exit_trigger
    if trig == "never":
        # Tier 3 without tier 2: keep the basis floor exactly as it is and add
        # only a deep hard stop. The 2026-08-14 stuck-campaign review found the
        # cost is all tail -- 4 of 40 assignments never resolved and ate 51% of
        # every held-day -- so a rule that fires deep and rarely is the shape
        # the data asks for, and the ITM ladder is not.
        return False
    if trig == "tier1_impossible":
        return tier1_failed
    if trig == "pct_below_basis":
        nb = _net_basis(pos)
        return (nb is not None and cfg.exit_trigger_pct is not None
                and spot < nb * (1.0 - cfg.exit_trigger_pct))
    if trig == "days_stuck":
        return (cfg.exit_trigger_days is not None
                and pos.get("stuck_days", 0) >= cfg.exit_trigger_days)
    return False


def tier3_stop_hit(pos, spot, cfg) -> bool:
    """Has the tier-2 exit failed, i.e. is it time to sell the shares?

    Deliberately reads `exit_strike` off the POSITION rather than the live
    short: when the stock falls, the tier-2 call goes OTM and cheap and the
    take-profit usually buys it back first, so a live-short read would lose
    the trigger at exactly the moment it is needed.
    """
    mode = cfg.exit_stop
    if mode == "none" or spot is None or not pos.get("shares"):
        return False
    if mode == "pct_below_basis":
        nb = _net_basis(pos)
        return nb is not None and spot < nb * (1.0 - cfg.exit_stop_buffer)
    k = pos.get("exit_strike")
    if k is None:
        return False
    if mode == "spot_below_strike":
        return spot < k
    if mode == "spot_below_strike_buffer":
        return spot < k * (1.0 - cfg.exit_stop_buffer)
    return False


def _call_cfg(cfg):
    """cfg as covered CALLS should see it. call_take_profit_pct=None inherits
    take_profit_pct, so the default path is the same object and byte-identical."""
    override = getattr(cfg, "call_take_profit_pct", None)
    if override is None:
        return cfg
    from dataclasses import replace
    return replace(cfg, take_profit_pct=override)


def step_one_day(state, market, day, cfg, *, selector, n_slots) -> StepResult:
    """One trading day: manage held positions, drop finished campaigns, fill
    empty slots, mark equity. Mutates `state`; returns today's outputs. The
    SAME logic the batch backtest runs — reading through the Market seam."""
    if cfg.rank_by == "iv_rank":
        # The guard has to be HERE, not only in run_portfolio_wheel: the live
        # bot calls step_one_day directly (live/run_daily.py, paper_step), so a
        # guard in the batch driver alone leaves the live path wide open. With
        # no history every name scores NEUTRAL, the tie breaks on
        # market.universe.index(tk), and the run prints a full plausible set of
        # trades chosen by universe ORDER. A market must therefore declare the
        # capability out loud -- absent declaration is not consent, it is what
        # LiveMarket and ~20 test fakes all looked like.
        rankable = getattr(market, "iv_rankable", None)
        if rankable is None or not rankable():
            raise ValueError(
                "rank_by='iv_rank' needs a market that can rank on IV — this "
                f"{type(market).__name__} cannot (no usable IV history). "
                "Backtest: pass iv_history=IVHistory.from_chains(chains). "
                "Live: pass iv_history= into LiveMarket.")
    if cfg.rank_by == "vrp":
        # Same declaration rule as iv_rank, and it has to be HERE because the
        # live bot calls step_one_day directly. With no history every name
        # scores NEUTRAL, the tie breaks on universe order, and the run prints
        # a full plausible set of trades chosen by nothing at all.
        rankable = getattr(market, "vrp_rankable", None)
        if rankable is None or not rankable():
            raise ValueError(
                "rank_by='vrp' needs a market that can read implied vol — this "
                f"{type(market).__name__} cannot (no usable IV history). "
                "Backtest: pass iv_history=IVHistory.from_chains(chains). "
                "Live: pass iv_history= into LiveMarket.")
    if cfg.rank_by == "vrp_viable":
        # Same declaration rule as vrp, and for the same reason it has to be
        # HERE: the live bot calls step_one_day directly. Tier 1 reads VRP
        # exactly like rank_by="vrp" does, so a market that cannot rank on IV
        # would score every tier-1 candidate NEUTRAL rather than raising.
        rankable = getattr(market, "vrp_rankable", None)
        if rankable is None or not rankable():
            raise ValueError(
                "rank_by='vrp_viable' needs a market that can read implied "
                f"vol — this {type(market).__name__} cannot (no usable IV "
                "history). Backtest: pass iv_history=IVHistory.from_chains"
                "(chains). Live: pass iv_history= into LiveMarket.")
    if cfg.earnings_blackout:
        # Same reasoning as the iv_rank guard above, and a QUIETER failure if
        # it is missing. `earnings_dates` is defined on every real market, so
        # the use site's getattr finds it and the gate looks wired; with no
        # calendar it answers None for every ticker, in_blackout reads unknown,
        # and unknown ALLOWS. The result is a gate that is on, blocks nothing,
        # and -- unlike the iv_rank case, which at least logs
        # entry_ranked_iv_unknown ~531x a day -- logs nothing at all, because
        # per-ticker "unknown allows" is the correct behaviour it is imitating.
        # The calendar is an offline data/ artifact that ships separately from
        # the code, so running without it is the easy mistake, not a rare one.
        known = getattr(market, "earnings_known", None)
        if known is None or not known():
            raise ValueError(
                "earnings_blackout=True needs a market that can answer the "
                f"earnings gate — this {type(market).__name__} cannot (no "
                "usable calendar). Backtest: pass earnings=EarningsCalendar."
                "load(). Live: pass earnings= into LiveMarket, and check "
                "data/earnings/ is present on the machine running it.")
    d = pd.Timestamp(day)
    cash = state.cash
    positions = state.positions
    campaign = state.campaign
    days_flat = state.days_flat
    days_shares_uncovered = state.days_shares_uncovered
    idle_slot_days = state.idle_slot_days
    pool_depth_sum = state.pool_depth_sum
    pool_depth_sessions = state.pool_depth_sessions
    pool_starved_days = state.pool_starved_days
    scan_names = state.scan_names
    drop_no_chain = state.drop_no_chain
    drop_weather = state.drop_weather
    drop_no_contract = state.drop_no_contract
    rank_viable_sum = state.rank_viable_sum
    rank_viable_sessions = state.rank_viable_sessions
    prev_d = state.prev_d
    mult = cfg.contract_multiplier
    trades, warnings, route_events = [], [], []
    ccfg = _call_cfg(cfg)          # covered calls' view of take-profit

    if prev_d is not None and cfg.cash_yield > 0:
        cash *= (1 + cfg.cash_yield / 365) ** (d - prev_d).days

    # 1) manage every held position (TP -> expiry -> covered call)
    closed_today = set()
    # A5: seed with today's intraday closes -- same-day anti-churn must see
    # what the 10:00 manager did, not just what this step does
    day_norm = pd.Timestamp(d).normalize()
    for e in state.intraday_closed:
        if pd.Timestamp(e["date"]).normalize() == day_norm:
            closed_today.add(e["contract"])
    for pos in positions:
        tk = pos["ticker"]
        # A10: BEFORE the last_spot overwrite -- the guard's evidence is the
        # stored spot vs the fresh series' restated history
        ca = _ca_guard(pos, market, tk, d, prev_d, warnings)
        if ca == "frozen":
            continue          # no TP, no settlement, no covered call, no mark
        day_chain = market.chain(tk, d)
        spot = market.spot(tk, d, pos["last_spot"])
        pos["last_spot"] = spot
        short = pos["short"]
        if short is not None:
            c, n = short["contract"], short["contracts"]
            mark = option_mark(day_chain, d, c) if day_chain is not None else None
            # B-exit: a close leg cannot refuse (the position is already
            # held), so price the same size-vs-market ratio the entry side
            # already caps instead of leaving it unpriced. Adjusts the ASK
            # the buyback prices at; None knobs -> impact is always 0.0.
            if mark is not None and day_chain is not None:
                impact = exit_impact_slippage(day_chain, d, c, n, cfg)
                if impact > 0.0:
                    mark = replace(mark, ask=mark.ask + impact)
            # Covered calls may carry their own take-profit (spec 2026-08-14):
            # the shared 60% buys the call back and KEEPS you in the stock,
            # while holding to expiry gets you called away, which is an exit.
            leg_cfg = ccfg if getattr(c, "right", None) == "C" else cfg
            dec = try_take_profit(mark=mark, credit=short["credit"], contracts=n,
                                  cfg=leg_cfg, day=d, expiry=c.expiry)
            if dec.filled:
                cash, fully = close_short_fill(pos, dec, cash, trades)
                short = pos["short"]
                if fully:
                    closed_today.add(c)
                    # A5 skeptic F1: the step must leave a note for ITSELF
                    # too -- in the save-succeeded/snapshot-failed retry
                    # window the same day is re-stepped and this exact
                    # contract would be re-sold. Same date-stamped list the
                    # intraday manager writes; stale entries pruned.
                    state.intraday_closed = (
                        [e for e in state.intraday_closed
                         if pd.Timestamp(e["date"]).normalize() == day_norm]
                        + [{"date": day_norm, "contract": c}])
            if short is not None and d >= c.expiry:
                if ca == "defer":
                    # A10 suspect day: a >=25% gap with clean (or unjudgeable)
                    # restatement. Marks and TP ran normally above; only
                    # SETTLEMENT waits one session -- booking strike-vs-close
                    # arithmetic across a possible corporate action is the
                    # $42,750 class. A real crash re-checks clean tomorrow and
                    # settles via the late-expiry path (tested).
                    continue
                # re-read the size: a partial TP fill above shrank the leg, and
                # settling the stale pre-fill `n` would assign contracts that
                # were already bought back (A17/I3)
                n = short["contracts"]
                # Settlement reads the EXPIRY DAY's own close and nothing else.
                # It used to take `spot`, which degrades to the CARRIED
                # pos["last_spot"] when the close is missing — so on a data
                # outage a deep-ITM put was booked PUT_EXPIRED (worthless) with
                # no warning and no gap record. Measured on the real TMO leg:
                # identical market reality, two ledgers, $153,750 apart, and the
                # wrong one silent. zombie_check cannot catch it (it fires at 50%
                # of the universe; one ticker is 0.18%). settle_price() is not
                # the answer either — it returns the last close ON OR BEFORE the
                # expiry, i.e. a different day's price. Refuse instead: leave the
                # leg open and warn, so a later run with restored history settles
                # it correctly. (2026-07-31)
                settle_spot = market.spot(tk, c.expiry, None)
                if settle_spot is None:
                    # A15: the batch market (and ONLY the batch market)
                    # offers a bounded reach-back -- without it the batch
                    # driver, which has no later run, zombies the leg to the
                    # residual finalizer. Live falls through to refuse+warn.
                    bounded = getattr(market, "bounded_settle_price", None)
                    if bounded is not None:
                        settle_spot = bounded(tk, c.expiry)
                        if settle_spot is not None:
                            warnings.append((d, "expiry_settled_reachback", c))
                if settle_spot is None:
                    warnings.append((d, "expiry_unsettleable", c))
                    continue
                if d > c.expiry:
                    warnings.append((d, "expiry_resolved_late", c))
                if c.right == "P":
                    if settle_spot < c.strike:
                        # A13: per-event assignment fee ($0 at Schwab, VERIFY)
                        cash -= c.strike * mult * n + cfg.fee_per_assignment
                        pos["shares"] += mult * n; pos["phase"] = "CALL"
                        pos["basis"] = c.strike
                        # A9: the notice arrives after this session's close and
                        # the shares settle next session -- the covered-call
                        # block below is skipped while assigned_d == today.
                        # PERSISTED (not an in-step flag): live re-steps the
                        # same day in the snapshot-failed retry window, and a
                        # reconciled real-money assignment (A8b) arrives from
                        # another process. Keyed on the BOOKING session, so a
                        # late-booked expiry defers from when the engine learns.
                        pos["assigned_d"] = str(d.date())
                        trades.append(Trade(d, "ASSIGNED", c, n, c.strike, cash, pos["campaign"]))
                    else:
                        trades.append(Trade(d, "PUT_EXPIRED", c, n, 0.0, cash, pos["campaign"]))
                else:
                    if settle_spot > c.strike:
                        cash += c.strike * mult * n - cfg.fee_per_assignment
                        pos["shares"] -= mult * n; pos["phase"] = "PUT"
                        pos["basis"] = None
                        trades.append(Trade(d, "CALLED_AWAY", c, n, c.strike, cash, pos["campaign"]))
                    else:
                        trades.append(Trade(d, "CALL_EXPIRED", c, n, 0.0, cash, pos["campaign"]))
                pos["short"] = None; short = None
        # 1b) TIER 3 (spec 2026-08-14): the tier-2 exit call has failed -- the
        # stock fell through its strike, so it will not carry the shares away.
        # Placed AFTER take-profit deliberately: on a crater the TP buys the
        # call back cheaply for a gain, and only then are the shares sold, so
        # the option profit is banked before the exit.
        if (cfg.assignment_exit and pos["shares"] >= mult
                and pos.get("assigned_d") != str(d.date())
                and tier3_stop_hit(pos, spot, cfg)):
            blocked = False
            if pos["short"] is not None:
                sc = pos["short"]["contract"]
                smk = option_mark(day_chain, d, sc) if day_chain is not None else None
                if smk is None:
                    # FAIL CLOSED. Selling the shares now would leave an
                    # uncovered short call against an account with no margin --
                    # unlimited liability. A slot stuck one more day is merely
                    # the problem this ladder exists to reduce.
                    warnings.append((d, "exit_blocked_unclosable_call", tk))
                    blocked = True
                else:
                    sn = pos["short"]["contracts"]
                    cost = buy_cost(smk, sn, cfg)
                    cash -= cost
                    pos["premium"] -= cost
                    trades.append(Trade(d, "CLOSE_CALL" if sc.right == "C"
                                        else "CLOSE_PUT", sc, sn, smk.ask,
                                        cash, pos["campaign"]))
                    closed_today.add(sc)
                    pos["short"] = None
            if not blocked:
                sh = pos["shares"]
                cash += sh * spot          # at the day's close; no slippage
                trades.append(Trade(d, "SOLD_SHARES", tk, sh, spot, cash,
                                    pos["campaign"]))
                pos["shares"] = 0
                pos["phase"] = "PUT"
                pos["basis"] = None
                pos["exit_strike"] = None
                pos["stuck_days"] = 0
                if cfg.exit_cooldown_days > 0:
                    state.exit_cooldown[tk] = d + pd.Timedelta(
                        days=cfg.exit_cooldown_days)
                continue          # nothing left to cover this session

        if (pos["short"] is None and pos["phase"] == "CALL"
                and pos["shares"] >= mult and day_chain is not None
                and pos.get("assigned_d") != str(d.date())):
            # A9: the WHOLE block is gated, not just the sale -- a structural
            # one-session deferral must not fire the A4/A3b naked-shares
            # warnings (each one emails the owner daily).
            floor = None
            if cfg.call_min_strike == "basis" and pos["basis"] is not None:
                floor = pos["basis"] - pos["premium"] / pos["shares"]
            # n depends only on shares/mult, not on whether a contract is
            # found -- computed unconditionally so the tier-2 fallback below
            # (which can reach the write with a DIFFERENT `c`/`mark` than the
            # ones selected here) always has it.
            n = pos["shares"] // mult
            c = select_contract(day_chain, d, "C", cfg.call_delta,
                                cfg.target_dte, tk, min_strike=floor)
            mark = option_mark(day_chain, d, c) if c is not None else None
            # B-exit: priced BEFORE the feasibility checks below -- a covered
            # call is exempt from the A2 liquidity gate (refusing one leaves
            # shares naked), but if it is too large to sell at a real price,
            # that must show up in whether it clears the credit/TP-
            # reachability floors, not only in the cash booked once it is
            # written. None knobs -> impact is 0.0.
            if mark is not None:
                impact = exit_impact_slippage(day_chain, d, c, n, cfg)
                if impact > 0.0:
                    mark = replace(mark, bid=max(0.0, mark.bid - impact))
            if floor is not None and c is None:
                # A4: the basis floor sits above every strike the chain
                # carries -- the income half of the wheel cannot start and the
                # shares sit naked. Silent for months live (TMO class); never
                # silent again. (No warning when floor is None: a plain-wheel
                # run has no floor and no defect. No separate "no_mark" branch:
                # select_contract picks rows from the same frame option_mark
                # re-scans with the same key, and ingest guarantees float
                # bid/ask/mid -- a selected contract always marks; the skeptic
                # proved the branch dead.)
                warnings.append((d, "covered_call_unreachable", tk))
            # TWO independent rules, kept separable so their counters stay
            # honest. `tp_ok` is A3b (can this call's own take-profit exit be
            # reached); `credit_ok_` is the tp-INDEPENDENT write floor. A3b
            # goes silent at call_take_profit_pct >= 1.0 -- tp_exit_floor is
            # None and tp_exit_feasible admits $0.00 -- and calls are exempt
            # from the A2 liquidity gate, so without the second rule that arm
            # writes worthless calls at -friction/contract. At tp 0.50/0.60
            # A3b's floor is the higher of the two and this changes nothing.
            tp_ok = mark is None or tp_exit_feasible(mark.bid, ccfg)[0]
            credit_ok_ = mark is None or write_credit_ok(mark.bid, cfg)[0]
            feasible = tp_ok and credit_ok_
            # TIER 2 (spec 2026-08-14): tier 1 is impossible -- either the floor
            # sits above every listed strike (covered_call_unreachable) or the
            # far-OTM call it forces is worth a penny and A3b refuses it as
            # unclosable (call_gated_unclosable, 78% of the measured jam).
            # Abandon the floor and write ITM instead: premium ~ (S-K) + time
            # value, and being called away returns K, so the position exits at
            # roughly spot plus a tip, with a cushion down to K.
            tier1_failed = (floor is not None and c is None) or \
                           (mark is not None and not feasible)
            if cfg.assignment_exit and tier2_should_fire(pos, tier1_failed,
                                                         spot, cfg):
                c2 = select_contract(day_chain, d, "C", cfg.exit_call_delta,
                                     cfg.target_dte, tk)
                m2 = option_mark(day_chain, d, c2) if c2 is not None else None
                # B-exit: same treatment as the tier-1 contract above.
                if m2 is not None:
                    impact2 = exit_impact_slippage(day_chain, d, c2, n, cfg)
                    if impact2 > 0.0:
                        m2 = replace(m2, bid=max(0.0, m2.bid - impact2))
                if (c2 is not None and m2 is not None and c2 not in closed_today
                        and tp_exit_feasible(m2.bid, ccfg)[0]
                        and write_credit_ok(m2.bid, cfg)[0]):
                    c, mark, feasible = c2, m2, True
                    pos["exit_strike"] = float(c2.strike)
            if mark is not None and not feasible:
                # A3b (owner 2026-08-01): a covered call whose own TP exit is
                # unreachable at the tick or a guaranteed net loss is never
                # written -- the shares stay honestly naked for the day
                # (counted below, retried daily). Calls remain EXEMPT from
                # the A2 liquidity gate: refusing a call leaves shares naked,
                # so only arithmetic impossibility may refuse one.
                #
                # A3b FIRST when both rules refuse: its floor is the higher one
                # at every take-profit the repo holds, so the historical counts
                # are unchanged, and `call_gated_no_credit` can only appear on
                # an arm where A3b has gone silent. The counter is therefore
                # self-documenting rather than a second name for the same
                # event -- which is the mistake that made 1,333 -> 0 look like
                # evidence on 2026-08-14.
                warnings.append((d, "call_gated_unclosable" if not tp_ok
                                 else "call_gated_no_credit", tk))
            if c is not None and c not in closed_today and mark is not None \
                    and feasible:
                # n and the B-exit impact adjustment on `mark` were already
                # applied above, before tp_ok/credit_ok_ were computed.
                proceeds = sell_proceeds(mark, n, cfg)
                cash += proceeds; pos["premium"] += proceeds
                pos["short"] = {"contract": c, "contracts": n,
                                "credit": mark.bid, "last_mid": mark.mid}
                trades.append(Trade(d, "SELL_CALL", c, n, mark.bid, cash, pos["campaign"]))
                pos["stuck_days"] = 0
            elif cfg.assignment_exit:
                # Feeds the days_stuck trigger variant. Counts only sessions
                # where a call was genuinely wanted and not written.
                pos["stuck_days"] = pos.get("stuck_days", 0) + 1

    positions[:] = [p for p in positions
                    if not (p["short"] is None and p["shares"] == 0 and p["phase"] == "PUT")]

    # 2) routing entry: fill empty slots with the best good-to-rent tickers
    held_tickers = {p["ticker"] for p in positions}
    # Capacity accounting. `free_at_start` is the demand, the first iteration's
    # pool is the supply, and what is left empty at the end is the shortfall.
    # Measured on the FIRST iteration only: later ones re-scan a universe with
    # one more name held and a smaller budget, so they answer a different
    # question and would flatter the mean.
    free_at_start = n_slots - len(positions)
    first_pool_depth = None
    first_viable_count = None   # rank_by="vrp_viable": tier-1 count, first pass
    # THE SILENT DROPS. Everything upstream of the liquidity floor leaves the
    # scan with a bare `continue` and logs nothing, so the refusal counts on a
    # scorecard describe only the instrumented tail of the funnel: on the
    # 2026-08-16 live-tier run the liquidity floor showed 2,151 refusals while
    # roughly 29 candidates a session disappeared before it, uncounted. A
    # number that is visible because it is measured, standing next to bigger
    # ones that are not, is how a gate gets blamed for a funnel. Counted on the
    # FIRST pass only, for the same reason as the pool depth above.
    drops = {"no_chain": 0, "weather": 0, "no_contract": 0, "scanned": 0}
    while len(positions) < n_slots:
        empty_slots = n_slots - len(positions)
        committed = sum(p["short"]["contract"].strike * mult * p["short"]["contracts"]
                        for p in positions
                        if p["short"] is not None and p["short"]["contract"].right == "P")
        available = cash - committed
        pool = []
        pref_tier: dict = {}      # ticker -> 0/1, only when prefer_chop_half is on
        viable_count = 0          # rank_by="vrp_viable": tier-1 count this pass
        first_pass = first_pool_depth is None
        for tk in market.universe:
            if tk in held_tickers:
                continue
            if first_pass:
                drops["scanned"] += 1
            # Post-exit cooldown (spec 2026-08-14): a name just stopped out is
            # not a fresh candidate the same session. Deduped warning so the
            # n_slots while-loop does not re-log it per iteration.
            until = state.exit_cooldown.get(tk)
            if until is not None:
                if d < until:
                    if (d, "entry_in_exit_cooldown", tk) not in warnings:
                        warnings.append((d, "entry_in_exit_cooldown", tk))
                    continue
                state.exit_cooldown.pop(tk, None)
            day_chain = market.chain(tk, d)
            if day_chain is None or not market.eligible(tk, d):
                if first_pass:
                    drops["no_chain"] += 1
                continue
            row = market.regime_row(tk, d)
            if selector == "chop":
                # Every chop_* knob, from one mapping (regime.state.GATE_KNOBS).
                # Passing three by hand is how a fourth gets added to the config
                # and never reaches the engine.
                if not is_good_renting_weather(row, **weather_kwargs(cfg)):
                    if first_pass:
                        drops["weather"] += 1
                    continue
            else:
                if row is not None and is_unpaid_decline(row["trend"], row["vol"]):
                    if first_pass:
                        drops["weather"] += 1
                    continue
            c = select_contract(day_chain, d, "P", cfg.put_delta, cfg.target_dte, tk)
            mark = option_mark(day_chain, d, c) if c is not None else None
            if c is None or c in closed_today or mark is None:
                if first_pass:
                    drops["no_contract"] += 1
                continue
            liq, why = liquidity_ok(day_chain, d, c, cfg)
            if not liq:
                # A2: veto, never substitute -- and never silently. A fully
                # gated day must not print like a quiet one (paper_step
                # surfaces these warnings in the run log). Deduped: the
                # n_slots while-loop revisits gated tickers every iteration
                # (skeptic F4).
                if (d, "entry_gated_illiquid", tk) not in warnings:
                    warnings.append((d, "entry_gated_illiquid", tk))
                continue
            if not tp_exit_feasible(mark.bid, cfg)[0]:
                # A3: this entry's own take-profit exit is unreachable at the
                # minimum tick or a guaranteed net loss -- unclosable by
                # construction (the WBD 25P $0.01-credit case).
                if (d, "entry_gated_unclosable", tk) not in warnings:
                    warnings.append((d, "entry_gated_unclosable", tk))
                continue
            if not write_credit_ok(mark.bid, cfg)[0]:
                # The tp-INDEPENDENT half of A3. On the put leg A2 and the
                # intrinsic/yield floors would usually catch a $0.00 bid, but
                # all three default off and a config may turn them off, while
                # A3 above goes silent at take_profit_pct >= 1.0. A write that
                # cannot clear its own friction is refused on arithmetic, not
                # judgment -- same veto discipline as A2/A3, never a
                # substitution, never silent, deduped for the same reason.
                if (d, "entry_gated_no_credit", tk) not in warnings:
                    warnings.append((d, "entry_gated_no_credit", tk))
                continue
            if not credit_ok(mark.bid, c.strike, cfg)[0]:
                # Intrinsic filter: this credit is mostly moneyness, not
                # volatility -- a stock purchase wearing a premium costume.
                # Veto like A2/A3: never substitute another strike, never
                # silent. Deduped for the same reason as the A2 warning (the
                # n_slots while-loop revisits gated tickers each iteration).
                if (d, "entry_gated_intrinsic", tk) not in warnings:
                    warnings.append((d, "entry_gated_intrinsic", tk))
                continue
            if not yield_ok(mark.bid, c.strike, (c.expiry - d).days, cfg)[0]:
                # Collateral-yield floor: a negligible return on the cash the
                # strike locks up (the owner's $10-against-$100k case). Veto
                # like A2/A3 -- never substitute another strike, never
                # silent. Placed AFTER select_contract so dte is the real
                # expiry, not target_dte. Deduped for the same reason as the
                # A2 warning (the n_slots while-loop revisits gated tickers
                # each iteration).
                if (d, "entry_gated_low_yield", tk) not in warnings:
                    warnings.append((d, "entry_gated_low_yield", tk))
                continue
            if cfg.earnings_blackout:
                # A scheduled print inside the put's life. Veto like A2/A3 --
                # never substitute a different expiry that dodges the date,
                # never silent. Placed AFTER select_contract so the window's
                # right edge is the real expiry, not target_dte. Deduped for
                # the same reason as the A2 warning (the n_slots while-loop
                # revisits gated tickers each iteration).
                # getattr, not a required method: markets grow capabilities the
                # way bounded_settle_price does, so the ~20 test fakes and any
                # market without a calendar leave the gate inert rather than
                # raising. A market that HAS the method returns None for a
                # ticker it has no data on -- unknown allows, and the caller
                # counts it (see earnings.py).
                cal = getattr(market, "earnings_dates", None)
                if cal is not None and in_blackout(cal(tk), d, c.expiry):
                    if (d, "entry_gated_earnings", tk) not in warnings:
                        warnings.append((d, "entry_gated_earnings", tk))
                    continue
            if cfg.min_iv_rank is not None:
                # This ticker's implied vol is cheap against its own trailing
                # 252 observations -- we would be selling insurance into a calm
                # tape, which measured -11.8 bps mean return on collateral
                # against +13.7 in the top decile. Veto like A2/A3: never
                # substitute a different strike or expiry to find richer vol,
                # never silent. Deduped for the same reason as the A2 warning.
                # getattr like the earnings calendar: a market without the
                # capability leaves the gate inert rather than raising, and a
                # market that HAS it returns None for a name it cannot rank --
                # unknown allows, and the caller counts it (see iv_rank.py).
                rk = getattr(market, "iv_rank", None)
                if rk is not None and not iv_rank_ok(rk(tk, d), cfg)[0]:
                    if (d, "entry_gated_iv_rank", tk) not in warnings:
                        warnings.append((d, "entry_gated_iv_rank", tk))
                    continue
            if row is None:
                # deduped (A12 skeptic F4): the pool now includes unaffordable
                # tickers and rebuilds per while-iteration
                if (d, "route_state_unknown", tk) not in warnings:
                    warnings.append((d, "route_state_unknown", tk))
                pct = -1.0
            else:
                pct = float(row["vol_pctile"])
            if cfg.rank_by == "iv_rank":
                # Rank on the price PAID for risk, not the risk taken. This
                # reorders the pool; it never removes a member -- the veto
                # (min_iv_rank, above) was measured and rejected precisely
                # because refusing entries leaves slots idle. getattr like the
                # gate: a market without the capability ranks everything
                # neutral rather than raising, and run_portfolio_wheel refuses
                # to start in that state so it cannot happen silently.
                rk = getattr(market, "iv_rank", None)
                key = rk(tk, d) if rk is not None else None
                if key is None:
                    # Unmeasurable -> neutral, and COUNTED. How many names were
                    # ranked on a neutral is the only way to tell afterwards
                    # whether the unknown-handling decision moved the result.
                    # Deduped for the same reason as the gate warnings: the
                    # n_slots while-loop re-ranks the pool every iteration.
                    if (d, "entry_ranked_iv_unknown", tk) not in warnings:
                        warnings.append((d, "entry_ranked_iv_unknown", tk))
                    key = NEUTRAL_IV_RANK
            elif cfg.rank_by == "vrp":
                # Rank on the SPREAD -- what the option pays against what the
                # stock actually delivers -- rather than on the risk taken
                # (vol_pctile) or the price level (iv_rank), neither of which is
                # the edge. Reorders the pool; never removes a member, because
                # every refusal this project has measured cost more in campaigns
                # than it saved in quality.
                rk = getattr(market, "vrp", None)
                key = rk(tk, d, None if row is None else row.get("realized_vol")) \
                    if rk is not None else None
                if key is None:
                    # Unmeasurable -> NEUTRAL, and COUNTED. How many names were
                    # ranked on a neutral is the only way to tell afterwards
                    # whether the unknown-handling decision moved the result.
                    if (d, "entry_ranked_vrp_unknown", tk) not in warnings:
                        warnings.append((d, "entry_ranked_vrp_unknown", tk))
                key = vrp_sort_key(key)
            elif cfg.rank_by == "vrp_viable":
                # Batch 5 Arm 2: a two-tier reorder, not a new veto. Candidates
                # whose annualized credit yield on collateral clears
                # VRP_VIABLE_MIN_ANN_YIELD -- the SAME number the
                # min_ann_yield_on_collateral gate computes above, never a
                # second formula -- sort by VRP descending, ahead of everyone
                # else, who sort in ordinary vol_pctile order. Every candidate
                # stays in the pool; an empty tier 1 degrades byte-identically
                # to vol_pctile order (VRP_VIABLE_TIER_OFFSET keeps the two
                # tiers from ever crossing inside the one sortable key).
                ann_yield = ann_yield_on_collateral(mark.bid, c.strike,
                                                    (c.expiry - d).days)
                if vrp_is_viable(ann_yield):
                    if first_pass:
                        viable_count += 1
                    rk = getattr(market, "vrp", None)
                    vrp_val = rk(tk, d, None if row is None
                                else row.get("realized_vol")) \
                        if rk is not None else None
                    if vrp_val is None:
                        # Unmeasurable -> NEUTRAL, and COUNTED, same rule as
                        # rank_by="vrp" above -- it is the same statistic.
                        if (d, "entry_ranked_vrp_unknown", tk) not in warnings:
                            warnings.append((d, "entry_ranked_vrp_unknown", tk))
                    key = vrp_viable_sort_key(vrp_val)
                else:
                    key = pct
            elif cfg.rank_by == "none":
                # Batch 4: the inertness CONTROL -- the sort key is a constant,
                # so the pool tuple's second element decides alone. That
                # element is `market.universe.index(tk)`: the FIXED universe
                # order as loaded, never pool-build order or dict order, so
                # the tie-order is itself an inspectable ordering (T3) --
                # read the universe file and you have read the sort. Every
                # gate above still applies; this deletes the sort, not a name.
                key = 0.0
            else:
                key = pct
            # A2b: measured here (day_chain is in scope) and carried, so the
            # sizing block below never re-reads the frame. None = uncapped.
            cap = liquidity_size_cap(day_chain, d, c, cfg)
            if cfg.prefer_chop_half is not None:
                # 0 = the preferred chop half, 1 = everything else. Carried in a
                # SIDE dict rather than in the tuple: the pool tuple is unpacked
                # positionally in three places downstream (including the
                # concentration fallback, which indexes it by number), so
                # widening it is a silent reindex bug waiting to happen -- and
                # was, until the wiring test caught it.
                pref_tier[tk] = 0 if chop_half_of(row) == cfg.prefer_chop_half else 1
            pool.append((-key, market.universe.index(tk), tk, c, mark, cap))
        if first_pool_depth is None:
            first_pool_depth = len(pool)
            first_viable_count = viable_count
        if not pool:
            break
        if cfg.prefer_chop_half is None:
            pool.sort()
        else:
            # The preferred half first, then the ordinary rank key and the fixed
            # tie order INSIDE each half -- so this is strictly a reordering and
            # every candidate stays in the pool. A veto would leave the slot idle
            # when only the other half is available, which is how min_iv_rank
            # cost 18 points of return on 2026-08-04.
            pool.sort(key=lambda p: (pref_tier.get(p[2], 1), p[0], p[1]))
        prw = getattr(cfg, "prefer_rank_window", None)
        if prw is not None:
            # Batch 4: ranks [a, b] of the sorted pool to the FRONT, ordinary
            # rank order inside each tier, the rest behind -- a pure
            # re-ordering (prefer_chop_half shape), never a veto. AFTER both
            # sort branches so it re-orders whatever ordering is in force,
            # BEFORE the rank_window slice below. Python slices make the
            # fallback structural: a pool shorter than `a` yields an empty
            # preferred tier and an unchanged queue, so this cannot starve
            # and needs no warning counter -- there is nothing to count.
            plo, phi = int(prw[0]), int(prw[1])
            pool = pool[plo - 1:phi] + pool[:plo - 1] + pool[phi:]
        rw = getattr(cfg, "rank_window", None)
        if rw is not None:
            # Batch-3 H-B2-3: entries come from ranks [a, b] of the SORTED
            # pool, re-applied to the re-ranked pool every fill-iteration.
            # AFTER both sort branches, BEFORE affordability -- the fallback
            # below then searches only the slice, so the window is a hard
            # scope: falling back to ranks it excludes would silently turn
            # the mechanic into top-of-list under thin pools (zombie-gate
            # shape). Depth accounting above reads the UNSLICED pool, so
            # pool_depth_mean stays comparable across windowed and plain arms.
            lo, hi = int(rw[0]), int(rw[1])
            if len(pool) < lo:
                # The window sits beyond today's pool: the slot stays honestly
                # idle, and loudly -- a starving window is the finding, not a
                # nuisance to be papered over.
                if (d, "entry_rank_window_empty", "pool") not in warnings:
                    warnings.append((d, "entry_rank_window_empty", "pool"))
                break
            pool = pool[lo - 1:hi]
        # A12: equal split FIRST (k == empty_slots is the old budget exactly,
        # so behavior is byte-identical whenever anything is affordable). When
        # NOTHING fits, re-split over fewer effective slots down to one --
        # $5k/N5 offered $1,000/slot, afforded nothing, and sat 82% idle
        # while a $3k contract was listed; the grid then measured which slots
        # could buy anything, not N. Concentration only when the alternative
        # is idleness. Fallback pick order = richest-ranked-first (owner
        # decision 2026-08-01; was least-concentration, skeptic F3).
        budget = available / empty_slots
        # H-B2-1: the notional ceiling, priced off the EQUAL split -- never
        # off the fallback's k-split, which grows exactly when the cap is
        # supposed to bind. Checked in BOTH loops below so the concentration
        # fallback cannot route around it.
        _snc = getattr(cfg, "slot_notional_cap_mult", None)
        cap_notional = None if _snc is None else float(_snc) * budget
        candidates = []
        for (negpct, idx, tk_, c_, mk_, cap_) in pool:
            if cap_notional is not None and c_.strike * mult > cap_notional:
                # Refused under its own name, deduped like every entry gate;
                # the loop moves to the next candidate, so the refusal can
                # only idle a slot when nothing else fits.
                if (d, "entry_gated_notional", tk_) not in warnings:
                    warnings.append((d, "entry_gated_notional", tk_))
                continue
            n_ = int(budget // (c_.strike * mult))
            if n_ <= 0:
                continue                      # unaffordable at this split
            if cap_ is not None and cap_ < n_:
                n_ = cap_                     # A2b: write it smaller, not never
            if n_ <= 0:
                # A2b: affordable, but a single contract is already too large a
                # share of the listed market (or a measured field was missing).
                # Refuse under its own reason -- never silently, and never by
                # substituting a different strike. Deduped for the same reason
                # as the A2 warning: the n_slots while-loop revisits the pool.
                if (d, "entry_gated_size", tk_) not in warnings:
                    warnings.append((d, "entry_gated_size", tk_))
                continue
            candidates.append((negpct, idx, tk_, c_, mk_, cap_, n_))
        if not candidates:
            # Fallback (owner 2026-08-01, skeptic F3): the BEST-RANKED name
            # that fits at ANY concentration wins, sized at the largest k
            # (least concentration) that affords it -- not the cheapest name
            # at the least concentration. Pool is already rank-sorted.
            for cand in pool:
                if cap_notional is not None and \
                        cand[3].strike * mult > cap_notional:
                    # The cap's real bite: the unbounded fallback is where a
                    # 30k contract used to enter a 20k slot. Same warning,
                    # same dedupe; the fallback tries the next-ranked name.
                    if (d, "entry_gated_notional", cand[2]) not in warnings:
                        warnings.append((d, "entry_gated_notional", cand[2]))
                    continue
                cap_ = cand[5]
                for k in range(empty_slots - 1, 0, -1):
                    n = int((available / k) // (cand[3].strike * mult))
                    if cap_ is not None and cap_ < n:
                        n = cap_            # A2b binds on the fallback path too
                    if n > 0:
                        candidates = [(*cand, n)]
                        break
                if candidates:
                    break
        if not candidates:
            break
        _, _, tk, c, mark, _cap, n = candidates[0]
        campaign += 1
        proceeds = sell_proceeds(mark, n, cfg)
        cash += proceeds
        positions.append({"ticker": tk, "shares": 0, "phase": "PUT", "basis": None,
                          "premium": proceeds, "campaign": campaign,
                          "last_spot": market.spot(tk, d, 0.0),
                          "short": {"contract": c, "contracts": n,
                                    "credit": mark.bid, "last_mid": mark.mid}})
        trades.append(Trade(d, "SELL_PUT", c, n, mark.bid, cash, campaign))
        if at_risky_window_edge(market.chain(tk, d), d, c, cfg.put_delta):
            # B11: the surveyed window clipped the delta ladder -- this entry
            # is riskier than configured, loudly
            warnings.append((d, "strike_window_edge", tk))
        route_events.append((d, [(t_[2], -t_[0]) for t_ in candidates], tk))
        held_tickers.add(tk)

    # 2b) capacity accounting. A slot left empty here is the ONLY way a refusal
    # can idle capacity -- the loop rescans all 530 names for every slot, so
    # refusing one name simply promotes the next. That is worth counting rather
    # than asserting: the claim "gates cost turnover by leaving slots idle" was
    # made repeatedly in this project's notes and has never had a number under
    # it.
    if free_at_start > 0:
        pool_depth_sessions += 1
        pool_depth_sum += first_pool_depth or 0
        if (first_pool_depth or 0) < free_at_start:
            pool_starved_days += 1
        scan_names += drops["scanned"]
        drop_no_chain += drops["no_chain"]
        drop_weather += drops["weather"]
        drop_no_contract += drops["no_contract"]
        if cfg.rank_by == "vrp_viable":
            rank_viable_sessions += 1
            rank_viable_sum += first_viable_count or 0
    idle_slot_days += n_slots - len(positions)

    # 3) flat/uncovered accounting + equity mark
    if not positions:
        days_flat += 1
    for pos in positions:
        if pos["short"] is None and pos["phase"] == "CALL" and pos["shares"] >= mult:
            days_shares_uncovered += 1
    liab, shares_val = 0.0, 0.0
    for pos in positions:
        if pos["short"] is not None:
            # A10f: a frozen position's chain quotes may be in restated units
            # (the reason it is frozen) -- never overwrite the stored marks
            # with them. The leg carries its pre-freeze mark into liab until a
            # human clears the freeze; the freeze is already loud every run.
            if pos.get("ca_frozen"):
                short_ = pos["short"]
                liab += short_.get("last_ask", short_["last_mid"]) \
                    * mult * short_["contracts"]
                shares_val += pos["shares"] * pos["last_spot"]
                continue
            day_chain = market.chain(pos["ticker"], d)
            mk = option_mark(day_chain, d, pos["short"]["contract"]) \
                if day_chain is not None else None
            if mk is not None:
                pos["short"]["last_mid"] = mk.mid
                pos["short"]["last_ask"] = mk.ask
                # C1: stamp WHEN this mark was taken -- never on the carried
                # path below, so `mark_asof < today` IS the carried-mark
                # discriminator (TMO sat at $11.30 for six snapshots with
                # nothing saying the price was days old). quote_time is the
                # chain row's per-leg time when the live path provides one.
                pos["short"]["mark_asof"] = str(d.date())
                if mk.quote_time is not None:
                    pos["short"]["mark_quote_time"] = mk.quote_time
            # Marked at the ASK (owner decision 2026-07-29, option B). A short
            # option is a liability dischargeable only by BUYING it back, and you
            # buy at the offer; the midpoint books half a spread the account can
            # never capture. Measured on the live paper run: +$31,752 reported at
            # mid became +$12,061 at exitable prices, and 7 of 25 accounts turned
            # out to be losing. This deliberately breaks the byte-identical
            # anchor -- every portfolio-engine number produced before this line
            # changed is non-comparable with one produced after.
            # `.get` fallback: legs recorded under the old scheme carry no
            # last_ask until their next successful mark.
            short_ = pos["short"]
            liab += short_.get("last_ask", short_["last_mid"]) * mult * short_["contracts"]
        shares_val += pos["shares"] * pos["last_spot"]
    equity_val = cash + shares_val - liab

    state.cash = cash
    state.positions = positions
    state.campaign = campaign
    state.days_flat = days_flat
    state.days_shares_uncovered = days_shares_uncovered
    state.idle_slot_days = idle_slot_days
    state.pool_depth_sum = pool_depth_sum
    state.pool_depth_sessions = pool_depth_sessions
    state.pool_starved_days = pool_starved_days
    state.scan_names = scan_names
    state.drop_no_chain = drop_no_chain
    state.drop_weather = drop_weather
    state.drop_no_contract = drop_no_contract
    state.rank_viable_sum = rank_viable_sum
    state.rank_viable_sessions = rank_viable_sessions
    state.prev_d = d
    return StepResult(trades, equity_val, warnings, route_events)


def run_portfolio_wheel(chains: dict, cfg: WheelConfig, regime_states: dict,
                        clean_start: dict | None = None,
                        selector: str = "vol_pctile",
                        n_slots: int = 1,
                        universe: list | None = None,
                        earnings=None,
                        iv_history=None) -> PortfolioResult:
    if selector not in ("vol_pctile", "chop"):
        raise ValueError(f"selector must be 'vol_pctile' or 'chop', got {selector!r}")
    if cfg.rank_by not in ("vol_pctile", "iv_rank", "vrp", "none", "vrp_viable"):
        # A typo'd sort key must never fall back to the default: the run would
        # print a full set of trades that silently answers a different question
        # than the one the arm claims to be testing.
        raise ValueError(f"rank_by must be 'vol_pctile', 'iv_rank', 'vrp', "
                         f"'none' or 'vrp_viable', got {cfg.rank_by!r}")
    if n_slots < 1:
        raise ValueError(f"n_slots must be >= 1, got {n_slots}")
    # Same stance as rank_by above: a typo'd weather knob must stop the run, not
    # quietly become "off" and print a full set of trades answering a different
    # question. These three are the only chop_* knobs with a closed value set;
    # the numeric ones are checked where they are used.
    for fld in ("chop_half", "prefer_chop_half"):
        v = getattr(cfg, fld, None)
        if v is not None and v not in CHOP_HALVES:
            raise ValueError(f"{fld} must be one of {CHOP_HALVES} or None, "
                             f"got {v!r}")
    if getattr(cfg, "chop_fast_pair", None) not in FAST_PAIRS:
        raise ValueError(
            f"chop_fast_pair must be one of "
            f"{sorted(k for k in FAST_PAIRS if k)} or None, "
            f"got {cfg.chop_fast_pair!r}")
    if getattr(cfg, "chop_min_ma50_slope", None) is not None and \
            cfg.chop_ma50_slope_window not in (None, 5, 10, 20):
        raise ValueError(
            f"chop_ma50_slope_window must be 5, 10 or 20 (the windows "
            f"regime_series precomputes), got {cfg.chop_ma50_slope_window!r}")
    for fld in ("chop_drawdown_band", "chop_dip_z_band"):
        b = getattr(cfg, fld, None)
        if b is not None and (len(b) != 2 or float(b[0]) > float(b[1])):
            raise ValueError(f"{fld} must be [lo, hi] with lo <= hi, got {b!r}")
    rw = getattr(cfg, "rank_window", None)
    if rw is not None and (len(rw) != 2 or int(rw[0]) < 1
                           or int(rw[0]) > int(rw[1])):
        # Same stance as the band knobs: a malformed window must stop the run,
        # not quietly become top-of-list and print trades answering a
        # different question than the variant asks.
        raise ValueError(f"rank_window must be [a, b] with 1 <= a <= b "
                         f"(1-indexed ranks), got {rw!r}")
    prw = getattr(cfg, "prefer_rank_window", None)
    if prw is not None and (len(prw) != 2 or int(prw[0]) < 1
                            or int(prw[0]) > int(prw[1])):
        # Same stance as rank_window: a malformed preference must stop the
        # run, not quietly become top-of-list and print trades answering a
        # different question than the variant asks.
        raise ValueError(f"prefer_rank_window must be [a, b] with 1 <= a <= b "
                         f"(1-indexed ranks), got {prw!r}")
    snc = getattr(cfg, "slot_notional_cap_mult", None)
    if snc is not None and not float(snc) > 0.0:
        # A cap of zero refuses every entry and a negative one is nonsense;
        # both must stop the run, not print a full set of empty sessions.
        raise ValueError(f"slot_notional_cap_mult must be > 0 when set, "
                         f"got {snc!r}")
    if getattr(cfg, "assignment_exit", False):
        # A typo'd policy must never silently fall back to a different one: the
        # run would print a full set of trades answering a question nobody
        # asked, which is the same failure class as a zombie gate.
        if cfg.exit_trigger not in ("tier1_impossible", "pct_below_basis",
                                    "days_stuck", "never"):
            raise ValueError(
                f"exit_trigger must be 'tier1_impossible', 'pct_below_basis', "
                f"'days_stuck' or 'never', got {cfg.exit_trigger!r}")
        if cfg.exit_stop not in ("spot_below_strike", "spot_below_strike_buffer",
                                 "pct_below_basis", "none"):
            raise ValueError(
                f"exit_stop must be 'spot_below_strike', "
                f"'spot_below_strike_buffer', 'pct_below_basis' or 'none', "
                f"got {cfg.exit_stop!r}")
        if cfg.exit_trigger == "pct_below_basis" and cfg.exit_trigger_pct is None:
            raise ValueError("exit_trigger='pct_below_basis' needs "
                             "exit_trigger_pct")
        if cfg.exit_trigger == "days_stuck" and cfg.exit_trigger_days is None:
            raise ValueError("exit_trigger='days_stuck' needs exit_trigger_days")
    if cfg.roll_tested_puts or cfg.put_stop_mult is not None or \
            cfg.liquidate_assignment or cfg.any_regime_gate:
        raise ValueError("portfolio supports the plain+basis wheel only — "
                         "roll/stop/gates/liquidate are solo mechanics")
    if universe is None:
        # DEFAULT: the validated 9-ticker rotation. Sort by fixed tie order and
        # enforce the rotation/reserved/state allow-list.
        universe = sorted(chains, key=lambda t: ROTATION_TIE_ORDER.index(t)
                          if t in ROTATION_TIE_ORDER else len(ROTATION_TIE_ORDER))
        for t in universe:
            if t in RESERVED_TICKERS:
                raise ValueError(f"{t} is a reserved one-shot ticker — never a "
                                 f"rotation universe member")
            if t not in ROTATION_TIE_ORDER:
                raise ValueError(f"{t} is not in the rotation universe "
                                 f"{ROTATION_TIE_ORDER}")
            if t not in regime_states:
                raise ValueError(f"universe member {t} has no regime_states — "
                                 f"routing without state is a bug, not a run")
    else:
        # EXPANDED-BACKTEST: the passed list is the ordering + allow-list. Filter
        # to tickers present in `chains` (preserving passed order); drop the
        # rotation/reserved restrictions but keep the regime_states requirement.
        universe = [t for t in universe if t in chains]
        for t in universe:
            if t not in regime_states:
                raise ValueError(f"universe member {t} has no regime_states — "
                                 f"routing without state is a bug, not a run")
    clean_start = {**DEFAULT_CLEAN_START, **(clean_start or {})}
    if cfg.earnings_blackout and earnings is None:
        # Fail loudly rather than run a no-op gate. Same stance as the A2
        # liquidity gate: a threshold that is SET but unmeasurable must refuse,
        # never silently pass. A gate believed to be on while it is structurally
        # inert is the zombie-gate failure class (see earnings.py).
        raise ValueError("earnings_blackout=True needs an earnings calendar — "
                         "pass earnings=EarningsCalendar.load() (build it with "
                         "scripts/pull_earnings.py)")
    if cfg.min_iv_rank is not None and iv_history is None:
        # Same stance as the earnings gate above: a floor that is SET but has
        # no history to measure against would pass every entry as "unknown",
        # which reads in the log exactly like a gate that is working.
        raise ValueError("min_iv_rank needs an IV history — pass "
                         "iv_history=IVHistory.from_chains(chains)")
    if cfg.rank_by == "iv_rank" and iv_history is None:
        # Same stance as the floor above, and it matters MORE for a sort: with
        # no history every name ranks NEUTRAL, so the pool keeps its universe
        # tie order and the run prints a full, plausible set of trades. A veto
        # with no history at least trades like the baseline; a ranker with no
        # history looks like it is working and is answering nothing.
        raise ValueError("rank_by='iv_rank' needs an IV history — pass "
                         "iv_history=IVHistory.from_chains(chains)")
    if cfg.rank_by == "vrp" and iv_history is None:
        # The variance risk premium is IV over realized vol; with no IV there
        # is no numerator, every name scores NEUTRAL, and the sort degrades to
        # universe order while looking exactly like a working ranker.
        raise ValueError("rank_by='vrp' needs an IV history — pass "
                         "iv_history=IVHistory.from_chains(chains)")
    if cfg.rank_by == "vrp_viable" and iv_history is None:
        # Same stance as vrp above -- tier 1 reads the same statistic, so with
        # no IV history it degrades the same way.
        raise ValueError("rank_by='vrp_viable' needs an IV history — pass "
                         "iv_history=IVHistory.from_chains(chains)")

    market = BatchMarket(chains, regime_states, clean_start, universe,
                         earnings=earnings, iv_history=iv_history)
    dates = sorted({pd.Timestamp(d) for t in universe
                    for d in pd.to_datetime(chains[t]["date"]).unique()})
    mult = cfg.contract_multiplier

    state = PortfolioState(cash=cfg.starting_capital, positions=[])
    warnings, route_events, trades, equity = [], [], [], {}
    for d in dates:
        r = step_one_day(state, market, d, cfg, selector=selector, n_slots=n_slots)
        trades.extend(r.trades)
        warnings.extend(r.warnings)
        route_events.extend(r.route_events)
        equity[d] = r.equity

    # residual-settlement finalizer (batch-only; the live bot never runs this)
    cash = state.cash
    residual_settled = False
    final_shares = {}
    for pos in state.positions:
        if pos["short"] is not None:
            last = dates[-1]
            day_chain = market.chain(pos["ticker"], last)
            mk = option_mark(day_chain, last, pos["short"]["contract"]) \
                if day_chain is not None else None
            # At the ASK, for the same reason the daily mark is (option B): this
            # finalizer BUYS the residual book back, and a buyer pays the offer.
            exit_px = mk.ask if mk is not None else \
                pos["short"].get("last_ask", pos["short"]["last_mid"])
            cash -= exit_px * mult * pos["short"]["contracts"]
            residual_settled = True
        if pos["shares"]:
            final_shares[pos["ticker"]] = final_shares.get(pos["ticker"], 0) + pos["shares"]
    return PortfolioResult(pd.Series(equity), trades, cash, final_shares,
                           residual_settled, days_flat=state.days_flat,
                           warnings=warnings,
                           days_shares_uncovered=state.days_shares_uncovered,
                           route_events=route_events,
                           n_campaigns_opened=state.campaign,
                           idle_slot_days=state.idle_slot_days,
                           pool_depth_sum=state.pool_depth_sum,
                           pool_depth_sessions=state.pool_depth_sessions,
                           pool_starved_days=state.pool_starved_days,
                           scan_names=state.scan_names,
                           drop_no_chain=state.drop_no_chain,
                           drop_weather=state.drop_weather,
                           drop_no_contract=state.drop_no_contract,
                           rank_viable_sum=state.rank_viable_sum,
                           rank_viable_sessions=state.rank_viable_sessions)
