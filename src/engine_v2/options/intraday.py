"""Intraday (hourly) option marks for take-profit timing. Pulls whole-chain OHLC
(trade prices; single-strike quotes aren't available on STANDARD) and indexes by
contract. Also the two-pass orchestration. Isolated from the gate."""
from __future__ import annotations
import pandas as pd
from .theta_client import ThetaError

_RIGHT = {"CALL": "C", "PUT": "P", "C": "C", "P": "P"}

def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["timestamp"]).dt.tz_localize(None),
        "expiry": pd.to_datetime(df["expiration"]).dt.normalize(),
        "strike": df["strike"].astype(float),
        "right": df["right"].map(_RIGHT),
        "close": df["close"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "volume": df["volume"].astype(float),
    })
    return out.dropna(subset=["close"]).sort_values(["expiry","strike","right","timestamp"]).reset_index(drop=True)

def pull_option_intraday(client, symbol, expiration, start, end,
                         interval="1h", strike_range=10) -> pd.DataFrame:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    lo = start
    while lo <= end:
        hi = min(lo + pd.Timedelta(days=25), end)
        try:
            raw = client.get_csv("/v3/option/history/ohlc", symbol=symbol,
                                 expiration=pd.Timestamp(expiration).strftime("%Y-%m-%d"),
                                 start_date=lo.strftime("%Y-%m-%d"),
                                 end_date=hi.strftime("%Y-%m-%d"),
                                 interval=interval, strike_range=strike_range)
            if len(raw):
                frames.append(_normalize(raw))
        except ThetaError as e:
            if e.code != 472:
                raise
        lo = hi + pd.Timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["timestamp","expiry","strike","right","close","high","low","volume"])
    return pd.concat(frames, ignore_index=True)

def intraday_marks(df: pd.DataFrame) -> dict:
    # volume > 0 and close > 0 only: a bar with no trade is not a price. On
    # illiquid tickers ~60% of hourly bars are volume-0/close-0 placeholders;
    # letting them through hands the TP check phantom fills (see the engine's
    # zero-bar guard — this filter is the first line, that guard the second).
    df = df[(df["volume"] > 0) & (df["close"] > 0)]
    out = {}
    for (exp, strike, right), g in df.groupby(["expiry","strike","right"]):
        out[(pd.Timestamp(exp), float(strike), right)] = \
            g[["timestamp","close"]].sort_values("timestamp").reset_index(drop=True)
    return out

def held_contracts(result) -> list:
    """Each short position -> (expiry, strike, right, open_date, close_date)."""
    OPEN = {"SELL_PUT", "SELL_CALL", "ROLL_OPEN"}
    CLOSE = {"CLOSE_PUT", "CLOSE_CALL", "ASSIGNED", "PUT_EXPIRED", "CALLED_AWAY",
             "CALL_EXPIRED", "ROLL_CLOSE", "STOP_CLOSE"}
    held, cur = [], None
    for t in result.trades:
        c = t.contract
        if t.action in OPEN:
            cur = (pd.Timestamp(c.expiry), float(c.strike), c.right, pd.Timestamp(t.date))
        elif t.action in CLOSE and cur is not None:
            held.append((cur[0], cur[1], cur[2], cur[3], pd.Timestamp(t.date)))
            cur = None
    if cur is not None:  # still open at window end -> hold through the last activity date
        last_date = max((pd.Timestamp(t.date) for t in result.trades), default=cur[3])
        held.append((cur[0], cur[1], cur[2], cur[3], max(last_date, cur[3])))
    return held

def run_wheel_intraday(chain, cfg, intraday_df, regime_states=None):
    from .wheel import run_wheel
    return run_wheel(chain, cfg, intraday=intraday_marks(intraday_df),
                     regime_states=regime_states)

def run_regime_router_intraday(chain, cfg, intraday_df, regime_states):
    from .regime_router import run_regime_router
    return run_regime_router(chain, cfg, regime_states,
                             intraday=intraday_marks(intraday_df))
