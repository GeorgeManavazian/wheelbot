"""Our own implied-volatility solver -- the one that owns the SCALE.

Why this exists (owner ruling 2026-08-07, [[2026-08-07 - The IV provenance rule,
the solver is the scale]]): IV rank is a percentile of a ticker against its own
trailing 252 observations, so every observation in a series must be produced the
same way. Vendors do not agree -- but measured over 63,016 ranked ticker-days, two
different IV models fed identical quotes pick the SAME top-ranked name on 95.5% of
days (spearman 0.9988). So vendor identity is cheap and a mid-window MODEL CHANGE is
not: it shifts every post-change observation relative to its own past and pins ranks
toward 0 or 1.

The live bot pulls Schwab daily while the history came from a vendor. That splice is
guaranteed unless WE compute the IV from THEIR quotes. Hence this module: vendors
supply quotes, this supplies IV, and the scale never changes when the source does.

The convention is fitted, not assumed -- 6,240 live Schwab contracts across 42
tickers, 2026-08-06 (scratchpad/diag_schwab_iv_fit.py):

    calendar days / 365       median err -0.0015   <- fitted
    trading days / 252        median err -0.0742   <- decisively wrong
    mark vs mid               indistinguishable
    rate/dividend zeroed      mean abs err 0.0169 -> 0.0189 (barely matters)

Restricted to the OTM 0.20-0.40 delta band the bot actually trades, plain European
Black-Scholes matches Schwab to a median absolute 0.0063, p90 0.0144 -- including
dividend payers (q>=2.5%: 0.0089; q>=4%: 0.0052).

OUTSIDE that band it falls apart: ITM p90 0.1627, and dividend payers ITM 0.0705
median (Ford reads +0.154 ITM vs +0.031 in-band). An American engine and discrete
dated dividends would be needed there. `select_contract` targets a 0.30-delta PUT,
which is OTM essentially always, so the series never records those contracts -- and
that restriction is the whole correctness argument, so it is ENFORCED here rather
than documented. If the observation ever needs to move ITM, this module is void.

Spec: this file + scratchpad/diag_schwab_iv_fit.py
"""
import math

import pytest

from src.engine_v2.options.iv_solve import implied_vol_put


def _bs_put(S, K, T, r, q, sigma):
    """Textbook European put -- the reference the solver must invert."""
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    nd = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return K * math.exp(-r * T) * nd(-d2) - S * math.exp(-q * T) * nd(-d1)


def test_recovers_the_volatility_a_price_was_made_with():
    # 11 DTE is the FROZEN target; 0.30-delta put on a $100 name sits ~$96.
    price = _bs_put(100.0, 96.0, 11 / 365, 0.04, 0.01, 0.2500)
    assert implied_vol_put(price, underlying=100.0, strike=96.0, dte=11,
                           rate=0.04, div_yield=0.01) == pytest.approx(0.2500, abs=1e-4)


def test_refuses_an_in_the_money_put():
    """The OTM restriction is the correctness argument, so it is enforced.

    European BS + a continuous yield is accurate in the 0.20-0.40 delta OTM band
    (median abs 0.0063) and NOT outside it (ITM p90 0.1627; dividend payers ITM
    0.0705 median). A number returned here would be quietly wrong, which is the
    failure mode this project keeps rediscovering."""
    price = _bs_put(100.0, 104.0, 11 / 365, 0.04, 0.01, 0.2500)
    with pytest.raises(ValueError, match="in the money"):
        implied_vol_put(price, underlying=100.0, strike=104.0, dte=11,
                        rate=0.04, div_yield=0.01)


def _gdx_band_contracts():
    """OTM puts in the 0.20-0.40 delta band from the captured Schwab GDX chain."""
    import json
    raw = json.load(open("fixtures/option_chain_gdx_puts.json"))
    S = raw["underlyingPrice"]
    r, q = raw["interestRate"] / 100.0, raw["dividendYield"] / 100.0
    out = []
    for exp_key, strikes in (raw.get("putExpDateMap") or {}).items():
        for _sk, cts in strikes.items():
            c = cts[0]
            K, d, v = c["strikePrice"], c.get("delta"), c.get("volatility")
            bid, ask, mark = c.get("bid"), c.get("ask"), c.get("mark")
            if None in (d, v, bid, ask, mark) or v != v or d != d:
                continue
            if K >= S or not 0.20 <= abs(d) <= 0.40 or bid <= 0 or mark <= 0:
                continue
            if (ask - bid) / mark >= 0.5 or c["daysToExpiration"] < 5:
                continue
            out.append((mark, S, K, c["daysToExpiration"], r, q, v / 100.0))
    return out


def test_matches_schwabs_own_iv_on_the_captured_chain():
    """Anchor to reality, not to my own reference implementation.

    Measured on 6,240 live Schwab contracts across 42 tickers (2026-08-06): in this
    band the median absolute error is 0.0063 and p90 is 0.0144. The generous 0.02
    bound below still fails loudly if the time convention drifts -- trading-days/252
    lands 0.0742 out."""
    rows = _gdx_band_contracts()
    assert len(rows) >= 5, f"fixture yielded too few band contracts: {len(rows)}"
    errs = sorted(abs(implied_vol_put(p, underlying=S, strike=K, dte=dte,
                                      rate=r, div_yield=q) - schwab)
                  for (p, S, K, dte, r, q, schwab) in rows)
    median = errs[len(errs) // 2]
    assert median < 0.02, f"median abs error {median:.4f} over {len(errs)} contracts"
