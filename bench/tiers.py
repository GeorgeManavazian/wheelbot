"""Tiers: the conditions an experiment runs under.

A tier is one bundle of (universe, window, max DTE headroom). It exists so that
"which names, over which years" is ONE choice, made once, rather than three
fields a variant can set independently -- because two variants that differ in
any of them are not comparable, and the old harnesses differed in all three.

  smoke  12 liquid ETFs, 6 months.  Seconds. For the inner loop while you are
         still deciding whether a mechanic does anything at all.
         NOT EVIDENCE. `bench/policy.toml` refuses it for promotion by default,
         and a smoke scorecard is stamped `evidence = false` so it cannot be
         mistaken for one later.

  mid    60 names, ~2 years. Minutes. Enough signal to kill a bad idea without
         paying for the full corpus. Not evidence.

  live   531 names, 2024-08-01 -> 2026-07-01. THE DEFAULT, and the only window
         where the live config is faithfully reproducible. See below.

  full   531 names, 2022-06-01 -> 2026-07-01. Twice the history, at the cost of
         a structural divergence. See below.

WHY `live` IS THE DEFAULT AND NOT `full` -- measured 2026-08-14. The master
config gates on open interest (`liq_min_open_interest = 250`), and
`liquidity_ok` REFUSES any contract whose open_interest is missing when a
threshold is set. The OI store (`data/options/open_interest/`) covers:

    2021-08 onward ....  86 of 534 tickers
    2024-08 onward ... 533 of 534 tickers

So a run of the live config starting 2022-06 spends its first 26 months able to
enter only 86 names, and refuses the other 443 outright -- not because they were
illiquid, but because nobody had pulled their OI yet. That is not a strategy
result. It is the same class of error as the A2b rel-spread gate refusing names
for being cheap: a gate firing on data availability rather than on the thing it
claims to measure.

`full` is kept because 4.1 years of history is worth having and because a
variant that does not touch liquidity is unaffected by any of this. Running it
is fine; running it and reading the first 26 months as a strategy finding is
not, so `run._divergences` measures the actual per-row OI coverage of every run
and writes it into the scorecard.

WHY EVERY WINDOW ENDS 2026-07-01. The ThetaData Options STANDARD subscription
lapsed ~2026-07-25, so the corpus is frozen by circumstance. That is
inconvenient for coverage and excellent for reproducibility: the exam data
cannot drift under the bench. If the subscription is renewed, EXTENDING these
windows is a decision with consequences -- every existing scorecard becomes
incomparable -- so it must be made deliberately, here, in one place.

NO HELD-OUT SPLIT, and it is deliberate (owner call, 2026-08-09). Every arm sees
the whole window, so a bench winner is a RANKING, not a forecast. The forward
paper run is the exam. Any scorecard that gets read as a prediction is being
read wrong, and `scorecard.render` says so on every printout.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# The 12 names at the head of bench/universe.UNIVERSE: the largest, most liquid
# ETFs in the store, chosen because their chains are dense enough that a smoke
# run exercises the gates instead of skipping them for lack of rows.
SMOKE_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "VOO", "IVV",
                 "VTI", "VEA", "VWO", "EFA", "FXI", "GLD"]

MID_COUNT = 60          # first 60 of UNIVERSE -- ETFs then large-cap singles


@dataclass(frozen=True)
class Tier:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_tickers: int | None       # None = the whole universe
    explicit: tuple | None      # a fixed ticker list, or None
    is_evidence: bool
    blurb: str

    def tickers(self) -> list[str]:
        from bench.universe import UNIVERSE
        if self.explicit is not None:
            return [t for t in self.explicit if t in set(UNIVERSE)] or list(self.explicit)
        if self.n_tickers is None:
            return list(UNIVERSE)
        return list(UNIVERSE)[:self.n_tickers]


#: The month the OI store reaches near-full ticker coverage. Every window that
#: intends to run the live config faithfully starts on or after this.
OI_COMPLETE_FROM = pd.Timestamp("2024-08-01")

TIERS: dict[str, Tier] = {
    "smoke": Tier("smoke", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-01"),
                  None, tuple(SMOKE_TICKERS), False,
                  "12 ETFs, 6 months -- the inner loop. Never evidence."),
    "mid": Tier("mid", OI_COMPLETE_FROM, pd.Timestamp("2026-07-01"),
                MID_COUNT, None, False,
                "60 names, 23 months -- kill a bad idea cheaply."),
    "live": Tier("live", OI_COMPLETE_FROM, pd.Timestamp("2026-07-01"),
                 None, None, True,
                 "531 names, 23 months -- the only faithful window."),
    "full": Tier("full", pd.Timestamp("2022-06-01"), pd.Timestamp("2026-07-01"),
                 None, None, True,
                 "531 names, 4.1 years -- long history, OI-blind before 2024-08."),
}

#: `live`, not `full`. A default that quietly refuses 443 of 531 names for its
#: first 26 months is a default that produces confident wrong answers.
DEFAULT_TIER = "live"


def get(name: str) -> Tier:
    try:
        return TIERS[name]
    except KeyError:
        raise ValueError(
            f"unknown tier `{name}`. Known: {', '.join(TIERS)}. A tier is not a "
            f"free-form window on purpose -- two variants run over different "
            f"windows are not comparable, and the bench refuses to pretend "
            f"otherwise.") from None


def max_dte_for(target_dte: int) -> int:
    """Chain rows above this cannot be selected by an arm at `target_dte`.

    `select.derived_band(t) = (max(5, t-2), t+4)`, so t+4 is the ceiling; the
    extra headroom covers marking a held contract as its dte counts down."""
    return int(target_dte) + 4
