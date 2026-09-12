"""Modern-era bar loader. Rejects any query before 2007-01-01."""
from __future__ import annotations
import pandas as pd

MODERN_ERA_START = pd.Timestamp("2007-01-01")

class PreModernEraError(ValueError):
    pass

def load_bars(tickers, start, end, source="parquet",
              path="fixtures/bars_2007_2010_small.parquet") -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    if start_ts < MODERN_ERA_START:
        raise PreModernEraError(
            f"start={start} < MODERN_ERA_START={MODERN_ERA_START.date()}"
        )
    if source != "parquet":
        raise NotImplementedError(f"source={source} not yet supported")
    df = pd.read_parquet(path)
    missing = [t for t in tickers if t not in df.columns.get_level_values(0)]
    if missing:
        raise KeyError(f"tickers not in bar file: {missing}")
    df = df.loc[start_ts:pd.Timestamp(end), tickers]
    return df
