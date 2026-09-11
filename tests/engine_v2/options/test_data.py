"""Tests for per-ticker data discovery helpers."""

from src.engine_v2.options.data import available_tickers, intraday_path, chain_path


def test_available_tickers(tmp_path):
    for n in ("spy_greeks_eod_all.parquet", "gdx_greeks_eod_all.parquet",
              "gdx_ohlc_1h_all.parquet", "junk.parquet"):
        (tmp_path / n).touch()
    assert available_tickers(str(tmp_path)) == ["GDX", "SPY"]
    assert intraday_path("GDX", str(tmp_path)) is not None
    assert intraday_path("SPY", str(tmp_path)) is None
    assert chain_path("GDX", str(tmp_path)).endswith("gdx_greeks_eod_all.parquet")
