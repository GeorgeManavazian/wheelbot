import io
import pandas as pd
from src.engine_v2.options.chain import normalize_greeks_eod, Contract, Mark

RAW = (
 'symbol,expiration,strike,right,timestamp,open,high,low,close,volume,count,'
 'bid_size,bid_exchange,bid,bid_condition,ask_size,ask_exchange,ask,ask_condition,'
 'delta,implied_vol,underlying_timestamp,underlying_price\n'
 '"SPY","2024-01-19",477.000,"CALL",2024-01-16T16:14:46,1.74,2.10,0.86,1.26,43870,4393,'
 '306,60,1.26,50,1,1,1.27,50,0.3611,0.1196,2024-01-16T17:15:18,474.93\n'
 '"SPY","2024-01-19",477.000,"PUT",2024-01-16T16:09:57,3.06,4.44,2.20,2.90,13740,1544,'
 '306,1,2.96,50,112,69,3.00,50,-0.6497,0.1107,2024-01-16T17:15:18,474.93\n'
 '"SPY","2024-01-19",999.000,"PUT",2024-01-16T16:00:00,0,0,0,0,0,0,'
 '0,0,0,50,0,0,0,50,-0.99,0.5,2024-01-16T17:15:18,474.93\n'   # bad: bid/ask 0 -> dropped
)

def _norm():
    return normalize_greeks_eod(pd.read_csv(io.StringIO(RAW)))

def test_columns_and_types():
    df = _norm()
    assert list(df.columns) == ["date","expiry","dte","strike","right","bid","ask","mid",
                                "close","delta","iv","underlying"]
    assert set(df["right"]) <= {"P","C"}
    assert df["date"].dt.tz is None

def test_values_and_derived():
    df = _norm().set_index("right")
    assert df.loc["P","mid"] == (2.96 + 3.00) / 2
    assert df.loc["P","underlying"] == 474.93
    # date from underlying_timestamp (2024-01-16); dte to 2024-01-19 = 3
    assert df.loc["P","date"] == pd.Timestamp("2024-01-16")
    assert df.loc["P","dte"] == 3
    assert df.loc["P","iv"] == 0.1107

def test_bad_rows_dropped():
    df = _norm()
    assert (df["strike"] == 999.0).sum() == 0  # zero-bid/ask row removed
