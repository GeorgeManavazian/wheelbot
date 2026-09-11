import urllib.error
import io
import pandas as pd
import pytest
from src.engine_v2.options.theta_client import ThetaClient, ThetaError

class _Resp:
    def __init__(self, code, body): self.status = code; self._b = body.encode()
    def read(self): return self._b
    def getcode(self): return self.status
    def __enter__(self): return self
    def __exit__(self, *a): return False

def test_get_csv_parses(monkeypatch):
    c = ThetaClient()
    monkeypatch.setattr(c, "_open", lambda url: _Resp(200, "a,b\n1,2\n3,4\n"))
    df = c.get_csv("/v3/x", symbol="SPY")
    assert list(df.columns) == ["a", "b"]
    assert df.iloc[1]["a"] == 3

def test_get_csv_raises_on_error(monkeypatch):
    c = ThetaClient()
    monkeypatch.setattr(c, "_open", lambda url: _Resp(404, "Not Found\n..."))
    with pytest.raises(ThetaError):
        c.get_csv("/v3/x")

def test_get_csv_maps_httperror_to_thetaerror(monkeypatch):
    # urllib raises HTTPError on 4xx/5xx before we can inspect the response;
    # ThetaData uses custom codes like 472 (no data). Must surface as ThetaError
    # carrying .code so the puller can skip a single no-data expiration.
    def _raise(url):
        raise urllib.error.HTTPError(url, 472, "No data", {}, io.BytesIO(b"no data\n"))
    c = ThetaClient()
    monkeypatch.setattr(c, "_open", _raise)
    with pytest.raises(ThetaError) as ei:
        c.get_csv("/v3/x")
    assert ei.value.code == 472

def test_list_expirations_parses(monkeypatch):
    c = ThetaClient()
    monkeypatch.setattr(c, "_open",
        lambda url: _Resp(200, 'symbol,expiration\n"SPY","2024-01-19"\n"SPY","2024-02-16"\n'))
    exps = c.list_expirations("SPY")
    assert exps == [pd.Timestamp("2024-01-19"), pd.Timestamp("2024-02-16")]
