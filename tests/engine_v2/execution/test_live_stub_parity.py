import random
from src.engine_v2.execution.sim import Order, fill_order
from src.engine_v2.execution.live_stub import live_fill

def test_live_stub_matches_sim_bytewise():
    random.seed(42)
    for _ in range(200):
        side = random.choice(["buy", "sell"])
        kind = random.choice(["market", "limit"])
        qty = random.uniform(1, 10_000)
        bid = random.uniform(10, 500)
        ask = bid + random.uniform(0.01, 0.20)
        lim = None if kind == "market" else random.uniform(bid - 0.05, ask + 0.05)
        adv = random.uniform(1e6, 1e10)
        o = Order("TEST", side, kind, qty, lim)
        f_sim = fill_order(o, bid, ask, adv)
        f_live = live_fill(o, bid, ask, adv)
        assert f_sim == f_live, f"parity break on {o} bid={bid} ask={ask}"
