"""Fake futures data for smoke-testing the machinery (demo mode). NOT for judging strategies."""
import numpy as np
import pandas as pd


def make_bars(days=120, interval_min=5, start_price=5000.0, seed=1, trend_days=True):
    rng = np.random.default_rng(seed)
    rows = []
    price = start_price
    day0 = pd.Timestamp("2026-01-05")
    d = 0
    while len(rows) < days * (390 // interval_min):
        day = day0 + pd.Timedelta(days=d); d += 1
        if day.weekday() >= 5:
            continue
        drift = rng.normal(0, 0.0004) if trend_days else 0.0
        vol = abs(rng.normal(0.0009, 0.0003)) + 0.0004
        for m in range(9 * 60 + 30, 16 * 60, interval_min):
            t = (day + pd.Timedelta(minutes=m)).tz_localize("America/New_York")
            ret = drift + rng.normal(0, vol)
            o = price
            c = o * (1 + ret)
            h = max(o, c) * (1 + abs(rng.normal(0, vol / 2)))
            l = min(o, c) * (1 - abs(rng.normal(0, vol / 2)))
            rows.append((t, round(o / 0.25) * 0.25, round(h / 0.25) * 0.25, round(l / 0.25) * 0.25, round(c / 0.25) * 0.25, int(abs(rng.normal(800, 300)))))
            price = c
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"]).set_index("time")
    return df
