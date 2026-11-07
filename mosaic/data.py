"""Data collector: pulls bars from Yahoo Finance, keeps growing a local history, and imports CSVs
from other sources (Databento, TopstepX exports, FirstRate, Kibot ...)."""
import os
import numpy as np
import pandas as pd
from .config import CONTRACTS, DATA_DIR, YF_PERIODS

TZ = "America/New_York"
COLS = ["open", "high", "low", "close", "volume"]


def store_path(symbol, interval):
    return os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")


def load_bars(symbol, interval) -> pd.DataFrame:
    p = store_path(symbol, interval)
    if not os.path.exists(p):
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)
    df.index.name = "time"
    return df[COLS].astype(float)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df.high >= df.low) & (df.close > 0)]
    df["volume"] = df["volume"].fillna(0)
    return df[COLS]


def save_bars(symbol, interval, new: pd.DataFrame) -> pd.DataFrame:
    """Merge new bars into the stored history (never throws old data away)."""
    old = load_bars(symbol, interval)
    new = new.copy()
    new.index = pd.to_datetime(new.index, utc=True).tz_convert(TZ)
    merged = _clean(pd.concat([old, new]))
    merged.index.name = "time"
    merged.to_csv(store_path(symbol, interval))
    return merged


def fetch_yf(symbol, interval, period=None) -> pd.DataFrame:
    import yfinance as yf
    ticker = CONTRACTS[symbol]["ticker"]
    period = period or YF_PERIODS.get(interval, "60d")
    raw = yf.download(ticker, interval=interval, period=period, auto_adjust=False,
                      progress=False, threads=False)
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=COLS)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    raw = raw[COLS]
    idx = pd.to_datetime(raw.index)
    raw.index = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(TZ)
    return _clean(raw)


def collect(symbols, intervals, log=print):
    """Pull everything Yahoo allows and merge it into the local store. Run this daily to grow history."""
    summary = {}
    for sym in symbols:
        for iv in intervals:
            try:
                new = fetch_yf(sym, iv)
                merged = save_bars(sym, iv, new)
                summary[(sym, iv)] = len(merged)
                span = f"{merged.index[0]:%Y-%m-%d} -> {merged.index[-1]:%Y-%m-%d}" if len(merged) else "empty"
                log(f"  {sym} {iv}: +{len(new)} new, {len(merged)} bars stored ({span})")
            except Exception as e:  # keep going with other symbols
                log(f"  {sym} {iv}: FAILED ({e})")
    return summary


def import_csv(path, symbol, interval, time_col=None, tz="UTC", log=print):
    """Import bars from any CSV. Finds the time/OHLCV columns by name (case-insensitive)."""
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    time_col = time_col or next((low[k] for k in ("time", "timestamp", "datetime", "date", "ts_event") if k in low), None)
    if time_col is None:
        raise ValueError("Could not find a time column; pass time_col=")
    mapping = {}
    for want in COLS:
        for cand in (want, want[0], want.capitalize()):
            if cand.lower() in low:
                mapping[low[cand.lower()]] = want
                break
    out = df.rename(columns=mapping)
    for c in COLS:
        if c not in out:
            out[c] = 0.0 if c == "volume" else np.nan
    t = pd.to_datetime(out[time_col], utc=(tz == "UTC"))
    if t.dt.tz is None:
        t = t.dt.tz_localize(tz)
    out.index = t
    merged = save_bars(symbol, interval, out[COLS].astype(float))
    log(f"  imported {len(out)} rows -> {symbol} {interval} now {len(merged)} bars")
    return merged


def rth_only(df: pd.DataFrame, start_min, end_min) -> pd.DataFrame:
    mins = df.index.hour * 60 + df.index.minute
    return df[(mins >= start_min) & (mins < end_min)]
