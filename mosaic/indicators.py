"""Technical indicators + session bookkeeping. Everything is vectorized (fast)."""
import numpy as np
import pandas as pd


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def sma(s, n):
    return s.rolling(n).mean()


def true_range(df):
    pc = df.close.shift(1)
    return pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def adx(df, n=14):
    up = df.high.diff()
    dn = -df.low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df).ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0)


def efficiency_ratio(s, n=20):
    """1 = straight-line trend, 0 = pure chop."""
    change = (s - s.shift(n)).abs()
    noise = s.diff().abs().rolling(n).sum()
    return (change / noise.replace(0, np.nan)).fillna(0)


def session_columns(df, rth_start_min):
    """Adds: date (session date), mins (minutes since midnight ET), bar_of_day (bars since RTH open)."""
    df = df.copy()
    df["mins"] = df.index.hour * 60 + df.index.minute
    df["date"] = df.index.date
    df["day_id"] = pd.factorize(df["date"])[0]
    in_rth = df["mins"] >= rth_start_min
    df["bar_of_day"] = in_rth.groupby(df["day_id"]).cumsum() - 1
    df.loc[~in_rth, "bar_of_day"] = -1
    return df


def vwap(df):
    """Session VWAP (resets each trading date). Falls back to cumulative typical price if no volume."""
    tp = (df.high + df.low + df.close) / 3
    vol = df.volume.where(df.volume > 0, 1.0)
    g = df["day_id"]
    return (tp * vol).groupby(g).cumsum() / vol.groupby(g).cumsum()


def opening_range(df, rth_start_min, minutes):
    """Per-day high/low of the first `minutes` of RTH. NaN until the range is complete."""
    in_or = (df["mins"] >= rth_start_min) & (df["mins"] < rth_start_min + minutes)
    g = df["day_id"]
    or_high = df.high.where(in_or).groupby(g).transform("max")
    or_low = df.low.where(in_or).groupby(g).transform("min")
    done = df["mins"] >= rth_start_min + minutes
    return or_high.where(done), or_low.where(done)


def prev_day_levels(df):
    daily = df.groupby("day_id").agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
    prev = daily.shift(1)
    return (df["day_id"].map(prev["h"]), df["day_id"].map(prev["l"]), df["day_id"].map(prev["c"]))


def prepare(df, rth_start_min):
    """One-time indicator pack shared by every piece (so the search stays fast)."""
    df = session_columns(df, rth_start_min)
    df["atr"] = atr(df, 14)
    df["atr_median"] = df["atr"].rolling(300, min_periods=50).median()
    df["adx"] = adx(df, 14)
    df["er"] = efficiency_ratio(df.close, 30)
    df["vwap"] = vwap(df)
    df["ema_trend"] = ema(df.close, 200)
    pdh, pdl, pdc = prev_day_levels(df)
    df["pdh"], df["pdl"], df["pdc"] = pdh, pdl, pdc
    return df
