"""The puzzle pieces. Every strategy = ENTRY + FILTERS + EXIT + SIZE.

Entry  : when to go long (+1) / short (-1). Signal fires on a bar close; the trade opens at the NEXT bar's open.
Filter : takes signals and removes the ones it doesn't like (wrong time, wrong market mood, wrong direction).
Exit   : stop / target / trailing / breakeven / time-out rules (the engine does the mechanics).
Size   : how many contracts.
Each piece has a `grid` of parameter values the search is allowed to try. Keep grids small on purpose:
more knobs = easier to fit the past and fail the future."""
import itertools
import numpy as np
import pandas as pd
from . import indicators as ind


class Piece:
    kind = ""
    name = ""
    grid: dict = {}
    mechanism = ""        # which of the pack's five mechanisms this piece relies on
    hypothesis = ""       # why it should make money (who is on the other side)
    fails_when = ""       # what would make the mechanism stop working

    def __init__(self, **params):
        self.params = {**self.defaults(), **params}

    @classmethod
    def defaults(cls):
        return {k: v[len(v) // 2] for k, v in cls.grid.items()}

    @classmethod
    def variants(cls):
        keys = list(cls.grid)
        for combo in itertools.product(*[cls.grid[k] for k in keys]):
            yield cls(**dict(zip(keys, combo)))

    def neighbours(self):
        """Same piece with each numeric parameter nudged one grid step up/down (robustness check)."""
        out = []
        for k, vals in self.grid.items():
            if not isinstance(vals[0], (int, float)) or len(vals) < 2:
                continue
            i = vals.index(self.params[k]) if self.params[k] in vals else None
            if i is None:
                continue
            for j in (i - 1, i + 1):
                if 0 <= j < len(vals):
                    p = dict(self.params); p[k] = vals[j]
                    out.append(type(self)(**p))
        return out

    def spec(self):
        return {"name": self.name, "params": self.params}

    def label(self):
        ps = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({ps})"


def _cross_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_dn(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


# ----------------------------------------------------------------------------- ENTRIES
class Entry(Piece):
    kind = "entry"

    def signals(self, df, cfg) -> np.ndarray:
        raise NotImplementedError


class OpeningRangeBreakout(Entry):
    """Classic day-trade: price breaks above/below the high/low of the first N minutes."""
    name = "OpeningRangeBreakout"
    mechanism = "information / behavioral"
    hypothesis = "Overnight news gets priced in during the first minutes; a break of that range shows which side won and trend-followers pile on."
    fails_when = "Range breaks get faded (choppy, low-volume days); the pattern is widely known, so edge decays as more bots trade it."
    grid = {"or_minutes": [15, 30, 60]}

    def signals(self, df, cfg):
        hi, lo = ind.opening_range(df, cfg.rth_start_min, self.params["or_minutes"])
        long = _cross_up(df.close, hi)
        short = _cross_dn(df.close, lo)
        return np.where(long, 1, np.where(short, -1, 0)).astype(np.int8)


class EMACrossMomentum(Entry):
    """Fast average crosses slow average = momentum shift."""
    name = "EMACrossMomentum"
    mechanism = "behavioral (under-reaction)"
    hypothesis = "Traders under-react to new information, so a move that has started tends to continue for a while."
    fails_when = "Mean-reverting regimes; whipsaws when volatility is low relative to the averages."
    grid = {"fast": [5, 9, 13], "slow": [21, 34, 55]}

    def signals(self, df, cfg):
        f, s = ind.ema(df.close, self.params["fast"]), ind.ema(df.close, self.params["slow"])
        return np.where(_cross_up(f, s), 1, np.where(_cross_dn(f, s), -1, 0)).astype(np.int8)


class VWAPPullback(Entry):
    """In an uptrend (price above trend EMA), buy when price dips to VWAP and closes back above it. Mirror for shorts."""
    name = "VWAPPullback"
    mechanism = "liquidity provision / institutional flow"
    hypothesis = "Institutions execute around VWAP, so a pullback to it in a trend attracts buyers who want the average price."
    fails_when = "Trend day reverses; VWAP is less relevant on low-volume or holiday sessions."
    grid = {"touch_atr": [0.25, 0.5, 1.0]}

    def signals(self, df, cfg):
        near = (df.low <= df.vwap + self.params["touch_atr"] * df.atr)
        near_s = (df.high >= df.vwap - self.params["touch_atr"] * df.atr)
        up = df.close > df.ema_trend
        long = up & near & (df.close > df.vwap) & (df.close.shift(1) <= df.vwap.shift(1))
        short = (~up) & near_s & (df.close < df.vwap) & (df.close.shift(1) >= df.vwap.shift(1))
        return np.where(long, 1, np.where(short, -1, 0)).astype(np.int8)


class RSIReversion(Entry):
    """Buy oversold bounces, sell overbought fades (mean reversion)."""
    name = "RSIReversion"
    mechanism = "liquidity provision"
    hypothesis = "Short bursts of selling or buying overshoot; providing liquidity against the overshoot earns the snap-back."
    fails_when = "Trend days (overshoot keeps going); news-driven moves where the first push is right."
    grid = {"n": [7, 14], "band": [20, 25, 30]}

    def signals(self, df, cfg):
        r = ind.rsi(df.close, self.params["n"])
        b = self.params["band"]
        long = _cross_up(r, pd.Series(b, index=df.index))
        short = _cross_dn(r, pd.Series(100 - b, index=df.index))
        return np.where(long, 1, np.where(short, -1, 0)).astype(np.int8)


class BollingerReversion(Entry):
    """Price pokes outside the bands and comes back inside = snap-back trade."""
    name = "BollingerReversion"
    mechanism = "liquidity provision"
    hypothesis = "Moves beyond 2 standard deviations are usually noise and revert."
    fails_when = "Volatility expansion: the bands are measured on calm data and the move is a regime change."
    grid = {"n": [20, 40], "k": [2.0, 2.5]}

    def signals(self, df, cfg):
        m = ind.sma(df.close, self.params["n"])
        sd = df.close.rolling(self.params["n"]).std()
        up, lo = m + self.params["k"] * sd, m - self.params["k"] * sd
        long = (df.close > lo) & (df.close.shift(1) <= lo.shift(1))
        short = (df.close < up) & (df.close.shift(1) >= up.shift(1))
        return np.where(long, 1, np.where(short, -1, 0)).astype(np.int8)


class PrevDayBreak(Entry):
    """Break of yesterday's high (long) or low (short)."""
    name = "PrevDayBreak"
    mechanism = "behavioral (reference prices)"
    hypothesis = "Yesterday's high and low are watched by everyone; stops sit beyond them, so a break triggers a burst of order flow."
    fails_when = "Breaks that fail immediately (stop hunts); low-volume drift through the level."
    grid = {"buffer_atr": [0.0, 0.25]}

    def signals(self, df, cfg):
        b = self.params["buffer_atr"] * df.atr
        long = _cross_up(df.close, df.pdh + b)
        short = _cross_dn(df.close, df.pdl - b)
        return np.where(long, 1, np.where(short, -1, 0)).astype(np.int8)


class DonchianBreakout(Entry):
    """Close beyond the highest high / lowest low of the last N bars."""
    name = "DonchianBreakout"
    mechanism = "risk premium / trend"
    hypothesis = "Classic trend-following: new N-bar extremes mark the start of moves that pay a few large wins for many small losses."
    fails_when = "Sideways markets; too-short lookbacks make it a noise trader."
    grid = {"n": [20, 40, 80]}

    def signals(self, df, cfg):
        n = self.params["n"]
        hh = df.high.rolling(n).max().shift(1)
        ll = df.low.rolling(n).min().shift(1)
        return np.where(df.close > hh, 1, np.where(df.close < ll, -1, 0)).astype(np.int8)


class OvernightGapFade(Entry):
    """Open gaps up or down vs yesterday's close by more than X ATR -> trade back toward the close."""
    name = "OvernightGapFade"
    mechanism = "liquidity provision / behavioral"
    hypothesis = "Overnight moves on thin volume overshoot; day-session volume pulls price back toward the prior close."
    fails_when = "Gap-and-go days driven by real news; gaps in the direction of a strong trend."
    grid = {"gap_atr": [0.75, 1.0, 1.5]}

    def signals(self, df, cfg):
        first = df["bar_of_day"].values == 0
        gap = (df.open - df.pdc) / df.atr.replace(0, np.nan)
        k = self.params["gap_atr"]
        long = first & (gap <= -k)
        short = first & (gap >= k)
        return np.where(long, 1, np.where(short, -1, 0)).astype(np.int8)


class ConfirmedBreakout(Entry):
    """Donchian breakout that ALSO needs a volume surge (institutional participation)."""
    name = "ConfirmedBreakout"
    mechanism = "information / institutional flow"
    hypothesis = "A breakout on 2x normal volume is more likely to be real participation than a stop run."
    fails_when = "Volume data is poor (Yahoo futures volume is patchy) or everyone fades high-volume breaks."
    grid = {"n": [20, 40], "vol_mult": [1.5, 2.0]}

    def signals(self, df, cfg):
        n = self.params["n"]
        hh = df.high.rolling(n).max().shift(1); ll = df.low.rolling(n).min().shift(1)
        vmed = df.volume.rolling(100, min_periods=20).median()
        surge = df.volume >= self.params["vol_mult"] * vmed
        return np.where((df.close > hh) & surge, 1, np.where((df.close < ll) & surge, -1, 0)).astype(np.int8)


ENTRIES = [OpeningRangeBreakout, EMACrossMomentum, VWAPPullback, RSIReversion,
           BollingerReversion, PrevDayBreak, DonchianBreakout, OvernightGapFade, ConfirmedBreakout]


# ----------------------------------------------------------------------------- FILTERS
class Filter(Piece):
    kind = "filter"

    def apply(self, sig: np.ndarray, df, cfg) -> np.ndarray:
        raise NotImplementedError


class SessionWindow(Filter):
    """Only take entries between two clock times (ET). Morning is usually cleaner than lunch."""
    name = "SessionWindow"
    grid = {"start": ["09:30", "09:45", "10:00"], "end": ["11:30", "13:00", "15:30"]}

    def apply(self, sig, df, cfg):
        s = int(self.params["start"][:2]) * 60 + int(self.params["start"][3:])
        e = int(self.params["end"][:2]) * 60 + int(self.params["end"][3:])
        ok = (df["mins"].values >= s) & (df["mins"].values < e)
        return np.where(ok, sig, 0).astype(np.int8)


class TrendRegime(Filter):
    """'trend' = only trade when ADX says the market is moving. 'chop' = only when it isn't."""
    name = "TrendRegime"
    grid = {"mode": ["trend", "chop"], "adx": [20, 25]}

    def apply(self, sig, df, cfg):
        a = df["adx"].values
        ok = a >= self.params["adx"] if self.params["mode"] == "trend" else a < self.params["adx"]
        return np.where(ok, sig, 0).astype(np.int8)


class VolatilityBand(Filter):
    """Skip dead markets (too quiet) and crazy ones (too wild). ATR compared to its usual level."""
    name = "VolatilityBand"
    grid = {"lo": [0.6, 0.8], "hi": [1.5, 2.0, 99.0]}

    def apply(self, sig, df, cfg):
        ratio = (df["atr"] / df["atr_median"]).values
        ok = (ratio >= self.params["lo"]) & (ratio <= self.params["hi"])
        return np.where(ok, sig, 0).astype(np.int8)


class TrendAlign(Filter):
    """Only longs above the 200-EMA, only shorts below it (trade with the bigger trend)."""
    name = "TrendAlign"
    grid = {"on": [True]}

    def apply(self, sig, df, cfg):
        up = (df.close > df.ema_trend).values
        return np.where((sig == 1) & up, 1, np.where((sig == -1) & ~up, -1, 0)).astype(np.int8)


class LongOnly(Filter):
    name = "LongOnly"
    grid = {"on": [True]}

    def apply(self, sig, df, cfg):
        return np.where(sig == 1, 1, 0).astype(np.int8)


class EventDay(Filter):
    """Skip the first minutes after scheduled economic releases (8:30 and 10:00 ET prints, FOMC at 14:00).
    The pack's 'event time vs publication time' point: a release moves price before any chart signal."""
    name = "EventDay"
    grid = {"skip_minutes": [15, 30]}
    mechanism = "information (avoid being the slow side)"
    hypothesis = "Right after a release the first move is driven by people who parsed the number faster than a 5-minute chart can."
    fails_when = "Low-impact days where skipping just costs trades; use a real economic calendar to make this day-specific."

    def apply(self, sig, df, cfg):
        k = self.params["skip_minutes"]
        m = df["mins"].values
        hot = ((m >= 600) & (m < 600 + k)) | ((m >= 840) & (m < 840 + k))   # 10:00 and 14:00 ET
        # ponytail: fixed clock windows; swap in a dated calendar (FOMC/CPI/NFP) when you have one
        return np.where(hot, 0, sig).astype(np.int8)


FILTERS = [SessionWindow, TrendRegime, VolatilityBand, TrendAlign, LongOnly, EventDay]


# ----------------------------------------------------------------------------- EXITS
class Exit(Piece):
    """Exits are just parameter bundles; engine.run() executes them.
    stop_atr / target_atr in ATR units; trail_atr=0 disables trailing; be_atr=0 disables breakeven;
    max_bars=0 disables the time-out."""
    kind = "exit"

    def rules(self):
        return {"stop_atr": 1.5, "target_atr": 3.0, "trail_atr": 0.0, "be_atr": 0.0, "max_bars": 0, **self.params}


class FixedBracket(Exit):
    """Stop and target at fixed ATR distances."""
    name = "FixedBracket"
    grid = {"stop_atr": [1.0, 1.5, 2.0], "target_atr": [1.5, 2.0, 3.0]}


class TrailingATR(Exit):
    """No target; a stop that follows price at a set ATR distance. Lets winners run."""
    name = "TrailingATR"
    grid = {"stop_atr": [1.5, 2.0], "trail_atr": [1.5, 2.0, 3.0], "target_atr": [99.0]}


class BreakevenBracket(Exit):
    """Bracket, but once the trade is up be_atr the stop jumps to entry (free trade)."""
    name = "BreakevenBracket"
    grid = {"stop_atr": [1.0, 1.5], "target_atr": [2.0, 3.0], "be_atr": [0.75, 1.0]}


class TimeBoxed(Exit):
    """Bracket + give up after N bars if nothing happened (frees capital from dead trades)."""
    name = "TimeBoxed"
    grid = {"stop_atr": [1.0, 1.5], "target_atr": [2.0, 3.0], "max_bars": [12, 24, 48]}


EXITS = [FixedBracket, TrailingATR, BreakevenBracket, TimeBoxed]


# ----------------------------------------------------------------------------- SIZE
class Size(Piece):
    kind = "size"

    def contracts(self, stop_points, point_value, cfg) -> int:
        raise NotImplementedError


class FixedContracts(Size):
    """Always the same number of contracts."""
    name = "FixedContracts"
    grid = {"n": [1, 2]}

    def contracts(self, stop_points, point_value, cfg):
        return int(min(self.params["n"], cfg.max_contracts))


class RiskBased(Size):
    """Contracts = risk-per-trade dollars / (stop distance in $). Bigger stop -> fewer contracts."""
    name = "RiskBased"
    grid = {"risk": [100, 150, 250]}

    def contracts(self, stop_points, point_value, cfg):
        risk_per_ct = max(stop_points * point_value, 1e-9)
        n = int(self.params["risk"] // risk_per_ct)
        return int(max(1, min(n, cfg.max_contracts)))


SIZES = [FixedContracts, RiskBased]

ALL = {p.name: p for p in ENTRIES + FILTERS + EXITS + SIZES}


def from_spec(spec: dict) -> Piece:
    return ALL[spec["name"]](**spec.get("params", {}))
