"""Extra checks taken from the learning pack's reading list.

PBO   Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado, Zhu 2015), via CSCV:
      split the data into S slices, form every half/half train-test combination, and count how often
      the strategy that looks best on the train half ranks in the BOTTOM half on the test half.
      0 = selection is trustworthy, 0.5 = coin flip, near 1 = the search is picking noise.
DSR   Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): the Sharpe you'd expect the best of N
      random tries to show, then the probability the real Sharpe beats it. Uses skew/kurtosis.
Stress  Rerun with worse fills: 2x slippage, 1.5x fees, a 1-bar delay. A real edge survives friction.
Monte Carlo  Reshuffle the unseen trades thousands of times under the prop-firm rules: how often do you
      hit the profit target before the max-loss limit?
Data quality  Gaps, duplicate bars, zero-range bars, outliers, timezone sanity (HRT: the data is the
      product; a predictive model on bad data is a bad model).
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from . import engine


# ----------------------------------------------------------------------------- PBO / CSCV
def pbo(perf_matrix, n_slices=8):
    """perf_matrix: array [n_slices, n_strategies] of per-slice performance (e.g. net $ or Sharpe).
    Returns probability of backtest overfitting plus the logit list."""
    S, N = perf_matrix.shape
    if S < 4 or N < 2:
        return float("nan"), []
    half = S // 2
    logits = []
    for train_idx in itertools.combinations(range(S), half):
        test_idx = [i for i in range(S) if i not in train_idx]
        tr = perf_matrix[list(train_idx)].sum(axis=0)
        te = perf_matrix[test_idx].sum(axis=0)
        best = int(np.argmax(tr))
        rank = (te < te[best]).sum() + 0.5 * (te == te[best]).sum()   # how many it beats on test
        w = (rank + 0.5) / (N + 1)                                        # relative rank in (0,1)
        logits.append(float(np.log(w / (1 - w))))
    logits = np.array(logits)
    return float((logits <= 0).mean()), logits.tolist()


def slice_perf(df, sig, strat, spec, cfg, n_slices):
    """Net $ in each chronological slice for one strategy (one column of the PBO matrix)."""
    days = df["day_id"].values
    edges = np.linspace(0, days.max() + 1, n_slices + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        s, e = int(np.searchsorted(days, a)), int(np.searchsorted(days, b))
        out.append(engine.run(df, sig, strat.exit.rules(), strat.size, spec, cfg, s, e).metrics["net"])
    return out


# ----------------------------------------------------------------------------- Deflated Sharpe
def deflated_sharpe(daily_pnl, n_trials, var_trials_sharpe=None):
    """Probability that the observed (daily) Sharpe is real after trying n_trials ideas.
    Returns dict with the per-day sharpe, the expected max sharpe of N random tries, and the PSR."""
    d = np.asarray(daily_pnl, dtype=float)
    T = len(d)
    if T < 10 or d.std() == 0:
        return {"sr": 0.0, "sr0": 0.0, "dsr": 0.0, "T": T}
    sr = d.mean() / d.std(ddof=1)
    g3, g4 = float(skew(d)), float(kurtosis(d, fisher=False))
    v = var_trials_sharpe if var_trials_sharpe is not None else 0.25 * sr ** 2 + 1e-9  # ponytail: unknown trial variance, 0.5 sr as proxy
    emc = 0.5772156649
    n = max(n_trials, 2)
    sr0 = np.sqrt(v) * ((1 - emc) * norm.ppf(1 - 1 / n) + emc * norm.ppf(1 - 1 / (n * np.e)))
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-9))
    z = (sr - sr0) * np.sqrt(T - 1) / denom
    return {"sr": float(sr), "sr0": float(sr0), "dsr": float(norm.cdf(z)), "T": T,
            "sr_annual": float(sr * np.sqrt(252))}


# ----------------------------------------------------------------------------- Stress
def stress(df, folds, spec, cfg, cache=None):
    """Rerun each unseen slice (with the settings that were chosen for it) under worse execution.
    folds: list of (strategy, Result-with-.fold). Returns {name: net$}; 'as tested' matches the report."""
    import copy
    c2 = copy.copy(cfg); c2.slippage_ticks = cfg.slippage_ticks * cfg.stress_slippage_mult
    s2 = dict(spec); s2["fee_per_side"] = spec["fee_per_side"] * cfg.stress_fee_mult
    cases = {"as tested": (cfg, spec, 0),
             f"slippage x{cfg.stress_slippage_mult:g}": (c2, spec, 0),
             f"fees x{cfg.stress_fee_mult:g}": (cfg, s2, 0),
             "1-bar delay": (cfg, spec, 1),
             "all three": (c2, s2, 1)}
    out = {}
    for name, (c, s, delay) in cases.items():
        total = 0.0
        for strat, res in folds:
            a, b = res.fold
            sig = strat.signals(df, c, cache)
            total += engine.run(df, sig, strat.exit.rules(), strat.size, s, c, a, b, delay=delay).metrics["net"]
        out[name] = float(total)
    return out


# ----------------------------------------------------------------------------- Monte Carlo
def monte_carlo(trades, cfg, runs=2000, seed=11):
    """Reshuffle trade outcomes (grouped by day so one bad day stays one bad day) and play them
    forward under the prop-firm rules. Returns odds of hitting the profit target first vs the max loss."""
    if trades is None or len(trades) == 0:
        return {}
    rng = np.random.default_rng(seed)
    t = trades.copy()
    t["date"] = pd.to_datetime(t["entry_time"]).dt.date
    days = [g["pnl"].values for _, g in t.groupby("date")]
    if len(days) < 5:
        return {}
    target, maxloss, dll = cfg.profit_target, cfg.max_loss_limit, cfg.daily_loss_limit
    horizon = max(len(days), 60)
    hit_target = hit_loss = 0; days_to_target = []; max_dds = []
    for _ in range(runs):
        bal = 0.0; peak = 0.0; dd = 0.0; outcome = None
        order = rng.integers(0, len(days), size=horizon)
        for k, di in enumerate(order):
            day_pnl = 0.0
            for p in days[di]:
                day_pnl += p
                if day_pnl <= -dll:
                    break
            bal += day_pnl
            if cfg.trailing_mode == "intraday":
                peak = max(peak, bal)
            dd = max(dd, peak - bal)
            if bal <= peak - maxloss:
                outcome = "loss"; break
            if cfg.trailing_mode == "eod":
                peak = max(peak, bal)
            if bal >= target:
                outcome = "target"; days_to_target.append(k + 1); break
        max_dds.append(dd)
        if outcome == "target": hit_target += 1
        elif outcome == "loss": hit_loss += 1
    return {"p_target_first": hit_target / runs, "p_breach_first": hit_loss / runs,
            "p_neither": 1 - (hit_target + hit_loss) / runs, "horizon_days": horizon,
            "median_days_to_target": float(np.median(days_to_target)) if days_to_target else float("nan"),
            "dd_95pct": float(np.percentile(max_dds, 95))}


# ----------------------------------------------------------------------------- Data quality
def data_quality(df, interval_minutes):
    """Plain-English list of data problems. Empty list = clean."""
    issues = []
    n = len(df)
    if n == 0:
        return ["no data"]
    dup = int(df.index.duplicated().sum())
    if dup: issues.append(f"{dup} duplicate timestamps")
    zero = int((df.high == df.low).sum())
    if zero / n > 0.02: issues.append(f"{zero} bars with zero range ({zero/n:.1%}) - stale or bad feed?")
    rng = (df.high - df.low)
    med = rng.rolling(200, min_periods=20).median()
    wild = int((rng > 15 * med).sum())
    if wild: issues.append(f"{wild} bars with a range 15x the usual - check for bad prints")
    gap = df.index.to_series().diff().dt.total_seconds().div(60)
    same_day = df.index.date[1:] == df.index.date[:-1]
    intraday_gaps = int(((gap.values[1:] > interval_minutes * 3) & same_day).sum())
    if intraday_gaps: issues.append(f"{intraday_gaps} missing stretches inside a session")
    if df.index.tz is None or "America/New_York" not in str(df.index.tz): issues.append("timestamps are not in US Eastern time")
    first = df.groupby(df.index.date).apply(lambda g: g.index.min().hour * 60 + g.index.min().minute)
    if (first > 9 * 60 + 35).mean() > 0.1: issues.append("more than 10% of days start after 9:35 ET - partial sessions")
    return issues


if __name__ == "__main__":   # self-check
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, size=(8, 50))
    p, _ = pbo(noise)
    assert 0.3 < p < 0.7, p                            # pure noise -> ~0.5
    skill = noise.copy(); skill[:, 0] += 3.0
    p2, _ = pbo(skill)
    assert p2 < 0.1, p2                                # one real winner -> low PBO
    d = deflated_sharpe(rng.normal(0.1, 1, 500), n_trials=200)
    assert 0 <= d["dsr"] <= 1
    print("validate self-check ok", p, p2, round(d["dsr"], 2))
