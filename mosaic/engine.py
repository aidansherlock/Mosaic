"""Bar-by-bar backtest engine with real-world friction and prop-firm rules.

Conservative on purpose:
- entries fill at the NEXT bar's open, plus slippage
- stops fill at the stop price minus slippage (or the open, if the bar gapped through)
- if a bar touches both stop and target, we assume the STOP hit first
- targets are limit orders (no slippage) but only count if price traded through them
- daily loss limit -> flatten and stop trading for the day
- trailing max loss -> account breached, run stops (and is scored as a failure)
"""
import numpy as np
import pandas as pd


class Result:
    def __init__(self, trades, daily, breached, cfg):
        self.trades = trades            # DataFrame, one row per closed trade
        self.daily = daily              # Series of $ pnl per trading date
        self.breached = breached        # True if the trailing max-loss was hit
        self.metrics = compute_metrics(trades, daily, breached, cfg)


def compute_metrics(trades, daily, breached, cfg):
    m = {"trades": int(len(trades)), "breached": bool(breached)}
    if len(trades) == 0:
        m.update(net=0.0, win_rate=0.0, profit_factor=0.0, expectancy=0.0, max_dd=0.0,
                 sharpe=0.0, avg_win=0.0, avg_loss=0.0, worst_day=0.0, days=int(len(daily)),
                 per_day=0.0, score=-1e9)
        return m
    pnl = trades["pnl"].astype(float).values
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gp, gl = wins.sum(), -losses.sum()
    eq = np.cumsum(pnl)
    dd = eq - np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    d = daily.values if len(daily) else np.array([0.0])
    sharpe = float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0
    m.update(
        net=float(pnl.sum()),
        win_rate=float(len(wins) / len(pnl)),
        profit_factor=float(gp / gl) if gl > 0 else (9.99 if gp > 0 else 0.0),
        expectancy=float(pnl.mean()),
        max_dd=float(-dd.min()) if len(dd) else 0.0,
        sharpe=sharpe,
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        worst_day=float(d.min()),
        days=int(len(daily)),
        per_day=float(pnl.sum() / max(len(daily), 1)),
    )
    # R-multiples: results in units of the risk taken per trade (the learning pack's "R")
    if "risk" in trades and trades["risk"].gt(0).any():
        risk = trades["risk"].astype(float).values
        r = np.where(risk > 0, pnl / np.where(risk > 0, risk, 1.0), np.nan)
        r = r[np.isfinite(r)]
        m["avg_win_R"] = float(r[r > 0].mean()) if (r > 0).any() else 0.0
        m["avg_loss_R"] = float(r[r <= 0].mean()) if (r <= 0).any() else 0.0
        m["expectancy_R"] = float(r.mean()) if len(r) else 0.0
    # longest stretch (in trades) spent below a previous equity high
    hi, run_, worst = -np.inf, 0, 0
    for v in eq:
        if v >= hi: hi, run_ = v, 0
        else: run_ += 1; worst = max(worst, run_)
    m["longest_underwater_trades"] = int(worst)
    streak = best_streak = 0
    for v in pnl:
        streak = streak + 1 if v <= 0 else 0
        best_streak = max(best_streak, streak)
    m["max_consecutive_losses"] = int(best_streak)
    k = max(1, int(len(pnl) * 0.05))
    m["tail_5pct_avg"] = float(np.sort(pnl)[:k].mean())
    # lag-1 autocorrelation of daily pnl (Lo 2002: serial correlation breaks simple Sharpe scaling)
    m["daily_autocorr"] = float(np.corrcoef(d[:-1], d[1:])[0, 1]) if len(d) > 5 and d[:-1].std() > 0 and d[1:].std() > 0 else 0.0
    m["score"] = score(m, cfg)
    return m


def score(m, cfg):
    """One number to rank strategies. Rewards steady profit, punishes drawdown and rule breaches."""
    if m["breached"]:
        return -1e6 + m["net"]
    if m["trades"] < cfg.min_trades:
        return -1e5 + m["net"]
    pf = min(m["profit_factor"], 3.0)
    dd_pen = m["max_dd"] / max(cfg.max_loss_limit, 1.0)   # 1.0 = would have blown the account
    return (pf - 1.0) * 100 + m["sharpe"] * 20 - dd_pen * 100 + m["expectancy"] * 0.5


def run(df, sig, exit_rules, sizer, spec, cfg, start=0, end=None, delay=0):
    """df: prepared bars (see indicators.prepare). sig: int8 signals aligned to df.
    spec: contract spec dict (point_value, tick, fee_per_side). delay: extra bars between the
    signal and the fill (stress test). Returns Result."""
    end = len(df) if end is None else end
    if delay:
        sig = np.concatenate([np.zeros(delay, dtype=np.int8), sig[:-delay]])
    o = df.open.values; h = df.high.values; l = df.low.values; c = df.close.values
    atr = df.atr.values; mins = df.mins.values; day = df.day_id.values
    idx = df.index
    pv, tick, fee = spec["point_value"], spec["tick"], spec["fee_per_side"]
    slip = cfg.slippage_ticks * tick
    r = exit_rules
    stop_k, tgt_k, trail_k, be_k, max_bars = r["stop_atr"], r["target_atr"], r["trail_atr"], r["be_atr"], int(r["max_bars"])
    last_entry, flatten = cfg.last_entry_min, cfg.flatten_min
    dll, mll, mode = cfg.daily_loss_limit, cfg.max_loss_limit, cfg.trailing_mode

    trades = []
    daily = {}
    pos = 0; qty = 0; entry = 0.0; stop = 0.0; target = 0.0; bars_held = 0; entry_i = 0; extreme = 0.0; init_stop = 0.0
    day_pnl = 0.0; cur_day = day[start] if start < end else -1; locked = False
    balance = 0.0; peak = 0.0; breached = False

    def close_trade(i, px, reason):
        nonlocal pos, qty, day_pnl, balance, peak, breached
        pnl = (px - entry) * pos * pv * qty - fee * qty * 2
        trades.append(dict(entry_time=idx[entry_i], exit_time=idx[i], side="long" if pos == 1 else "short",
                           qty=qty, entry=entry, exit=px, pnl=pnl, bars=i - entry_i, reason=reason,
                           risk=abs(entry - init_stop) * pv * qty))
        day_pnl += pnl
        balance += pnl
        if mode == "intraday":
            peak = max(peak, balance)
        pos = 0; qty = 0

    for i in range(start, end):
        if day[i] != cur_day:
            if cur_day != -1:
                daily[idx[i - 1].date()] = day_pnl
                if mode == "eod":
                    peak = max(peak, balance)
            cur_day = day[i]; day_pnl = 0.0; locked = False
        if breached:
            break

        if pos != 0:
            bars_held += 1
            exited = False
            # gap through stop at the open
            if (pos == 1 and o[i] <= stop) or (pos == -1 and o[i] >= stop):
                close_trade(i, o[i] - slip * pos, "stop_gap"); exited = True
            elif (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                close_trade(i, stop - slip * pos, "stop"); exited = True
            elif (pos == 1 and h[i] >= target) or (pos == -1 and l[i] <= target):
                close_trade(i, target, "target"); exited = True
            elif max_bars and bars_held >= max_bars:
                close_trade(i, c[i] - slip * pos, "time"); exited = True
            elif mins[i] >= flatten:
                close_trade(i, c[i] - slip * pos, "flatten"); exited = True
            if not exited:
                # trailing / breakeven updates use this bar's extreme, apply from next bar
                a = atr[entry_i]
                if pos == 1:
                    extreme = max(extreme, h[i])
                    if trail_k: stop = max(stop, extreme - trail_k * a)
                    if be_k and extreme >= entry + be_k * a: stop = max(stop, entry + tick)
                else:
                    extreme = min(extreme, l[i])
                    if trail_k: stop = min(stop, extreme + trail_k * a)
                    if be_k and extreme <= entry - be_k * a: stop = min(stop, entry - tick)
            # daily loss check (mid-bar realism: we lock after the trade that crossed it)
            if day_pnl <= -dll:
                locked = True
            if balance <= peak - mll:
                breached = True
                continue

        if pos == 0 and not locked and i > start:
            s = sig[i - 1]
            if s != 0 and day[i - 1] == day[i] and mins[i] < last_entry and mins[i] >= cfg.rth_start_min:
                a = atr[i - 1]
                if not np.isfinite(a) or a <= 0:
                    continue
                stop_pts = stop_k * a
                n = sizer.contracts(stop_pts, pv, cfg)
                if n <= 0:
                    continue
                pos = int(s); qty = n; entry_i = i
                entry = o[i] + slip * pos
                stop = entry - stop_pts * pos
                init_stop = stop
                target = entry + tgt_k * a * pos
                extreme = entry; bars_held = 0
                # a trade opened at the flatten time is pointless
                if mins[i] >= flatten:
                    pos = 0; qty = 0

    if pos != 0:
        close_trade(end - 1, c[end - 1] - slip * pos, "end")
    if cur_day != -1 and end > start:
        daily[idx[end - 1].date()] = day_pnl
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "side", "qty", "entry", "exit", "pnl", "bars", "reason", "risk"])
    daily_s = pd.Series(daily, dtype=float)
    return Result(trades_df, daily_s, breached, cfg)
