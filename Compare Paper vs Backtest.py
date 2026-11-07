"""Reconcile paper (or live) trades against the backtest the strategy was approved on.
The learning pack's last validation step: compare simulated decisions against paper logs.
Usage: python "Compare Paper vs Backtest.py" [--symbol MES] [--live]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from mosaic.config import LOG_DIR, STRAT_DIR, load_settings

args = sys.argv[1:]
cfg = load_settings()
sym = args[args.index("--symbol") + 1] if "--symbol" in args else cfg.live_symbol
mode = "live" if "--live" in args else "paper"
tp = os.path.join(LOG_DIR, f"{mode}_trades_{sym}.csv")
sp = os.path.join(STRAT_DIR, f"best_{sym}.json")
if not (os.path.exists(tp) and os.path.exists(sp)):
    sys.exit(f"Need both {tp} and {sp}")
t = pd.read_csv(tp)
info = json.load(open(sp))
m = info["oos_metrics"]
pnl = t["pnl"].astype(float).values
wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
n = len(pnl)
se = pnl.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
print(f"{mode.upper()} {sym}: {n} trades\n")
print(f"{'metric':22s} {'backtest':>12s} {mode:>12s}")
for label, b, l in [("profit factor", m["profit_factor"], pf), ("$ per trade", m["expectancy"], pnl.mean()),
                    ("win rate", m["win_rate"], (pnl > 0).mean()), ("avg win $", m["avg_win"], wins.mean() if len(wins) else 0),
                    ("avg loss $", m["avg_loss"], losses.mean() if len(losses) else 0)]:
    print(f"{label:22s} {b:12.2f} {l:12.2f}")
print(f"\n$ per trade is {pnl.mean():.0f} +/- {2*se:.0f} (95% range) vs backtest {m['expectancy']:.0f}.")
if n < 30:
    print("Under 30 trades: too early to judge. Keep paper trading.")
elif pnl.mean() + 2 * se < m["expectancy"]:
    print("VERDICT: live results are significantly worse than the backtest. Something differs (fills, data, timing). Do not go live.")
elif pf < 1.0:
    print("VERDICT: losing in paper. Re-run research; the edge may have decayed.")
else:
    print("VERDICT: paper results are consistent with the backtest so far.")
by_reason = t.groupby("reason")["pnl"].agg(["count", "sum", "mean"])
print("\nBy exit reason:\n", by_reason.round(1).to_string())
