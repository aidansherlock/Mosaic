"""Paper / live trading loop with hard safety rails.

Every cycle:
  1. pull the latest bars, drop the unfinished one
  2. if in a trade: check stop/target, trail, breakeven, time-out, flatten time
  3. if flat and allowed: look at the LAST CLOSED bar's signal and enter
Rails: daily loss limit, max trades/day, no entries after last_entry, flat by flatten_at,
       kill switch (a file named STOP.txt in the Mosaic folder), drift monitor (pauses when live
       results fall well below the backtest), and real orders only with --live AND
       live_trading_enabled=true in settings.json.
"""
import json, os, time
from datetime import datetime
import pandas as pd
from .config import BASE_DIR, CONTRACTS, LOG_DIR, STRAT_DIR, YF_PERIODS, load_settings
from . import data, indicators
from .strategy import Strategy
from .broker import PaperBroker, TopstepXBroker

KILL = os.path.join(BASE_DIR, "STOP.txt")
TZ = "America/New_York"


class Logger:
    def __init__(self, name):
        self.path = os.path.join(LOG_DIR, f"{name}_{datetime.now():%Y%m%d}.log")

    def __call__(self, msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class DriftMonitor:
    """Compares live results to the backtest. If the last N trades look much worse, pause."""

    def __init__(self, expected_per_trade, window=25, log=print):
        self.exp, self.window, self.log = expected_per_trade, window, log
        self.pnls = []

    def add(self, pnl):
        self.pnls.append(pnl)

    def paused(self):
        if len(self.pnls) < self.window:
            return False
        recent = self.pnls[-self.window:]
        avg = sum(recent) / len(recent)
        gp = sum(p for p in recent if p > 0); gl = -sum(p for p in recent if p <= 0)
        pf = gp / gl if gl > 0 else 9.9
        if pf < 0.8 and avg < 0.25 * self.exp:
            self.log(f"DRIFT: last {self.window} trades PF {pf:.2f}, avg ${avg:.0f} vs backtest ${self.exp:.0f}. Pausing. Re-run research.")
            return True
        return False


def latest_bars(cfg, symbol, interval, broker=None):
    """Recent bars, finished ones only."""
    if isinstance(broker, TopstepXBroker):
        unit_number = int(interval.rstrip("m")) if interval.endswith("m") else 60
        rows = broker.bars(unit=2, unit_number=unit_number)
        if rows:
            df = pd.DataFrame(rows).rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(TZ)
            return df.set_index("time")[data.COLS].astype(float).sort_index()
    df = data.fetch_yf(symbol, interval, period="10d" if interval != "1h" else "60d")
    if len(df) == 0:
        return df
    minutes = int(interval.rstrip("m")) if interval.endswith("m") else 60
    now = pd.Timestamp.now(tz=TZ)
    if df.index[-1] + pd.Timedelta(minutes=minutes) > now:   # unfinished bar
        df = df.iloc[:-1]
    return df


def run_trader(symbol=None, live=False, once=False):
    cfg = load_settings()
    symbol = symbol or cfg.live_symbol
    spec = CONTRACTS[symbol]
    log = Logger("live" if live else "paper")
    path = os.path.join(STRAT_DIR, f"best_{symbol}.json")
    if not os.path.exists(path):
        log(f"No saved strategy at {path}. Run 'Run Research.py' first."); return
    strat, info = Strategy.load(path)
    if live and not cfg.live_trading_enabled:
        log("Refusing: --live given but live_trading_enabled is false in settings.json."); return
    if live and info.get("verdict") != "PASS":
        log(f"Refusing live trading: saved strategy verdict is {info.get('verdict')} (needs PASS). Paper trade it instead."); return
    log(f"Strategy: {strat.label()}")
    log(f"Backtest (unseen data): PF {info['oos_metrics']['profit_factor']:.2f}, ${info['oos_metrics']['expectancy']:.0f}/trade, verdict {info.get('verdict')}")
    if live:
        log("!!! LIVE MODE: real orders will be sent. Type YES to continue.")
        if input("> ").strip() != "YES":
            log("Cancelled."); return
    broker = (TopstepXBroker if live else PaperBroker)(spec, cfg, log)
    broker.connect()
    drift = DriftMonitor(info["oos_metrics"]["expectancy"], log=log)
    rules = strat.exit.rules()
    interval = info.get("interval", cfg.research_interval)

    last_bar_time = None
    day = None; day_pnl = 0.0; trades_today = 0; locked = False; entry_atr = None
    log(f"Watching {symbol} {interval} bars every {cfg.poll_seconds}s. Create STOP.txt to halt.")
    while True:
        try:
            if os.path.exists(KILL):
                log("STOP.txt found - flattening and exiting.")
                if broker.pos:
                    px = latest_bars(cfg, symbol, interval, broker).close.iloc[-1]
                    pnl = broker.flatten(px, datetime.now(), "kill")
                break
            raw = latest_bars(cfg, symbol, interval, broker)
            if len(raw) < 250:
                log(f"only {len(raw)} bars available, waiting..."); time.sleep(cfg.poll_seconds); continue
            bar_time = raw.index[-1]
            if bar_time == last_bar_time:
                time.sleep(cfg.poll_seconds); continue
            last_bar_time = bar_time
            df = indicators.prepare(raw, cfg.rth_start_min)
            bar = df.iloc[-1]
            now_min = int(bar["mins"]) + (int(interval.rstrip("m")) if interval.endswith("m") else 60)  # bar close time
            if bar.name.date() != day:
                day = bar.name.date(); day_pnl = 0.0; trades_today = 0; locked = False
                log(f"--- new session {day} ---")

            # manage open trade
            if broker.pos:
                p = broker.pos
                pnl = broker.on_bar(bar, bar.name)
                if pnl is None:
                    a = entry_atr or bar["atr"]
                    new_stop = p.stop
                    if rules["trail_atr"]:
                        new_stop = max(new_stop, p.extreme - rules["trail_atr"] * a) if p.side == 1 else min(new_stop, p.extreme + rules["trail_atr"] * a)
                    if rules["be_atr"] and ((p.side == 1 and p.extreme >= p.entry + rules["be_atr"] * a) or (p.side == -1 and p.extreme <= p.entry - rules["be_atr"] * a)):
                        new_stop = max(new_stop, p.entry + spec["tick"]) if p.side == 1 else min(new_stop, p.entry - spec["tick"])
                    if new_stop != p.stop:
                        broker.update_stop(new_stop); log(f"stop moved to {new_stop:.2f}")
                    if rules["max_bars"] and p.bars >= rules["max_bars"]:
                        pnl = broker.flatten(bar["close"], bar.name, "time")
                    elif now_min >= cfg.flatten_min:
                        pnl = broker.flatten(bar["close"], bar.name, "flatten")
                if pnl is not None:
                    day_pnl += pnl; drift.add(pnl)
                    log(f"day pnl ${day_pnl:,.0f} ({trades_today} trades)")
                    if day_pnl <= -cfg.daily_loss_limit:
                        locked = True; log("DAILY LOSS LIMIT hit - no more trades today.")

            # new entry
            if not broker.pos and not locked and not drift.paused():
                sig = strat.signals(df, cfg)
                s = int(sig[-1])
                can = (s != 0 and now_min >= cfg.rth_start_min and now_min < cfg.last_entry_min
                       and trades_today < cfg.max_trades_per_day and pd.notna(bar["atr"]) and bar["atr"] > 0)
                if can:
                    a = float(bar["atr"]); entry_atr = a
                    stop_pts = rules["stop_atr"] * a
                    qty = strat.size.contracts(stop_pts, spec["point_value"], cfg)
                    px = float(bar["close"])
                    stop = px - stop_pts * s; target = px + rules["target_atr"] * a * s
                    broker.place_bracket(s, qty, px, stop, target, bar.name)
                    trades_today += 1
            _write_state(symbol, live, broker, day_pnl, trades_today, locked, bar_time)
        except KeyboardInterrupt:
            log("Ctrl+C - flattening and exiting.")
            if broker.pos:
                try:
                    px = latest_bars(cfg, symbol, interval, broker).close.iloc[-1]
                    broker.flatten(px, datetime.now(), "manual")
                except Exception as e:
                    log(f"!! could not flatten automatically: {e}. CLOSE THE POSITION BY HAND.")
            break
        except Exception as e:
            log(f"error: {e!r} (retrying in {cfg.poll_seconds}s)")
        if once:
            break
        time.sleep(cfg.poll_seconds)
    _write_trades(symbol, live, broker)


def _write_state(symbol, live, broker, day_pnl, n, locked, bar_time):
    st = {"symbol": symbol, "mode": "live" if live else "paper", "updated": datetime.now().isoformat(timespec="seconds"),
          "last_bar": str(bar_time), "day_pnl": day_pnl, "trades_today": n, "locked": locked,
          "position": None if not broker.pos else {"side": broker.pos.side, "qty": broker.pos.qty, "entry": broker.pos.entry,
                                                    "stop": broker.pos.stop, "target": broker.pos.target}}
    with open(os.path.join(LOG_DIR, "state.json"), "w") as f:
        json.dump(st, f, indent=2, default=str)


def _write_trades(symbol, live, broker):
    if broker.closed:
        p = os.path.join(LOG_DIR, f"{'live' if live else 'paper'}_trades_{symbol}.csv")
        pd.DataFrame(broker.closed).to_csv(p, mode="a", header=not os.path.exists(p), index=False)
