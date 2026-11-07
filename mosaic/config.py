"""All settings in one place. Edit settings.json (created on first run) rather than this file."""
import json, os
from dataclasses import dataclass, field, asdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STRAT_DIR = os.path.join(BASE_DIR, "strategies")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")
for d in (DATA_DIR, RESULTS_DIR, STRAT_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

# Contract specs. fee_per_side = commission + exchange + clearing, per contract, one direction.
# These are typical numbers; check your firm's fee sheet and adjust in settings.json.
CONTRACTS = {
    "MES": dict(ticker="MES=F", point_value=5.0,  tick=0.25, fee_per_side=1.40, name="Micro S&P 500"),
    "MNQ": dict(ticker="MNQ=F", point_value=2.0,  tick=0.25, fee_per_side=1.40, name="Micro Nasdaq"),
    "M2K": dict(ticker="M2K=F", point_value=5.0,  tick=0.10, fee_per_side=1.40, name="Micro Russell"),
    "ES":  dict(ticker="ES=F",  point_value=50.0, tick=0.25, fee_per_side=2.50, name="E-mini S&P 500"),
    "NQ":  dict(ticker="NQ=F",  point_value=20.0, tick=0.25, fee_per_side=2.50, name="E-mini Nasdaq"),
}

# How far back Yahoo Finance lets you pull each bar size.
YF_PERIODS = {"1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d"}


@dataclass
class Settings:
    # --- markets ---
    symbols: list = field(default_factory=lambda: ["MES", "MNQ"])
    intervals: list = field(default_factory=lambda: ["5m", "15m", "1h"])
    research_interval: str = "5m"          # bar size the puzzle tester works on
    # --- realism ---
    slippage_ticks: float = 1.0            # per side, on market/stop fills
    # --- prop-firm rules (defaults = Topstep 50K Combine) ---
    account_size: float = 50000.0
    daily_loss_limit: float = 1000.0       # hit it -> flat and locked out for the day
    max_loss_limit: float = 2000.0         # trailing max loss (breach = account gone)
    trailing_mode: str = "eod"             # "eod" (Topstep style) or "intraday" (Apex style)
    max_contracts: int = 5                 # per-position cap (Topstep 50K: 5 minis = 50 micros; stay small)
    risk_per_trade: float = 150.0          # $ risked per trade when sizing by risk
    # --- session (US Eastern) ---
    rth_start: str = "09:30"
    rth_end: str = "16:00"
    last_entry: str = "15:30"              # no new trades after this
    flatten_at: str = "15:55"              # everything closed by here
    # --- research controls ---
    folds: int = 5                         # walk-forward slices
    embargo_days: int = 1                  # gap between training and test data (no straddling trades)
    cscv_slices: int = 8                   # slices for the overfitting probability (PBO) test
    mc_runs: int = 2000                    # Monte Carlo paths for the breach / payout odds
    profit_target: float = 3000.0          # $ you need to bank before a payout (Topstep 50K: $3,000)
    stress_slippage_mult: float = 2.0      # stress test: slippage x this
    stress_fee_mult: float = 1.5           # stress test: fees x this
    min_trades: int = 30                   # fewer OOS trades than this = not trusted
    max_pieces: int = 4                    # cap on pieces per combined strategy
    top_entries: int = 4                   # entries that survive stage 1
    top_combos: int = 10                   # combos that get the expensive checks
    luck_runs: int = 100                   # random-entry runs for the luck test
    # --- live / paper ---
    live_trading_enabled: bool = False     # must be true AND pass --live to place real orders
    max_trades_per_day: int = 6
    poll_seconds: int = 30
    topstepx_base_url: str = "https://api.topstepx.com"
    topstepx_username: str = ""
    topstepx_api_key: str = ""
    topstepx_account_id: int = 0
    live_symbol: str = "MES"

    def hm(self, key):
        h, m = self.__dict__[key].split(":")
        return int(h) * 60 + int(m)

    @property
    def rth_start_min(self): return self.hm("rth_start")
    @property
    def rth_end_min(self): return self.hm("rth_end")
    @property
    def last_entry_min(self): return self.hm("last_entry")
    @property
    def flatten_min(self): return self.hm("flatten_at")


def load_settings() -> Settings:
    s = Settings()
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH) as f:
            data = json.load(f)
        for k, v in data.items():
            if k == "contracts":
                for sym, spec in v.items():
                    CONTRACTS.setdefault(sym, {}).update(spec)
            elif hasattr(s, k):
                setattr(s, k, v)
    else:
        save_settings(s)
    return s


def save_settings(s: Settings):
    data = asdict(s)
    data["contracts"] = CONTRACTS
    with open(SETTINGS_PATH, "w") as f:
        json.dump(data, f, indent=2)
