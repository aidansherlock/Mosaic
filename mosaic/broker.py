"""Brokers. PaperBroker = fake fills, real logic. TopstepXBroker = real orders through the TopstepX
(ProjectX Gateway) API.

!! The TopstepX endpoint names/field values below follow the public ProjectX Gateway docs as I know
!! them. Before the first live order, open the API docs from your TopstepX dashboard and confirm each
!! one (they are marked VERIFY). The broker refuses to trade unless connect() succeeds cleanly.
"""
import time
import requests


class Position:
    def __init__(self, side, qty, entry, stop, target, opened_at):
        self.side, self.qty, self.entry, self.stop, self.target, self.opened_at = side, qty, entry, stop, target, opened_at
        self.extreme = entry
        self.bars = 0


class PaperBroker:
    """Simulates fills using the latest bar. Same rules as the backtest engine."""

    def __init__(self, spec, cfg, log=print):
        self.spec, self.cfg, self.log = spec, cfg, log
        self.pos = None
        self.closed = []

    def connect(self):
        self.log("Paper broker ready (no real orders).")
        return True

    def place_bracket(self, side, qty, price, stop, target, when):
        slip = self.cfg.slippage_ticks * self.spec["tick"]
        fill = price + slip * side
        self.pos = Position(side, qty, fill, stop, target, when)
        self.log(f"PAPER {'BUY' if side == 1 else 'SELL'} {qty} @ {fill:.2f} stop {stop:.2f} target {target:.2f}")
        return True

    def update_stop(self, new_stop):
        if self.pos:
            self.pos.stop = new_stop

    def on_bar(self, bar, when):
        """Check whether this bar hit the stop/target. Returns closed pnl or None."""
        p = self.pos
        if not p:
            return None
        p.bars += 1
        slip = self.cfg.slippage_ticks * self.spec["tick"]
        exit_px, reason = None, None
        if (p.side == 1 and bar["low"] <= p.stop) or (p.side == -1 and bar["high"] >= p.stop):
            exit_px, reason = p.stop - slip * p.side, "stop"
        elif (p.side == 1 and bar["high"] >= p.target) or (p.side == -1 and bar["low"] <= p.target):
            exit_px, reason = p.target, "target"
        if exit_px is None:
            p.extreme = max(p.extreme, bar["high"]) if p.side == 1 else min(p.extreme, bar["low"])
            return None
        return self._close(exit_px, reason, when)

    def flatten(self, price, when, reason="flatten"):
        if not self.pos:
            return None
        slip = self.cfg.slippage_ticks * self.spec["tick"]
        return self._close(price - slip * self.pos.side, reason, when)

    def _close(self, px, reason, when):
        p = self.pos
        pnl = (px - p.entry) * p.side * self.spec["point_value"] * p.qty - self.spec["fee_per_side"] * p.qty * 2
        self.closed.append({"side": p.side, "qty": p.qty, "entry": p.entry, "exit": px, "pnl": pnl,
                            "reason": reason, "opened": p.opened_at, "closed": when})
        self.log(f"PAPER CLOSE {reason} @ {px:.2f} pnl ${pnl:,.2f}")
        self.pos = None
        return pnl


class TopstepXBroker(PaperBroker):
    """Real orders on TopstepX. Entry = market order; protection = a stop order + a limit order.
    We poll positions each cycle; when one side fills we cancel the other."""
    ORDER_LIMIT, ORDER_MARKET, ORDER_STOP = 1, 2, 4     # VERIFY against docs
    SIDE_BUY, SIDE_SELL = 0, 1                           # VERIFY against docs

    def __init__(self, spec, cfg, log=print):
        super().__init__(spec, cfg, log)
        self.base = cfg.topstepx_base_url.rstrip("/")
        self.token = None
        self.account_id = cfg.topstepx_account_id
        self.contract_id = None
        self.protect_ids = []

    # ---- plumbing
    def _post(self, path, body):
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        r = requests.post(f"{self.base}{path}", json=body, headers=h, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(f"{path} failed: {data.get('errorMessage') or data}")
        return data

    def connect(self):
        if not (self.cfg.topstepx_username and self.cfg.topstepx_api_key):
            raise RuntimeError("Fill topstepx_username / topstepx_api_key in settings.json first.")
        d = self._post("/api/Auth/loginKey", {"userName": self.cfg.topstepx_username, "apiKey": self.cfg.topstepx_api_key})  # VERIFY
        self.token = d.get("token")
        if not self.token:
            raise RuntimeError("No token returned from login.")
        accts = self._post("/api/Account/search", {"onlyActiveAccounts": True}).get("accounts", [])   # VERIFY
        if not self.account_id:
            if len(accts) != 1:
                raise RuntimeError(f"Set topstepx_account_id in settings.json. Accounts: {[(a.get('id'), a.get('name')) for a in accts]}")
            self.account_id = accts[0]["id"]
        cs = self._post("/api/Contract/search", {"searchText": self.cfg.live_symbol, "live": False}).get("contracts", [])  # VERIFY
        front = [c for c in cs if c.get("activeContract")] or cs
        if not front:
            raise RuntimeError(f"No contract found for {self.cfg.live_symbol}")
        self.contract_id = front[0]["id"]
        self.log(f"TopstepX connected. account={self.account_id} contract={front[0].get('name')} ({self.contract_id})")
        return True

    def bars(self, unit=2, unit_number=5, limit=600):
        """Recent bars from the API (unit 2 = minute). Returns list of dicts."""
        from datetime import datetime, timedelta, timezone
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=10)
        d = self._post("/api/History/retrieveBars", {"contractId": self.contract_id, "live": False,
                       "startTime": start.isoformat(), "endTime": end.isoformat(), "unit": unit,
                       "unitNumber": unit_number, "limit": limit, "includePartialBar": False})  # VERIFY
        return d.get("bars", [])

    def _order(self, side, qty, otype, limit=None, stop=None, tag=None):
        body = {"accountId": self.account_id, "contractId": self.contract_id, "type": otype, "side": side, "size": qty,
                "limitPrice": limit, "stopPrice": stop, "customTag": tag}
        d = self._post("/api/Order/place", body)  # VERIFY
        return d.get("orderId")

    def place_bracket(self, side, qty, price, stop, target, when):
        api_side = self.SIDE_BUY if side == 1 else self.SIDE_SELL
        opp = self.SIDE_SELL if side == 1 else self.SIDE_BUY
        tag = f"mosaic-{int(time.time())}"
        entry_id = self._order(api_side, qty, self.ORDER_MARKET, tag=tag + "-e")
        if entry_id is None:
            raise RuntimeError("Entry order rejected")
        time.sleep(1.0)
        s_id = self._order(opp, qty, self.ORDER_STOP, stop=round(stop / self.spec["tick"]) * self.spec["tick"], tag=tag + "-s")
        t_id = self._order(opp, qty, self.ORDER_LIMIT, limit=round(target / self.spec["tick"]) * self.spec["tick"], tag=tag + "-t")
        if s_id is None:
            self.log("!! STOP ORDER REJECTED - flattening immediately")
            self.flatten(price, when, "no-stop")
            raise RuntimeError("Stop order rejected; position closed for safety")
        self.protect_ids = [i for i in (s_id, t_id) if i is not None]
        self.pos = Position(side, qty, price, stop, target, when)
        self.log(f"LIVE {'BUY' if side == 1 else 'SELL'} {qty} mkt, stop {stop:.2f} target {target:.2f}")
        return True

    def _open_position(self):
        d = self._post("/api/Position/searchOpen", {"accountId": self.account_id})  # VERIFY
        for p in d.get("positions", []):
            if p.get("contractId") == self.contract_id and p.get("size", 0) != 0:
                return p
        return None

    def _cancel_protection(self):
        for oid in self.protect_ids:
            try:
                self._post("/api/Order/cancel", {"accountId": self.account_id, "orderId": oid})  # VERIFY
            except Exception as e:
                self.log(f"cancel {oid}: {e}")
        self.protect_ids = []

    def update_stop(self, new_stop):
        """Replace the stop order at a new price (trailing / breakeven)."""
        if not self.pos or not self.protect_ids:
            return
        try:
            self._post("/api/Order/modify", {"accountId": self.account_id, "orderId": self.protect_ids[0],
                                             "stopPrice": round(new_stop / self.spec["tick"]) * self.spec["tick"]})  # VERIFY
            self.pos.stop = new_stop
        except Exception as e:
            self.log(f"modify stop failed: {e}")

    def on_bar(self, bar, when):
        p = self.pos
        if not p:
            return None
        p.bars += 1
        live = self._open_position()
        if live is None:
            # broker closed it (stop or target hit). Work out which from the bar.
            self._cancel_protection()
            hit_stop = (p.side == 1 and bar["low"] <= p.stop) or (p.side == -1 and bar["high"] >= p.stop)
            px = p.stop if hit_stop else p.target
            return self._close(px, "stop" if hit_stop else "target", when)
        p.extreme = max(p.extreme, bar["high"]) if p.side == 1 else min(p.extreme, bar["low"])
        return None

    def flatten(self, price, when, reason="flatten"):
        if not self.pos:
            return None
        self._cancel_protection()
        self._post("/api/Position/closeContract", {"accountId": self.account_id, "contractId": self.contract_id})  # VERIFY
        return self._close(price, reason, when)
