"""Continuous PUBLIC market-data connection with local, persistent PAPER wallets.

There are deliberately no private endpoints, API keys, or exchange order methods.
"""
import asyncio
import gzip
import hashlib
import json
import math
import os
import sqlite3
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from .core import Signals, digest, predict, size, utc_day, validate_bar
from .research import get_public

WS = "wss://data-stream.binance.vision/stream?streams=btcusdt@trade/btcusdt@depth20@100ms/btcusdt@kline_1m"


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts REAL, kind TEXT, payload TEXT);
          CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, ts REAL, payload TEXT);
          CREATE TABLE IF NOT EXISTS trades (id TEXT PRIMARY KEY, wallet TEXT, ts REAL, payload TEXT);
          CREATE TABLE IF NOT EXISTS expenses (id TEXT PRIMARY KEY, ts REAL, usd REAL CHECK(usd>=0), category TEXT, note TEXT);
        """)

    def event(self, ts, kind, payload):
        self.db.execute("INSERT INTO events(ts,kind,payload) VALUES(?,?,?)", (ts, kind, json.dumps(payload)))

    def get(self, key):
        r = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, key, obj):
        self.db.execute("INSERT INTO state VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(obj)))

    def commit(self):
        self.db.commit()


@contextmanager
def single_runner(directory):
    import fcntl
    p = Path(directory) / "runner.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("A runner already owns this directory") from e
        yield


class Book:
    def __init__(self):
        self.bids, self.asks, self.ts, self.update_id = [], [], 0, -1

    def update(self, data, now):
        if int(data["lastUpdateId"]) <= self.update_id:
            return False
        bids = [[float(p), float(q)] for p, q in data["bids"]]
        asks = [[float(p), float(q)] for p, q in data["asks"]]
        if not bids or not asks or any(not math.isfinite(v) or v <= 0 for x in bids + asks for v in x):
            raise ValueError("Invalid depth snapshot")
        if bids != sorted(bids, reverse=True) or asks != sorted(asks) or bids[0][0] >= asks[0][0]:
            raise ValueError("Crossed or unsorted depth snapshot")
        self.bids, self.asks, self.ts, self.update_id = bids, asks, now, int(data["lastUpdateId"])
        return True

    def stats(self):
        if not self.bids:
            return {}
        bid, ask = self.bids[0][0], self.asks[0][0]
        b, a = sum(q for _, q in self.bids), sum(q for _, q in self.asks)
        return dict(bid=bid, ask=ask, mid=(bid + ask) / 2, spread_bps=(ask - bid) / ((ask + bid) / 2) * 10000,
                    depth_imbalance=(b - a) / (b + a), bids_qty=b, asks_qty=a,
                    update_id=self.update_id, snapshot_received_ms=self.ts)

    def fill(self, side, qty, c):
        """Walk only 10% of each displayed level. Reject insufficient visible depth."""
        levels = self.asks if side == "buy" else self.bids
        remaining, cost = qty, 0
        for price, available in levels:
            taken = min(remaining, available * c["book_participation"])
            cost += taken * price
            remaining -= taken
            if remaining < 1e-10:
                break
        if remaining > 1e-9 or qty <= 0:
            return None
        slip = c["slippage_bps_per_side"] / 10000
        return cost / qty * (1 + slip if side == "buy" else 1 - slip)


class PaperEngine:
    def __init__(self, c, model, store):
        self.c, self.model, self.store = c, model, store
        self.signals, self.book = Signals(c), Book()
        self.trade_sizes = deque(maxlen=5000)
        self.trade_id, self.last_market_ms, self.session_cvd = None, 0, 0.0
        self.minute_flow = dict(buy=0., sell=0., big_buy=0., big_sell=0., count=0)
        self.big_threshold, self.trade_counter = None, 0
        self.halt = False
        source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ("core.py", "live.py")}
        fingerprint = digest({"config": c, "model": model, "source": source_hashes})
        old = store.get("fingerprint")
        if old and old != fingerprint:
            raise ValueError("Config/model changed: create a NEW runtime directory and keep the old ledger")
        store.put("fingerprint", fingerprint)
        store.put("source_hashes", source_hashes)
        store.put("config", c)
        store.put("trading_enabled", bool(model["deploy_enabled"]))
        self.wallets = store.get("wallets")
        if self.wallets is None:
            names = ["primary"]
            self.wallets = {name: dict(cash=c["initial_cash"], position=None, pending=None,
                                      day=None, day_equity=c["initial_cash"], daily_trades=0,
                                      loss_streak=0, realized_pnl=0., fees=0.) for name in names}
        self.disconnect("process_start")

    def disconnect(self, reason):
        self.book = Book()
        self.last_market_ms = 0
        for w in self.wallets.values():
            if w["position"]:
                # Resume by closing at the next fresh book, never invent fills while offline.
                w["pending"] = dict(side="sell", ready_ms=0, reason="recovery_exit")
            else:
                w["pending"] = None
        self.store.event(time.time() * 1000, "connection_gap", {"reason": reason})
        self.store.put("runner_status", "stopped" if reason == "runner_stopped" else "reconnecting")
        self.save()

    def save(self):
        self.store.put("wallets", self.wallets)
        self.store.commit()

    def bootstrap(self, bars):
        for b in bars:
            self.signals.push(b)
        # Do not replay decisions or trade historical bootstrap bars.
        self.store.event(time.time() * 1000, "warmup", {"bars": len(bars)})
        self.store.commit()

    def trade(self, data, now):
        trade_id, event_ms = int(data["t"]), int(data["E"])
        if self.trade_id is not None and trade_id <= self.trade_id:
            return
        if self.trade_id is not None and trade_id != self.trade_id + 1:
            self.store.event(now, "trade_gap", {"previous": self.trade_id, "next": trade_id})
            self.minute_flow = dict(buy=0., sell=0., big_buy=0., big_sell=0., count=0)
            self.session_cvd = 0
        self.trade_id = trade_id
        self.last_market_ms = event_ms
        qty, price = float(data["q"]), float(data["p"])
        if qty <= 0 or price <= 0 or not math.isfinite(qty * price):
            raise ValueError("Invalid trade")
        side = "sell" if data["m"] else "buy"
        amount = qty * price
        # Threshold from PREVIOUS trades only, refreshed every 100 observations.
        if len(self.trade_sizes) >= 1000 and self.trade_counter % 100 == 0:
            values = sorted(self.trade_sizes)
            self.big_threshold = values[int(.99 * (len(values) - 1))]
        self.minute_flow[side] += qty
        self.minute_flow["count"] += 1
        if self.big_threshold is not None and amount >= self.big_threshold:
            self.minute_flow["big_" + side] += qty
        self.session_cvd += qty if side == "buy" else -qty
        self.trade_sizes.append(amount)
        self.trade_counter += 1

    def on_bar(self, b, now):
        if b["t"] + 60000 > now:
            raise ValueError("Unclosed future candle")
        signals = self.signals.push(b)
        stats = self.book.stats()
        for s in signals:
            p = predict(self.model, s, self.c)
            record = {**s, "prediction": p, "received_ms": now, "book": stats,
                      "observed_trade_flow_since_previous_bar_delivery": self.minute_flow.copy(),
                      "session_cvd_btc": self.session_cvd,
                      "big_trade_threshold_quote": self.big_threshold,
                      "big_trade_used_by_model": False, "depth_used_by_model": False}
            cur = self.store.db.execute("INSERT OR IGNORE INTO decisions VALUES(?,?,?)", (s["id"], now, json.dumps(record)))
            if not cur.rowcount:
                continue
            targets = []
            if self.model["deploy_enabled"] and s["strategy"] == self.model["selected"] and p["eligible"]:
                targets.append("primary")
            for name in targets:
                w = self.wallets[name]
                reason = self.entry_blocker(w, s, now)
                if reason:
                    self.store.event(now, "entry_rejected", {"wallet": name, "signal": s["id"], "reason": reason})
                    continue
                w["pending"] = dict(side="buy", ready_ms=now + self.c["latency_ms"],
                                    expires_ms=now + self.c["max_signal_age_seconds"] * 1000,
                                    signal=s, prediction=p)
        self.minute_flow = dict(buy=0., sell=0., big_buy=0., big_sell=0., count=0)
        self.store.put("last_closed_bar_ms", b["t"])
        self.save()

    def entry_blocker(self, w, s, now):
        if self.halt:
            return "operator_halt"
        if w["position"] or w["pending"]:
            return "existing_position_or_order"
        if now - s["signal_ms"] > self.c["max_signal_age_seconds"] * 1000:
            return "stale_signal"
        if not self.book.ts or now - self.book.ts > self.c["max_book_age_seconds"] * 1000:
            return "stale_book"
        if abs(now - self.last_market_ms) > 5000:
            return "stale_market_clock"
        if self.book.stats()["spread_bps"] > self.c["max_spread_bps"]:
            return "wide_spread"
        self.reset_day(w, now)
        if w["daily_trades"] >= self.c["max_daily_trades"] or w["loss_streak"] >= self.c["max_consecutive_losses"]:
            return "daily_trade_or_streak_limit"
        if w["cash"] <= w["day_equity"] * (1 - self.c["max_daily_loss_fraction"]):
            return "daily_loss_limit"
        return None

    def reset_day(self, w, now):
        if w["day"] != utc_day(now):
            w["day"] = utc_day(now)
            w["day_equity"] = w["cash"] + ((w["position"]["qty"] * self.book.bids[0][0]) if w["position"] else 0)
            w["daily_trades"], w["loss_streak"] = 0, 0

    def on_book(self, data, now):
        if not self.book.update(data, now):
            return
        for name, w in self.wallets.items():
            self.reset_day(w, now)
            pending, pos = w["pending"], w["position"]
            if pending and now >= pending["ready_ms"]:
                if pending["side"] == "buy":
                    if self.halt or now > pending["expires_ms"] or abs(now - self.last_market_ms) > 5000:
                        self.store.event(now, "entry_expired", {"wallet": name})
                        w["pending"] = None
                    elif self.book.stats()["spread_bps"] <= self.c["max_spread_bps"]:
                        self.enter(name, w, pending, now)
                elif pos:
                    self.exit(name, w, pending["reason"], now)
            pos = w["position"]
            if pos and w["pending"] is None:
                bid = self.book.bids[0][0]
                reason = ("operator_halt" if self.halt else "stop" if bid <= pos["stop"] else
                          "target" if bid >= pos["target"] else "timeout" if now >= pos["deadline_ms"] else None)
                if reason:
                    w["pending"] = dict(side="sell", ready_ms=now + self.c["latency_ms"], reason=reason)
        # Snapshot state at most once a second; fills are committed immediately.
        previous = self.store.get("heartbeat") or {"received_ms": 0}
        if now - previous["received_ms"] >= 1000:
            self.store.put("heartbeat", {"received_ms": now, "book": self.book.stats(),
                                        "last_market_event_ms": self.last_market_ms, "halt": self.halt})
            self.save()

    def enter(self, name, w, pending, now):
        s = pending["signal"]
        estimate = self.book.asks[0][0] * (1 + self.c["slippage_bps_per_side"] / 10000)
        qty = size(w["cash"], w["cash"], estimate, s["stop_fraction"], self.c)
        price = self.book.fill("buy", qty, self.c)
        if price is None or qty * price * (1 + self.c["fee_rate_per_side"]) > w["cash"]:
            self.store.event(now, "entry_rejected", {"wallet": name, "reason": "insufficient_depth_or_cash"})
            w["pending"] = None
            self.save()
            return
        # The depth walk may increase price: enforce risk/allocation against final fill too.
        if qty > size(w["cash"], w["cash"], price, s["stop_fraction"], self.c) + 1e-12:
            qty = size(w["cash"], w["cash"], price, s["stop_fraction"], self.c)
            price = self.book.fill("buy", qty, self.c)
        if price is None or not qty:
            w["pending"] = None
            self.save()
            return
        fee = qty * price * self.c["fee_rate_per_side"]
        w["cash"] -= qty * price + fee
        w["fees"] += fee
        w["daily_trades"] += 1
        w["position"] = dict(id=s["id"], strategy=s["strategy"], entry_ms=now, entry=price, qty=qty,
                             entry_fee=fee, stop=price * (1 - s["stop_fraction"]),
                             target=price * (1 + s["target_fraction"]),
                             deadline_ms=now + self.c["hold_minutes"] * 60000,
                             prediction=pending["prediction"])
        w["pending"] = None
        self.store.event(now, "paper_entry", {"wallet": name, **w["position"]})
        self.save()

    def exit(self, name, w, reason, now):
        pos = w["position"]
        price = self.book.fill("sell", pos["qty"], self.c)
        if price is None:
            # Keep the position and order outstanding, expose this in reports.
            return
        fee = pos["qty"] * price * self.c["fee_rate_per_side"]
        gross = pos["qty"] * (price - pos["entry"])
        net = gross - pos["entry_fee"] - fee
        w["cash"] += pos["qty"] * price - fee
        w["fees"] += fee
        w["realized_pnl"] += net
        w["loss_streak"] = w["loss_streak"] + 1 if net < 0 else 0
        trade = {**pos, "exit_ms": now, "exit": price, "exit_fee": fee,
                 "gross_pnl": gross, "net_pnl": net, "reason": reason, "kind": "paper_fill"}
        self.store.db.execute("INSERT INTO trades VALUES(?,?,?,?)", (name + ":" + pos["id"], name, now, json.dumps(trade)))
        w["position"], w["pending"] = None, None
        self.save()


def report(store, directory, usd_per_usdt=1.0):
    wallets = store.get("wallets") or {}
    heartbeat = store.get("heartbeat")
    expenses = store.db.execute("SELECT COALESCE(SUM(usd),0) FROM expenses").fetchone()[0]
    result = dict(kind="forward_paper_not_realized_cash_profit", generated_utc=datetime.now(timezone.utc).isoformat(),
                  runner_status=store.get("runner_status"), source_hashes=store.get("source_hashes"),
                  trading_enabled=bool(store.get("trading_enabled")),
                  quote_currency="USDT", usd_per_usdt_assumption=usd_per_usdt,
                  recorded_real_expenses_usd=expenses, cost_coverage="recorded_only; unallocated AI/electricity/labor not automatically known",
                  actual_exchange_trading_profit=None, wallets={}, heartbeat=heartbeat)
    for name, w in wallets.items():
        pos = w["position"]
        mark = heartbeat["book"]["bid"] if heartbeat else None
        fee = store.get("config")["fee_rate_per_side"]
        equity = w["cash"] + (pos["qty"] * mark * (1 - fee) if pos and mark else 0)
        trade_rows = [json.loads(r[0]) for r in store.db.execute("SELECT payload FROM trades WHERE wallet=? ORDER BY ts", (name,))]
        gross = sum(t["gross_pnl"] for t in trade_rows)
        initial = store.get("config")["initial_cash"]
        unrealized = equity - initial - w["realized_pnl"]
        result["wallets"][name] = dict(cash=w["cash"], realized_gross_pnl=gross,
                                       realized_net_pnl=w["realized_pnl"], paid_simulated_fees=w["fees"],
                                       unrealized_net_pnl_at_last_bid=unrealized,
                                       total_equity_pnl_at_last_bid=equity-initial,
                                       equity_at_last_bid_less_exit_fee=equity, open_position=pos, pending=w["pending"],
                                       closed_trades=len(trade_rows),
                                       net_after_recorded_costs_usd=w["realized_pnl"] * usd_per_usdt - expenses,
                                       equity_pnl_after_recorded_costs_usd=(equity-initial)*usd_per_usdt-expenses,
                                       note="Paper wallet only; do not combine profits from independent experiment versions.")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    result["heartbeat_age_seconds"] = (time.time()*1000-heartbeat["received_ms"])/1000 if heartbeat else None
    # Unique temporary path allows an independent observer to generate a report concurrently.
    temporary = out / f"status.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, out / "status.json")
    lines = ["# 实时模拟实验账本", "", "真实交易收益：未产生（没有交易所订单）。以下为模拟成交。", "",
             f"已登记真实支出：{expenses:.2f} USD；未登记支出不自动视为零。", "",
             "| 独立账本 | 已平仓笔数 | 已实现模拟净收益 USDT | 模拟手续费 USDT |", "|---|---:|---:|---:|"]
    for name, w in result["wallets"].items():
        lines.append(f"| {name} | {w['closed_trades']} | {w['realized_net_pnl']:.4f} | {w['paid_simulated_fees']:.4f} |")
    lines += ["", "未平仓仓位、待成交订单、行情时间见 status.json。",
              "当前交易准入：" + ("通过" if result["trading_enabled"] else "未通过，仅采集数据和记录候选信号"),
              "USDT/USD 换算是显式假设。"]
    (out / "STATUS.md").write_text("\n".join(lines) + "\n")
    return result


async def run(c, model, directory, minutes=0, direct=False):
    import websockets
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    if model["config_hash"] != digest(c):
        raise ValueError("Frozen model/config mismatch")
    # Stop new entries at the END of the promised local deadline date.
    from datetime import timedelta
    deadline = (datetime.fromisoformat(c["deadline_local"]).replace(tzinfo=ZoneInfo(c["deadline_timezone"]))
                + timedelta(days=1)).timestamp()*1000
    with single_runner(out):
        store = Store(out / "ledger.sqlite")
        engine = PaperEngine(c, model, store)
        started = time.monotonic()
        reconnect_delay, raw_file, raw_hour, last_report = 1, None, None, 0
        try:
            while not minutes or time.monotonic() - started < minutes * 60:
                if time.time()*1000 >= deadline and all(not w["position"] for w in engine.wallets.values()):
                    (out / "COMPLETED").write_text("Experiment deadline reached; all paper wallets flat.\n")
                    break
                try:
                    server, _, _ = await asyncio.to_thread(get_public, "time", None, direct)
                    rows, _, _ = await asyncio.to_thread(get_public, "klines", {"symbol": c["symbol"], "interval": "1m", "limit": 120}, direct)
                    from .core import kline
                    engine.signals = Signals(c)
                    engine.bootstrap([kline(r) for r in rows if int(r[6]) < int(server["serverTime"])])
                    async with websockets.connect(WS, proxy=None if direct else True, open_timeout=20, ping_interval=20,
                                                  ping_timeout=20, max_queue=1024) as ws:
                        reconnect_delay = 1
                        store.event(time.time() * 1000, "connected", {"url": WS})
                        store.put("runner_status", "running")
                        while not minutes or time.monotonic() - started < minutes * 60:
                            raw = await asyncio.wait_for(ws.recv(), timeout=10)
                            now = time.time() * 1000
                            message = json.loads(raw)
                            hour = datetime.now(timezone.utc).strftime("%Y%m%d-%H")
                            if raw_hour != hour:
                                if raw_file:
                                    raw_file.close()
                                raw_file = gzip.open(out / f"raw-{hour}.jsonl.gz", "at", encoding="utf-8")
                                raw_hour = hour
                            raw_file.write(json.dumps({"received_ms": now, "message": message}) + "\n")
                            stream, d = message["stream"], message["data"]
                            engine.halt = (out / "HALT").exists() or now >= deadline
                            if "@trade" in stream:
                                engine.trade(d, now)
                            elif "@depth" in stream:
                                engine.on_book(d, now)
                            elif "@kline" in stream and d["k"]["x"]:
                                k = d["k"]
                                b = dict(t=int(k["t"]), o=float(k["o"]), h=float(k["h"]), l=float(k["l"]),
                                         c=float(k["c"]), v=float(k["v"]), q=float(k["q"]), n=int(k["n"]), buy=float(k["V"]))
                                engine.last_market_ms = int(d["E"])
                                engine.on_bar(b, now)
                            if now - last_report > 30000:
                                import shutil
                                if shutil.disk_usage(out).free < 1024**3:
                                    (out / "HALT").write_text("Disk free space below 1 GiB; stop new entries.\n")
                                report(store, out)
                                raw_file.flush()
                                last_report = now
                            if now >= deadline and all(not w["position"] and not w["pending"] for w in engine.wallets.values()):
                                (out / "COMPLETED").write_text("Experiment deadline reached; all paper wallets flat.\n")
                                return
                except (OSError, ValueError, KeyError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as e:
                    engine.disconnect(type(e).__name__ + ": " + str(e)[:250])
                    report(store, out)
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(30, reconnect_delay * 2)
        finally:
            # Positions persist. A later restart liquidates them against a fresh book.
            engine.disconnect("runner_stopped")
            store.put("stopped_utc", datetime.now(timezone.utc).isoformat())
            report(store, out)
            if raw_file:
                raw_file.close()
            store.commit()
            store.db.close()
