"""Causal signals, conditional probability estimates and cost-aware bar replay.

All returns are simulated; historical bars contain no historical order book.
"""
import hashlib
import json
import math
import random
import statistics as st
from collections import defaultdict, deque
from datetime import datetime, timezone


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_config(path):
    with open(path) as f:
        c = json.load(f)
    if c["mode"] != "paper_only" or c["symbol"] != "BTCUSDT" or c["venue"] != "binance_spot":
        raise ValueError("This implementation only supports paper-only Binance BTCUSDT spot.")
    for k in ("fee_rate_per_side", "slippage_bps_per_side", "historical_half_spread_bps"):
        if c[k] < 0:
            raise ValueError(k)
    if not (0 < c["risk_fraction"] <= .01 and 0 < c["max_allocation"] <= 1):
        raise ValueError("Invalid unlevered risk/allocation")
    return c


def utc_day(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def kline(row):
    b = dict(t=int(row[0]), o=float(row[1]), h=float(row[2]), l=float(row[3]),
             c=float(row[4]), v=float(row[5]), q=float(row[7]), n=int(row[8]), buy=float(row[9]))
    validate_bar(b)
    return b


def validate_bar(b):
    if not all(math.isfinite(b[k]) for k in ("o", "h", "l", "c", "v", "q", "buy")):
        raise ValueError("Non-finite kline")
    if not (0 < b["l"] <= min(b["o"], b["c"]) <= max(b["o"], b["c"]) <= b["h"]):
        raise ValueError("Invalid OHLC")
    if b["v"] < 0 or not 0 <= b["buy"] <= b["v"] + 1e-8 or b["q"] < 0:
        raise ValueError("Invalid volume")


def cell_key(s):
    f = s["features"]
    return f'{s["strategy"]}|delta={int(f["delta_ratio"] >= .2)}|trend={int(f["efficiency"] >= .3)}'


class Signals:
    """An OB is activated only AFTER a closing-price breakout has occurred."""
    def __init__(self, c):
        self.c = c
        self.bars = deque(maxlen=120)
        self.zone = None
        self.last_signal = {}

    def push(self, b):
        validate_bar(b)
        if self.bars and b["t"] <= self.bars[-1]["t"]:
            return []
        if self.bars and b["t"] != self.bars[-1]["t"] + 60000:
            self.bars.clear()
            self.zone = None
        old = list(self.bars)
        self.bars.append(b)
        if len(old) < self.c["warmup_bars"]:
            return []
        ranges = [max(x["h"] - x["l"], abs(x["h"] - y["c"]), abs(x["l"] - y["c"]))
                  for y, x in zip(old[-21:-1], old[-20:])]
        atr = st.mean(ranges)
        hi, lo = max(x["h"] for x in old[-20:]), min(x["l"] for x in old[-20:])
        path = sum(abs(x["c"] - y["c"]) for y, x in zip(old[-21:-1], old[-20:]))
        efficiency = (old[-1]["c"] - old[-21]["c"]) / max(path, 1e-12)
        vwap = sum(x["q"] for x in old[-60:]) / max(sum(x["v"] for x in old[-60:]), 1e-12)
        delta = (2 * b["buy"] - b["v"]) / max(b["v"], 1e-12)
        relvol = b["v"] / max(st.mean(x["v"] for x in old[-20:]), 1e-12)
        f = dict(delta_ratio=delta, relative_volume=relvol, efficiency=efficiency,
                 atr=atr, prior_vwap=vwap, prior_high=hi, prior_low=lo)
        setups = []
        if self.zone and (b["c"] < self.zone["low"] or b["t"] - self.zone["t"] > 30 * 60000):
            self.zone = None
        if (self.zone and b["t"] > self.zone["t"] and b["l"] <= self.zone["high"]
                and b["h"] >= self.zone["low"] and b["c"] > self.zone["high"]
                and b["c"] > vwap and efficiency > .15 and delta > .2 and relvol > 1):
            setups.append(("flow_pullback", self.zone["low"]))
        if b["l"] < lo and b["c"] > lo and delta > .1 and relvol > 1:
            setups.append(("sweep_reclaim", b["l"]))
        if b["c"] > hi:
            setups.append(("breakout", hi - atr))
            reverse = next((x for x in reversed(old[-10:]) if x["c"] < x["o"]), None)
            if reverse:
                self.zone = dict(low=reverse["l"], high=reverse["h"], t=b["t"])
        result = []
        for strategy, structural_stop in setups:
            if b["t"] - self.last_signal.get(strategy, 0) < self.c["signal_cooldown_minutes"] * 60000:
                continue
            d = max(self.c["min_stop_fraction"], atr * self.c["stop_atr_multiple"] / b["c"],
                    (b["c"] - structural_stop) / b["c"])
            if d > self.c["max_stop_fraction"]:
                continue
            self.last_signal[strategy] = b["t"]
            s = dict(strategy=strategy, signal_ms=b["t"] + 60000, reference=b["c"],
                     stop_fraction=d, target_fraction=d * self.c["target_r"], features=f.copy())
            s["id"] = digest({"version": self.c["version"], "strategy": strategy, "t": s["signal_ms"]})[:24]
            s["cell"] = cell_key(s)
            result.append(s)
        return result


def cost_return(entry, exit_price, fee):
    """Net quote P&L divided by entry notional; fee paid on both notionals."""
    return (exit_price - entry - fee * (entry + exit_price)) / entry


def label(s, future, c, stress=1):
    h = c["hold_minutes"]
    if len(future) < h or future[0]["t"] != s["signal_ms"]:
        return None
    slip = (c["slippage_bps_per_side"] * stress + c["historical_half_spread_bps"]) / 10000
    entry = future[0]["o"] * (1 + slip)
    stop, target = entry * (1 - s["stop_fraction"]), entry * (1 + s["target_fraction"])
    outcome, raw_exit = "timeout", future[h - 1]["c"]
    for j, b in enumerate(future[:h]):
        if b["t"] != s["signal_ms"] + j * 60000:
            return None
        if b["o"] <= stop:
            outcome, raw_exit = "stop", b["o"]
            break
        # Both touched in one bar: stop first, including an entry-minute ambiguity.
        if b["l"] <= stop:
            outcome, raw_exit = "stop", stop
            break
        if b["h"] >= target:
            outcome, raw_exit = "target", target
            break
    exit_price = raw_exit * (1 - slip)
    return dict(entry=entry, exit=exit_price, exit_ms=future[j]["t"] + 60000,
                outcome=outcome, net_return=cost_return(entry, exit_price, c["fee_rate_per_side"]),
                fees_per_btc=c["fee_rate_per_side"] * (entry + exit_price))


def dataset(bars, c):
    engine, rows = Signals(c), []
    for i, b in enumerate(bars):
        for s in engine.push(b):
            y = label(s, bars[i + 1:i + 1 + c["hold_minutes"]], c)
            if y:
                stress = label(s, bars[i + 1:i + 1 + c["hold_minutes"]], c, 2)
                rows.append({**s, **y, "stress_net_return": stress["net_return"], "stress_label": stress})
    return rows


def summarize(rows):
    n = len(rows)
    returns = [r["net_return"] for r in rows]
    return dict(n=n, days=len({utc_day(r["signal_ms"]) for r in rows}),
                sum_return=sum(returns), wins=sum(r > 0 for r in returns),
                outcomes={k: sum(r["outcome"] == k for r in rows) for k in ("target", "stop", "timeout")})


def fit(rows, c, cutoff):
    rows = [r for r in rows if r["exit_ms"] <= cutoff]
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["cell"]].append(r)
    return dict(version=c["version"], config_hash=digest(c), trained_through_ms=cutoff,
                cells={k: summarize(v) for k, v in grouped.items()},
                parents={k: summarize([r for r in rows if r["strategy"] == k]) for k in c["strategies"]},
                selected=None, deploy_enabled=False)


def predict(model, s, c):
    parent = model["parents"].get(s["strategy"], {"n": 0})
    cell = model["cells"].get(s["cell"], {"n": 0})
    if parent["n"] < c["min_parent_samples"] or cell["n"] < c["min_cell_samples"]:
        return dict(eligible=False, reason="insufficient_training", n=cell["n"])
    a, n = c["prior_strength"], cell["n"]
    prior_p = (parent["wins"] + 1) / (parent["n"] + 2)
    p = (cell["wins"] + a * prior_p) / (n + a)
    mu = (cell["sum_return"] + a * parent["sum_return"] / parent["n"]) / (n + a)
    outcomes = {k: (cell["outcomes"][k] + a * parent["outcomes"][k] / parent["n"]) / (n + a)
                for k in ("target", "stop", "timeout")}
    eligible = mu >= c["min_expected_net_return"]
    return dict(eligible=eligible, reason="positive_edge" if eligible else "cost_or_edge_rejected",
                p_profit=p, expected_net_return=mu, p_outcomes=outcomes, n=n, parent_n=parent["n"])


def size(cash, equity, entry, stop_fraction, c):
    # Include entry+exit fees and a slippage reserve in risk sizing, not only the stop distance.
    risk = stop_fraction + 2 * c["fee_rate_per_side"] + 2 * c["slippage_bps_per_side"] / 10000
    notional = min(equity * c["risk_fraction"] / risk, equity * c["max_allocation"],
                   cash / (1 + c["fee_rate_per_side"]))
    qty = math.floor(max(0, notional / entry) / c["qty_step"]) * c["qty_step"]
    return qty if qty >= c["min_qty"] and qty * entry >= c["min_notional"] else 0


def bootstrap_days(daily, seed=42):
    if len(daily) < 2:
        return [None, None]
    rng = random.Random(seed)
    draws = sorted(st.mean(rng.choices(daily, k=len(daily))) for _ in range(2000))
    return [draws[49], draws[1949]]


def replay(rows, model, strategy, c, start, end, filtered=True, stress=False):
    equity = c["initial_cash"]
    free_at, loss_streak, count, day_equity, day = 0, 0, 0, equity, None
    history, daily = [], defaultdict(float)
    for t in range(start, end, 86400000):
        daily[utc_day(t)] = 0.0
    for r in sorted(rows, key=lambda x: (x["signal_ms"], x["strategy"])):
        if stress:
            # Use the stressed entry, exit, timing, fees and sizing consistently.
            r = {**r, **r["stress_label"]}
        if r["strategy"] != strategy or not start <= r["signal_ms"] < end or r["exit_ms"] > end:
            continue
        d = utc_day(r["signal_ms"])
        if d != day:
            day, day_equity, loss_streak, count = d, equity, 0, 0
        if r["signal_ms"] < free_at or count >= c["max_daily_trades"]:
            continue
        if loss_streak >= c["max_consecutive_losses"] or equity <= day_equity * (1 - c["max_daily_loss_fraction"]):
            continue
        p = predict(model, r, c)
        if filtered and not p["eligible"]:
            continue
        qty = size(equity, equity, r["entry"], r["stop_fraction"], c)
        if not qty:
            continue
        ret = r["net_return"]
        pnl = qty * r["entry"] * ret
        equity += pnl
        daily[d] += pnl
        loss_streak = loss_streak + 1 if pnl < 0 else 0
        free_at, count = r["exit_ms"], count + 1
        history.append(dict(signal_ms=r["signal_ms"], exit_ms=r["exit_ms"], strategy=strategy,
                            entry=r["entry"], exit=r["exit"], qty=qty, net_pnl=pnl,
                            fees=qty * r["fees_per_btc"], equity=equity, prediction=p,
                            outcome=r["outcome"]))
    gains = sum(max(x["net_pnl"], 0) for x in history)
    losses = -sum(min(x["net_pnl"], 0) for x in history)
    peak, dd = c["initial_cash"], 0
    for x in history:
        peak = max(peak, x["equity"])
        dd = max(dd, 1 - x["equity"] / peak)
    bs = [(x["prediction"]["p_profit"] - (x["net_pnl"] > 0)) ** 2 for x in history if "p_profit" in x["prediction"]]
    return dict(trades=len(history), net_pnl=equity - c["initial_cash"], return_fraction=equity / c["initial_cash"] - 1,
                fees=sum(x["fees"] for x in history), profit_factor=gains / losses if losses else None,
                realized_drawdown=dd, daily_pnl=dict(daily), mean_daily_pnl_ci95=bootstrap_days(list(daily.values())),
                brier=st.mean(bs) if bs else None, ledger=history)
