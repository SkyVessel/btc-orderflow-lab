"""Frozen BTC v0.2 study: causal 5m features, calibrated probabilities, 1m fills.

This research module never enables live or paper orders in the v0.1 runner.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .core import bootstrap_days, digest, kline, label, size, utc_day

FEATURES = ["delta_ratio", "log_relative_volume", "trend_efficiency", "atr_percent",
            "distance_vwap_atr", "close_location", "lower_wick_fraction",
            "momentum_15m_atr", "momentum_60m_atr", "hour_sin", "hour_cos",
            "fee_to_stop", "is_pullback", "is_sweep"]


def aggregate(bars):
    result = []
    for i in range(0, len(bars), 5):
        group = bars[i:i+5]
        if len(group) != 5 or group[0]["t"] % 300000 or any(b["t"] != group[0]["t"]+j*60000 for j, b in enumerate(group)):
            raise ValueError("Five-minute aggregation requires complete aligned minutes")
        result.append(dict(t=group[0]["t"], o=group[0]["o"], h=max(b["h"] for b in group),
                           l=min(b["l"] for b in group), c=group[-1]["c"],
                           v=sum(b["v"] for b in group), q=sum(b["q"] for b in group),
                           buy=sum(b["buy"] for b in group), minute_index=i+4))
    return result


def signals(bars, protocol):
    five, output, last = aggregate(bars), [], {}
    zone = None
    for i in range(60, len(five)):
        b, old = five[i], five[i-60:i]
        recent = old[-12:]
        hi, lo = max(x["h"] for x in recent), min(x["l"] for x in recent)
        atr = sum(max(x["h"]-x["l"], abs(x["h"]-y["c"]), abs(x["l"]-y["c"]))
                  for y, x in zip(old[-21:-1], old[-20:])) / 20
        if atr <= 0 or b["v"] <= 0:
            continue
        path = sum(abs(x["c"]-y["c"]) for y, x in zip(old[-13:-1], recent))
        er = (old[-1]["c"]-old[-13]["c"])/max(path, 1e-12)
        vwap = sum(x["q"] for x in recent)/max(sum(x["v"] for x in recent), 1e-12)
        relvol = b["v"]/max(sum(x["v"] for x in old[-20:])/20, 1e-12)
        delta = (2*b["buy"]-b["v"])/b["v"]
        location = (b["c"]-b["l"])/max(b["h"]-b["l"], 1e-12)
        if zone and (b["c"] < zone["low"] or b["t"]-zone["t"] > 60*60000):
            zone = None
        setups = []
        if (zone and b["t"] > zone["t"] and b["l"] <= zone["high"] and b["h"] >= zone["low"]
                and b["c"] > zone["high"] and b["c"] > vwap and er > .15 and delta > .1 and relvol > 1):
            setups.append(("value_pullback", zone["low"]))
        if b["l"] < lo and b["c"] > lo and b["c"] > b["o"] and delta > .1 and relvol > 1.2 and location > .6 and er > -.3:
            setups.append(("sweep_reclaim", b["l"]))
        if b["c"] > hi:
            if delta > .1 and relvol > 1.2 and er > 0:
                setups.append(("impulse_breakout", hi-atr))
            reverse = next((x for x in reversed(old[-10:]) if x["c"] < x["o"]), None)
            if reverse:
                zone = dict(low=reverse["l"], high=reverse["h"], t=b["t"])
        for strategy, structural in setups:
            if b["t"]-last.get(strategy, 0) < 15*60000:
                continue
            stop = max(protocol["min_stop_fraction"], protocol["atr_stop_multiple"]*atr/b["c"], (b["c"]-structural)/b["c"])
            if stop > protocol["max_stop_fraction"]:
                continue
            last[strategy] = b["t"]
            hour = (b["t"] % 86400000)/86400000*2*math.pi
            x = [delta, math.log(max(relvol, 1e-8)), er, atr/b["c"]*100,
                 (b["c"]-vwap)/atr, 2*location-1, (min(b["o"], b["c"])-b["l"])/max(b["h"]-b["l"], 1e-12),
                 (b["c"]-old[-3]["c"])/atr, (b["c"]-old[-12]["c"])/atr,
                 math.sin(hour), math.cos(hour), 2*protocol["fees_per_side"]/stop,
                 float(strategy == "value_pullback"), float(strategy == "sweep_reclaim")]
            output.append(dict(id=digest([strategy,b["t"]])[:24], strategy=strategy, signal_ms=b["t"]+300000,
                               minute_index=b["minute_index"], stop_fraction=stop,
                               target_fraction=protocol["target_r"]*stop, x=x))
    return output


def make_rows(bars, candidates, protocol, hold):
    c = dict(hold_minutes=hold, slippage_bps_per_side=protocol["slippage_bps"],
             historical_half_spread_bps=protocol["half_spread_bps"], fee_rate_per_side=protocol["fees_per_side"])
    rows = []
    for s in candidates:
        i = s["minute_index"]+1
        future = bars[i:i+hold]
        y, stress = label(s, future, c), label(s, future, c, 2)
        if y is not None and stress is not None:
            rows.append({**s, **y, "stress_label": stress})
    return rows


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -35, 35)))


def logistic(X, y, penalty, steps=60):
    beta = np.zeros(X.shape[1])
    penalties = np.full(X.shape[1], penalty)
    penalties[0] = 1e-6
    for _ in range(steps):
        p = sigmoid(X@beta)
        weights = np.maximum(p*(1-p), 1e-6)
        h = X.T@(weights[:,None]*X)+np.diag(penalties)
        gradient = X.T@(p-y)+penalties*beta
        delta = np.linalg.solve(h, gradient)
        # Damped Newton descent avoids overshooting on near-separable samples.
        old_loss = np.sum(np.logaddexp(0, X@beta)-y*(X@beta))+.5*np.sum(penalties*beta**2)
        scale = 1.0
        while scale > 1e-5:
            trial = beta-scale*delta
            loss = np.sum(np.logaddexp(0, X@trial)-y*(X@trial))+.5*np.sum(penalties*trial**2)
            if loss <= old_loss:
                beta = trial
                break
            scale *= .5
        if np.max(np.abs(scale*delta)) < 1e-7:
            break
    return beta


def fit_model(train, calibration, penalty):
    raw = np.array([r["x"] for r in train])
    mean, sd = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-8)
    def transform(rows):
        x = np.clip((np.array([r["x"] for r in rows])-mean)/sd, -5, 5)
        return np.column_stack([np.ones(len(rows)), x])
    y = np.array([r["net_return"] > 0 for r in train], dtype=float)
    beta = logistic(transform(train), y, penalty)
    logits = transform(calibration)@beta
    cal_y = np.array([r["net_return"] > 0 for r in calibration], dtype=float)
    cal_beta = logistic(np.column_stack([np.ones(len(logits)), logits]), cal_y, 1.0)
    payoffs = {}
    for strategy in sorted({r["strategy"] for r in train}):
        parts = {}
        for win in (False, True):
            parent = [r["net_return"]/r["stop_fraction"] for r in train if (r["net_return"] > 0) == win]
            group = [r["net_return"]/r["stop_fraction"] for r in train if r["strategy"] == strategy and (r["net_return"] > 0) == win]
            if not parent:
                raise ValueError("Training requires both profit and loss cases")
            parts["win" if win else "loss"] = (sum(group)+30*np.mean(parent))/(len(group)+30)
        payoffs[strategy] = parts
    return dict(mean=mean.tolist(), sd=sd.tolist(), coefficients=beta.tolist(), calibration=cal_beta.tolist(),
                payoff_r=payoffs, features=FEATURES, l2=penalty, training_n=len(train), calibration_n=len(calibration),
                constant_p=float((sum(cal_y)+1)/(len(cal_y)+2)))


def score(rows, model):
    if not rows:
        return []
    x = np.clip((np.array([r["x"] for r in rows])-np.array(model["mean"]))/np.array(model["sd"]), -5, 5)
    z = np.column_stack([np.ones(len(rows)), x])@np.array(model["coefficients"])
    p = sigmoid(model["calibration"][0]+model["calibration"][1]*z)
    result = []
    for row, prob in zip(rows, p):
        payoff = model["payoff_r"][row["strategy"]]
        result.append({**row, "p_profit": float(prob),
                       "expected_net_return": float((prob*payoff["win"]+(1-prob)*payoff["loss"])*row["stop_fraction"])})
    return result


def probability_metrics(rows, constant):
    if not rows:
        return dict(n=0, brier=None, constant_brier=None, bins=[])
    bins = []
    for i in range(10):
        group = [r for r in rows if min(9, int(10*r["p_profit"])) == i]
        if group:
            bins.append(dict(lower=i/10, n=len(group), mean_prediction=float(np.mean([r["p_profit"] for r in group])),
                             observed_profit_rate=float(np.mean([r["net_return"] > 0 for r in group]))))
    return dict(n=len(rows), brier=float(np.mean([(r["p_profit"]-(r["net_return"] > 0))**2 for r in rows])),
                constant_brier=float(np.mean([(constant-(r["net_return"] > 0))**2 for r in rows])), bins=bins,
                caveat="Overlapping candidate labels; calibration sample count is not independent trade count")


def replay(rows, protocol, start, end, threshold, stress=False):
    cash = protocol["initial_cash"]
    c = {**protocol, "fee_rate_per_side": protocol["fees_per_side"], "slippage_bps_per_side": protocol["slippage_bps"]*(2 if stress else 1),
         "qty_step": .00001, "min_qty": .00001, "min_notional": 5}
    free_at, day, day_cash, count, streak = 0, None, cash, 0, 0
    ledger, daily = [], {utc_day(t): 0. for t in range(start, end, 86400000)}
    for original in sorted(rows, key=lambda r: (r["signal_ms"], -r["expected_net_return"], r["strategy"])):
        r = {**original, **original["stress_label"]} if stress else original
        if not start <= r["signal_ms"] < end or r["exit_ms"] > end or r["signal_ms"] < free_at:
            continue
        if threshold is not None and r["expected_net_return"] < threshold:
            continue
        today = utc_day(r["signal_ms"])
        if today != day:
            day, day_cash, count, streak = today, cash, 0, 0
        if count >= c["max_daily_trades"] or streak >= c["max_consecutive_losses"] or cash <= day_cash*(1-c["max_daily_loss_fraction"]):
            continue
        qty = size(cash, cash, r["entry"], r["stop_fraction"], c)
        if not qty:
            continue
        pnl, fees = qty*r["entry"]*r["net_return"], qty*r["fees_per_btc"]
        cash += pnl
        daily[today] += pnl
        count += 1
        streak = streak+1 if pnl < 0 else 0
        free_at = r["exit_ms"]
        ledger.append({k: r[k] for k in ("id", "strategy", "signal_ms", "exit_ms", "entry", "exit", "outcome", "p_profit", "expected_net_return")} |
                      dict(qty=qty, net_pnl=pnl, fees=fees, equity=cash))
    gain, loss = sum(max(t["net_pnl"],0) for t in ledger), -sum(min(t["net_pnl"],0) for t in ledger)
    peak, dd = c["initial_cash"], 0
    for t in ledger:
        peak = max(peak,t["equity"])
        dd = max(dd,1-t["equity"]/peak)
    return dict(trades=len(ledger), net_pnl=cash-c["initial_cash"], gross_pnl=cash-c["initial_cash"]+sum(t["fees"] for t in ledger),
                fees=sum(t["fees"] for t in ledger), return_fraction=cash/c["initial_cash"]-1,
                profit_factor=gain/loss if loss else None, realized_drawdown=dd, daily_pnl=daily,
                mean_daily_pnl_ci95=bootstrap_days(list(daily.values())), ledger=ledger)


def partition(rows, start, end):
    return [r for r in rows if start <= r["signal_ms"] < end and r["exit_ms"] <= end]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/history-v0.2")
    p.add_argument("--protocol", default="research/v0.2/protocol.json")
    p.add_argument("--output", default="research/v0.2/results")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(Path(args.protocol).read_text())
    data = Path(args.data)
    manifest = json.loads((data/"manifest.json").read_text())
    raw = []
    for record in manifest["snapshots"]:
        content = (data/record["file"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("Raw data checksum mismatch")
        raw.extend(json.loads(content))
    bars = sorted((kline(x) for x in raw), key=lambda b:b["t"])
    start, end = manifest["start_ms"], manifest["end_ms"]
    if [b["t"] for b in bars] != list(range(start,end,60000)):
        raise ValueError("Missing minutes")
    if utc_day(end) != protocol["historical_end_utc_exclusive"] or (end-start)//86400000 != protocol["days"]:
        raise ValueError("Data interval differs from registered protocol")
    fit_end = start+90*86400000
    cal_end, val_end = fit_end+30*86400000, fit_end+60*86400000
    candidates = signals(bars, protocol)
    trials, models, labels = [], {}, {}
    for hold in protocol["holding_minutes"]:
        rows = make_rows(bars, candidates, protocol, hold)
        labels[hold] = rows
        train, cal = partition(rows,start,fit_end), partition(rows,fit_end,cal_end)
        if len(train) < protocol["minimum_train_candidates"] or len(cal) < 50:
            raise ValueError("Insufficient training/calibration candidates")
        for penalty in protocol["logistic_l2"]:
            model = fit_model(train,cal,penalty)
            key = f"hold={hold}|l2={penalty}"
            models[key] = model
            validation = score(partition(rows,cal_end,val_end),model)
            metrics = probability_metrics(validation,model["constant_p"])
            for threshold in protocol["minimum_expected_net_returns"]:
                replayed = replay(validation,protocol,cal_end,val_end,threshold)
                trial = dict(model_key=key, hold=hold, l2=penalty, threshold=threshold, validation=replayed,
                             validation_probability=metrics)
                trials.append(trial)
                print(json.dumps({"trial":len(trials),"model":key,"threshold":threshold,"trades":replayed["trades"],"net":replayed["net_pnl"]}),flush=True)
    sufficient = [t for t in trials if t["validation"]["trades"] >= protocol["minimum_validation_trades"]]
    selected = max(sufficient or trials,key=lambda t:t["validation"]["net_pnl"])
    model = models[selected["model_key"]]
    # Only now score the single selected model on the final untouched partition.
    test_rows = score(partition(labels[selected["hold"]],val_end,end),model)
    test = replay(test_rows,protocol,val_end,end,selected["threshold"])
    stress = replay(test_rows,protocol,val_end,end,selected["threshold"],stress=True)
    baseline = replay(test_rows,protocol,val_end,end,None)
    metrics = probability_metrics(test_rows,model["constant_p"])
    checks = dict(validation_positive=selected["validation"]["net_pnl"]>0,
                  validation_samples=selected["validation"]["trades"]>=protocol["minimum_validation_trades"],
                  test_positive=test["net_pnl"]>0,test_samples=test["trades"]>=protocol["minimum_test_trades"],
                  profit_factor=test["profit_factor"] is not None and test["profit_factor"]>=protocol["minimum_profit_factor"],
                  uncertainty=test["mean_daily_pnl_ci95"][0]>0,stress_positive=stress["net_pnl"]>0,
                  calibration=metrics["brier"] is not None and metrics["brier"]<=metrics["constant_brier"],
                  drawdown=test["realized_drawdown"]<=protocol["maximum_drawdown"],beats_unfiltered=test["net_pnl"]>baseline["net_pnl"])
    result=dict(version=protocol["version"],kind="historical_paper_replay",protocol_hash=digest(protocol),
                data=manifest,split_ms=dict(start=start,fit_end=fit_end,calibration_end=cal_end,validation_end=val_end,end=end),
                candidates=len(candidates),trials=trials,selected={k:v for k,v in selected.items() if k!='validation'},
                test=test,stress=stress,unfiltered_baseline=baseline,test_probability=metrics,checks=checks,
                qualified=all(checks.values()),actual_exchange_profit=None,
                limitations=["Historical L2/big trade unavailable", "Candidate labels overlap; evaluate uncertainty by day",
                             "Minute simulation assumes spread/slippage; exit-only drawdown", "12 validation policies tried; selection can overfit"])
    (out/"report.json").write_text(json.dumps(result,indent=2))
    model.update(version=protocol["version"],hold_minutes=selected["hold"],threshold=selected["threshold"],
                 protocol_hash=digest(protocol),qualified=result["qualified"],deploy_enabled=False,
                 note="Research model; requires a separately verified execution adapter and forward test before deployment")
    (out/"model.json").write_text(json.dumps(model,indent=2))
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2))
    (out/"source_hashes.json").write_text(json.dumps({name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                                     for name in ("train_v2.py","core.py")},indent=2))
    print(json.dumps({"selected":selected["model_key"],"threshold":selected["threshold"],"test_net":test["net_pnl"],"checks":checks},indent=2))


if __name__ == "__main__":
    main()
