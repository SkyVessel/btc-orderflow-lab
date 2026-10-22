"""Predeclared absorption/reversal experiment; retains selection before test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
from .absorption import FEATURES, signals
from .core import digest, kline, utc_day
from .train_v2 import fit_model, make_rows, probability_metrics, replay, score


def load_bars(directory):
    data = Path(directory)
    manifest = json.loads((data/"manifest.json").read_text())
    rows = []
    for record in manifest["snapshots"]:
        raw = (data/record["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record["sha256"]:
            raise ValueError("Original response checksum mismatch")
        rows.extend(json.loads(raw))
    bars = sorted((kline(x) for x in rows),key=lambda b:b["t"])
    if [b["t"] for b in bars] != list(range(manifest["start_ms"],manifest["end_ms"],60000)):
        raise ValueError("Missing, duplicate or misaligned historical minute")
    return bars,manifest


def partition(rows,start,end,hold):
    # Purge the FULL potential outcome horizon for both ordinary and stressed execution.
    return [r for r in rows if start <= r["signal_ms"] and r["signal_ms"]+hold*60000 <= end
            and r["exit_ms"]<=end and r["stress_label"]["exit_ms"]<=end]


def select_trial(trials,p):
    enough = [x for x in trials if x["validation"]["trades"]>=p["minimum_validation_trades"]]
    traded = [x for x in trials if x["validation"]["trades"]>0]
    return max(enough or traded or trials,key=lambda x:x["validation"]["net_pnl"])


def checks_for(selected,test,stress,metrics,baseline,p):
    ci = test["mean_daily_pnl_ci95"][0]
    return dict(validation_positive=selected["validation"]["net_pnl"]>0,
                validation_samples=selected["validation"]["trades"]>=p["minimum_validation_trades"],
                test_positive=test["net_pnl"]>0,test_samples=test["trades"]>=p["minimum_test_trades"],
                profit_factor=test["profit_factor"] is not None and test["profit_factor"]>=p["minimum_profit_factor"],
                uncertainty=ci is not None and ci>0,stress_positive=stress["net_pnl"]>0,
                drawdown=test["realized_drawdown"]<=p["maximum_drawdown"],
                probability_filter_valid=(selected["threshold"] is None or (metrics["brier"] is not None and metrics["brier"]<=metrics["constant_brier"])),
                beats_unfiltered=(selected["threshold"] is None or test["net_pnl"]>baseline["net_pnl"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",default="data/history-v0.3-year")
    parser.add_argument("--protocol",default="research/v0.3/protocol.json")
    parser.add_argument("--output",default="research/v0.3/results")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    p = json.loads(Path(args.protocol).read_text())
    bars,manifest = load_bars(args.data)
    start,end = manifest["start_ms"],manifest["end_ms"]
    if utc_day(end)!=p["historical_end_utc_exclusive"] or end-start!=p["days"]*86400000:
        raise ValueError("Data does not match protocol")
    boundaries=[start]
    for stage in ("fit","calibration","validation","holdout"):
        boundaries.append(boundaries[-1]+p["split_days"][stage]*86400000)
    if boundaries[-1]!=end:
        raise ValueError("Split mismatch")
    _,fit_end,cal_end,val_end,_ = boundaries
    candidates = signals(bars,p)
    models,labels,trials = {},{},[]
    source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ("train_v3.py","absorption.py","train_v2.py","core.py")}
    (out/"protocol.json").write_text(json.dumps(p,indent=2))
    (out/"source_hashes.json").write_text(json.dumps(source_hashes,indent=2))
    for hold in p["holding_minutes"]:
        # Outcome arrays are derived once, but final-partition labels are not inspected for selection.
        rows=make_rows(bars,candidates,p,hold)
        labels[hold]=rows
        train,cal=partition(rows,start,fit_end,hold),partition(rows,fit_end,cal_end,hold)
        if len(train)<p["minimum_train_candidates"] or len(cal)<p["minimum_calibration_candidates"]:
            raise ValueError(f"Insufficient training/calibration data: {len(train)}/{len(cal)}")
        if set(r["strategy"] for r in train)!=set(p["strategies"]):
            raise ValueError("A strategy is absent from training")
        model=fit_model(train,cal,p["logistic_l2"])
        model["features"]=FEATURES
        models[hold]=model
        validation=score(partition(rows,cal_end,val_end,hold),model)
        for strategy in p["strategies"]:
            subset=[r for r in validation if r["strategy"]==strategy]
            for threshold in p["minimum_expected_net_returns"]:
                trial=dict(strategy=strategy,hold=hold,threshold=threshold,
                           validation=replay(subset,p,cal_end,val_end,threshold),
                           probability=probability_metrics(subset,model["constant_p"]))
                trials.append(trial)
                print(json.dumps({"trial":len(trials),"strategy":strategy,"hold":hold,"threshold":threshold,
                                  "trades":trial["validation"]["trades"],"net":trial["validation"]["net_pnl"]}),flush=True)
    if len(trials)!=p["max_policy_candidates"]:
        raise ValueError("Unexpected search budget")
    selected=select_trial(trials,p)
    locked={k:selected[k] for k in ("strategy","hold","threshold")}
    locked.update(protocol_hash=digest(p),model_hash=digest(models[selected["hold"]]),source_hashes=source_hashes)
    (out/"selection-before-test.json").write_text(json.dumps(locked,indent=2))
    model=models[selected["hold"]]
    test_rows=score([r for r in partition(labels[selected["hold"]],val_end,end,selected["hold"])
                     if r["strategy"]==selected["strategy"]],model)
    test=replay(test_rows,p,val_end,end,selected["threshold"])
    stress=replay(test_rows,p,val_end,end,selected["threshold"],stress=True)
    baseline=replay(test_rows,p,val_end,end,None)
    metrics=probability_metrics(test_rows,model["constant_p"])
    checks=checks_for(selected,test,stress,metrics,baseline,p)
    report=dict(version=p["version"],kind="historical_paper_not_actual_profit",data=manifest,
                protocol_hash=digest(p),source_hashes=source_hashes,split_ms=boundaries,
                candidate_count=len(candidates),candidate_by_strategy={s:sum(r["strategy"]==s for r in candidates) for s in p["strategies"]},
                trials=trials,selected=locked,selected_validation=selected["validation"],test=test,stress=stress,
                unfiltered_baseline=baseline,test_probability=metrics,checks=checks,qualified=all(checks.values()),
                actual_exchange_profit=None,real_operating_costs="Not allocated; historical trading PnL only",
                limitations=["Absorption proxy: no historical L2 replenishment or MBO verification",
                             "Later historical period already used by previous versions; chronological transfer remains untested",
                             "Overlapping labels and selected validation winner; daily bootstrap is exploratory",
                             "Minute fills assume spread/slippage; drawdown measured at exits, not intratrade"])
    (out/"report.json").write_text(json.dumps(report,indent=2))
    model.update(version=p["version"],selected=locked,qualified=report["qualified"],deploy_enabled=False,
                 note="No execution adapter activated; user requires qualified backtest before Dot handoff")
    (out/"model.json").write_text(json.dumps(model,indent=2))
    print(json.dumps({"selected":{k:locked[k] for k in ("strategy","hold","threshold")},
                      "test_net":test["net_pnl"],"checks":checks},indent=2))


if __name__=="__main__":
    main()
