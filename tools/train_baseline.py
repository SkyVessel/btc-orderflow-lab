"""Portable v0.2-structure scratch baseline on an explicitly supplied new snapshot.
No forward trading, no automatic strategy optimization, no claim of an unseen test.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "v2"))
from data import dump, sha, valid_ohlc
from model import MLP
from zones import btc_context
from regions import extract, region_labels, state_labels, state_matrix
from execution import net_prediction, plan, replay, activity
from run_v2 import metrics, score_regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = Path(args.data); out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT / "v2/protocol.json").read_text())
    cfg.update(version=out.name, parent_version="bitx-v0.2", mode="historical_paper_only", deploy_enabled=False,
               test_status="retrospective recent-data baseline; not guaranteed unseen", policy_models=["scratch"], maximum_candidates=3)
    dump(out / "protocol.json", cfg)
    manifest = json.loads((data / "manifest.json").read_text())
    for file, expected in manifest["files"].items():
        if sha(data / file) != expected:raise ValueError("Input hash mismatch: " + file)
    bars = pd.read_csv(data / "bitx_5m.csv.gz"); btc = pd.read_csv(data / "btc_5m.csv.gz")
    valid_ohlc(bars); valid_ohlc(btc)
    days = sorted(bars.date.unique())
    if len(days) < 30:raise ValueError("Need at least 30 complete sessions for this baseline")
    a = int(len(days)*.6); b = int(len(days)*.8)
    split = dict(train=days[:a], validation=days[a:b], test=days[b:])
    dump(out / "split_before_training.json", split); dump(out / "data_manifest.json", manifest)
    source_paths = [Path(__file__), ROOT / "model.py", ROOT / "zones.py", ROOT / "data.py", ROOT / "backtest.py"] + list((ROOT / "v2").glob("*.py"))
    dump(out / "source_hashes.json", {str(p.relative_to(ROOT)):sha(p) for p in source_paths})
    events, _, states = extract(bars, cfg)
    regions, incomplete = region_labels(events, bars, cfg)
    states = state_labels(states, bars, cfg)
    train_regions = [r for r in regions if r["date"] in split["train"]]
    train = [r for r in states if r["date"] in split["train"]]
    if not train_regions or not train:raise ValueError("Insufficient training labels")
    m = cfg["model"]; kw = dict(lr=m["learning_rate"], batch=m["batch_size"], l2=m["l2"], seed=m["seed"])
    region = MLP(17,1,m["seed"],True)
    loss = dict(region=region.fit(np.asarray([r["features"] for r in train_regions]), np.asarray([r["label"] for r in train_regions]), m["region_finetune_epochs"], reset_scaler=True, **kw))
    context = btc_context(btc); returns = MLP(35,3,m["seed"]+1,False)
    tx = state_matrix(train,context,region)
    loss["returns"] = returns.fit(tx,np.asarray([r["gross_returns"] for r in train])*100,m["return_epochs"],reset_scaler=True,**kw)
    region.save(out / "scratch_region.npz"); returns.save(out / "scratch_returns.npz")
    dump(out / "training_losses.json",loss)
    def scored(which):
        rows = [r for r in states if r["date"] in split[which]]
        pred = returns.predict(state_matrix(rows,context,region))/100
        return [{**r,"predicted_returns":v.tolist()} for r,v in zip(rows,pred)]
    valid = scored("validation"); thresholds = net_prediction(returns.predict(tx)/100,cfg).max(axis=1)
    trials=[]
    for q in cfg["entry_score_quantiles"]:
        threshold=float(np.quantile(thresholds,q))
        result=replay(bars,plan(valid,split["validation"],cfg,threshold),split["validation"],cfg)
        trials.append(dict(quantile=q,threshold=threshold,activity=activity(result,cfg),result={k:v for k,v in result.items() if k not in ["ledger","daily"]}))
    eligible=[t for t in trials if t["activity"]["passed"]]
    selected=max(eligible or trials,key=lambda t:t["result"]["mean_daily_pnl"])
    dump(out / "validation_trials.json",trials)
    dump(out / "selection_before_test.json",dict(selected=selected,validation_activity_passed=bool(eligible),models={p.name:sha(p) for p in out.glob("*.npz")}))
    test_rows=scored("test"); orders=plan(test_rows,split["test"],cfg,selected["threshold"])
    test=replay(bars,orders,split["test"],cfg); stress=replay(bars,orders,split["test"],cfg,cfg["stress_multiplier"])
    for name,result in [("test",test),("stress",stress)]:
        pd.DataFrame(result["ledger"]).to_csv(out / f"{name}_trades.csv",index=False)
        pd.DataFrame(result["daily"]).to_csv(out / f"{name}_daily.csv",index=False)
    rs=[r for r in regions if r["date"] in split["test"] and r["touch_number"]==1]
    region_scores=score_regions(rs,region) if rs else []
    act=activity(test,cfg)
    report=dict(version=cfg["version"],evaluation_status=cfg["test_status"],test={k:v for k,v in test.items() if k not in ["ledger","daily"]},
        activity=act,region_first_touch=metrics(region_scores),incomplete_region_labels=incomplete,stress_net_pnl=stress["net_pnl"],
        profit_goal_passed=bool(eligible and act["passed"] and test["target_days"]==test["days"] and stress["net_pnl"]>0),
        actual_profit=None,business_profit_after_all_costs=None,forward_started=False)
    dump(out / "report.json",report)
    dump(out / "artifact_hashes.json",{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":main()
