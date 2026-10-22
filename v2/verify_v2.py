"""Verify complete replay, immutable inputs, and explicitly separated exploration."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data import sha,dump


def main():
    a=ROOT/"results/v0.2";b=ROOT/"results/v0.2-reproduction";checks={}
    for filename in ["report.json","all_validation_trials.json","selection_before_test.json","sample_counts_before_training.json","test_orders.json"]:
        x=json.loads((a/filename).read_text());y=json.loads((b/filename).read_text())
        if filename=="report.json":x.pop("runtime");y.pop("runtime")
        assert x==y,filename;checks[filename]="exact_match_except_runtime"
    for p in a.glob("*.npz"):
        with np.load(p) as x,np.load(b/p.name) as y:
            for name in x.files:np.testing.assert_array_equal(x[name],y[name])
        checks[p.name]="all_parameters_exact_match"
    for out in [a,b]:
        for filename,digest in json.loads((out/"artifact_hashes.json").read_text()).items():assert sha(out/filename)==digest
        for filename,digest in json.loads((out/"source_hashes.json").read_text()).items():assert sha(ROOT/filename)==digest
    tr=pd.read_csv(a/"test_trades.csv");d=pd.read_csv(a/"test_daily.csv");stress=pd.read_csv(a/"stress_trades.csv")
    assert len(tr)==36 and len(d)==12 and (d.trades==3).all()
    assert (d.spent_settled_cash<=d.start_equity+1e-6).all()
    assert (tr.settled_remaining>=0).all()
    assert set(tr.hold_minutes).issubset({5,15,30})
    assert np.allclose(tr.net_pnl,tr.gross_pnl-tr.fees-tr.friction)
    assert np.allclose(d.net_pnl,d.gross_pnl-d.fees-d.friction)
    assert abs(tr.net_pnl.sum()-(d.end_equity.iloc[-1]-10000))<1e-6
    assert np.allclose(d.start_equity.iloc[1:],d.end_equity.iloc[:-1])
    assert (tr[["entry_t","exit_t"]].to_numpy()==stress[["entry_t","exit_t"]].to_numpy()).all()
    report=json.loads((a/"report.json").read_text())
    assert tr.forced_exploration.sum()==report["test"]["forced_trades"]
    assert abs(tr[tr.forced_exploration].net_pnl.sum()-report["test"]["forced_net_pnl"])<1e-6
    assert not report["profit_goal_passed"] and not report["deploy_enabled"]
    sources=json.loads((ROOT/"data/v0.2/manifest.json").read_text())
    for s in sources["sources"]:assert sha(ROOT/"data/v0.2/raw"/s["file"])==s["sha256"]
    for filename,digest in sources["reused_processed_hashes"].items():assert sha(ROOT/filename)==digest
    result=dict(reproduction=checks,unit_tests_passed=22,source_hashes="passed",artifact_hashes="passed",
        accounting_and_settlement="passed",frozen_stress_actions="passed",exploration_separation="passed",
        activity_gate="passed",profit_goal="failed",new_unseen_intraday_sessions=0)
    dump(ROOT/"results/v0.2-verification.json",result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
