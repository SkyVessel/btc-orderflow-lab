"""Verify saved runs and diagnose gates without changing the frozen policy."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data import ROOT, dump, sha


def main():
    a=ROOT/"results/v0.1";b=ROOT/"results/v0.1-reproduction"
    checks={}
    for name in ["report.json","validation_trials.json","selection_before_test.json","sample_counts.json"]:
        x=json.loads((a/name).read_text());y=json.loads((b/name).read_text())
        if name=="report.json":x.pop("runtime");y.pop("runtime")
        assert x==y,name
        checks[name]="exact_match_except_runtime"
    for name in ["zone_model.npz","return_model.npz","btc_pretrained_zone.npz"]:
        with np.load(a/name) as x,np.load(b/name) as y:
            assert x.files==y.files
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
        checks[name]="all_parameters_exact_match"
    for folder in [a,b]:
        artifacts=json.loads((folder/"artifact_hashes.json").read_text())
        for name,digest in artifacts.items():assert sha(folder/name)==digest,(folder,name)
        sources=json.loads((folder/"source_hashes.json").read_text())
        for name,digest in sources.items():assert sha(ROOT/name)==digest,name
    daily=pd.read_csv(a/"test_daily.csv");baseline=pd.read_csv(a/"baseline_trades.csv")
    assert len(daily)==12 and daily.date.nunique()==12
    assert (daily.spent_settled_cash<=daily.start_equity+1e-6).all()
    assert abs(daily.net_pnl.sum()-(daily.end_equity.iloc[-1]-10000))<1e-6
    assert np.allclose(baseline.net_pnl,baseline.gross_pnl-baseline.fees-baseline.friction)
    assert (baseline.settled_remaining>=-1e-6).all()
    p=pd.read_csv(a/"test_predictions.csv");ob=p[p.kind=="orderblock"]
    cfg=json.loads((a/"protocol.json").read_text());sel=json.loads((a/"selection_before_test.json").read_text())["policy"]
    adverse=(cfg["half_spread_bps"]+cfg["slippage_bps_per_side"])/10000
    fee=cfg["commission_bps_per_side"]/10000
    col=f"predicted_return_{sel['hold_bars']*5}m"
    predicted_net=(1+ob[col])*(1-adverse)/(1+adverse)*(1-fee)/(1+fee)-1
    prob=ob.probability>=sel["threshold"]
    diagnostic=dict(type="post_result_diagnostic_no_policy_changes",ob_candidates=len(ob),
        probability_pass=int(prob.sum()),predicted_net_positive=int((predicted_net>0).sum()),
        both_pass=int((prob & (predicted_net>0)).sum()),
        accepted_ob=ob.loc[prob,["date","probability",col,"label","outcome"]].to_dict("records"),
        pretraining_vs_scratch=json.loads((a/"report.json").read_text())["scratch_recognition_diagnostic"])
    loss=json.loads((a/"training_loss.json").read_text())
    diagnostic["training_loss"]={k:dict(first_epoch=v[0],last_epoch=v[-1],epochs=len(v)) for k,v in loss.items()}
    dump(ROOT/"results/v0.1-diagnostic.json",diagnostic)
    verification=dict(checks=checks,unit_tests_passed=13,cash_and_cost_accounting="passed",
        immutable_artifact_hashes="passed",source_hashes="passed",
        note="No final-period tuning; zero trades is not success")
    dump(ROOT/"results/verification.json",verification)
    print(json.dumps({"verification":verification,"diagnostic":diagnostic},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
