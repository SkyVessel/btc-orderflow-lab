"""Active, versioned historical exploration and frozen retrospective comparison."""
import sys,json,time,resource,argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from data import dump,sha
from model import MLP
from zones import btc_context
from backtest import wilson
from data_v2 import ROOT,load
from regions import extract,region_labels,state_labels,state_matrix,base_features,region_features,REGION_FEATURES,STATE_FEATURES
from execution import net_prediction,plan,replay,activity


def brief(r):return {k:v for k,v in r.items() if k not in ["daily","ledger"]}
def arr(rows,key):return np.asarray([r[key] for r in rows],dtype=float)


def metrics(rows,threshold=.5):
    result={}
    for kind in ["orderblock","pressure"]:
        q=[r for r in rows if r["kind"]==kind];s=[r for r in q if r["probability"]>=threshold]
        wins=sum(r["label"] for r in s)
        result[kind]=dict(scored=len(q),accepted=len(s),successes=wins,precision=wins/len(s) if s else None,
            coverage=len(s)/len(q) if q else None,all_candidate_success_rate=float(np.mean([r["label"] for r in q])) if q else None,
            brier=float(np.mean([(r["probability"]-r["label"])**2 for r in q])) if q else None,
            wilson95=wilson(wins,len(s)),timeout=sum(r["outcome"]=="timeout" for r in q))
    return result


def score_regions(rows,model):
    p=model.predict(arr(rows,"features")).ravel()
    return [{**r,"probability":float(v)} for r,v in zip(rows,p)]


def score_states(rows,context,region,returns):
    x=state_matrix(rows,context,region);y=returns.predict(x)/100
    return [{**r,"predicted_returns":v.tolist()} for r,v in zip(rows,y)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="results/v0.2");args=parser.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    cfg=json.loads((HERE/"protocol.json").read_text());dump(out/"protocol.json",cfg)
    files=list(HERE.glob("*.py"))+[HERE/"protocol.json",ROOT/"model.py",ROOT/"zones.py",ROOT/"data.py",ROOT/"backtest.py"]
    dump(out/"source_hashes.json",{str(p.relative_to(ROOT)):sha(p) for p in files})
    bars,btc,hourly,source=load(cfg);context=btc_context(btc)
    days=sorted(bars.date.unique());a,b,_=cfg["split_sessions"]
    split={"train":days[:a],"validation":days[a:a+b],"test":days[a+b:]}
    dump(out/"splits_before_training.json",split)
    events,registered,states=extract(bars,cfg)
    regions,excluded=region_labels(events,bars,cfg)
    states=state_labels(states,bars,cfg)
    he,hz,_=extract(hourly,cfg,True);hr,hexc=region_labels(he,hourly,cfg,True)
    assert max(r["label_end_t"] for r in hr)<min(bars.t)
    train_regions=[r for r in regions if r["date"] in split["train"]]
    train_states=[r for r in states if r["date"] in split["train"]]
    valid_states=[r for r in states if r["date"] in split["validation"]]
    assert max(r["label_end_t"] for r in train_states)<min(r["signal_t"] for r in valid_states)
    counts=dict(hourly_pretraining=len(hr),registered_regions=len(registered),total_touches=len(events),
        labelled_regions=len(regions),unique_labelled_regions=len(set(r["zone_id"] for r in regions)),
        first_touch_labels=sum(r["touch_number"]==1 for r in regions),incomplete_region_labels=excluded,
        hourly_incomplete_labels=hexc,train_regions=len(train_regions),train_states=len(train_states),
        all_states=len(states),state_action_labels=len(states)*3)
    dump(out/"sample_counts_before_training.json",counts);print("samples",counts,flush=True)
    m=cfg["model"];kw=dict(lr=m["learning_rate"],batch=m["batch_size"],l2=m["l2"],seed=m["seed"])
    models={};histories={};trials=[]
    for name in cfg["policy_models"]:
        region=MLP(17,1,m["seed"],True);history={}
        if name=="hourly_pretrained":
            history["pretrain"]=region.fit(arr(hr,"features"),arr(hr,"label"),m["region_pretrain_epochs"],reset_scaler=True,**kw)
        history["finetune"]=region.fit(arr(train_regions,"features"),arr(train_regions,"label"),m["region_finetune_epochs"],
                                      reset_scaler=name=="scratch",**kw)
        returns=MLP(35,3,m["seed"]+1,False)
        tx=state_matrix(train_states,context,region)
        history["return"]=returns.fit(tx,arr(train_states,"gross_returns")*100,m["return_epochs"],reset_scaler=True,**kw)
        region.save(out/f"{name}_region.npz");returns.save(out/f"{name}_returns.npz")
        models[name]=(region,returns);histories[name]=history
        predicted=returns.predict(tx)/100
        train_scores=net_prediction(predicted,cfg).max(axis=1)
        valid=score_states(valid_states,context,region,returns)
        for quantile in cfg["entry_score_quantiles"]:
            threshold=float(np.quantile(train_scores,quantile))
            orders=plan(valid,split["validation"],cfg,threshold)
            result=replay(bars,orders,split["validation"],cfg)
            trials.append(dict(model=name,quantile=quantile,threshold=threshold,result=brief(result),activity=activity(result,cfg)))
    assert len(trials)==cfg["maximum_candidates"]
    eligible=[t for t in trials if t["activity"]["passed"]]
    selected=max(eligible,key=lambda t:t["result"]["mean_daily_pnl"]) if eligible else max(trials,key=lambda t:t["result"]["trades"])
    locked={k:selected[k] for k in ["model","quantile","threshold"]}
    locked.update(validation_activity_passed=bool(eligible),model_hashes={p.name:sha(p) for p in out.glob("*.npz")})
    dump(out/"all_validation_trials.json",trials);dump(out/"selection_before_test.json",locked)
    dump(out/"training_losses.json",histories)
    print("frozen selection",selected,flush=True)
    region,returns=models[selected["model"]]
    final_states=score_states([r for r in states if r["date"] in split["test"]],context,region,returns)
    orders=plan(final_states,split["test"],cfg,selected["threshold"])
    test=replay(bars,orders,split["test"],cfg);stress=replay(bars,orders,split["test"],cfg,cfg["stress_multiplier"])
    test_regions=score_regions([r for r in regions if r["date"] in split["test"]],region)
    first=[r for r in test_regions if r["touch_number"]==1]
    baseline={}
    for hold in cfg["holding_bars"]:
        bp=plan(final_states,split["test"],cfg,0,fixed_hold=hold)
        br=replay(bars,bp,split["test"],cfg);baseline[str(hold*5)]=brief(br)
        pd.DataFrame(br["daily"]).to_csv(out/f"baseline_{hold*5}m_daily.csv",index=False)
    # Same previously evaluated v0.1 events, unchanged labels, for a matched recognition diagnostic.
    oldp=pd.read_csv(ROOT/"results/v0.1/test_predictions.csv")
    sessions={d:g.sort_values("t").to_dict("records") for d,g in bars.groupby("date")};matched=[]
    for r in oldp.to_dict("records"):
        aa=sessions[r["date"]];i=r["signal_index"]
        base,atr=base_features(aa,i)
        created_index=next(j for j,z in enumerate(aa) if z["t"]+300==r["created_t"])
        z=dict(kind=r["kind"],low=r["low"],high=r["high"],created_index=created_index,touches=1)
        f=region_features(base,aa[i],atr,z,i,cfg)
        matched.append({**r,"probability":float(region.predict(np.asarray([f]))[0,0])})
    for name,r in [("test",test),("stress",stress)]:
        pd.DataFrame(r["daily"]).to_csv(out/f"{name}_daily.csv",index=False)
        pd.DataFrame(r["ledger"]).to_csv(out/f"{name}_trades.csv",index=False)
    dump(out/"test_orders.json",orders);dump(out/"registered_regions.json",registered)
    pd.DataFrame([{k:v for k,v in r.items() if k!="features"} for r in test_regions]).to_csv(out/"test_region_predictions.csv",index=False)
    old=json.loads((ROOT/"results/v0.1/report.json").read_text())
    act=activity(test,cfg)
    report=dict(version=cfg["version"],data=source,splits={k:[v[0],v[-1],len(v)] for k,v in split.items()},counts=counts,
        selected=locked,selected_validation=selected["result"],test=brief(test),stress=brief(stress),activity=act,
        region_first_touch=metrics(first),region_all_touches=metrics(test_regions),matched_v01_regions=metrics(matched),
        baseline_fixed_schedule=baseline,v01_comparison=dict(test=old["test"],recognition=old["recognition"]),
        activity_failure=not(act["passed"] and bool(eligible)),
        profit_goal_passed=bool(act["passed"] and eligible and test["mean_daily_pnl"]>=cfg["daily_target"] and test["target_days"]==test["days"] and stress["net_pnl"]>0),
        parameters=region.parameters+returns.parameters,
        selected_model_bytes=sum((out/f"{selected['model']}_{s}.npz").stat().st_size for s in ["region","returns"]),
        runtime=dict(seconds=time.perf_counter()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2),
        business_profit_after_all_costs=None,actual_profit=None,deploy_enabled=False,
        limitations=["没有新增五分钟交易日；更长的小时线用于跨周期区域预训练，日线用于分红调整。",
            "最终12天已被v0.1查看，本轮属于同日期事后比较，不是新样本外证明。",
            "探索窗口每天要求参与，截止点可以在预测净收益为负时买入；强制探索收益独立列出。",
            "区域可多次回踩；主准确率只计每区域第一次触碰，多次触碰仅作训练扩样和次要指标。",
            "每边5bp交易成本是假设，没有历史可执行报价；压力情景采用同一冻结订单计划、双倍成本。",
            "现金全额整数股，无同日卖出款复用；每段10000美元开始，段内不按天重置，无单日止损。",
            "三个交易窗口、每笔最多三分之一本金是本版执行设定；收益对比同时列出相同资金和时点规则基准。",
            "小时预训练标签跨最多6个交易时段内小时bar，可能跨夜；不能当成5–30分钟成交。",
            "当前仍是监督式状态/动作收益学习，不是在线强化学习；没有实时任务。",
            "区域成功率是数值代理定义。条件变化后新区域成功率不能直接与旧1个样本相比，因此另列旧同一批区域重评分。",
            "AI、电费和人工未分摊，项目全成本净利润未知。"])
    dump(out/"report.json",report)
    make_report(out,report)
    dump(out/"artifact_hashes.json",{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k:report[k] for k in ["test","activity","region_first_touch","matched_v01_regions","profit_goal_passed"]},ensure_ascii=False,indent=2),flush=True)


def make_report(out,r):
    t=r["test"];a=r["activity"]
    pct=lambda x:"无样本" if x is None else f"{x:.1%}"
    lines=["# BITX v0.2：扩样与主动交易实验","",
        f"最后12个交易日：**{t['trades']}笔交易，净利润 ${t['net_pnl']:.2f}，日均 ${t['mean_daily_pnl']:.2f}，日赚$200达标 {t['target_days']}/12（{t['daily_target_rate']:.1%}）**。",
        f"交易量门槛：{'通过' if a['passed'] else '失败'}；盈利目标：{'通过' if r['profit_goal_passed'] else '失败'}。没有启动前瞻交易。","",
        "## 数据扩充","",
        f"- 小时历史：{r['data']['hourly_dates']}，{r['data']['hourly_rows']}根、{r['data']['hourly_sessions']}个完整交易日；日线{r['data']['daily_rows']}根，自2023-06-27开始。",
        f"- 小时预训练仅使用2026-07-29之前的{r['data']['hourly_pretrain_rows']}根，得到{r['counts']['hourly_pretraining']}个标签。",
        "- 五分钟历史仍为60个交易日、4680根。更早的五分钟请求被公开接口拒绝，已保存错误响应。没有付费或账户接入。",
        f"- 不完整小时交易日排除：{r['data']['quarantined_hourly_dates']}。小时线与重叠五分钟聚合价格核对通过。",
        f"- 区域标签{r['counts']['labelled_regions']}个，其中独立首触{r['counts']['first_touch_labels']}个；训练区域{r['counts']['train_regions']}个。训练决策状态{r['counts']['train_states']}个，每个评价5/15/30分钟三个动作。","",
        "## 区域识别","",
        "成功定义沿用区域外一个ATR反应目标与相反方向一个ATR失效线，最多观察30分钟；同时触及按失败、超时计失败。","",
        "| 指标 | OB接受/成功 | OB成功率 | 压力区接受/成功 | 压力区成功率 |",
        "|---|---:|---:|---:|---:|"]
    for title,key in [("新区域独立首次触碰","region_first_touch"),("包括重复回踩（相关样本）","region_all_touches"),("v0.1同一批区域重新评分","matched_v01_regions")]:
        ob=r[key]["orderblock"];p=r[key]["pressure"]
        lines.append(f"| {title} | {ob['accepted']}/{ob['successes']} | {pct(ob['precision'])} | {p['accepted']}/{p['successes']} | {pct(p['precision'])} |")
    lines += ["","列中的“接受/成功”为模型接受数量/其中成功数量；置信区间、覆盖率、Brier详见report.json。v0.1原结果为OB接受1个成功0个、压力区接受12个成功9个。","",
        "## 交易与探索","",
        "- 预登记最低要求：平均每日至少2笔，且至少80%交易日有交易；不满足直接失败。",
        f"- 实际平均每日{a['average_trades_per_day']:.2f}笔，活跃日比例{a['active_day_fraction']:.1%}，空仓日{t['no_trade_days']}。",
        "- 每日窗口为纽约时间10:40–11:00、12:10–12:30、14:00–14:20；首次达到训练分位分数时入场，否则在截止点进行标记的探索交易。没有用事后最高收益时刻选入场。",
        f"- 强制探索{t['forced_trades']}笔，净利润${t['forced_net_pnl']:.2f}；分数触发交易净利润${t['score_selected_net_pnl']:.2f}。",
        f"- 毛利润${t['gross_pnl']:.2f}，佣金假设${t['fees']:.2f}，点差/滑点假设${t['friction']:.2f}；双倍成本净利润${r['stress']['net_pnl']:.2f}。",
        f"- 账户期末${t['end_equity']:.2f}，相对已实现峰值的盘中低价估值回撤{t['low_mark_drawdown']:.2%}；最好一天${t['best_day']:.2f}、最差一天${t['worst_day']:.2f}。","",
        "同一时段、资金和成本下的固定持仓基准：","",
        "| 持有分钟 | 交易次数 | 净利润 | 日目标达标率 |","|---|---:|---:|---:|"]
    for h,b in r["baseline_fixed_schedule"].items():lines.append(f"| {h} | {b['trades']} | ${b['net_pnl']:.2f} | {b['daily_target_rate']:.1%} |")
    lines += ["",f"选参规则：{r['selected']}。6个候选仅用中间12天比较，结果与模型哈希先锁定，再回放最终12天。",
        f"本轮模型共{r['parameters']}个参数，文件{r['selected_model_bytes']}字节；耗时{r['runtime']['seconds']:.2f}秒，峰值内存{r['runtime']['peak_rss_mib']:.1f}MiB。","","## 局限"]
    lines += ["- "+s for s in r["limitations"]]
    lines += ["","复现：`python v2/run_v2.py --output results/v0.2-new-reproduction`。逐笔账本test_trades.csv，每日账本test_daily.csv，成本压力账本stress_trades.csv，候选all_validation_trials.json。"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":main()
