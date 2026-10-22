"""Reproducible pretrain / fine-tune / chronological validation / frozen replay."""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import ROOT, dump, sha, prepare
from model import MLP
from zones import FEATURES, btc_context, candidates, labels
from backtest import recognition, replay


def array(rows,key):
    return np.asarray([r[key] for r in rows],dtype=float)


def score(rows,zone,returns):
    if not rows:return []
    x=array(rows,"features");p=zone.predict(x).ravel();ret=returns.predict(x)/100
    return [{**r,"probability":float(p[i]),"predicted_returns":ret[i].tolist()} for i,r in enumerate(rows)]


def brief(result):
    return {k:v for k,v in result.items() if k not in ["daily","ledger"]}


def write_report(out,r):
    test=r["test"];stress=r["stress"];rec=r["recognition"]
    def pct(x):return "无样本" if x is None else f"{x:.1%}"
    lines=["# BITX v0.1：历史预训练与日内回测", "",
        "本报告为历史纸面模拟；没有真实订单，没有启动实时学习。", "",
        f"结果：最终 {test['days']} 个交易日，净利润 **${test['net_pnl']:.2f}**；日均 **${test['mean_daily_pnl']:.2f}**；每日净赚 $200 达标 **{test['target_days']}/{test['days']}（{test['daily_target_rate']:.1%}）**。",
        f"双倍假设成本净利润 ${stress['net_pnl']:.2f}。净利润达标判定：**{'通过' if r['profit_gate'] else '未通过'}**。", "",
        "## 区域有效性", "",
        "“准确度”指预先登记、之后首次触碰的区域，在未来30分钟内先到达规定反应目标的比例；不是机构订单识别真值。超时记失败，同时保留超时数量。", "",
        "| 类型 | 登记 | 触碰 | 有完整结果 | 模型接受 | 接受区域成功率 | 全部候选成功率 | 覆盖率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for kind,name in [("orderblock","看涨 Order block"),("pressure","压力区")]:
        q=rec[kind]
        lines.append(f"| {name} | {q['registered']} | {q['touched']} | {q['scored']} | {q['accepted']} | {pct(q['accepted_success_rate'])} | {pct(q['all_region_success_rate'])} | {pct(q['coverage'])} |")
        lines += []
    lines += ["", "概率误差及区间：", ""]
    for kind,q in rec.items():
        lines.append(f"- {kind}：接受区域成功率 Wilson 95% 区间 {q['accepted_wilson95']}；Brier {q['brier']}；训练期固定概率基准 {q['constant_brier']}。同日事件相关，区间可能偏窄。")
    lines += ["", "## 收益与成本", "",
        f"- 毛利润 ${test['gross_pnl']:.2f}；佣金假设 ${test['fees']:.2f}；点差/滑点假设 ${test['spread_slippage']:.2f}。",
        f"- {test['trades']} 笔交易，{test['no_trade_days']} 个无交易日；无交易日纳入达标率分母。",
        f"- 期末账户 ${test['end_equity']:.2f}，盘中低价估值最大回撤 {test['max_drawdown_low_mark']:.2%}。",
        f"- 最好一天 ${test['best_day']:.2f}，最差一天 ${test['worst_day']:.2f}。",
        f"- 不用模型筛选、采用相同持有期和资金约束的区域规则基准净利润 ${r['baseline']['net_pnl']:.2f}。",
        "- 交易成本基准：每边3bp半点差、1bp滑点、1bp佣金，共每边5bp；双倍压力情景每边10bp。没有历史报价，均为假设。",
        "- 数据购买支出 $0；AI、电费、人工未分摊。项目全成本净利润保持 null，不能称商业利润已达标。",
        "- 基金内部费用已体现在实际ETF价格中，不另重复扣费。日内平仓不领取隔夜分红；区域按日重置。", "",
        "## 训练与数据", "",
        f"- BTC 预训练：{r['btc_pretrain']['start']} 至 {r['btc_pretrain']['end_exclusive']}（右端不含），{r['btc_pretrain']['samples']} 个区域样本。只使用早于BITX训练起点的数据。",
        f"- BITX：{r['data']['start']} 至 {r['data']['end']}，{r['data']['bars']} 根5分钟K线，60个完整常规交易时段。",
        f"- 时间分段：{r['splits']}。候选策略9个，选择只使用中间12天，参数在读取最终段收益前保存。",
        f"- 选定规则：{r['selected']}。",
        f"- 双模型共 {r['model']['parameters']} 个可训练参数，保存文件 {r['model']['files_bytes']} 字节；纯float32权重等价 {r['model']['float32_weight_bytes']} 字节。",
        f"- 本次运行 {r['runtime']['elapsed_seconds']:.1f} 秒，进程峰值内存约 {r['runtime']['peak_rss_mib']:.1f} MiB。",
        "- 使用NumPy实现的小型神经网络，自动反向传播优化参数；区域分类器先BTC预训练再BITX微调，收益网络只用BITX训练。当前版本不是强化学习，也不会后台自行更新。", "",
        "## 边界与可复现性", ""]
    lines += ["- "+s for s in r["limitations"]]
    lines += ["", "逐笔账本见 test_trades.csv，每日收益见 test_daily.csv；所有候选见 validation_trials.json，区域与概率见 test_predictions.csv，来源与哈希见 ../../data/manifest.json。", "",
        "运行命令：`python run.py --output results/reproduction-01`（新目录，禁止覆盖旧结果）。"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="results/v0.1")
    args=parser.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();cfg=json.loads((ROOT/"protocol.json").read_text())
    dump(out/"protocol.json",cfg)
    hashes={p.name:sha(p) for p in ROOT.glob("*.py")};hashes["protocol.json"]=sha(ROOT/"protocol.json")
    dump(out/"source_hashes.json",hashes)
    x,b,manifest=prepare(cfg);days=sorted(x.date.unique())
    splits={"train":days[:36],"validation":days[36:48],"test":days[48:]}
    dump(out/"split_before_training.json",splits)
    print("data",len(x),len(b),flush=True)
    context=btc_context(b)
    pre=b[(b.date>=cfg["btc_pretrain_start"]) & (b.date<cfg["btc_pretrain_end_exclusive"]) &
          (b.minute>=570) & (b.minute<960)].copy()
    # BTC is 24/7; match weekday/time-of-day sampling without pretending US holidays trade ETFs.
    pre=pre[pd.to_datetime(pre.date).dt.dayofweek<5]
    be,br=candidates(pre,context,cfg);bl,bexcluded=labels(be,pre,cfg)
    print("BTC region samples",len(bl),flush=True)
    xe,xr=candidates(x,context,cfg);xl,xexcluded=labels(xe,x,cfg)
    training=[r for r in xl if r["date"] in splits["train"]]
    validation=[r for r in xl if r["date"] in splits["validation"]]
    # Entire sessions are separated; all labels end within their own session.
    assert max(r["label_end_t"] for r in training)<=min(r["signal_t"] for r in validation)
    assert max(r["label_end_t"] for r in bl)<min(x.t)
    assert len(training)>=40 and len(validation)>=10
    m=cfg["models"];seed=m["seed"]
    zone=MLP(len(FEATURES),1,seed,True)
    kwargs=dict(lr=m["learning_rate"],batch=m["batch_size"],l2=m["l2"],seed=seed)
    history={}
    history["BTC_pretrain"]=zone.fit(array(bl,"features"),array(bl,"label"),m["btc_epochs"],reset_scaler=True,**kwargs)
    zone.save(out/"btc_pretrained_zone.npz")
    history["BITX_finetune"]=zone.fit(array(training,"features"),array(training,"label"),m["bitx_zone_epochs"],**kwargs)
    returns=MLP(len(FEATURES),3,seed+1,False)
    history["BITX_returns"]=returns.fit(array(training,"features"),array(training,"gross_returns")*100,m["return_epochs"],reset_scaler=True,**kwargs)
    zone.save(out/"zone_model.npz");returns.save(out/"return_model.npz")
    dump(out/"training_loss.json",history);dump(out/"features.json",FEATURES)
    validation=score(validation,zone,returns);trials=[]
    for h in cfg["holding_bars"]:
        for threshold in cfg["zone_thresholds"]:
            policy=dict(hold_bars=h,threshold=threshold)
            result=replay(x,validation,splits["validation"],cfg,policy)
            trials.append(dict(policy=policy,result=brief(result)))
    eligible=[t for t in trials if t["result"]["trades"]>=5] or trials
    selected=max(eligible,key=lambda t:(t["result"]["mean_daily_pnl"],-t["result"]["trades"]))
    locked=dict(policy=selected["policy"],zone_hash=sha(out/"zone_model.npz"),return_hash=sha(out/"return_model.npz"),source_hashes=hashes)
    dump(out/"validation_trials.json",trials);dump(out/"selection_before_test.json",locked)
    print("selection frozen",selected,flush=True)
    test_rows=score([r for r in xl if r["date"] in splits["test"]],zone,returns)
    assert all(r["label_end_t"]<=max(x[x.date==r["date"]].t)+300 for r in test_rows)
    test=replay(x,test_rows,splits["test"],cfg,selected["policy"])
    stress=replay(x,test_rows,splits["test"],cfg,selected["policy"],stress=cfg["stress_multiplier"])
    baseline=replay(x,test_rows,splits["test"],cfg,selected["policy"],baseline=True)
    basep={k:float(np.mean([r["label"] for r in training if r["kind"]==k])) for k in ["orderblock","pressure"]}
    rec=recognition(test_rows,[r for r in xr if r["date"] in splits["test"]],selected["policy"]["threshold"],basep)
    for name,value in [("test",test),("stress",stress),("baseline",baseline)]:
        pd.DataFrame(value["daily"]).to_csv(out/f"{name}_daily.csv",index=False)
        pd.DataFrame(value["ledger"]).to_csv(out/f"{name}_trades.csv",index=False)
    pred=[{k:v for k,v in r.items() if k not in ["features","gross_returns","predicted_returns"]} |
          {f"predicted_return_{5*h}m":r["predicted_returns"][j] for j,h in enumerate(cfg["holding_bars"])} for r in test_rows]
    pd.DataFrame(pred).to_csv(out/"test_predictions.csv",index=False)
    dump(out/"bitx_zones.json",xr)
    dump(out/"sample_counts.json",dict(btc_pretrain=len(bl),bitx_train=len(training),bitx_validation=len(validation),bitx_test=len(test_rows),btc_incomplete_excluded=bexcluded,bitx_incomplete_excluded=xexcluded))
    # Save a matched scratch model comparator, without selecting or revising the primary model.
    scratch=MLP(len(FEATURES),1,seed,True)
    scratch.fit(array(training,"features"),array(training,"label"),m["bitx_zone_epochs"],reset_scaler=True,**kwargs)
    scratch_rows=score([r for r in xl if r["date"] in splits["test"]],scratch,returns)
    scratch_rec=recognition(scratch_rows,[r for r in xr if r["date"] in splits["test"]],selected["policy"]["threshold"],basep)
    params=zone.parameters+returns.parameters
    r=dict(version=cfg["version"],data=dict(start=days[0],end=days[-1],bars=len(x)),
           splits={k:[v[0],v[-1],len(v)] for k,v in splits.items()},
           btc_pretrain=dict(start=cfg["btc_pretrain_start"],end_exclusive=cfg["btc_pretrain_end_exclusive"],samples=len(bl)),
           selected=selected["policy"],selected_validation=selected["result"],test=brief(test),stress=brief(stress),baseline=brief(baseline),
           recognition=rec,scratch_recognition_diagnostic=scratch_rec,
           profit_gate=bool(test["mean_daily_pnl"]>=cfg["daily_target"] and test["target_days"]==test["days"] and stress["net_pnl"]>0),
           business_profit_after_all_costs=None,actual_trading_profit=None,deploy_enabled=False,
           model=dict(parameters=params,float32_weight_bytes=params*4,actual_weight_dtype="float64",
                      files_bytes=sum((out/p).stat().st_size for p in ["zone_model.npz","return_model.npz"])),
           runtime=dict(elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2),
           limitations=["60个BITX交易日只用于初步研究，最终段12天很短；历史成交价不能确认实际可执行报价。",
             "BTC同期行情已被旧实验研究，最终段只对本次模型参数隔离，不称为全新市场留出。",
             "BTC预训练用USDT现货，BITX为USD杠杆ETF，存在跨市场差异；未假设精确盘中2倍关系。",
             "区域形态与价格反应是代理定义，没有历史L2或机构订单真值；ETF主动买卖方向缺失。",
             "5分钟K线内先后顺序未知，同时触及成功和失效线时按失败；压力区只评分，不卖空。",
             "收益模型在BITX区域样本上监督训练；不是在线强化学习。没有部署后台进程。",
             "本金每段独立初始10000美元，段内连续复利、不每日重置；卖出款下一交易日可用，费用预留。",
             "每笔最多用日初权益20%，最多占已结束信号K线成交量1%，是本版固定执行假设，无单日止损。",
             "价格为当前回溯快照，缺少历史修订版本；10月21日日线close缺失，未补写，5分钟价格完整。",
             "最短5分钟、最长30分钟固定持仓；收益达标率包括空仓日，费用敏感性单独报告。"])
    dump(out/"report.json",r);write_report(out,r)
    dump(out/"artifact_hashes.json",{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({"test":brief(test),"recognition":rec,"model":r["model"],"runtime":r["runtime"]},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
