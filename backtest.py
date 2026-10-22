"""Sequential cash-account ledger. No sale-proceeds reuse or account resets."""
import math
import numpy as np


def wilson(success,n):
    if not n:return None
    z=1.96;p=success/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-delta),min(1,center+delta)]


def recognition(rows,register,threshold,base_probability):
    result={}
    for kind in ["orderblock","pressure"]:
        r=[x for x in rows if x["kind"]==kind]
        s=[x for x in r if x["probability"]>=threshold]
        n=len(s);wins=sum(x["label"] for x in s)
        reg=[x for x in register if x["kind"]==kind]
        result[kind]=dict(registered=len(reg),touched=sum(z["status"]=="touched" for z in reg),
            scored=len(r),accepted=n,coverage=n/len(r) if r else None,
            accepted_successes=wins,accepted_success_rate=wins/n if n else None,
            accepted_wilson95=wilson(wins,n),
            all_region_success_rate=float(np.mean([x["label"] for x in r])) if r else None,
            classification_accuracy=float(np.mean([(x["probability"]>=threshold)==bool(x["label"]) for x in r])) if r else None,
            recall=wins/sum(x["label"] for x in r) if sum(x["label"] for x in r) else None,
            brier=float(np.mean([(x["probability"]-x["label"])**2 for x in r])) if r else None,
            constant_brier=float(np.mean([(base_probability[kind]-x["label"])**2 for x in r])) if r else None,
            timeout=sum(x["outcome"]=="timeout" for x in r),
            ambiguous_failures=sum(x["ambiguous"] for x in r),
            note="Outcome-based precision, not institutional-order ground truth; Wilson interval ignores event dependence")
    return result


def replay(bars,rows,days,cfg,policy,stress=1,baseline=False):
    cash=cfg["initial_cash"];trades=[];daily=[];peak=cash;maxdd=0.;status="active"
    fee=cfg["commission_bps_per_side"]*stress/10000
    adverse=(cfg["half_spread_bps"]+cfg["slippage_bps_per_side"])*stress/10000
    sessions={d:g.sort_values("t").to_dict("records") for d,g in bars.groupby("date")}
    bydate={d:[] for d in days}
    for r in rows:
        if r["date"] in bydate:bydate[r["date"]].append(r)
    for date in days:
        start_cash=cash;settled=cash;pending=0.;gross=0.;fees=0.;friction=0.;count=0;last_exit=-1
        a=sessions[date];budget=start_cash*cfg["allocation_fraction"]
        for r in sorted(bydate[date],key=lambda z:(z["signal_t"],z["id"])):
            if r["kind"]!="orderblock" or r["signal_t"]<last_exit:continue
            # Cost hurdle and selection depend on predictions, never observed outcome labels.
            h=policy["hold_bars"];hi=cfg["holding_bars"].index(h)
            expected_net=(1+r["predicted_returns"][hi])*(1-adverse)/(1+adverse)*(1-fee)/(1+fee)-1
            if not baseline and (r["probability"]<policy["threshold"] or expected_net<=0):continue
            i=r["signal_index"]+1
            if i+h>len(a):continue
            entrybar=a[i];exitbar=a[i+h-1]
            assert entrybar["t"]==r["signal_t"] and r["created_t"]<=r["signal_t"]
            if entrybar["o"]<r["low"]-r["atr"]:continue
            ep=entrybar["o"]*(1+adverse);xp=exitbar["c"]*(1-adverse)
            # Size from the completed signal bar, never future bar volume.
            cap=math.floor(a[i-1]["v"]*cfg["max_participation"])
            qty=min(math.floor(min(budget,settled)/(ep*(1+fee))),cap)
            if qty<1:continue
            entry_fee=qty*ep*fee;exit_fee=qty*xp*fee
            paid=qty*ep+entry_fee;received=qty*xp-exit_fee
            settled-=paid
            assert settled>=-1e-7
            # Intratrade conservative low-mark liquidation drawdown.
            for b in a[i:i+h]:
                equity=settled+pending+qty*b["l"]*(1-adverse)*(1-fee)
                maxdd=max(maxdd,(peak-equity)/peak)
            pending+=received
            net=received-paid;raw_gross=qty*(exitbar["c"]-entrybar["o"])
            trade_friction=qty*(ep-entrybar["o"]+exitbar["c"]-xp)
            gross+=raw_gross;fees+=entry_fee+exit_fee;friction+=trade_friction;count+=1
            last_exit=exitbar["t"]+300
            eq=settled+pending;peak=max(peak,eq);maxdd=max(maxdd,(peak-eq)/peak)
            trades.append(dict(id=r["id"],date=date,entry_t=entrybar["t"],exit_t=last_exit,
                hold_minutes=5*h,shares=qty,raw_entry=entrybar["o"],raw_exit=exitbar["c"],
                fill_entry=ep,fill_exit=xp,fees=entry_fee+exit_fee,friction=trade_friction,
                gross_pnl=raw_gross,net_pnl=net,settled_remaining=settled,unsettled_proceeds=pending,
                equity=eq,probability=r["probability"],expected_net_return=expected_net))
        cash=settled+pending
        assert abs(cash-start_cash-(gross-fees-friction))<1e-6
        assert cash>=0
        if cash<=0:status="capital_exhausted"
        elif cash<min(b["o"] for b in a)*(1+adverse)*(1+fee):status="insufficient_for_one_share"
        daily.append(dict(date=date,start_equity=start_cash,end_equity=cash,gross_pnl=gross,
                          fees=fees,friction=friction,net_pnl=cash-start_cash,
                          target_met=bool(cash-start_cash>=cfg["daily_target"]),trades=count,
                          spent_settled_cash=start_cash-settled,status=status))
    n=len(daily);hits=sum(d["target_met"] for d in daily)
    pnl=[d["net_pnl"] for d in daily]
    return dict(start_equity=cfg["initial_cash"],end_equity=cash,net_pnl=cash-cfg["initial_cash"],
                gross_pnl=sum(d["gross_pnl"] for d in daily),fees=sum(d["fees"] for d in daily),
                spread_slippage=sum(d["friction"] for d in daily),days=n,trades=len(trades),
                target_days=hits,daily_target_rate=hits/n if n else None,target_rate_wilson95=wilson(hits,n),
                mean_daily_pnl=float(np.mean(pnl)) if n else None,best_day=max(pnl) if n else None,
                worst_day=min(pnl) if n else None,no_trade_days=sum(d["trades"]==0 for d in daily),
                max_drawdown_low_mark=maxdd,status=status,daily=daily,ledger=trades)
