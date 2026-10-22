"""Declared exploration windows, cash settlement and frozen stress replays."""
import math
import numpy as np


def net_prediction(returns,cfg):
    adverse=(cfg["half_spread_bps"]+cfg["slippage_bps_per_side"])/10000
    fee=cfg["commission_bps_per_side"]/10000
    return (1+np.asarray(returns))*(1-adverse)/(1+adverse)*(1-fee)/(1+fee)-1


def plan(rows,days,cfg,threshold,fixed_hold=None):
    result=[]
    for day in days:
        subset=sorted([r for r in rows if r["date"]==day],key=lambda r:r["signal_t"])
        last_exit=-1
        for slot,(start,end) in enumerate(cfg["windows_minutes_ny"]):
            # No future maximum: first qualified observation, else a known fixed deadline.
            choices=[r for r in subset if start<=r["minute"]<=end and r["signal_t"]>=last_exit]
            for r in choices:
                scores=net_prediction(r["predicted_returns"],cfg)
                hi=int(np.argmax(scores));expected=float(scores[hi])
                forced=r["minute"]==end
                if fixed_hold is not None:
                    if not forced:continue
                    hold=fixed_hold;forced=False;reason="fixed_schedule_baseline"
                elif expected>=threshold:
                    hold=cfg["holding_bars"][hi];reason="score_threshold"
                    forced=False
                elif forced:
                    hold=cfg["holding_bars"][hi];reason="forced_exploration_deadline"
                else:continue
                result.append(dict(date=day,slot=slot,signal_t=r["signal_t"],signal_index=r["signal_index"],
                    hold_bars=hold,reason=reason,forced_exploration=forced,expected_net=expected,
                    has_ob=bool(r["snapshots"].get("orderblock")),has_pressure=bool(r["snapshots"].get("pressure"))))
                last_exit=r["signal_t"]+hold*300
                break
    return result


def activity(result,cfg):
    avg=result["trades"]/result["days"]
    active=(result["days"]-result["no_trade_days"])/result["days"]
    return dict(average_trades_per_day=avg,active_day_fraction=active,
                passed=bool(avg>=cfg["minimum_trades_per_day_average"] and active>=cfg["minimum_active_day_fraction"]))


def replay(bars,orders,days,cfg,stress=1):
    sessions={d:g.sort_values("t").to_dict("records") for d,g in bars.groupby("date")}
    cash=cfg["initial_cash"];peak=cash;maxdd=0.;daily=[];ledger=[]
    fee=cfg["commission_bps_per_side"]*stress/10000
    adverse=(cfg["half_spread_bps"]+cfg["slippage_bps_per_side"])*stress/10000
    for day in days:
        a=sessions[day];start=cash;settled=cash;pending=0.;last_exit=-1;daytrades=[]
        for order in sorted([r for r in orders if r["date"]==day],key=lambda r:r["signal_t"]):
            i=order["signal_index"]+1;h=order["hold_bars"]
            assert h in cfg["holding_bars"] and i+h<=len(a)
            assert a[i]["t"]==order["signal_t"] and a[i]["t"]>=last_exit
            ep=a[i]["o"]*(1+adverse);xp=a[i+h-1]["c"]*(1-adverse)
            qty=min(math.floor(min(start*cfg["allocation_fraction"],settled)/(ep*(1+fee))),
                    math.floor(a[i-1]["v"]*cfg["max_participation"]))
            if qty<1:continue
            paid=qty*ep*(1+fee);received=qty*xp*(1-fee);settled-=paid
            assert settled>=-1e-7
            for b in a[i:i+h]:
                equity=settled+pending+qty*b["l"]*(1-adverse)*(1-fee)
                maxdd=max(maxdd,(peak-equity)/peak)
            pending+=received;equity=settled+pending;peak=max(peak,equity)
            maxdd=max(maxdd,(peak-equity)/peak)
            fees=qty*(ep+xp)*fee
            friction=qty*(ep-a[i]["o"]+a[i+h-1]["c"]-xp)
            gross=qty*(a[i+h-1]["c"]-a[i]["o"])
            last_exit=a[i+h-1]["t"]+300
            tr={**order,"shares":qty,"entry_t":a[i]["t"],"exit_t":last_exit,"hold_minutes":h*5,
                "fill_entry":ep,"fill_exit":xp,"raw_entry":a[i]["o"],"raw_exit":a[i+h-1]["c"],
                "gross_pnl":gross,"fees":fees,"friction":friction,"net_pnl":received-paid,
                "settled_remaining":settled,"unsettled_proceeds":pending,"equity":equity}
            ledger.append(tr);daytrades.append(tr)
        cash=settled+pending;net=cash-start
        gross=sum(t["gross_pnl"] for t in daytrades);fees=sum(t["fees"] for t in daytrades)
        friction=sum(t["friction"] for t in daytrades)
        assert abs(net-(gross-fees-friction))<1e-6 and cash>=0
        status="capital_exhausted" if cash==0 else "insufficient_for_one_share" if cash<min(b["o"] for b in a)*(1+adverse)*(1+fee) else "active"
        daily.append(dict(date=day,start_equity=start,end_equity=cash,gross_pnl=gross,fees=fees,
            friction=friction,net_pnl=net,trades=len(daytrades),target_met=bool(net>=cfg["daily_target"]),
            spent_settled_cash=start-settled,forced_trades=sum(t["forced_exploration"] for t in daytrades),status=status))
    hits=sum(d["target_met"] for d in daily)
    return dict(start_equity=cfg["initial_cash"],end_equity=cash,days=len(days),trades=len(ledger),
        net_pnl=cash-cfg["initial_cash"],gross_pnl=sum(t["gross_pnl"] for t in ledger),fees=sum(t["fees"] for t in ledger),
        friction=sum(t["friction"] for t in ledger),target_days=hits,daily_target_rate=hits/len(days),
        mean_daily_pnl=(cash-cfg["initial_cash"])/len(days),best_day=max(d["net_pnl"] for d in daily),
        worst_day=min(d["net_pnl"] for d in daily),no_trade_days=sum(d["trades"]==0 for d in daily),
        low_mark_drawdown=maxdd,forced_trades=sum(t["forced_exploration"] for t in ledger),
        forced_net_pnl=sum(t["net_pnl"] for t in ledger if t["forced_exploration"]),
        score_selected_net_pnl=sum(t["net_pnl"] for t in ledger if not t["forced_exploration"]),
        status=daily[-1]["status"],daily=daily,ledger=ledger)
