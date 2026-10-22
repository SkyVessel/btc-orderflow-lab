"""Causal numerical zones. Future data are used only by the separate label function."""
import numpy as np
import pandas as pd

FEATURES = ["kind", "width_atr", "age", "location_atr", "body_atr", "range_atr", "log_rel_volume",
            "return1_atr", "return3_atr", "return6_atr", "atr_pct", "range_position",
            "pressure_distance_atr", "ob_distance_atr", "session_fraction", "btc_return1_pct",
            "btc_return3_pct", "btc_return6_pct", "btc_atr_pct", "btc_delta1", "btc_delta3",
            "btc_delta6", "btc_log_rel_volume", "btc_efficiency", "asset_efficiency", "relative_btc3_pct"]


def btc_context(b):
    b=b.sort_values("t").copy()
    prev=b.c.shift(1)
    tr=pd.concat([b.h-b.l,(b.h-prev).abs(),(b.l-prev).abs()],axis=1).max(axis=1)
    out=pd.DataFrame({"t":b.t})
    for h in [1,3,6]:
        out[f"r{h}"]=(b.c/b.c.shift(h)-1)*100
        out[f"d{h}"]=(2*b.buy-b.v).rolling(h).sum()/b.v.rolling(h).sum().clip(lower=1e-9)
    out["atrpct"]=tr.rolling(20).mean()/b.c*100
    out["logv"]=np.log((b.v+1e-9)/(b.v.shift(1).rolling(20).mean()+1e-9))
    out["eff"]=(b.c-b.c.shift(6))/b.c.diff().abs().rolling(6).sum().clip(lower=1e-9)
    return out.set_index("t")


def candidates(bars,context,cfg):
    """Zones are registered after close; only their later first touch is scored."""
    events,register=[],[]
    for date,df in bars.groupby("date",sort=True):
        df=df.sort_values("t");a=df.to_dict("records"); active=[]
        for i,b in enumerate(a):
            if i<20:
                continue
            prev=a[i-20:i]
            tr=[max(z["h"]-z["l"],abs(z["h"]-a[j-1]["c"]),abs(z["l"]-a[j-1]["c"]))
                for j,z in enumerate(a[i-19:i+1],start=i-19)]
            atr=max(float(np.mean(tr)),1e-9)
            ctx=context.loc[b["t"]]
            prior_hi=max(z["h"] for z in prev[-cfg["zone_lookback_bars"]:])
            prior_lo=min(z["l"] for z in prev[-cfg["zone_lookback_bars"]:])
            for z in active:
                if z["status"]!="active":
                    continue
                if i-z["created_index"]>cfg["zone_expiry_bars"]:
                    z["status"]="expired";continue
                touch=b["l"]<=z["high"] and b["h"]>=z["low"]
                if touch:
                    z["status"]="touched";z["touch_t"]=b["t"]+300
                    pressures=[q["low"] for q in active if q["kind"]=="pressure" and q["status"] in ["active","touched"]]
                    obs=[q["high"] for q in active if q["kind"]=="orderblock" and q["status"] in ["active","touched"]]
                    pdist=min((abs(p-b["c"]) for p in pressures),default=5*atr)/atr
                    odist=min((abs(p-b["c"]) for p in obs),default=5*atr)/atr
                    path=sum(abs(a[j]["c"]-a[j-1]["c"]) for j in range(i-5,i+1))
                    f=[1 if z["kind"]=="orderblock" else -1,(z["high"]-z["low"])/atr,
                       (i-z["created_index"])/cfg["zone_expiry_bars"],(b["c"]-(z["low"]+z["high"])/2)/atr,
                       (b["c"]-b["o"])/atr,(b["h"]-b["l"])/atr,
                       np.log((b["v"]+1e-9)/(np.mean([p["v"] for p in prev])+1e-9))]
                    f += [(b["c"]-a[i-h]["c"])/atr for h in [1,3,6]]
                    f += [atr/b["c"]*100,(b["c"]-prior_lo)/max(prior_hi-prior_lo,1e-9),pdist,odist,
                          (b["minute"]-570)/390,ctx.r1,ctx.r3,ctx.r6,ctx.atrpct,
                          ctx.d1,ctx.d3,ctx.d6,ctx.logv,ctx.eff,
                          (b["c"]-a[i-6]["c"])/max(path,1e-9),(b["c"]/a[i-3]["c"]-1)*100-ctx.r3]
                    assert len(f)==len(FEATURES) and np.isfinite(f).all()
                    events.append(dict(id=z["id"],date=date,kind=z["kind"],signal_t=b["t"]+300,
                                       signal_index=i,created_t=z["created_t"],low=z["low"],high=z["high"],
                                       reference=b["c"],atr=atr,features=f))
                elif ((z["kind"]=="orderblock" and b["c"]<z["low"]) or
                      (z["kind"]=="pressure" and b["c"]>z["high"])):
                    z["status"]="invalid_before_touch"
            # Bullish order block = last down candle preceding a confirmed prior-range breakout.
            new=[]
            if b["c"]>prior_hi+cfg["ob_breakout_atr"]*atr:
                base=next((q for q in reversed(a[max(0,i-6):i]) if q["c"]<q["o"]),None)
                if base and base["h"]-base["l"]<=cfg["ob_max_width_atr"]*atr:
                    new.append(("orderblock",base["l"],base["h"],base["t"]))
            # Five-bar swing high is known only at the close two bars AFTER the pivot.
            pivot=a[i-2]
            if all(pivot["h"]>a[j]["h"] for j in [i-4,i-3,i-1,i]):
                half=cfg["zone_width_atr"]*atr/2
                if b["c"]<pivot["h"]-half:
                    new.append(("pressure",pivot["h"]-half,pivot["h"]+half,pivot["t"]))
            for kind,low,high,base_t in new:
                if any(z["kind"]==kind and z["base_t"]==base_t for z in active):
                    continue
                z=dict(id=f"{date}-{kind}-{b['t']}",date=date,kind=kind,low=low,high=high,
                       created_t=b["t"]+300,created_index=i,base_t=base_t,status="active")
                active.append(z);register.append(z)
        for z in active:
            if z["status"]=="active":z["status"]="session_end_untouched"
    return events,register


def labels(events,bars,cfg):
    sessions={d:g.sort_values("t").to_dict("records") for d,g in bars.groupby("date")}
    scored=[];excluded=0
    for e in events:
        a=sessions[e["date"]];i=e["signal_index"]
        future=a[i+1:i+1+cfg["zone_outcome_bars"]]
        if len(future)!=cfg["zone_outcome_bars"] or any(
            b["t"]!=e["signal_t"]+300*j for j,b in enumerate(future)):
            excluded+=1;continue
        direction=1 if e["kind"]=="orderblock" else -1
        target=e["high"]+e["atr"]*cfg["zone_success_atr"] if direction==1 else e["low"]-e["atr"]*cfg["zone_success_atr"]
        stop=e["low"]-e["atr"]*cfg["zone_failure_atr"] if direction==1 else e["high"]+e["atr"]*cfg["zone_failure_atr"]
        outcome="timeout";ambiguous=False
        # A touch bar that already invalidates the zone is a failure, never a fresh opportunity.
        touch=a[i]
        failed=(touch["l"]<=stop if direction==1 else touch["h"]>=stop)
        if failed: outcome="failure_on_touch"
        else:
            for b in future:
                hit_fail=b["l"]<=stop if direction==1 else b["h"]>=stop
                hit_ok=b["h"]>=target if direction==1 else b["l"]<=target
                if hit_fail:
                    outcome="failure";ambiguous=hit_ok;break
                if hit_ok:outcome="success";break
        returns=[future[h-1]["c"]/future[0]["o"]-1 for h in cfg["holding_bars"]]
        scored.append({**e,"label":int(outcome=="success"),"outcome":outcome,
                       "ambiguous":ambiguous,"gross_returns":returns,
                       "label_end_t":future[-1]["t"]+300})
    return scored,excluded
