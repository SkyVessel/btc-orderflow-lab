"""Causal multi-touch zones and dense state/action training labels."""
import numpy as np

REGION_FEATURES=["body_atr","range_atr","atr_pct","log_relative_volume","r1_atr","r3_atr","r6_atr",
    "range_position","efficiency","clock","upper_wick_atr","lower_wick_atr",
    "kind","width_atr","age","location_atr","touch_count"]
STATE_FEATURES=REGION_FEATURES[:12]+["btc_r1","btc_r3","btc_r6","btc_d1","btc_d3","btc_d6","btc_atrpct","btc_logv","btc_eff"]+[
    "ob_exists","ob_probability","ob_distance","ob_width","ob_age",
    "pressure_exists","pressure_probability","pressure_distance","pressure_width","pressure_age",
    "ob_count","pressure_count","relative_btc3","from_open_atr"]


def base_features(a,i):
    b=a[i];prev=a[max(0,i-20):i]
    tr=[max(z["h"]-z["l"],abs(z["h"]-a[j-1]["c"]),abs(z["l"]-a[j-1]["c"]))
        for j,z in enumerate(a[max(1,i-19):i+1],start=max(1,i-19))]
    atr=max(float(np.mean(tr)),1e-9)
    high=max(z["h"] for z in prev[-12:]);low=min(z["l"] for z in prev[-12:])
    path=sum(abs(a[j]["c"]-a[j-1]["c"]) for j in range(i-5,i+1))
    f=[(b["c"]-b["o"])/atr,(b["h"]-b["l"])/atr,100*atr/b["c"],
       np.log((b["v"]+1e-9)/(np.mean([z["v"] for z in prev])+1e-9))]
    f += [(b["c"]-a[i-h]["c"])/atr for h in [1,3,6]]
    f += [(b["c"]-low)/max(high-low,1e-9),(b["c"]-a[i-6]["c"])/max(path,1e-9),
          (b["minute"]-570)/390,(b["h"]-max(b["o"],b["c"]))/atr,(min(b["o"],b["c"])-b["l"])/atr]
    return f,atr


def region_features(base,b,atr,z,i,cfg):
    f=base+[1 if z["kind"]=="orderblock" else -1,(z["high"]-z["low"])/atr,
        (i-z["created_index"])/cfg["region_expiry_bars"],(b["c"]-(z["low"]+z["high"])/2)/atr,
        max(1,z["touches"])/cfg["region_max_touches"]]
    assert len(f)==17 and np.isfinite(f).all()
    return f


def extract(bars,cfg,hourly=False):
    groups="segment" if hourly else "date"
    events,register,states=[],[],[]
    for key,df in bars.groupby(groups,sort=True):
        a=df.sort_values("t").to_dict("records");active=[];known=set()
        for i,b in enumerate(a):
            if i<cfg["region_warmup"]:continue
            base,atr=base_features(a,i)
            duration=1800 if hourly and b["minute"]==930 else (3600 if hourly else 300)
            signal=b["t"]+duration
            for z in active:
                if z["status"]!="active":continue
                if i-z["created_index"]>cfg["region_expiry_bars"]:
                    z["status"]="expired";continue
                touch=(b["l"]<=z["high"] and b["h"]>=z["low"])
                if touch and z["touches"]<cfg["region_max_touches"] and i-z["last_touch_index"]>=cfg["region_touch_cooldown_bars"]:
                    z["touches"]+=1;z["last_touch_index"]=i
                    events.append(dict(id=f"{z['id']}-touch{z['touches']}",zone_id=z["id"],group=key,
                        date=b["date"],kind=z["kind"],touch_number=z["touches"],signal_index=i,signal_t=signal,
                        created_t=z["created_t"],low=z["low"],high=z["high"],reference=b["c"],atr=atr,
                        features=region_features(base,b,atr,z,i,cfg)))
                if ((z["kind"]=="orderblock" and b["c"]<z["low"]-atr) or
                    (z["kind"]=="pressure" and b["c"]>z["high"]+atr)):
                    z["status"]="invalid"
            new=[];look=cfg["region_lookback"]
            if b["c"]>max(z["h"] for z in a[i-look:i]):
                down=next((z for z in reversed(a[i-look:i]) if z["c"]<z["o"]),None)
                if down and down["h"]-down["l"]<=cfg["region_max_width_atr"]*atr:
                    new.append(("orderblock",down["l"],down["h"],down["t"]))
            pivot=a[i-2]
            if all(pivot["h"]>a[j]["h"] for j in [i-4,i-3,i-1,i]):
                half=cfg["pressure_width_atr"]*atr/2
                if b["c"]<pivot["h"]-half:
                    new.append(("pressure",pivot["h"]-half,pivot["h"]+half,pivot["t"]))
            for kind,lo,hi,t in new:
                if (kind,t) in known:continue
                known.add((kind,t))
                z=dict(id=f"{key}-{kind}-{t}",group=key,date=b["date"],kind=kind,low=lo,high=hi,
                    created_t=signal,created_index=i,base_t=t,status="active",touches=0,last_touch_index=-1000)
                active.append(z);register.append(z)
            if not hourly:
                snapshots={}
                counts={}
                for kind in ["orderblock","pressure"]:
                    pool=[z for z in active if z["status"]=="active" and z["kind"]==kind]
                    counts[kind]=len(pool)
                    if pool:
                        z=min(pool,key=lambda z:abs((z["low"]+z["high"])/2-b["c"]))
                        snapshots[kind]=dict(features=region_features(base,b,atr,z,i,cfg),
                            distance=(b["c"]-(z["low"]+z["high"])/2)/atr,width=(z["high"]-z["low"])/atr,
                            age=(i-z["created_index"])/cfg["region_expiry_bars"])
                states.append(dict(date=b["date"],signal_t=signal,signal_index=i,minute=b["minute"]+5,
                    base=base,atr=atr,snapshots=snapshots,counts=counts,
                    return3_pct=(b["c"]/a[i-3]["c"]-1)*100,from_open_atr=(b["c"]-a[0]["o"])/atr))
        for z in active:
            if z["status"]=="active":z["status"]="group_end"
    return events,register,states


def region_labels(events,bars,cfg,hourly=False):
    key="segment" if hourly else "date"
    groups={g:d.sort_values("t").to_dict("records") for g,d in bars.groupby(key)}
    rows=[];excluded=0
    for e in events:
        a=groups[e["group"]];i=e["signal_index"];future=a[i+1:i+1+cfg["label_bars"]]
        if len(future)!=cfg["label_bars"]:
            excluded+=1;continue
        if not hourly:assert all(b["t"]==e["signal_t"]+j*300 for j,b in enumerate(future))
        bullish=e["kind"]=="orderblock"
        target=e["high"]+e["atr"] if bullish else e["low"]-e["atr"]
        stop=e["low"]-e["atr"] if bullish else e["high"]+e["atr"]
        outcome="timeout";ambiguous=False
        if (a[i]["l"]<=stop if bullish else a[i]["h"]>=stop):outcome="failure_on_touch"
        else:
            for b in future:
                fail=b["l"]<=stop if bullish else b["h"]>=stop
                success=b["h"]>=target if bullish else b["l"]<=target
                if fail:outcome="failure";ambiguous=success;break
                if success:outcome="success";break
        rows.append({**e,"label":int(outcome=="success"),"outcome":outcome,"ambiguous":ambiguous,
            "label_end_t":future[-1]["t"]+(1800 if hourly and future[-1]["minute"]==930 else 3600 if hourly else 300)})
    return rows,excluded


def state_matrix(states,context,region_model):
    rows=[]
    for s in states:
        c=context.loc[s["signal_t"]-300]
        f=s["base"]+[c.r1,c.r3,c.r6,c.d1,c.d3,c.d6,c.atrpct,c.logv,c.eff]
        for kind in ["orderblock","pressure"]:
            q=s["snapshots"].get(kind)
            if q:
                p=float(region_model.predict(np.asarray([q["features"]]))[0,0])
                f += [1,p,q["distance"],q["width"],q["age"]]
            else:f += [0,.5,0,0,0]
        f += [s["counts"]["orderblock"]/10,s["counts"]["pressure"]/10,s["return3_pct"]-c.r3,s["from_open_atr"]]
        assert len(f)==35 and np.isfinite(f).all()
        rows.append(f)
    return np.asarray(rows)


def state_labels(states,bars,cfg):
    groups={d:g.sort_values("t").to_dict("records") for d,g in bars.groupby("date")}
    result=[]
    for s in states:
        a=groups[s["date"]];i=s["signal_index"]+1
        if i+max(cfg["holding_bars"])>len(a):continue
        assert a[i]["t"]==s["signal_t"]
        returns=[a[i+h-1]["c"]/a[i]["o"]-1 for h in cfg["holding_bars"]]
        result.append({**s,"gross_returns":returns,"label_end_t":a[i+max(cfg["holding_bars"])-1]["t"]+300})
    return result
