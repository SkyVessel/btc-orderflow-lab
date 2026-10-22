"""Causal sell-pressure/price-response proxies, confirmed by a later closed bar."""
import math
from .core import digest
from .train_v2 import aggregate

FEATURES = ["event_delta", "log_event_relative_sell", "event_close_change_atr", "event_close_location",
            "event_lower_wick", "event_low_vs_prior_low_atr", "confirmation_delta", "log_confirmation_relvol",
            "confirmation_rise_atr", "confirmation_close_location", "trend_efficiency", "atr_percent",
            "distance_vwap_atr", "momentum_60m_atr", "fee_to_stop", "confirmation_wait_bars",
            "hour_sin", "hour_cos", "is_exhaustion"]


def context(b, old):
    recent = old[-12:]
    atr = sum(max(x["h"]-x["l"], abs(x["h"]-y["c"]), abs(x["l"]-y["c"]))
              for y,x in zip(old[-21:-1],old[-20:]))/20
    path = sum(abs(x["c"]-y["c"]) for y,x in zip(old[-13:-1],recent))
    return dict(atr=atr, prior_low=min(x["l"] for x in recent),
                er=(old[-1]["c"]-old[-13]["c"])/max(path,1e-12),
                vwap=sum(x["q"] for x in recent)/max(sum(x["v"] for x in recent),1e-12),
                delta=(2*b["buy"]-b["v"])/max(b["v"],1e-12),
                relsell=(b["v"]-b["buy"])/max(sum(x["v"]-x["buy"] for x in old[-20:])/20,1e-12),
                relvol=b["v"]/max(sum(x["v"] for x in old[-20:])/20,1e-12),
                location=(b["c"]-b["l"])/max(b["h"]-b["l"],1e-12))


def event_types(b, f, p):
    if f["atr"] <= 0 or b["v"] <= 0:
        return []
    a, e = p["absorption"], p["exhaustion"]
    result = []
    if (f["delta"] <= a["delta_max"] and f["relsell"] >= a["relative_sell_min"]
            and f["location"] >= a["close_location_min"]
            and b["c"]-b["o"] >= -a["max_close_drop_atr"]*f["atr"]
            and b["l"] <= f["prior_low"]+a["near_low_atr"]*f["atr"]):
        result.append("absorption_confirmed")
    if (f["delta"] <= e["delta_max"] and f["relsell"] >= e["relative_sell_min"]
            and f["location"] >= e["close_location_min"] and b["l"] < f["prior_low"]):
        result.append("exhaustion_reclaim")
    return result


def confirmed(event, b, f, index, p):
    elapsed = index-event["index"]
    limit = p["absorption" if event["strategy"]=="absorption_confirmed" else "exhaustion"]["confirmation_bars"]
    if not 1 <= elapsed <= limit or b["l"] < event["bar"]["l"]-.1*event["f"]["atr"]:
        return False
    if f["delta"] <= 0:
        return False
    if event["strategy"] == "absorption_confirmed":
        return b["c"] > event["bar"]["h"]
    return b["c"] > max(event["bar"]["c"],event["f"]["prior_low"])


def signals(bars, p):
    five = aggregate(bars)
    pending, output, last = [], [], {}
    for i in range(60,len(five)):
        b, old = five[i],five[i-60:i]
        f = context(b,old)
        if f["atr"] <= 0:
            continue
        remaining = []
        for event in pending:
            e, ef, strategy = event["bar"],event["f"],event["strategy"]
            limit = p["absorption" if strategy=="absorption_confirmed" else "exhaustion"]["confirmation_bars"]
            if i-event["index"] > limit or b["l"] < e["l"]-.1*ef["atr"]:
                continue
            if not confirmed(event,b,f,i,p):
                remaining.append(event)
                continue
            if b["t"]-last.get(strategy,0) < p["signal_cooldown_minutes"]*60000:
                continue
            stop = max(p["min_stop_fraction"],p["atr_stop_multiple"]*f["atr"]/b["c"],
                       (b["c"]-e["l"]+.1*ef["atr"])/b["c"])
            if stop > p["max_stop_fraction"]:
                continue
            hour = b["t"]%86400000/86400000*2*math.pi
            x = [ef["delta"],math.log(max(ef["relsell"],1e-8)),(e["c"]-e["o"])/ef["atr"],ef["location"],
                 (min(e["o"],e["c"])-e["l"])/max(e["h"]-e["l"],1e-12),
                 (e["l"]-ef["prior_low"])/ef["atr"],f["delta"],math.log(max(f["relvol"],1e-8)),
                 (b["c"]-e["c"])/ef["atr"],f["location"],f["er"],100*f["atr"]/b["c"],
                 (b["c"]-f["vwap"])/f["atr"],(b["c"]-old[-12]["c"])/f["atr"],
                 2*p["fees_per_side"]/stop,i-event["index"],math.sin(hour),math.cos(hour),float(strategy=="exhaustion_reclaim")]
            output.append(dict(id=digest([p["version"],strategy,b["t"]])[:24],strategy=strategy,signal_ms=b["t"]+300000,
                               event_start_ms=e["t"],event_end_ms=e["t"]+300000,minute_index=b["minute_index"],
                               stop_fraction=stop,target_fraction=p["target_r"]*stop,x=x))
            last[strategy] = b["t"]
        pending = remaining
        for strategy in event_types(b,f,p):
            pending.append(dict(strategy=strategy,bar=b,f=f,index=i))
    return output
