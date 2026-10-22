"""Supplement hourly/daily history without pretending it is intraday execution data."""
import sys
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import sha,dump,yahoo,local_time,valid_ohlc


def load(cfg):
    raw=ROOT/"data/v0.2/raw";records=[]
    urls={
        "BITX_2y_1h.json":"range=2y&interval=1h",
        "BITX_since_inception_1d.json":"period1=1687824000&period2=1792627200&interval=1d",
        "BITX_6mo_5m_probe.json":"range=6mo&interval=5m",
        "BITX_max_1d.json":"range=max&interval=1d"}
    for file,args in urls.items():
        p=raw/file;d=json.loads(p.read_text())["chart"]
        r=d.get("result")
        records.append(dict(file=file,sha256=sha(p),bytes=p.stat().st_size,
            retrieved_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),
            url="https://query2.finance.yahoo.com/v8/finance/chart/BITX?"+args+"&includePrePost=false&events=div%2Csplits",
            error=d.get("error"),returned_interval=r[0]["meta"].get("dataGranularity") if r else None,
            usage="rejected_wrong_interval" if file=="BITX_max_1d.json" else "recorded"))
    daily,dr=yahoo(raw/"BITX_since_inception_1d.json")
    hourly,hr=yahoo(raw/"BITX_2y_1h.json")
    assert dr["meta"]["dataGranularity"]=="1d"
    assert hr["meta"]["dataGranularity"]=="1h"
    daily=local_time(daily);hourly=local_time(hourly)
    discarded=hourly[hourly.date>=cfg["as_of_exclusive"]]
    hourly=hourly[(hourly.date<cfg["as_of_exclusive"]) & (hourly.minute>=570) & (hourly.minute<960)].copy()
    daily=daily[daily.date<cfg["as_of_exclusive"]].copy()
    # Current vintage total-return adjustment prevents distribution gaps from generating zones.
    daily["adjclose"]=np.asarray(dr["indicators"]["adjclose"][0]["adjclose"])[:len(daily)]
    factor=(daily.set_index("date").adjclose/daily.set_index("date").c).to_dict()
    missing=daily[daily[["o","h","l","c","adjclose"]].isna().any(axis=1)]
    assert not missing[missing.date<cfg["pretrain_end_exclusive"]].shape[0], "Missing daily adjustment before pretraining cutoff"
    # Quarantine whole incomplete sessions, including malformed early-close placeholders.
    incomplete_dates=sorted(hourly.loc[hourly[["o","h","l","c","v"]].isna().any(axis=1),"date"].unique())
    quarantined_rows=len(hourly[hourly.date.isin(incomplete_dates)])
    hourly=hourly[~hourly.date.isin(incomplete_dates)].copy()
    valid_ohlc(hourly)
    counts=hourly.groupby("date").size()
    assert (counts<=7).all() and (counts>=4).all()
    pre=hourly[hourly.date<cfg["pretrain_end_exclusive"]].copy()
    pre["factor"]=pre.date.map(factor)
    assert np.isfinite(pre.factor).all() and (pre.factor>0).all()
    for k in ["o","h","l","c"]:pre[k]*=pre.factor
    # Reset across weeks and quarantined days; no labels bridge missing sessions.
    pre["segment"]=pd.to_datetime(pre.date).dt.strftime("%G-W%V")+"-"+pre.date.map(
        lambda d:str(sum(q<d for q in incomplete_dates)))
    valid_ohlc(pre)
    # No duplicated original data; v0.1 source hashes must still match.
    old=json.loads((ROOT/"data/manifest.json").read_text())
    for record in old["bitx_sources"]:assert sha(ROOT/record["path"])==record["sha256"]
    intraday=pd.read_csv(ROOT/"data/bitx_5m.csv.gz")
    btc=pd.read_csv(ROOT/"data/btc_5m.csv.gz")
    valid_ohlc(intraday);valid_ohlc(btc)
    days=sorted(intraday.date.unique());assert len(days)==sum(cfg["split_sessions"])
    # Compare the hour source with contemporaneous 5m aggregation where available.
    idx=hourly.set_index("t");comparisons=[]
    for date,g in intraday.groupby("date"):
        for start,h in g.groupby((g.minute-570)//60):
            ts=int(h.t.iloc[0])
            if ts not in idx.index:continue
            one=idx.loc[ts]
            delta=max(abs(float(a)/float(one[k])-1) for k,a in
                [("o",h.o.iloc[0]),("h",h.h.max()),("l",h.l.min()),("c",h.c.iloc[-1])])
            comparisons.append(delta)
    assert comparisons and max(comparisons)<.015, "Material hourly/5m discrepancy"
    info=dict(sources=records,hourly_rows=len(hourly),hourly_sessions=len(counts),
        hourly_dates=[hourly.date.min(),hourly.date.max()],daily_rows=len(daily),daily_dates=[daily.date.min(),daily.date.max()],
        hourly_pretrain_rows=len(pre),hourly_pretrain_dates=[pre.date.min(),pre.date.max()],
        intraday_rows=len(intraday),intraday_dates=[days[0],days[-1]],new_intraday_dates=0,
        missing_daily_fields=missing.where(pd.notna(missing),None).astype(object).replace({np.nan:None}).to_dict("records"),
        discarded_partial_hour_rows=len(discarded),max_hourly_vs_5m_relative_ohlc_error=max(comparisons),
        quarantined_hourly_dates=incomplete_dates,quarantined_hourly_rows=quarantined_rows,
        reused_processed_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/"data/bitx_5m.csv.gz",ROOT/"data/btc_5m.csv.gz"]},
        note="Hourly data only pretrains scale-normalized regions, labels use next six trading bars including overnight; never used as 5-30 minute fills. Daily data only supplies corporate-action adjustment.",
        point_in_time_limitation="Yahoo current revised/adjusted vintage; no historical quote or revision archives")
    dump(ROOT/"data/v0.2/manifest.json",info)
    return intraday,btc,pre.reset_index(drop=True),info
