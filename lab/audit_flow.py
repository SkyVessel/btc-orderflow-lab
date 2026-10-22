"""Audit public aggregate trades against minute taker volumes; no outcome-based sampling."""
import argparse
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from .absorption import signals
from .research import get_public
from .train_v3 import load_bars


def summarize_trades(rows,start,end):
    selected=[r for r in rows if start<=int(r["T"])<end]
    ids=[int(r["a"]) for r in selected]
    if len(ids)!=len(set(ids)) or ids!=sorted(ids):
        raise ValueError("Duplicate or unordered aggregate trade IDs")
    if any(b!=a+1 for a,b in zip(ids,ids[1:])):
        raise ValueError("Aggregate trade ID gap")
    total=sum((Decimal(r["q"]) for r in selected),Decimal(0))
    buy=sum((Decimal(r["q"]) for r in selected if r["m"] is False),Decimal(0))
    return dict(aggregate_trades=len(selected),individual_matches=sum(int(r["l"])-int(r["f"])+1 for r in selected),
                volume=float(total),taker_buy=float(buy),delta=float(2*buy-total))


def audit_minute(b,directory,direct=False):
    out=Path(directory)/str(b["t"])
    out.mkdir(parents=True,exist_ok=True)
    params=dict(symbol="BTCUSDT",startTime=b["t"],endTime=b["t"]+59999,limit=1000)
    rows,provenance=[],[]
    for page in range(100):
        path=out/f"{page:03d}.json"
        meta=path.with_suffix(".meta")
        if path.exists() and meta.exists():
            raw=path.read_bytes();record=json.loads(meta.read_text())
            if hashlib.sha256(raw).hexdigest()!=record["sha256"]:
                raise ValueError("Cached trade response checksum mismatch")
            batch=json.loads(raw)
        else:
            batch,raw,url=get_public("aggTrades",params,direct)
            record=dict(url=url,retrieved_utc=datetime.now(timezone.utc).isoformat(),
                        sha256=hashlib.sha256(raw).hexdigest(),rows=len(batch),file=str(path.name))
            path.write_bytes(raw);meta.write_text(json.dumps(record,indent=2))
        provenance.append(record)
        if not batch:
            break
        if rows and int(batch[0]["a"])<=int(rows[-1]["a"]):
            raise ValueError("Pagination did not advance")
        rows.extend(batch)
        if int(batch[-1]["T"])>=b["t"]+60000 or len(batch)<1000:
            break
        params=dict(symbol="BTCUSDT",fromId=int(batch[-1]["a"])+1,limit=1000)
    else:
        raise ValueError("Audit window exceeded pagination bound; incomplete audit")
    result=summarize_trades(rows,b["t"],b["t"]+60000)
    volume_error=result["volume"]-b["v"]
    buy_error=result["taker_buy"]-b["buy"]
    return dict(minute_ms=b["t"],utc=datetime.fromtimestamp(b["t"]/1000,timezone.utc).isoformat(),
                expected=dict(volume=b["v"],taker_buy=b["buy"],delta=2*b["buy"]-b["v"],individual_matches=b["n"]),
                observed=result,volume_error=volume_error,taker_buy_error=buy_error,
                passed=abs(volume_error)<1e-6 and abs(buy_error)<1e-6,
                provenance=provenance,
                scope="Aggregate fills at same price/time/taker order; not individual orders or historical L2")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",default="data/history-v0.3")
    p.add_argument("--protocol",default="research/v0.3/protocol.json")
    p.add_argument("--output",default="research/v0.3/flow-audit.json")
    p.add_argument("--raw-output",default="data/flow-audit-v0.3")
    p.add_argument("--direct-network",action="store_true")
    args=p.parse_args()
    if Path(args.output).exists():
        raise ValueError("Do not overwrite an audit")
    protocol=json.loads(Path(args.protocol).read_text())
    bars,_=load_bars(args.data)
    candidates=signals(bars,protocol)
    events=sorted({r["event_start_ms"] for r in candidates})
    n=min(protocol["audit_minutes"],len(events))
    event_starts=[events[round(i*(len(events)-1)/max(1,n-1))] for i in range(n)]
    by_time={b["t"]:b for b in bars}
    # Evenly spaced event timestamps, then strongest selling minute within that event.
    # No future price, trade outcome, or realized return is consulted.
    chosen=[min([by_time[t+j*60000] for j in range(5)],key=lambda b:(2*b["buy"]-b["v"])/max(b["v"],1e-12)) for t in event_starts]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda b:audit_minute(b,args.raw_output,args.direct_network),chosen))
    report=dict(kind="Raw aggregate-trade vs minute-volume audit",selection="Evenly spaced signal events, strongest selling minute, no labels used",
                all_passed=bool(results) and all(r["passed"] for r in results),minutes=results,
                source="https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints")
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({"minutes":len(results),"all_passed":report["all_passed"],
                      "aggregate_trades":sum(r["observed"]["aggregate_trades"] for r in results)},indent=2))
    if not report["all_passed"]:
        raise SystemExit("Volume audit failed; investigate before interpreting backtest")


if __name__=="__main__":
    main()
