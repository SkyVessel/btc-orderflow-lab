"""Fetch a NEW public data vintage; never claim it reproduces an old snapshot."""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import dump, sha, yahoo, local_time, valid_ohlc


def download(url, path, direct):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with opener.open(request, timeout=30) as response:
        body = response.read()
    json.loads(body)
    path.write_bytes(body)
    return dict(file=path.name, url=url, sha256=sha(path), received_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--direct-network", action="store_true", help="Ignore proxy settings for this process only")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    raw = out / "raw"; raw.mkdir()
    source = []
    path = raw / "BITX_60d_5m.json"
    source.append(download("https://query2.finance.yahoo.com/v8/finance/chart/BITX?range=60d&interval=5m&includePrePost=false&events=div%2Csplits", path, args.direct_network))
    x, payload = yahoo(path)
    if payload["meta"]["dataGranularity"] != "5m":
        raise ValueError("Expected 5-minute BITX data")
    x = local_time(x)
    today = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    x = x[(x.date < today) & (x.minute >= 570) & (x.minute < 960)].copy()
    # This baseline evaluates complete regular 390-minute sessions only; exclusions are explicit.
    excluded = []
    keep = []
    for day, group in x.groupby("date"):
        group = group.sort_values("t")
        if len(group) != 78 or group[["o", "h", "l", "c", "v"]].isna().any().any() or not (np.diff(group.t) == 300).all():
            excluded.append(str(day)); continue
        valid_ohlc(group); keep.append(group)
    if not keep:
        raise ValueError("No complete sessions")
    x = pd.concat(keep, ignore_index=True)
    start = int(x.t.min() - 86400) * 1000
    end = int(x.t.max() + 300) * 1000
    rows = []
    while start < end:
        query = urllib.parse.urlencode(dict(symbol="BTCUSDT", interval="5m", startTime=start, endTime=end-1, limit=1000))
        path = raw / f"BTCUSDT_{start}.json"
        source.append(download("https://data-api.binance.vision/api/v3/klines?" + query, path, args.direct_network))
        batch = json.loads(path.read_text())
        if not isinstance(batch, list) or not batch:
            raise ValueError("BTC history unavailable or incomplete")
        if int(batch[-1][0]) < start:
            raise ValueError("BTC pagination did not advance")
        rows.extend(batch); start = int(batch[-1][0]) + 300000
        time.sleep(.1)
    b = pd.DataFrame([dict(t=int(r[0])//1000, o=float(r[1]), h=float(r[2]), l=float(r[3]), c=float(r[4]), v=float(r[5]), buy=float(r[9])) for r in rows])
    b = b.sort_values("t").reset_index(drop=True)
    valid_ohlc(b)
    if not (np.diff(b.t) == 300).all() or not set(x.t).issubset(set(b.t)):
        raise ValueError("BTC context gaps")
    if not ((b.buy >= 0) & (b.buy <= b.v + 1e-8)).all():
        raise ValueError("Invalid taker-buy volume")
    x.to_csv(out / "bitx_5m.csv.gz", index=False)
    b.to_csv(out / "btc_5m.csv.gz", index=False)
    dump(out / "manifest.json", dict(sources=source, bitx_sessions=int(x.date.nunique()), bitx_dates=[x.date.min(), x.date.max()], excluded_sessions=excluded,
        files={p.name:sha(p) for p in out.glob("*.csv.gz")},
        status="new current-vintage retrospective snapshot; not pristine holdout; no historical quotes",
        historical_data_already_used_through="2026-10-21", early_closes="excluded from this baseline; must be accounted for separately in prospective evaluation"))
    print(json.dumps(dict(output=str(out), sessions=int(x.date.nunique()), excluded=excluded)))


if __name__ == "__main__":
    main()
