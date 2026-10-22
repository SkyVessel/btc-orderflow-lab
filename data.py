"""Read immutable public snapshots; preserve availability and source limitations."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def yahoo(path):
    data = json.loads(Path(path).read_text())["chart"]
    if data.get("error"):
        raise ValueError(data["error"])
    r = data["result"][0]
    assert r["meta"]["symbol"] == "BITX"
    x = pd.DataFrame(r["indicators"]["quote"][0])
    x["t"] = np.asarray(r["timestamp"], dtype=np.int64)
    return x.rename(columns=dict(open="o", high="h", low="l", close="c", volume="v")), r


def valid_ohlc(x):
    cols = ["o", "h", "l", "c", "v"]
    assert np.isfinite(x[cols].to_numpy()).all(), "Non-finite OHLCV"
    assert (x[["o", "h", "l", "c"]] > 0).all().all()
    assert (x.v >= 0).all()
    assert (x.h + 1e-6 >= x[["o", "l", "c"]].max(axis=1)).all()
    assert (x.l - 1e-6 <= x[["o", "h", "c"]].min(axis=1)).all()
    assert not x.t.duplicated().any()


def local_time(x):
    times = pd.to_datetime(x.t, unit="s", utc=True).dt.tz_convert("America/New_York")
    x["date"] = times.dt.strftime("%Y-%m-%d")
    x["minute"] = times.dt.hour * 60 + times.dt.minute
    return x


def prepare(cfg):
    target = ROOT / "data"
    manifest_path = target / "manifest.json"
    raw_files = [target / "raw/BITX_60d_5m.json", target / "raw/BITX_3mo_1d.json"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["bitx_sources"]:
            assert sha(ROOT / record["path"]) == record["sha256"]
    else:
        manifest = {"bitx_sources": [dict(
            path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size,
            retrieved_utc=datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
            url="https://query2.finance.yahoo.com/v8/finance/chart/BITX?" +
                ("range=60d&interval=5m" if "5m" in p.name else "range=3mo&interval=1d") +
                "&includePrePost=false&events=div%2Csplits",
            transport="curl public endpoint; no credentials", original_publication_time=None,
            vintage="current retrospective snapshot, revision history unavailable") for p in raw_files]}
    x, raw = yahoo(raw_files[0])
    valid_ohlc(x)
    x = local_time(x)
    removed = x[(x.minute < 570) | (x.minute >= 960) | (x.date >= cfg["as_of_exclusive"])]
    x = x[(x.minute >= 570) & (x.minute < 960) & (x.date < cfg["as_of_exclusive"])].copy()
    periods = {datetime.fromtimestamp(p[0]["start"], timezone.utc).astimezone(
        __import__("zoneinfo").ZoneInfo("America/New_York")).strftime("%Y-%m-%d"):
        (p[0]["start"], p[0]["end"]) for p in raw["meta"]["tradingPeriods"]}
    for date, g in x.groupby("date"):
        start, end = periods[date]
        assert list(g.t) == list(range(start, end, 300)), f"Incomplete session {date}"
    days = sorted(x.date.unique())
    assert len(days) == sum(cfg[k] for k in ["train_sessions", "validation_sessions", "test_sessions"])
    # Cross-check daily OHLC against a separate endpoint response; volume can include auctions.
    daily, dr = yahoo(raw_files[1])
    daily = local_time(daily).set_index("date")
    checks = []
    for date, g in x.groupby("date"):
        d = daily.loc[date]
        vals = dict(o=g.o.iloc[0], h=g.h.max(), l=g.l.min(), c=g.c.iloc[-1])
        errors = {k: float(vals[k] / d[k] - 1) if pd.notna(d[k]) else None for k in vals}
        checks.append(dict(date=date, relative_ohlc_errors=errors,
                           intraday_volume=int(g.v.sum()), daily_volume=int(d.v)))
    # Material discrepancies must be investigated, not silently trained through.
    max_error = max(abs(e) for r in checks for e in r["relative_ohlc_errors"].values() if e is not None)
    assert max_error < 0.01, f"Daily-vs-intraday OHLC discrepancy {max_error}"
    btc_root = ROOT.parent / "btc-orderflow-lab/data"
    seen, records = {}, []
    start_ms = int(pd.Timestamp(cfg["btc_pretrain_start"], tz="UTC").timestamp()*1000)
    end_ms = int(pd.Timestamp(cfg["as_of_exclusive"], tz="UTC").timestamp()*1000)
    for name in ["history-v0.3-year", "history-v0.2", "history-initial"]:
        folder = btc_root / name
        source = json.loads((folder / "manifest.json").read_text())
        for s in source["snapshots"]:
            path = folder / s["file"]
            assert sha(path) == s["sha256"], str(path)
            for row in json.loads(path.read_bytes()):
                if start_ms <= row[0] < end_ms:
                    assert row[0] not in seen, "Duplicate BTC minute"
                    seen[row[0]] = [row[0]//1000] + [float(row[k]) for k in [1,2,3,4,5,7,8,9]]
        records.append(dict(path=str((folder/"manifest.json").relative_to(ROOT.parent)),
                            sha256=sha(folder/"manifest.json"), rows=source["rows"],
                            source="Binance BTCUSDT spot 1m", original_publication_time=None,
                            reused=True))
    assert sorted(seen) == list(range(start_ms, end_ms, 60000)), "BTC minute gap"
    b = pd.DataFrame(list(seen.values()), columns=["t","o","h","l","c","v","q","n","buy"])
    valid_ohlc(b)
    assert ((b.buy >= 0) & (b.buy <= b.v+1e-8)).all()
    minute_count = len(b)
    b["bucket"] = b.t // 300 * 300
    grouped = b.sort_values("t").groupby("bucket")
    assert (grouped.size() == 5).all()
    b = grouped.agg(o=("o","first"),h=("h","max"),l=("l","min"),c=("c","last"),
                    v=("v","sum"),q=("q","sum"),n=("n","sum"),buy=("buy","sum")).reset_index().rename(columns={"bucket":"t"})
    b = local_time(b)
    valid_ohlc(b)
    assert set(x.t).issubset(set(b.t)), "BTC alignment gap"
    manifest.update(btc_reused_sources=records, bitx_rows=len(x), bitx_sessions=days,
                    bitx_events=raw.get("events",{}), btc_1m_rows=minute_count, btc_5m_rows=len(b),
                    discarded_bitx_rows=removed[["t","date","minute"]].to_dict("records"),
                    historical_quotes_available=False, historical_L2_available=False,
                    etf_taker_side_available=False, btc_quote_currency="USDT", etf_quote_currency="USD",
                    cross_check={"max_relative_ohlc_error": max_error,"daily":checks,
                    "missing_daily_fields":[{"date":r["date"],"field":k} for r in checks for k,v in r["relative_ohlc_errors"].items() if v is None],
                    "note":"Same provider separate interval; not independent exchange verification. Intraday volume may omit auction trades."})
    dump(manifest_path, manifest)
    x.to_csv(target/"bitx_5m.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    b.to_csv(target/"btc_5m.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    return x.reset_index(drop=True), b.reset_index(drop=True), manifest
