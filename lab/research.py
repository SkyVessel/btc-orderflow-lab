"""Download public bars and run a predeclared time split. No account access."""
import concurrent.futures
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from .core import dataset, digest, fit, kline, replay, utc_day

BASE = "https://data-api.binance.vision/api/v3/"


def get_public(endpoint, params=None, direct=False):
    if endpoint not in {"klines", "aggTrades", "exchangeInfo", "time"}:
        raise ValueError("Read-only market endpoint allowlist")
    url = BASE + endpoint + "?" + urllib.parse.urlencode(params or {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}) if direct else urllib.request.ProxyHandler())
    for attempt in range(4):
        try:
            with opener.open(url, timeout=25) as r:
                raw = r.read()
            return json.loads(raw), raw, url
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            delay = min(30, float(e.headers.get("Retry-After", 2 ** attempt)))
        except (urllib.error.URLError, TimeoutError):
            delay = 2 ** attempt
        if attempt == 3:
            raise RuntimeError(f"Unable to fetch public data: {url}")
        time.sleep(delay)


def download(c, directory, days=60, direct=False, end_utc=None):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    # Complete UTC days only: exclude incomplete day/bar and make boundaries reproducible.
    server, _, _ = get_public("time", direct=direct)
    latest = (int(server["serverTime"]) // 86400000) * 86400000
    end = int(datetime.strptime(end_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()*1000) if end_utc else latest
    if end > latest or days < 1:
        raise ValueError("Only complete historical UTC days are allowed")
    start = end - days * 86400000
    params = [dict(symbol=c["symbol"], interval="1m", startTime=t,
                   endTime=min(end - 1, t + 1000 * 60000 - 1), limit=1000)
              for t in range(start, end, 1000 * 60000)]

    def page(p):
        name = f'bars-{p["startTime"]}.json'
        path, meta = out / name, out / (name + ".meta")
        if path.exists() and meta.exists():
            raw = path.read_bytes()
            record = json.loads(meta.read_text())
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("Cached raw data hash mismatch")
            return json.loads(raw), record
        result, raw, url = get_public("klines", p, direct)
        if not isinstance(result, list):
            raise ValueError(str(result))
        record = dict(file=name, url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                      sha256=hashlib.sha256(raw).hexdigest(), rows=len(result))
        path.write_bytes(raw)
        meta.write_text(json.dumps(record, indent=2))
        return result, record

    rows, manifest = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, (batch, record) in enumerate(pool.map(page, params)):
            rows.extend(batch)
            manifest.append(record)
            if (i + 1) % 10 == 0:
                print(json.dumps({"downloaded_pages": i + 1, "total_pages": len(params)}), flush=True)
    bars = sorted((kline(r) for r in rows if int(r[0]) < end), key=lambda x: x["t"])
    expected = list(range(start, end, 60000))
    if [b["t"] for b in bars] != expected:
        raise ValueError("Incomplete/duplicate/out-of-order minute coverage: do not silently fill gaps")
    payload = dict(source="Binance public REST klines", start_ms=start, end_ms=end,
                   interval="1m", symbol=c["symbol"], rows=len(bars), snapshots=manifest,
                   caveat="Historical aggregated taker volume, no historical L2 or individual big trades")
    (out / "manifest.json").write_text(json.dumps(payload, indent=2))
    (out / "bars.json").write_text(json.dumps(bars))
    return payload


def run_research(c, directory, output):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Research output must be new; retain every attempted version.")
    out.mkdir(parents=True, exist_ok=True)
    data = Path(directory)
    manifest = json.loads((data / "manifest.json").read_text())
    # Verify original responses; build the bars anew, rather than trusting a processed file.
    raw = []
    for m in manifest["snapshots"]:
        b = (data / m["file"]).read_bytes()
        if hashlib.sha256(b).hexdigest() != m["sha256"]:
            raise ValueError("Raw snapshot checksum failed")
        raw.extend(json.loads(b))
    bars = sorted((kline(x) for x in raw), key=lambda x: x["t"])
    start, end = manifest["start_ms"], manifest["end_ms"]
    if [x["t"] for x in bars] != list(range(start, end, 60000)):
        raise ValueError("Coverage failed")
    train_end = start + c["train_days"] * 86400000
    val_end = train_end + c["validation_days"] * 86400000
    if end != val_end + c["test_days"] * 86400000:
        raise ValueError("Data duration differs from preregistered split")
    rows = dataset(bars, c)
    model = fit(rows, c, train_end)
    validations = {s: replay(rows, model, s, c, train_end, val_end) for s in c["strategies"]}
    # Choose once on validation, then evaluate only that chosen model on the untouched test.
    selected = max(c["strategies"], key=lambda s: validations[s]["net_pnl"])
    model["selected"] = selected
    test = replay(rows, model, selected, c, val_end, end)
    stress = replay(rows, model, selected, c, val_end, end, stress=True)
    baseline = replay(rows, model, "breakout", c, val_end, end, filtered=False)
    val = validations[selected]
    ci = test["mean_daily_pnl_ci95"]
    checks = dict(validation_net_positive=val["net_pnl"] > 0,
                  validation_sample=val["trades"] >= c["minimum_validation_trades"],
                  test_sample=test["trades"] >= c["minimum_test_trades"],
                  test_net_positive=test["net_pnl"] > 0,
                  profit_factor=test["profit_factor"] is not None and test["profit_factor"] >= c["minimum_profit_factor"],
                  daily_ci_positive=ci[0] is not None and ci[0] > 0,
                  stress_positive=stress["net_pnl"] > 0,
                  beats_price_baseline=test["net_pnl"] > baseline["net_pnl"],
                  drawdown=test["realized_drawdown"] <= c["max_test_drawdown"])
    model["deploy_enabled"] = all(checks.values())
    model["checks"] = checks
    model["data_hash"] = digest(manifest)
    model["evaluated_through_ms"] = end
    report = dict(kind="historical_bar_replay_not_real_profit", quote_currency=c["quote_currency"],
                  config_hash=digest(c), data=manifest, training_end_ms=train_end,
                  validation_end_ms=val_end, test_end_ms=end, all_candidate_count=len(rows),
                  validation=validations, selected=selected, test=test, stress=stress, baseline=baseline,
                  checks=checks, deploy_enabled=model["deploy_enabled"],
                  limitations=["No historical depth/queue or individual-trade-size features",
                               "Stop first if stop and target occur in the same minute",
                               "Historical spread/slippage are assumptions; live fills use snapshots",
                               "Daily bootstrap is exploratory, not proof of stationarity",
                               "Drawdown measured at exits; intratrade drawdown may be larger"])
    (out / "config.json").write_text(json.dumps(c, indent=2))
    (out / "model.json").write_text(json.dumps(model, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "candidate_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    text = ["# 历史回放结果", "", "这是分钟级模拟结果，币种为 USDT；真实成交收益为空。", "",
            f"数据区间：{utc_day(start)} 至 {utc_day(end - 1)}，{len(bars):,} 根完整分钟 K 线。",
            f"训练/验证/留出：{c['train_days']}/{c['validation_days']}/{c['test_days']} 天。", "",
            "| 验证集候选 | 成交数 | 净收益 USDT |", "|---|---:|---:|"]
    text += [f"| {s} | {r['trades']} | {r['net_pnl']:.2f} |" for s, r in validations.items()]
    text += ["", f"选定：{selected}。留出净收益 {test['net_pnl']:.2f} USDT，成交 {test['trades']} 笔。",
             f"滑点加倍：{stress['net_pnl']:.2f}；纯价格基准：{baseline['net_pnl']:.2f}。",
             f"自动主账本准入：{'通过' if model['deploy_enabled'] else '未通过，仅采集行情与候选信号，不运行前瞻交易'}。",
             "", "检查：", json.dumps(checks, ensure_ascii=False, indent=2), "", "限制："]
    text += [f"- {x}" for x in report["limitations"]]
    (out / "REPORT.md").write_text("\n".join(text) + "\n")
    return {k: report[k] for k in ("selected", "checks", "deploy_enabled")}
