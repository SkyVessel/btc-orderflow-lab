import argparse
import asyncio
import json
import sqlite3
import time
from pathlib import Path
from .core import load_config
from .research import download, run_research
from .live import Store, report, run


def main():
    p = argparse.ArgumentParser(description="BTC public-data and paper-only trading lab")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("download", "research", "run"):
        x = sub.add_parser(name)
        x.add_argument("--config", default="config.json")
        if name == "download":
            x.add_argument("--output", default="data/history-initial")
            x.add_argument("--days", type=int, default=60)
            x.add_argument("--end-utc", help="Exclusive UTC end date, YYYY-MM-DD; defaults to today")
            x.add_argument("--direct-network", action="store_true")
        if name == "research":
            x.add_argument("--data", default="data/history-initial")
            x.add_argument("--output", default="research/initial")
        if name == "run":
            x.add_argument("--model", default="research/initial/model.json")
            x.add_argument("--runtime", default="runtime/v0.1")
            x.add_argument("--minutes", type=float, default=0, help="0: continuous until stopped")
            x.add_argument("--direct-network", action="store_true")
    x = sub.add_parser("report")
    x.add_argument("--runtime", default="runtime/v0.1")
    x.add_argument("--usd-per-usdt", type=float, default=1.0)
    x = sub.add_parser("expense")
    x.add_argument("--runtime", default="runtime/v0.1")
    x.add_argument("--id", required=True, help="Unique receipt/reference, prevents double charging")
    x.add_argument("--usd", required=True, type=float)
    x.add_argument("--category", required=True, choices=["data", "ai", "hosting", "electricity", "labor", "other"])
    x.add_argument("--note", default="")
    x = sub.add_parser("export")
    x.add_argument("--runtime", default="runtime/v0.1")
    x.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command in ("download", "research", "run"):
        c = load_config(args.config)
    if args.command == "download":
        print(json.dumps(download(c, args.output, args.days, args.direct_network, args.end_utc), indent=2))
    elif args.command == "research":
        print(json.dumps(run_research(c, args.data, args.output), indent=2))
    elif args.command == "run":
        model = json.loads(Path(args.model).read_text())
        asyncio.run(run(c, model, args.runtime, args.minutes, args.direct_network))
    else:
        path = Path(args.runtime) / "ledger.sqlite"
        if not path.exists():
            raise SystemExit("No existing ledger at this runtime")
        store = Store(path)
        if args.command == "expense":
            if args.usd < 0:
                raise SystemExit("Expenses must be nonnegative")
            try:
                store.db.execute("INSERT INTO expenses VALUES(?,?,?,?,?)", (args.id, time.time() * 1000, args.usd, args.category, args.note))
                store.commit()
            except sqlite3.IntegrityError:
                raise SystemExit("Expense id already exists or invalid amount; nothing charged")
            print(json.dumps(report(store, args.runtime), indent=2))
        elif args.command == "report":
            if args.usd_per_usdt <= 0:
                raise SystemExit("Positive FX assumption required")
            print(json.dumps(report(store, args.runtime, args.usd_per_usdt), indent=2))
        elif args.command == "export":
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=False)
            report(store, out)
            # Allowlist structured experiment records; exclude expense notes, credentials, raw feed.
            for table in ("decisions", "trades", "events"):
                rows = store.db.execute(f"SELECT * FROM {table}")
                names = [x[0] for x in rows.description]
                with (out / (table + ".jsonl")).open("w") as f:
                    for row in rows:
                        item = dict(zip(names, row))
                        item["payload"] = json.loads(item["payload"])
                        f.write(json.dumps(item) + "\n")
            (out / "fingerprint.txt").write_text(store.get("fingerprint") + "\n")
            print(str(out.resolve()))
        store.db.close()


if __name__ == "__main__":
    main()
