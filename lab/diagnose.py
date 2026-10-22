"""Post-run diagnostics; no selection, parameter changes, or new holdout claim."""
import argparse
import json
import statistics
from pathlib import Path
from .core import predict, replay


def diagnose(directory):
    root = Path(directory)
    c = json.loads((root / "config.json").read_text())
    model = json.loads((root / "model.json").read_text())
    report = json.loads((root / "report.json").read_text())
    rows = [json.loads(line) for line in (root / "candidate_labels.jsonl").read_text().splitlines()]
    start, middle, end = (report[k] for k in ("training_end_ms", "validation_end_ms", "test_end_ms"))
    ungated = {s: replay(rows, model, s, c, start, middle, filtered=False) for s in c["strategies"]}
    calibration = {}
    for s in c["strategies"]:
        pairs = []
        parent = model["parents"][s]
        base = (parent["wins"] + 1) / (parent["n"] + 2)
        for r in rows:
            if r["strategy"] == s and middle <= r["signal_ms"] < end and r["exit_ms"] <= end:
                p = predict(model, r, c)
                if "p_profit" in p:
                    pairs.append((p["p_profit"], int(r["net_return"] > 0)))
        bins = []
        for i in range(10):
            group = [(p, y) for p, y in pairs if min(9, int(p * 10)) == i]
            if group:
                bins.append(dict(lower=i / 10, n=len(group), mean_p=statistics.mean(p for p, _ in group),
                                 observed=statistics.mean(y for _, y in group)))
        calibration[s] = dict(n=len(pairs),
                              brier=statistics.mean((p - y) ** 2 for p, y in pairs) if pairs else None,
                              base_brier=statistics.mean((base - y) ** 2 for _, y in pairs) if pairs else None,
                              bins=bins, warning="Candidate windows overlap; not independent trades.")
    return dict(kind="post_run_diagnostics_no_retuning", validation_ungated=ungated,
                holdout_candidate_calibration=calibration)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research", default="research/initial")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = diagnose(args.research)
    # Publish aggregate diagnostics; preserve full simulated ledgers in local original output.
    for value in result["validation_ungated"].values():
        value.pop("ledger", None)
    with Path(args.output).open("x") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
