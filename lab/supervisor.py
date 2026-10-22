"""Restart the same paper runner; never resets its ledger or model."""
import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", default="runtime/v0.1")
    p.add_argument("--model", default="research/initial/model.json")
    p.add_argument("--config", default="config.json")
    p.add_argument("--direct-network", action="store_true")
    args = p.parse_args()
    out = Path(args.runtime)
    out.mkdir(parents=True, exist_ok=True)
    from .live import single_runner
    # A separate supervisor lock prevents two restart loops competing for the runner.
    with single_runner(out / "supervisor"):
        child, stopping = None, False

        def stop(*_):
            nonlocal stopping
            stopping = True
            if child and child.poll() is None:
                child.send_signal(signal.SIGINT)

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        cmd = [sys.executable, "-m", "lab", "run", "--runtime", args.runtime,
               "--model", args.model, "--config", args.config]
        if args.direct_network:
            cmd.append("--direct-network")
        failures = 0
        with (out / "supervisor.log").open("a", buffering=1) as log:
            while not stopping and not (out / "COMPLETED").exists():
                log.write(f"{time.time()} starting frozen paper runner\n")
                started = time.monotonic()
                child = subprocess.Popen(cmd, stdout=log, stderr=log)
                code = child.wait()
                log.write(f"{time.time()} runner exit={code}\n")
                if stopping:
                    break
                failures = failures + 1 if time.monotonic() - started < 60 else 0
                if failures >= 3:
                    log.write("Repeated startup failure: stopped; inspect configuration and network.\n")
                    raise SystemExit(1)
                time.sleep(10)


if __name__ == "__main__":
    main()
