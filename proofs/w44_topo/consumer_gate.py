"""Run each selected existing consumer in a fresh bounded CPU process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True, type=Path)
    p.add_argument("--files", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.tree, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.tree, text=True).strip()
    files = sorted(set(args.files.read_text().splitlines()))
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = args.out / "receipt.json"
    rows = json.loads(receipt.read_text())["rows"] if receipt.exists() else []
    done = {r["file"] for r in rows}
    for file in files:
        if file in done:
            continue
        if Path("/tmp/wrf_gpu2_quiet").exists():
            print("QUIET; stopped before next consumer", flush=True)
            break
        label = file.replace("/", "__").removesuffix(".py")
        started = time.monotonic()
        with (args.out / (label + ".log")).open("w") as log:
            try:
                result = subprocess.run([sys.executable, "-m", "pytest", "-q", file,
                    "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/w44-topo/pytest",
                    "--junitxml", str(args.out / (label + ".xml"))], cwd=args.tree,
                    stdout=log, stderr=subprocess.STDOUT, timeout=180)
                rc = result.returncode
            except subprocess.TimeoutExpired:
                rc = 124
        row = {"file": file, "rc": rc, "wall_s": round(time.monotonic() - started, 3)}
        rows.append(row)
        receipt.write_text(json.dumps({"source": pin, "expected": len(files), "rows": rows}, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    print("completed", len(rows), "of", len(files), "nonzero", sum(r["rc"] != 0 for r in rows), flush=True)


if __name__ == "__main__":
    main()
