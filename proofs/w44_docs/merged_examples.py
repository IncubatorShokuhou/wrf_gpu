#!/usr/bin/env python3
"""Exercise merged-tool documentation on CPU; no forecast or GPU invocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(ROOT / "src"))
    rows = []

    def command(name, argv):
        if Path("/tmp/wrf_gpu2_quiet").exists():
            raise SystemExit("QUIET")
        start = time.monotonic()
        log = out / f"{name}.log"
        with log.open("w") as f:
            p = subprocess.run([sys.executable, "-m", "gpuwrf", *argv], cwd=ROOT,
                               env=env, stdout=f, stderr=subprocess.STDOUT, timeout=90)
        rows.append({"name": name, "argv": argv, "rc": p.returncode,
                     "elapsed_s": time.monotonic()-start,
                     "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
        if p.returncode:
            raise RuntimeError(f"{name} failed: see {log}")

    case = ROOT / "examples/switzerland_d01"
    spec = out / "overrides.json"
    spec.write_text(json.dumps({"overrides": [{"nest": 1, "field": "TSK", "op": "add",
                                               "values": 0.1, "mask": "land"}]}))
    source_sha = hashlib.sha256((case / "wrfinput_d01").read_bytes()).hexdigest()
    command("override", ["override", "--input-dir", str(case), "--output-dir", str(out / "edited"), "--spec", str(spec)])
    spec_empty = out / "empty.json"
    spec_empty.write_text("{}\n")
    command("override_empty", ["override", "--input-dir", str(case), "--output-dir", str(out / "empty"), "--spec", str(spec_empty)])
    assert hashlib.sha256((out / "empty/wrfinput_d01").read_bytes()).hexdigest() == source_sha
    assert hashlib.sha256((case / "wrfinput_d01").read_bytes()).hexdigest() == source_sha

    variants = out / "variants.json"
    # Same relative-path convention as the documented runs/swiss-variants.json.
    variant_path = os.path.relpath(case / "namelist.input", out)
    variants.write_text(json.dumps({"variants": [{"name": "swiss", "namelist": variant_path}]}))
    command("precompile_queue", ["precompile", "--namelist-variants", str(variants),
            "--input-dir", str(case), "--cache-dir", str(out / "cache"),
            "--work-dir", str(out / "queue"), "--hours", "3"])
    queue = json.loads((out / "queue/report.json").read_text())
    # This command is a CPU plan only; cache must contain no executable.
    assert not any(p.is_file() for p in (out / "cache").rglob("*"))

    rad = out / "rad_case"
    rad.mkdir()
    for name in ("namelist.input", "wrfinput_d01", "wrfbdy_d01"):
        shutil.copy2(case / name, rad / name)
    with (rad / "namelist.input").open("a") as f:
        f.write("\n&gpuwrf\n rad_diag = 1,\n/\n")
    command("raddiag_short_dry", ["run", "--input-dir", str(rad), "--domain", "d01", "--minutes", "18",
            "--output-dir", str(out / "rad_dry"), "--dry-run"])
    # Execute the two literal Python examples from the pages, rather than copies.
    for page in ("docs/soil_porosity.md", "docs/noahmp_land_use.md"):
        for block in re.findall(r"(?ms)^```python\n(.*?)^```", (ROOT / page).read_text()):
            start = time.monotonic()
            subprocess.run([sys.executable, "-c", block], cwd=ROOT, env=env, check=True, timeout=60)
            rows.append({"name": page, "rc": 0, "elapsed_s": time.monotonic()-start,
                         "code_sha256": hashlib.sha256(block.encode()).hexdigest()})
    result = {"status": "PASS", "root": str(ROOT), "source_input_sha256": source_sha,
              "scope": "CPU file operations, queue and dry-run only; no model step/GPU/fidelity claim",
              "rows": rows, "queue_report": queue}
    (out / "receipt.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"PASS {len(rows)} CPU CLI/Python examples; no forecast/GPU")


if __name__ == "__main__":
    main()
