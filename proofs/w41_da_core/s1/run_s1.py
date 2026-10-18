"""S1 shared consumers on the w41-da-core candidate snapshot, memory-bounded.

One pytest process accumulates compiled programs (12 GB cgroup OOM after 30 v013 nodes), so the light files run
in one process and the heavy files in fresh processes of CHUNK nodes each, strictly sequentially (one RAM slot).
Writes one junit per process and S1_SUMMARY.json. Run from the snapshot root.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

OUT = Path(sys.argv[1])
CHUNK = 8
HEAVY = ["tests/test_v013_operational_smoke.py", "tests/test_v060_physics_dispatch.py",
         "tests/test_m7_restart_checkpoint_roundtrip.py"]
LIGHT = ["tests/test_cli.py", "tests/test_scheme_catalog_fail_closed.py", "tests/test_v034_implemented_notes.py",
         "tests/test_namelist_binding.py", "tests/test_namelist_check.py", "tests/test_netcdf_process_lock.py",
         "tests/contracts/", "tests/test_v018_mp_family_fail_closed.py"]
BASE = [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider"]


def pytest(args, tag):
    junit = OUT / f"junit_{tag}.xml"
    with open(OUT / f"log_{tag}.txt", "w") as log:
        rc = subprocess.run([*BASE, "--basetemp", str(OUT / "pytest" / tag), f"--junitxml={junit}", *args],
                            stdout=log, stderr=subprocess.STDOUT, timeout=5400).returncode
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    counts = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
    bad = [f"{c.get('classname')}::{c.get('name')}" for c in suite.iter("testcase")
           if c.find("failure") is not None or c.find("error") is not None]
    print(tag, rc, counts, bad[:5], flush=True)
    return {"tag": tag, "rc": rc, **counts, "bad": bad}


def main():
    (OUT / "pytest").mkdir(parents=True, exist_ok=True)       # pytest needs the --basetemp parent (E202)
    collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                                *HEAVY], capture_output=True, text=True).stdout   # one -q: node ids, not counts
    nodes = [line.strip() for line in collected.splitlines() if "::" in line]
    if len(nodes) != 110:
        raise SystemExit(f"expected 110 heavy nodes, collected {len(nodes)}")
    runs = [pytest(LIGHT, "light")]
    for i in range(0, len(nodes), CHUNK):
        runs.append(pytest(nodes[i:i + CHUNK], f"heavy{i // CHUNK:02d}"))
    total = {k: sum(r[k] for r in runs) for k in ("tests", "failures", "errors", "skipped")}
    summary = {"schema": "w41_da_core.s1_consumers.v1", "heavy_nodes": len(nodes), "chunk": CHUNK, "total": total,
               "bad": [b for r in runs for b in r["bad"]], "runs": runs}
    (OUT / "S1_SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("heavy_nodes", "total", "bad")}), flush=True)


if __name__ == "__main__":
    main()
