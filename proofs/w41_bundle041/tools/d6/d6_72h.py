"""RW3: frozen D6 + per-frame output integrity of the WN3 0227 3-nest 72 h GPU run vs the ORIGINAL CPU-WRF 72 h.

usage: JAX_PLATFORMS=cpu PYTHONPATH=<P>/src/src python d6_72h.py SOURCE CASE GPU_WRFOUT_DIR [--domains d01 d02 d03]
Same arithmetic as the v0.4.0/B1 release gate: d6/score.py is the SHA-pinned F20 scorer (4c025af8, frozen manifest
9df4fab8; every CPU field and frame, every frame's header/schema/nonfinite audit of BOTH arms) driven with read-only
Dataset reuse like B1's score_with_handles.py (7407b0dd). The ONLY differences: hours=72 and a sealed 219-frame CPU
manifest built here from the guarded staged copies (cpu72_receipt/STAGED_INDEX.json: guard SHA per file; leads 0-24
equal the F20 sealed 24 h manifest). CASE is created (input/ -> case inputs, cpu/ -> staged copies, gpu/ -> GPU dir).
No tolerance, field set or pairing change. Writes CASE/{CPU_OUTPUTS.json,cpu_run.json,compare_dNN.json,D6_dNN.json,
SCORING_IO_dNN.json}.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import netCDF4

assert os.environ.get("JAX_PLATFORMS") == "cpu"
assert not Path("/tmp/wrf_gpu2_quiet").exists(), "QUIET"
HERE = Path(__file__).resolve().parent
SCORER_SHA = "4c025af8382c9626e7f7e0de1e888a91bd0270376f5c837431d1000c70c2c21d"
INDEX = Path("<USER_HOME>/wrf_gpu2_lanes/w41-bundle041/cpu72_receipt/STAGED_INDEX.json")
INPUT = Path("<DATA_ROOT>/wrf_gpu2/scratch/w41-bundle041/cases/wn3_0227")
ORIGINAL = "<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run"

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("case", type=Path)
p.add_argument("gpu", type=Path)
p.add_argument("--domains", nargs="+", default=["d01", "d02", "d03"])
a = p.parse_args()
scorer_path = HERE / "score.py"
assert hashlib.sha256(scorer_path.read_bytes()).hexdigest() == SCORER_SHA, "scorer not the pinned F20 scorer"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


case = a.case
if not (case / "cpu_run.json").exists():
    case.mkdir(parents=True)
    (case / "cpu").mkdir()
    (case / "input").symlink_to(INPUT)
    (case / "gpu").symlink_to(a.gpu.resolve())
    index = json.loads(INDEX.read_text())
    assert not index["sha_mismatch_vs_f20"] and index["sealed24_checked"] == 75
    rows = []
    for dom in ("d01", "d02", "d03"):
        for name, rec in sorted(index["index"][dom].items(), key=lambda kv: kv[1]["lead_h"]):
            staged = Path(rec["staged"])
            digest = sha(staged)
            assert digest == rec["sha256"], f"staged copy changed: {name}"
            (case / "cpu" / name).symlink_to(staged)
            with netCDF4.Dataset(staged) as ds:
                times = ["".join(c.decode() for c in t) for t in ds["Times"][:]]
            rows.append(dict(file=name, bytes=staged.stat().st_size, sha256=digest, times=times))
    body = (json.dumps(rows, indent=1) + "\n").encode()
    (case / "CPU_OUTPUTS.json").write_bytes(body)
    inputs = {f.name: sha(f) for f in sorted(INPUT.iterdir())}
    (case / "cpu_run.json").write_text(json.dumps(dict(
        status="COMPLETE", case="WN3-0227-72h", original=ORIGINAL, frame_count=len(rows),
        input_sha256=inputs, output_manifest_sha256=hashlib.sha256(body).hexdigest(),
        guard_receipt=str(INDEX.parent / "facade_stage.json"),
        scope="Read-only reuse of the historical original CPU-WRF 72 h run (same frames as the v0.3.3 W3P full72 gate); "
              "no CPU model rerun", built_utc=datetime.now(timezone.utc).isoformat()), indent=1) + "\n")

spec = importlib.util.spec_from_file_location("frozen_d6", scorer_path)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
original = scorer.Dataset
netCDF4.set_chunk_cache(256 << 10, 521, .75)
verdicts = {}
for domain in a.domains:
    start = time.monotonic()
    with ExitStack() as stack:
        handles = {}

        class Held:
            def __init__(self, ds): self.ds = ds
            def __enter__(self): return self.ds
            def __exit__(self, *exc): return False

        def held(path, mode="r", *args, **kwargs):
            assert mode == "r" and not args and not kwargs
            key = str(path)
            if key not in handles:
                handles[key] = stack.enter_context(original(key, "r"))
            return Held(handles[key])

        scorer.Dataset = held
        result = scorer.score(a.source.resolve(), case.resolve(), domain, hours=72)
        opened = len(handles)
    scorer.Dataset = original
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), domain=domain, hours=72,
                   scope="Frozen F20 D6 at 72 h with read-only handles reused; no model/tolerance change",
                   scorer_sha256=SCORER_SHA, manifest_sha256=result["manifest_sha256"], verdict=result["verdict"],
                   paired_frames=result["paired_frames"], bounded_misses=len(result["failures"]),
                   failures=result["failures"], nonfinite=len(result["nonfinite"]), incomplete=result["incomplete"],
                   wall_s=time.monotonic() - start, handles=opened, GPU_executed=False)
    (case / f"SCORING_IO_{domain}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    verdicts[domain] = result["verdict"]
    print(json.dumps({k: receipt[k] for k in ("domain", "verdict", "paired_frames", "bounded_misses", "nonfinite",
                                              "wall_s")}), flush=True)
print(json.dumps(verdicts))
sys.exit(0 if all(v == "PASS" for v in verdicts.values()) else 1)
