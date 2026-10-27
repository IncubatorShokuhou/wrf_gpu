"""P1: d01-only nested-pipeline probe of the root ring-0 stage clock on CPU (o5-coast).

usage: JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=0 GPUWRF_JAX_CACHE=0 [GPUWRF_ROOT_SPEC_STAGE_CLOCK=1] \
       python p1_d01_probe.py CASE_DIR OUT_DIR HOURS
CASE_DIR holds wrfinput_d01/wrfbdy_d01/wrflowinp_d01/namelist.input (symlinks to our own CPU twin 0227 run dir).
Runs ONLY the root d01 through gpuwrf.integration.nested_pipeline.execute_nested_pipeline (max_dom=1), i.e. the same
root boundary configuration as the WN3 nested runs (_root_boundary_cadence_override: specified_bdy_cadence, adv
degrade, relax strength 1, interval cadence). d01 is one-way-independent of its children (feedback 0), so its
history is comparable with the CPU-WRF twin d01. Records backend, env and the git HEAD of the imported source."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
case, out, hours = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
out.mkdir(parents=True, exist_ok=False)
t0 = time.time()
import jax  # noqa: E402
import gpuwrf  # noqa: E402
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, execute_nested_pipeline  # noqa: E402

src = Path(gpuwrf.__file__).resolve().parents[2]
head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain", "src"], capture_output=True, text=True).stdout
receipt = dict(platform=jax.devices()[0].platform, gpuwrf=str(Path(gpuwrf.__file__).resolve()), head=head,
               src_dirty=bool(dirty.strip()), env={k: v for k, v in os.environ.items() if k.startswith(("GPUWRF_", "JAX_", "XLA_"))},
               case=str(case), hours=hours, started=t0)
(out / "receipt_start.json").write_text(json.dumps(receipt, indent=1))
cfg = NestedPipelineConfig(input_dir=case, output_dir=out / "wrfout", proof_dir=out / "proof", hours=hours, max_dom=1)
payload = execute_nested_pipeline(cfg)
receipt.update(wall_s=time.time() - t0, payload_keys=sorted(payload)[:40])
(out / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str))
print(json.dumps({k: receipt[k] for k in ("platform", "head", "src_dirty", "wall_s")}))
