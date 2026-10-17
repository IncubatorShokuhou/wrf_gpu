"""Prepared manager-only Swiss 3 h precipitation arm (run under official lock).

Source the driver bridge, then from an immutable reviewed checkout:
  scripts/with_gpu_lock.sh --label u2-diag -- timeout 1800 \
    env JAX_PLATFORMS=cuda python proofs/u2_diag/gpu_smoke.py
Window/lease approval, cold-compile cubin gate, CPU-WRF scoring and cache
cleanup belong to the manager. This script has NOT been executed on GPU.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1", "official GPU lock required"
ROOT=Path(__file__).resolve().parents[2]
LANE=Path("<USER_HOME>/wrf_gpu2_lanes/u2-diag")
SOURCE=Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
CASE=LANE/"gpu_bucket_case"
OUT=LANE/"gpu_bucket_3h"
assert not CASE.exists() and not OUT.exists(), "use a fresh prepared arm, never overwrite evidence"
assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT).strip(), "immutable clean checkout required"
commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
CASE.mkdir()
for p in SOURCE.iterdir():
    if p.name in ("wrf.exe","namelist.input","namelist.output") or p.name.startswith(("wrfout","wrfrst","rsl.")):
        continue
    if p.is_symlink() or p.name.startswith(("wrfinput","wrfbdy","wrflowinp")):
        (CASE/p.name).symlink_to(p.resolve())
text=(SOURCE/"namelist.input").read_text().replace("&physics","&physics\n prec_acc_dt=60.,\n do_radar_ref=1,",1)
text=text.replace("&time_control","&time_control\n nwp_diagnostics=1,",1)
(CASE/"namelist.input").write_text(text)
for key,path in (("GPUWRF_JAX_CACHE_DIR",LANE/"gpu_bucket_cache"),
                 ("JAX_COMPILATION_CACHE_DIR",LANE/"gpu_bucket_cache"),
                 ("GPUWRF_XLA_AUTOTUNE_CACHE_DIR",LANE/"gpu_bucket_autotune")):
    os.environ[key]=str(path)
os.environ["GPUWRF_FAST_DEFAULTS"]="1"
os.environ["GPUWRF_CENSUS"]="1"
os.environ["GPUWRF_JAX_CACHE"]="1"
sys.path.insert(0,str(ROOT/"src"))
from gpuwrf import cli
assert Path(cli.__file__).resolve().is_relative_to(ROOT/"src"), "wrong source import"
import jax
assert jax.devices()[0].platform == "gpu"
argv=["run","--input-dir",str(CASE),"--output-dir",str(OUT),"--scratch-dir",str(LANE/"gpu_bucket_scratch"),
      "--hours","3","--max-dom","1"]
receipt=dict(source_commit=commit,backend=jax.devices()[0].platform,device=str(jax.devices()[0]),
             namelist_sha256=hashlib.sha256(text.encode()).hexdigest(),argv=argv,
             work=dict(nx=42,ny=42,nz=44,dt_s=18,steps=600,mp=8,cu=0,bl=5,sf=4,ra_sw=4,ra_lw=4,radt_min=10,prec_acc_dt_min=60,
                       nwp_diagnostics=1,do_radar_ref=1,swint_opt=0,history_interval_min=60),
             scientific_reference=str(LANE/"cpu_reference_3h"),
             gates="pending execution, cubin scan, finite/census, all precipitation/NWP/REFL_10CM fields vs original CPU-WRF",
             speed_claim=False)
start=time.monotonic()
rc=cli.main(argv)
receipt.update(rc=rc,cli_wall_s=time.monotonic()-start)
(LANE/"gpu_bucket_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
raise SystemExit(rc)
