"""Stage the w41-bdyrec GPU requalification (CPU only, no model work).

usage: JAX_PLATFORMS=cpu python gpu_stage.py ROOT CANDIDATE_SNAPSHOT RELEASE_SNAPSHOT
ROOT (fresh, under <DATA_ROOT>/wrf_gpu2/scratch/w41-bdyrec) gets one case dir per arm in the
F20 layout the frozen D6 scorer reads (input/, cpu/, cpu_run.json, CPU_OUTPUTS.json,
inputs.json as read-only symlinks to the sealed F20 archive; gpu/, gpu_proof/, gpu_scratch/,
cache/ fresh). Short-input arms hold record-truncated copies of the same inputs (the 48 h
shape of the ALISIOS bundle incident) and share the cache of their full-input arm. Every
cache is seeded with a COPY of the v0.4.0 bundle's per-fusion autotune results, so fusions
that the record windows leave unchanged get the release tunes (E99/C34); the bundle itself
is only read.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from netCDF4 import Dataset

assert os.environ.get("JAX_PLATFORMS") == "cpu"
ROOT, CANDIDATE, RELEASE = (Path(a).resolve() for a in sys.argv[1:4])
F20 = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016")
BUNDLE = Path("<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0")
TUNES = BUNDLE / "jax/xla_gpu_per_fusion_autotune_cache_dir"
REFERENCE_SHA = {  # release GPU frames of the sealed v0.4.0 bundle priming (r-pkg040, src tree b542907a)
    "WN3-0227": Path("<USER_HOME>/wrf_gpu2_lanes/r-pkg040/bundle040/wrfout_sha_wn3_0227_3nest.txt"),
    "Swiss": Path("<USER_HOME>/wrf_gpu2_lanes/r-pkg040/bundle040/wrfout_sha_swiss_d01.txt"),
}
# case -> (F20 dir, short-input record counts {file: records}, pool fraction)
CASES = {
    "Swiss": ("Swiss", {"wrfbdy_d01": 3}, ".15"),
    "PROD": ("PROD", {"wrfbdy_d01": 9}, ".30"),
    "WN3-0227": ("WN3-0227", {"wrfbdy_d01": 8, "wrflowinp_d01": 9, "wrflowinp_d02": 9, "wrflowinp_d03": 9}, ".18"),
}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 23), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_detached(tree):
    assert subprocess.run(["git", "-C", str(tree), "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode
    assert not subprocess.check_output(["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=all",
                                        "--", "src"], text=True).strip(), f"dirty {tree}"
    assert (tree / "data/wrf_pristine").exists(), f"missing {tree}/data/wrf_pristine"
    return subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()


def truncate(src, dst, records):
    with Dataset(src) as a, Dataset(dst, "w", format=a.data_model) as b:
        b.setncatts({k: a.getncattr(k) for k in a.ncattrs()})
        for name, dim in a.dimensions.items():
            b.createDimension(name, None if dim.isunlimited() else len(dim))
        for name, var in a.variables.items():
            out = b.createVariable(name, var.datatype, var.dimensions)
            out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            data = var[:]
            out[:] = data[:records] if var.dimensions and var.dimensions[0] == "Time" else data
    with Dataset(dst) as check:
        assert len(check.dimensions["Time"]) == records, dst


def make_case(name, source, inputs=None):
    case = ROOT / name
    case.mkdir()
    for item in ("cpu", "cpu_run.json", "CPU_OUTPUTS.json", "inputs.json"):
        (case / item).symlink_to(source / item)
    if inputs is None:
        (case / "input").symlink_to(source / "input")
    else:
        (case / "input").mkdir()
        for path in sorted((source / "input").iterdir()):
            if path.name in inputs:
                truncate(path, case / "input" / path.name, inputs[path.name])
            else:
                (case / "input" / path.name).symlink_to(path)
    (case / "tmp").mkdir()  # gpu -> run24/wrfout is linked after phase 1 (frozen D6 scorer layout)
    return case


def seed_cache(case):
    for part in ("jax", "autotune", "triton", "cuda"):
        (case / "cache" / part).mkdir(parents=True)
    shutil.copytree(TUNES, case / "cache/jax" / TUNES.name)


ROOT.mkdir(parents=False, exist_ok=False)
SCORER = Path("<DATA_ROOT>/wrf_gpu2/scratch/v-prep/F20-e7e37390cc-tools")
(ROOT / "tools").mkdir()
for name, pin in (("score.py", "4c025af8382c9626e7f7e0de1e888a91bd0270376f5c837431d1000c70c2c21d"),
                  ("score_with_handles.py", "7407b0ddc611640b318e930f29b62673015f894fbc39d76f34d42856f92d8d19")):
    shutil.copy2(SCORER / name, ROOT / "tools" / name)
    assert sha(ROOT / "tools" / name) == pin, f"frozen F20 scorer {name} changed"
plan = {"candidate": str(CANDIDATE), "candidate_head": clean_detached(CANDIDATE),
        "release": str(RELEASE), "release_head": clean_detached(RELEASE),
        "seeded_tunes": {"source": str(TUNES), "entries": len(list(TUNES.iterdir()))}, "arms": {}}
for name, (folder, short, fraction) in CASES.items():
    source = F20 / folder
    case = make_case(name, source)
    seed_cache(case)
    short_case = make_case(f"{name}-short", source, short)
    (short_case / "cache").symlink_to(case / "cache", target_is_directory=True)
    plan["arms"][name] = {"case": str(case), "short": str(short_case), "fraction": fraction,
                          "domains": json.loads((source / "inputs.json").read_text())["domains"],
                          "short_records": short,
                          "short_input_sha256": {p.name: sha(p) for p in sorted((short_case / "input").iterdir())
                                                 if not p.is_symlink()}}
    if name in REFERENCE_SHA:
        plan["arms"][name]["release_frames"] = dict(
            line.split()[::-1] for line in REFERENCE_SHA[name].read_text().splitlines() if line.strip())
control = make_case("WN3-0227-release", F20 / "WN3-0227")
seed_cache(control)
plan["release_control"] = {"case": str(control), "fraction": ".18"}
(ROOT / "PLAN.json").write_text(json.dumps(plan, indent=1) + "\n")
print(json.dumps({k: plan[k] for k in ("candidate_head", "release_head")}))
