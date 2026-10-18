"""B1 v0.4.1 release benchmark, CPU setup at the manager's source pin (no GPU, no model work).

usage: JAX_PLATFORMS=cpu python setup.py ROOT PROGRAM_SHA
PROGRAM_SHA = the manager's pin (<USER_HOME>/wrf_gpu2_lanes/manager/B1_PIN, manager 2026-10-18T08:23:19Z); trigger.sh
runs this once per pin into a fresh ROOT. Builds
* ROOT/snap = immutable detached snapshot of PROGRAM_SHA; its src tree is recorded and session.sh refuses to start when
  the pin file or main's src tree has moved since;
* ROOT/wn3/<case> = the three WN3 3-nest gate cases (0227, 0502, 0614) as CLI input dirs, built ONLY from guarded
  copies (alisios_guard.facade_stage receipts), every file checked against the sealed F20 input SHA-256, no realpath
  inside an ALISIOS data root; PROD input = <DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725 (sha == F20 4/4);
* ROOT/d6/<WN3-case> = the F20 layout the frozen D6 scorer reads, for the par24 nested D6 gate: input -> ROOT/wn3/<case>,
  cpu/<frame> -> the guarded staged copy of the original CPU-WRF frame (sha == F20 CPU_OUTPUTS sealed sha, 75/case),
  CPU_OUTPUTS.json + cpu_run.json copied from F20, gpu -> ROOT/wn3_runs/par24/<case>/wrfout (filled by the run);
* ROOT/tools = the SHA-pinned F20 scorer (score.py + score_with_handles.py);
* ROOT/cache_prod, ROOT/cache_wn3 = EMPTY caches (the PROD arm 1 is a true cold compile; the WN3 prime compiles);
* ROOT/PLAN.json records everything.
"""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
ROOT, PROGRAM = Path(sys.argv[1]), sys.argv[2]
REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/w41-bdyrec")
F20 = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016")
L = Path("<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec")
ALISIOS = tuple(Path(p).resolve() for p in ("<DATA_ROOT>/server/work/src/alisios", "<DATA_ROOT>/server/work/alisios", "<DATA_ROOT>/alisios"))
PROD = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")
WN3 = {"wn3_0227": ("WN3-0227", Path("<DATA_ROOT>/wrf_gpu2/scratch/w41-bdyrec/wn3_0227_guarded/root0/wrf_gen/"
                                     "wg_20260227_18z_a1/run/run"), L / "alisios_stage_receipt/facade_stage.json"),
       "wn3_0502": ("WN3-0502", None, L / "b1_wn3_stage_receipt/facade_stage.json"),
       "wn3_0614": ("WN3-0614", None, L / "b1_wn3_stage_receipt/facade_stage.json")}
CPU_REFS = {"WN3-0227": L / "requal_refs_receipt", "WN3-0502": L / "b1_cpu_refs_receipt", "WN3-0614": L / "b1_cpu_refs_receipt"}
SCORER = Path("<DATA_ROOT>/wrf_gpu2/scratch/v-prep/F20-e7e37390cc-tools")
PINS = {"score.py": "4c025af8382c9626e7f7e0de1e888a91bd0270376f5c837431d1000c70c2c21d",
        "score_with_handles.py": "7407b0ddc611640b318e930f29b62673015f894fbc39d76f34d42856f92d8d19"}


def git(*args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 23), b""):
            digest.update(block)
    return digest.hexdigest()


def not_alisios(path):
    real = Path(path).resolve()
    assert not any(real == root or root in real.parents for root in ALISIOS), f"{path} -> ALISIOS {real}"
    return real


program = git("rev-parse", "--verify", f"{PROGRAM}^{{commit}}")
assert program == PROGRAM, f"pin {PROGRAM} is not a full commit sha ({program})"
ROOT.mkdir(parents=True, exist_ok=False)
snap = ROOT / "snap"
git("worktree", "add", "--detach", str(snap), program)
(snap / "data/wrf_pristine").symlink_to("<USER_HOME>/src/wrf_pristine")
assert not git("status", "--porcelain", "--untracked-files=all", "--", "src", cwd=snap), "dirty snapshot"
plan = {"program": program, "pin_source": "<USER_HOME>/wrf_gpu2_lanes/manager/B1_PIN", "snapshot": str(snap),
        "src_tree": git("rev-parse", "HEAD:src", cwd=snap), "gpuwrf_tree": git("rev-parse", "HEAD:src/gpuwrf", cwd=snap),
        "prod_input": str(PROD), "wn3": {}, "d6": {}}
for name, digest in json.loads((F20 / "PROD/inputs.json").read_text())["input_sha256"].items():
    assert sha(not_alisios(PROD / name)) == digest, f"PROD {name} sha"
staged = json.loads((L / "b1_wn3_stage_receipt/STAGED_INDEX.json").read_text())
assert not staged["sha_mismatch"], staged["sha_mismatch"]
for name, (f20, source, receipt) in WN3.items():
    folder = ROOT / "wn3" / name
    folder.mkdir(parents=True)
    for file, digest in json.loads((F20 / f20 / "inputs.json").read_text())["input_sha256"].items():
        real = not_alisios(source / file if source else staged["index"][f20][file])
        assert sha(real) == digest, f"{name} {file} sha"
        (folder / file).symlink_to(real)
    plan["wn3"][name] = {"case": str(folder), "f20": f20, "guard_receipt": str(receipt),
                         "files": sorted(p.name for p in folder.iterdir())}
(ROOT / "tools").mkdir()
for name, pin in PINS.items():
    shutil.copy2(SCORER / name, ROOT / "tools" / name)
    assert sha(ROOT / "tools" / name) == pin, f"frozen F20 scorer {name} changed"
for name, (f20, _, _) in WN3.items():  # par24 nested D6 gate: original CPU-WRF frames, guarded staged copies only
    case = ROOT / "d6" / f20
    (case / "cpu").mkdir(parents=True)
    (case / "input").symlink_to(ROOT / "wn3" / name)
    (case / "gpu").symlink_to(ROOT / "wn3_runs/par24" / name / "wrfout")
    for meta in ("CPU_OUTPUTS.json", "cpu_run.json"):
        shutil.copy2(F20 / f20 / meta, case / meta)
    assert sha(case / "CPU_OUTPUTS.json") == json.loads((case / "cpu_run.json").read_text())["output_manifest_sha256"]
    refs = json.loads((CPU_REFS[f20] / "STAGED_INDEX.json").read_text())
    assert not refs["sha_mismatch"], (f20, refs["sha_mismatch"])
    rows = json.loads((case / "CPU_OUTPUTS.json").read_text())
    for row in rows:
        real = not_alisios(refs["index"][f20][row["file"]])
        assert sha(real) == row["sha256"], f"{f20} {row['file']} sha"
        (case / "cpu" / row["file"]).symlink_to(real)
    plan["d6"][f20] = {"case": str(case), "launcher_case": name, "domains": ["d01", "d02", "d03"], "cpu_frames": len(rows),
                       "cpu_refs_receipt": str(CPU_REFS[f20] / "facade_stage.json")}
for cache in ("cache_prod", "cache_wn3"):
    for part in ("jax", "autotune", "triton", "cuda"):
        (ROOT / cache / part).mkdir(parents=True)
for part in ("tmp", "S1", "wn3_runs"):
    (ROOT / part).mkdir()
plan["scorer"] = {"tools": str(ROOT / "tools"), "pins": PINS, "source": str(SCORER)}
plan["baselines"] = {"S1_proxy_RC2": 3.088, "S1_whole_run_RC2": 3.604, "S1_proxy_LW9": 5.633, "S1_whole_LW9": 6.457,
                     "wn3_par_D28_s_per_case_h": 13.49, "cpu_wrf_3x4core_s_per_case_h": 123.385,
                     "refs": "MAILBOX 2026-10-05T04:23:38Z (RC2), FINDINGS A55 (LW9), D28 (W6 bench24_p3), wn3_parallel_analyze.py"}
(ROOT / "PLAN.json").write_text(json.dumps(plan, indent=1) + "\n")
print(json.dumps({k: plan[k] for k in ("program", "src_tree", "gpuwrf_tree")} | {"wn3": list(plan["wn3"]), "d6": list(plan["d6"])}))
