"""Composed default requal, CPU setup at REQUAL GO (no GPU, no model work).

usage: JAX_PLATFORMS=cpu python setup.py ROOT MAIN_SHA SWDDIR_SHA LANE_SHA STAGED_INDEX
(STAGED_INDEX '-' = dry setup before GO: PROD/WN3 cpu/ stay empty, nothing under ALISIOS is needed)
* candidate = immutable detached snapshot of MAIN_SHA (must contain LANE_SHA = the merged lane/w41-bdyrec head)
  + SWDDIR_SHA merged inside the snapshot (merge commit made there; a conflict refuses);
* ROOT/<case> for Swiss / PROD / WN3-0227 in the F20 layout the frozen D6 scorer reads (input/, cpu/, cpu_run.json,
  CPU_OUTPUTS.json, inputs.json), built ONLY from wrf_gpu2-owned files: Swiss = release-docs RD11, PROD inputs =
  <DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725 (sha == F20 4/4), WN3 inputs = the lane's guarded copies, PROD/WN3 cpu/ =
  guarded staged frames (stage_refs.py); every input and frame is checked against the sealed F20 SHA-256 and no
  realpath may reach an ALISIOS data root;
* fresh caches per case seeded with a COPY of the v0.4.0 bundle autotune results (bundle only read); the S1 arms
  share the PROD case cache directory (one cache path string, E136/E143);
* ROOT/tools = the SHA-pinned F20 scorer; ROOT/PLAN.json records everything.
"""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
ROOT, MAIN, SWD, LANE, STAGED = Path(sys.argv[1]), *sys.argv[2:5], sys.argv[5]
DRY = STAGED == "-"
REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/w41-bdyrec")
F20 = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016")
TUNES = Path("<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0/jax/xla_gpu_per_fusion_autotune_cache_dir")
SCORER = Path("<DATA_ROOT>/wrf_gpu2/scratch/v-prep/F20-e7e37390cc-tools")
PINS = {"score.py": "4c025af8382c9626e7f7e0de1e888a91bd0270376f5c837431d1000c70c2c21d",
        "score_with_handles.py": "7407b0ddc611640b318e930f29b62673015f894fbc39d76f34d42856f92d8d19"}
ALISIOS = tuple(Path(p).resolve() for p in ("<DATA_ROOT>/server/work/src/alisios", "<DATA_ROOT>/server/work/alisios", "<DATA_ROOT>/alisios"))
INPUTS = {"Swiss": Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu"),
          "PROD": Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725"),
          "WN3-0227": Path("<DATA_ROOT>/wrf_gpu2/scratch/w41-bdyrec/wn3_0227_guarded/root0/wrf_gen/wg_20260227_18z_a1/run/run")}


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


main_sha = git("rev-parse", "--verify", f"{MAIN}^{{commit}}")
swd_sha = git("rev-parse", "--verify", f"{SWD}^{{commit}}")
assert subprocess.run(["git", "merge-base", "--is-ancestor", LANE, main_sha], cwd=REPO).returncode == 0, \
    f"main {main_sha} does not contain lane/w41-bdyrec {LANE}"
ROOT.mkdir(parents=True, exist_ok=False)
snap = ROOT / "snap"
git("worktree", "add", "--detach", str(snap), main_sha)
merge = subprocess.run(["git", "-c", "user.name=w41-bdyrec", "-c", "user.email=w41-bdyrec@local", "merge", "--no-ff",
                        "--no-edit", "-m", f"composed requal: main {main_sha[:10]} + lane/w41-swddir {swd_sha[:10]}", swd_sha],
                       cwd=snap, capture_output=True, text=True)
assert merge.returncode == 0, f"swddir merge refused: {merge.stdout[-800:]} {merge.stderr[-800:]}"
(snap / "data/wrf_pristine").symlink_to("<USER_HOME>/src/wrf_pristine")
assert not git("status", "--porcelain", "--untracked-files=all", "--", "src", cwd=snap), "dirty snapshot"
head, tree = git("rev-parse", "HEAD", cwd=snap), git("rev-parse", "HEAD:src/gpuwrf", cwd=snap)

(ROOT / "tools").mkdir()
for name, pin in PINS.items():
    shutil.copy2(SCORER / name, ROOT / "tools" / name)
    assert sha(ROOT / "tools" / name) == pin, f"frozen F20 scorer {name} changed"
staged = {"index": {}, "sha_mismatch": []} if DRY else json.loads(Path(STAGED).read_text())
assert not staged["sha_mismatch"], staged["sha_mismatch"]
plan = {"candidate": str(snap), "candidate_head": head, "candidate_src_tree": tree, "main": main_sha,
        "swddir": swd_sha, "lane_bdyrec": LANE, "seeded_tunes": {"source": str(TUNES), "entries": len(list(TUNES.iterdir()))},
        "staged_refs": STAGED, "dry": DRY, "arms": {}}
for case, source in INPUTS.items():
    folder = ROOT / case
    folder.mkdir()
    for item in ("cpu_run.json", "CPU_OUTPUTS.json", "inputs.json"):
        shutil.copy2(F20 / case / item, folder / item)
    meta = json.loads((folder / "inputs.json").read_text())
    (folder / "input").mkdir()
    for name, digest in meta["input_sha256"].items():
        real = not_alisios(source / name)
        assert sha(real) == digest, f"{case} input {name} sha mismatch"
        (folder / "input" / name).symlink_to(real)
    (folder / "cpu").mkdir()
    rows = json.loads((folder / "CPU_OUTPUTS.json").read_text())
    for row in rows:
        if case == "Swiss":
            real = not_alisios(os.readlink(F20 / case / "cpu" / row["file"]))
        elif DRY:
            continue
        else:
            real = not_alisios(staged["index"][case][row["file"]])
        assert sha(real) == row["sha256"], f"{case} cpu {row['file']} sha mismatch"
        (folder / "cpu" / row["file"]).symlink_to(real)
    for part in ("jax", "autotune", "triton", "cuda"):
        (folder / "cache" / part).mkdir(parents=True)
    shutil.copytree(TUNES, folder / "cache/jax" / TUNES.name)
    (folder / "tmp").mkdir()
    plan["arms"][case] = {"case": str(folder), "domains": meta["domains"], "inputs": str(source),
                          "cpu_frames": len(rows)}
s1 = ROOT / "S1"
(s1 / "tmp").mkdir(parents=True)
plan["s1"] = {"root": str(s1), "case_input": str(INPUTS["PROD"]), "cache": str(ROOT / "PROD/cache"),
              "baseline": {"RC2": 3.088, "LW9": 5.633, "ref": "MAILBOX 2026-10-05T04:23:38Z (RC2), FINDINGS A55 (LW9)"}}
(ROOT / "PLAN.json").write_text(json.dumps(plan, indent=1) + "\n")
print(json.dumps({k: plan[k] for k in ("candidate_head", "candidate_src_tree", "main", "swddir")}))
