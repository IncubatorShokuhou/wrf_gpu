"""WN3 branch regression gate; GPU only under the official manager lock.

Supervisor imports stdlib only. Each measured arm uses a fresh process; the
supervisor forwards only the validated lock fd and kills its own child group
before returning on timeout/signal. CPU --dry-run does not import JAX.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n")


def array_hashes(payload):
    import jax
    import numpy as np
    result = {}
    for key, value in jax.tree_util.tree_flatten_with_path(payload)[0]:
        if isinstance(value, np.ndarray):
            if value.dtype.kind in "fc":
                assert np.all(np.isfinite(value)), str(key)
            result[jax.tree_util.keystr(key)] = {
                "shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    return result


def worker(spec, mode):
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
    import jax
    assert jax.devices()[0].platform == "gpu"
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, execute_nested_pipeline
    from gpuwrf.runtime.restart_store import RestartStore
    root = Path(spec["root"])
    same = mode in {"control", "resume"}
    stream = "control" if same else mode
    kwargs = dict(input_dir=Path(spec["input_dir"]), output_dir=root / stream / "outputs",
                  proof_dir=root / stream / "proof", hours=6, max_dom=3,
                  checkpoint_dir=root / stream / "checkpoints", checkpoint_interval_steps=200,
                  checkpoint_max_bytes=16 * 1024**3, checkpoint_reserve_bytes=200 * 1024**3,
                  aot_prefetch=False)
    if mode == "resume":
        kwargs["resume_checkpoint"] = root / "parent-seam" / "generation-seam"
    elif not same:
        kwargs["branch_from"] = root / "parent-seam" / "generation-seam"
        kwargs["branch_manifest"] = root / (mode + ".json")
        import gpuwrf.runtime.checkpoint_branch as branch
        original_apply = branch.apply_overrides
        def audited_apply(snapshot, changes, **options):
            before = array_hashes(snapshot["carries"])
            result, report = original_apply(snapshot, changes, **options)
            after = array_hashes(result["carries"])
            assert set(before) == set(after)
            changed = {key for key in before if before[key] != after[key]}
            expected = set() if mode == "null" else {
                f"['{name}']['state_fields']['qv']" for name in ("d01", "d02", "d03")}
            assert changed == expected, (changed, expected)
            write(root / (mode + "_seam.json"), {"passed": True, "changed": sorted(changed),
                  "leaf_count": len(before), "report": report})
            return result, report
        branch.apply_overrides = audited_apply
    started = time.perf_counter()
    result = execute_nested_pipeline(NestedPipelineConfig(**kwargs))
    write(root / (mode + "_run.json"), result)
    assert result["verdict"] == "PIPELINE_GREEN", result
    store = RestartStore(kwargs["checkpoint_dir"], 16 * 1024**3, 2, 200 * 1024**3)
    latest = max((json.loads((g / "manifest.json").read_text())["created_ns"], g)
                 for g in store.root.glob("generation-*"))[1]
    snapshot, manifest = store.read(latest, device=False)
    leaves = array_hashes(snapshot["carries"])
    outputs = {Path(p).name: sha(p) for entry in result["per_domain"].values() for p in entry["wrfout_files"]}
    record = {"mode": mode, "backend": jax.devices()[0].platform,
              "device": jax.devices()[0].device_kind, "wall_s": time.perf_counter() - started,
              "device_memory_stats": jax.devices()[0].memory_stats(),
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "own_steps": manifest["own_steps"], "leaves": leaves, "outputs": outputs,
              "result": result, "scope": "GPU regression/branch smoke; no CPU-WRF fidelity claim"}
    write(root / (mode + "_receipt.json"), record)
    if mode == "control":
        seam = next(g for g in store.root.glob("generation-*") if
                    json.loads((g / "manifest.json").read_text())["own_steps"]["d01"] == 200)
        destination = root / "parent-seam" / "generation-seam"
        destination.mkdir(parents=True)
        for name in ("manifest.json", "snapshot.pkl"):
            os.link(seam / name, destination / name)
        parent, _ = store.read(destination, device=False)
        write(root / "seam_arrays.json", array_hashes(parent["carries"]))


def supervisor(spec, *, dry_run):
    root, tree = Path(spec["root"]), Path(spec["tree"])
    assert subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip() == spec["pin"]
    assert not subprocess.check_output(["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    for name, expected in spec["inputs"].items():
        assert sha(Path(spec["input_dir"]) / name) == expected, name
    for name, expected in spec.get("tools", {}).items():
        assert sha(Path(__file__).resolve().parent / name) == expected, name
    if dry_run:
        print(json.dumps({"dry_run": "PASS", "pin": spec["pin"], "arms": ["control", "resume", "null", "override"],
                          "work": "WN3 0227 d01/d02/d03 dt54/18/6;6h control, seam3h;three3h continuations",
                          "no_gpu_import": "jax" not in sys.modules}))
        return
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
    fd = int(os.environ["GPUWRF_GPU_LOCK_FD"])
    a, b = os.fstat(fd), os.stat(os.environ["GPUWRF_GPU_LOCK_FILE"])
    assert fd >= 3 and (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
    root.mkdir(parents=True, exist_ok=False)
    write(root / "null.json", {"schema_version": 1})
    write(root / "override.json", {"schema_version": 1,
        "domains": {name: {"state": {"qv": {"multiply": .99}}} for name in ("d01", "d02", "d03")},
        "waivers": {name: {
            "hydrostatic_consistency": "Explicit unbalanced QV branch sensitivity smoke at fixed MU",
            "supersaturation_bound": "Source model-state supersaturation retained in research sensitivity smoke"}
            for name in ("d01", "d02", "d03")}})
    current = None

    def terminate(signum, frame):
        if current is not None and current.poll() is None:
            os.killpg(current.pid, signal.SIGTERM)
            try:
                current.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(current.pid, signal.SIGKILL)
                current.wait()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    deadline = time.monotonic() + int(spec.get("cap_s", 1800))
    for mode in ("control", "resume", "null", "override"):
        remaining = deadline - time.monotonic()
        assert remaining > 30, "GPU cap exhausted before next arm"
        with (root / (mode + ".log")).open("w") as log:
            current = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(args.spec), "--worker", mode],
                                       stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,), start_new_session=True)
            try:
                rc = current.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                terminate(signal.SIGTERM, None)
            assert rc == 0, (mode, rc)
    control = json.loads((root / "control_receipt.json").read_text())
    for mode in ("resume", "null"):
        item = json.loads((root / (mode + "_receipt.json")).read_text())
        assert item["leaves"] == control["leaves"], f"{mode}: final carry differs"
        for name, value in item["outputs"].items():
            assert value == control["outputs"][name], (mode, name)
    override = json.loads((root / "override_receipt.json").read_text())
    assert override["leaves"] != control["leaves"], "QV branch had no effect"
    assert all(override["result"]["per_domain"][n]["final_state_finite"] for n in ("d01", "d02", "d03"))
    write(root / "VERDICT.json", {"passed": True, "backend": "gpu", "null_resume_byte": True,
                                  "null_branch_byte": True, "qv_branch_finite": True,
                                  "pin": spec["pin"], "scope": "regression only; CPU-WRF physics gates carried from identical default program"})


def postprocess(spec):
    """Preserve small proofs, then remove only this arm's declared scratch."""
    lane = Path("<USER_HOME>/wrf_gpu2_lanes/w44-branch").resolve()
    root, cache, tree = (Path(spec[key]).resolve() for key in ("root", "cache", "tree"))
    assert json.loads((root / "VERDICT.json").read_text())["passed"]
    assert json.loads((lane / "GPU_E41/wave_gate.json").read_text())["e41_gate"] == "PASS"
    destination = lane / "GPU_RECEIPTS"
    destination.mkdir(exist_ok=False)
    for pattern in ("*.json", "*.log"):
        for path in root.glob(pattern):
            shutil.copy2(path, destination / path.name)
    write(destination / "spec.json", spec)
    assert root == Path("<DATA_ROOT>/wrf_gpu2/scratch/w44-branch/gpu01")
    assert cache == Path("<DATA_ROOT>/wrf_gpu2/scratch/w44-branch/cache01")
    assert tree == lane / "final_cpu"
    shutil.rmtree(root)
    shutil.rmtree(cache)
    subprocess.run(["git", "-C", "<USER_HOME>/src/wrf_gpu2", "worktree", "remove", "--force", spec["tree"]], check=True)
    write(destination / "CLEANUP.json", {"passed": True, "removed": [str(root), str(cache), str(tree)]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--worker", choices=("control", "resume", "null", "override"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.postprocess:
        assert os.environ.get("JAX_PLATFORMS") == "cpu"
        postprocess(spec)
    elif args.worker:
        worker(spec, args.worker)
    else:
        supervisor(spec, dry_run=args.dry_run)
