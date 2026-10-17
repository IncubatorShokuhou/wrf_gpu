"""Run the existing CPU release/real-case probes entirely inside this lane.

The upstream probe source is reused; only its artifact directory is relocated.
The portable/native trace policy is exactly proofs/s1_port/release_identity.py.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

root, output, mode = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
selected_mp = int(sys.argv[4]) if len(sys.argv)>4 else 17
lane = Path("<USER_HOME>/wrf_gpu2_lanes/s2-mpmore")
assert os.environ["JAX_PLATFORMS"] == "cpu"
assert (root / ".git").is_file(), "Use an immutable detached worktree"
if mode == "smoke":
    assert os.environ.get("GPUWRF_CENSUS") == "1", "Smoke guard gate requires GPUWRF_CENSUS=1 before load/compile"
sys.path[:0] = [str(root / "src"), str(root / "proofs/o1_nlbind")]
probe = root / "proofs/o1_nlbind/probe.py"

if mode == "release":
    # Preserve the current prescribed probe, changing only its scratch location.
    script = root / "proofs/s1_port/release_identity.py"
    source = script.read_text().replace("<USER_HOME>/wrf_gpu2_lanes/s1-port", str(lane))
    sys.argv = [str(script), str(root), str(output)]
    exec(compile(source, str(script), "exec"), {"__file__": str(script), "__name__": "__main__"})
else:
    source = probe.read_text().replace("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind", str(lane))
    arms = ["mp_physics=17", "mp_physics=18"] if mode == "identity" else [f"mp_physics={selected_mp}"]
    sys.argv = [str(probe), str(root), str(output), *arms]
    exec(compile(source, str(probe), "exec"), {"__file__": str(probe), "__name__": "__main__"})
result = json.loads(output.read_text())
assert all("error" not in arm for arm in result["arms"].values()), result
if mode in ("identity", "binding"):
    if mode == "identity":
        arms = list(result["arms"].values())
        assert arms[0]["bound"] == arms[1]["bound"] and arms[0]["sha"] == arms[1]["sha"], result
    # Exercise the other actual pipeline, rather than inferring its binding
    # from its import of the shared binder.
    import tempfile
    from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
    from gpuwrf.io.namelist_binding import require_namelist_honoured
    from gpuwrf.io.gen2_accessor import Gen2Run
    case_ref = Path(result["case"])
    with tempfile.TemporaryDirectory(dir=lane) as tmp:
        case = Path(tmp) / "case"
        case.mkdir()
        for item in case_ref.iterdir():
            if item.name != "namelist.input":
                (case / item.name).symlink_to(item)
        text = (case_ref / "namelist.input").read_text()
        import re
        text = re.sub(r"^\s*mp_physics\s*=.*$", f" mp_physics = {selected_mp},", text, flags=re.M | re.I)
        (case / "namelist.input").write_text(text)
        require_namelist_honoured(Gen2Run(case).namelist, ("d01",), replay=True)
        daily, _ = _build_real_case(DailyPipelineConfig(run_id=str(case), domain="d01", hours=1,
                                    output_dir=Path(tmp) / "out", proof_dir=Path(tmp) / "proof"))
        assert daily.namelist.mp_physics == (18 if selected_mp==17 else selected_mp)
        result["daily_pipeline"] = {"requested_mp": selected_mp, "bound_mp": daily.namelist.mp_physics,
                                     "grid": [daily.grid.nz, daily.grid.ny, daily.grid.nx]}
if mode == "smoke":
    arm = next(iter(result["arms"].values()))
    assert arm["bound"]["mp_physics"] == (18 if selected_mp==17 else selected_mp) and arm["nonfinite_values"] == 0, result
    assert arm.get("guard_total") == 0, result
result["lane_probe"] = {"upstream_probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
                        "gpu_executed": False, "scratch": str(lane)}
output.write_text(json.dumps(result, indent=2) + "\n")
