"""Real Swiss CAM-UW CPU execution and optional h1-4 T2 comparison.

No CUDA or CPU-WRF reference mutation. Uses the production loader and step loop.
CPU execution evidence only; full GPU D6 remains a separate gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

root, case, reference, output = (Path(s).resolve() for s in sys.argv[1:5])
sys.path.insert(0, str(root / "src"))
original_run = subprocess.run


def no_nvidia_tools(command, *args, **kwargs):
    words = command if isinstance(command, (list, tuple)) else [command]
    assert not any(Path(str(w)).name in {"nvidia-smi", "ptxas", "cuobjdump", "ncu", "nsys"}
                   for w in words), "CPU probe attempted NVIDIA tooling"
    return original_run(command, *args, **kwargs)


subprocess.run = no_nvidia_tools
import jax
import numpy as np
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.diagnostics.census import GUARDS, EVENTS, WORK, f2_negative_summary, f2_record
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.runtime import operational_mode as om

assert jax.devices()[0].platform == "cpu"
assert jax.config.x64_enabled, "CAM-UW is a WRF DOUBLE island"
assert Path(om.__file__).resolve().is_relative_to(root)
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip()
census_only = "--census-only" in sys.argv[5:]
t0 = time.perf_counter()
receipt = dict(started_utc=datetime.now(timezone.utc).isoformat(), platform="cpu", gpu_executed=False,
               source=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               reference="original CPU-WRF", case=str(case), frames=[], steps=[],
               cache="NONE GPUWRF_JAX_CACHE=0", nvidia_tools_called=False,
               scope="CPU forecast localization, not a GPU D6/speed claim")
receipt["census_only"] = census_only


def save():
    receipt["elapsed_s"] = time.perf_counter() - t0
    receipt["maxrss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


save()
cfg = NestedPipelineConfig(case, output.parent / "cpu_out", output.parent / "cpu_proof", hours=4, max_dom=1)
_, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
ns, result = bundles["d01"].namelist, carries["d01"]
assert ns.bl_pbl_physics == 9 and ns.mp_physics == 8 and ns.sf_surface_physics == 4
assert ns.rad_rk_tendf == 1
clock = om.build_clock_base(ns)
receipt["load_s"] = time.perf_counter() - t0
receipt["work"] = {k: getattr(ns, k) for k in ("dt_s", "bl_pbl_physics", "mp_physics", "sf_sfclay_physics",
                 "sf_surface_physics", "ra_lw_physics", "ra_sw_physics", "radiation_cadence_steps", "force_fp64", "disable_guards")}
receipt["input_sha256"] = {name: hashlib.sha256((case / name).read_bytes()).hexdigest()
                          for name in ("namelist.input", "wrfinput_d01", "wrfbdy_d01")}
print("LOADED", receipt["load_s"], receipt["work"], flush=True)
save()
done = 0
# Same traced-count executable for the smoke and longer intervals; no recompile for counts.
targets = (1, 2, 3, 200, 400, 600, 800)
assert float(ns.dt_s) == 18.
for target in targets:
    started = time.perf_counter()
    result = om._advance_chunk(result, ns, done + 1, clock, n_steps=target - done,
                               cadence=int(ns.radiation_cadence_steps))
    jax.block_until_ready(result)
    seconds = time.perf_counter() - started
    floats = [np.asarray(x) for x in jax.tree.leaves(result) if hasattr(x, "dtype")
              and np.issubdtype(x.dtype, np.floating)]
    nonfinite = sum(int(np.count_nonzero(~np.isfinite(x))) for x in floats)
    guards = getattr(getattr(result, "census", None), "guards", None)
    row = dict(end_step=target, steps=target - done, wall_s=seconds, nonfinite=nonfinite,
               guard_total=None if guards is None else int(np.asarray(guards).sum()))
    if guards is not None:
        host_census = jax.device_get(result.census)  # output/chunk boundary, outside compiled timestep loop
        row["guard_candidates"] = {name: dict(zip(EVENTS, map(int, counts)))
                                   for name, counts in zip(GUARDS, host_census.guards)
                                   if np.any(counts)}
        row["guard_count_scope"] = "pre-repair candidates; repaired is a predicate count, not executed repair"
        if host_census.f2_counts is not None:
            row["f2"] = f2_negative_summary(host_census, result.state.qv.shape)
            row["f2_events_and_water"] = f2_record(host_census, result.state.qv.shape)
        row["work_counts"] = dict(zip(WORK, map(int, host_census.work)))
        np.savez_compressed(output.parent / f"census_s{target}.npz", **{
            name: np.asarray(value) for name, value in host_census._asdict().items()
            if value is not None and name != "f2_qc_packet"})
    receipt["steps"].append(row)
    print("STEPS", {k: row[k] for k in ("end_step", "wall_s", "nonfinite", "guard_total")}, flush=True)
    save()
    assert nonfinite == 0, row
    done = target
    if census_only and target >= 200:
        np.savez_compressed(output.parent / f"moisture_s{target}.npz", **{
            name: np.asarray(getattr(result.state, name))
            for name in ("qv", "qc", "qr", "qi", "qs", "qg")})
    if target < 200 or census_only:
        continue
    hour = target // 200
    diag = om.compute_m9_diagnostics(result.state, ns, float(hour * 3600),
        noahmp_land=result.noahmp_land, noahmp_rad=result.noahmp_rad, clock_base=clock,
        radiation_diagnostics=result.radiation_diagnostics, history_diagnostics=result.history_diagnostics)
    data = {"T2": np.asarray(diag.t2), "TSK": np.asarray(diag.tsk), "GLW": np.asarray(diag.glw),
            "CLDFRA": np.asarray(result.radiation_diagnostics.cloud_fraction)}
    original = sorted(reference.glob("wrfout_d01_*"))[hour]
    frame = dict(lead_h=hour, reference=str(original), fields={})
    with Dataset(original) as cpu:
        cpu.set_auto_mask(False)
        for name, value in data.items():
            expected = np.asarray(cpu[name][0], np.float64)
            delta = value.astype(np.float64) - expected
            frame["fields"][name] = dict(rmse=float(np.sqrt(np.mean(delta ** 2))),
                bias=float(delta.mean()), max_abs=float(np.abs(delta).max()),
                nonzero=int(np.count_nonzero(value)))
    receipt["frames"].append(frame)
    np.savez_compressed(output.parent / f"cpu_h{hour}.npz", **data)
    print("FRAME", frame, flush=True)
    save()
receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
if not census_only:
    receipt["T2_h1_4_pass_1p5K"] = all(r["fields"]["T2"]["rmse"] <= 1.5 for r in receipt["frames"])
save()
