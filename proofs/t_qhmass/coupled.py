"""Two real Swiss CPU steps, with same-input pristine loading at each seam.

Uses the established FAST0/128-column/phase-cache-clear CPU configuration.
Site observers run outside compilation under disable_jit; this is not a GPU
program, timing benchmark or full CPU-WRF trajectory comparison.
"""
import argparse
import ctypes
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
assert os.environ.get("JAX_PLATFORMS") == "cpu", "Set JAX_PLATFORMS=cpu before this proof"
assert subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=ROOT,
                      capture_output=True).returncode == 1, "Measure a detached source snapshot"
assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT), \
    "Keep bookkeeping in the lane checkout; measure a clean snapshot"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "proofs/o1_nlbind")]
import cpu_safe
cpu_safe.install()
import jax
import numpy as np
from gpuwrf.contracts.moisture import active_moist_arrays
from gpuwrf.contracts.physics_registry import moist_species_for_mp
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime import operational_mode as om
from wrf_moist_oracle import build_oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mp", type=int, choices=(17, 18, 22, 38), required=True)
    parser.add_argument("--hail-source", type=Path,
                        help="Optional same-grid original CPU-WRF QHAIL/QNHAIL for an active carry diagnostic")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert jax.devices()[0].platform == "cpu"
    oracle = build_oracle(ROOT, args.out / "oracle")
    case = args.out / "case"
    case.mkdir()
    text, count = re.subn(r"(?im)^\s*mp_physics\s*=.*$", f" mp_physics = {args.mp},",
                         (args.case / "namelist.input").read_text())
    assert count == 1
    (case / "namelist.input").write_text(text)
    for path in args.case.iterdir():
        if path.name != "namelist.input":
            (case / path.name).symlink_to(path.resolve())
    report = dict(source=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  case=str(args.case), input_sha256=hashlib.sha256((args.case / "wrfinput_d01").read_bytes()).hexdigest(),
                  namelist_sha256=hashlib.sha256(text.encode()).hexdigest(), mp=args.mp,
                  platform="cpu", GPU_execution=False, cache="NONE", steps=[], sites=[],
                  scope="Full real Swiss CPU caller chain; same-input pristine loading, not WRF trajectory parity",
                  oracle=oracle.metadata)
    def save():
        (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    def clear():
        jax.clear_caches()
        gc.collect()
        ctypes.CDLL(None).malloc_trim(0)
    def phase(module, name):
        original = getattr(module, name)
        @functools.wraps(original)
        def measured(*a, **kw):
            clear()
            print("PHASE", name, "enter", flush=True)
            value = original(*a, **kw)
            jax.block_until_ready(value)
            clear()
            print("PHASE", name, "exit", flush=True)
            return value
        setattr(module, name, measured)
    for name in ("_refresh_rrtmg_driver", "_refresh_noahmp_rad", "noahmp_surface_step",
                 "_physics_step_forcing", "_rk_scan_step", "_apply_post_rk_microphysics"):
        phase(om, name)
    for name in ("solve_rrtmg_sw_column", "solve_rrtmg_lw_column"):
        phase(pc, name)
    om._M9_RRTMG_COLUMN_TILE_COLS = 128
    step = 0
    def reference(state, metrics):
        species = moist_species_for_mp(args.mp)
        moist = [np.asarray(getattr(state, name)) for name in species]
        # Assert the uniform State core has no hidden mass outside Registry.
        for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "qi2", "qi3"):
            value = getattr(state, name, None)
            if name not in species and value is not None:
                assert np.count_nonzero(np.asarray(value)) == 0, name
        pressure, faces, cq = oracle(moist, state.mu_total, metrics.c1h, metrics.c2h, metrics.dnw, metrics.p_top)
        row = dict(step=step, active_mass=species,
                   qh_nonzero=0 if state.qh is None else int(np.count_nonzero(np.asarray(state.qh))),
                   qh_max=0. if state.qh is None else float(np.max(np.asarray(state.qh))))
        if state.qh is not None:
            interior = np.ascontiguousarray(np.asarray(state.qh)[:, 5:-5, 5:-5])
            row["qh_interior_sha256"] = hashlib.sha256(interior.tobytes()).hexdigest()
        return pressure, faces, cq, row
    old_profile = pc._wrf_hydrostatic_pressure_profiles_from_state
    def profile(state, metrics, **kw):
        result = old_profile(state, metrics, **kw)
        p, faces, _, row = reference(state, metrics)
        for actual, expected in zip(result, (p, faces, faces[0]), strict=True):
            assert np.isfinite(actual).all() and np.isfinite(expected).all()
            np.testing.assert_array_equal(np.asarray(actual, np.float32), expected)
        row.update(site="phy_prep", pressure_BYTE=True)
        report["sites"].append(row)
        save()
        return result
    old_acoustic = om._acoustic_core_state_from_prep
    def acoustic(carry, prep, pressure, namelist, tendencies, **kw):
        result = old_acoustic(carry, prep, pressure, namelist, tendencies, **kw)
        _, _, cq, row = reference(prep.entry_state, namelist.metrics)
        # WRF calc_cq interleaves adjacent levels while the retained caller
        # sums masses first. Bound both orders and the reciprocal in REAL4.
        expected = 1. / (1. + cq[1:-1].astype(np.float64))
        actual = np.asarray(result.cqw)[1:-1]
        n = 2*len(row["active_mass"])+3
        eps = np.finfo(np.float32).eps
        bound = 2*n*eps/(1-n*eps)
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(actual, expected, rtol=bound, atol=0)
        row.update(site="calc_cq", max_abs=float(np.max(np.abs(actual-expected))), relative_bound=bound)
        report["sites"].append(row)
        save()
        return result
    old_psfc = om._psfc_from_state
    def psfc(state, metrics):
        result = old_psfc(state, metrics)
        _, faces, _, row = reference(state, metrics)
        eps = np.finfo(np.float32).eps
        n = state.qv.shape[0]+len(row["active_mass"])+3
        bound = 2*n*eps/(1-n*eps)
        assert np.isfinite(result).all() and np.isfinite(faces).all()
        np.testing.assert_allclose(result, faces[0], rtol=bound, atol=0)
        row.update(site="history_PSFC", max_abs_pa=float(np.max(np.abs(np.asarray(result)-faces[0]))), relative_bound=bound)
        report["sites"].append(row)
        save()
        return result
    config = NestedPipelineConfig(case, args.out / "output", args.out / "proof", hours=1, max_dom=1)
    start = time.monotonic()
    _, bundles, _, _, _, carries = _load_domains(config, ("d01",))
    namelist, carry = bundles["d01"].namelist, carries["d01"]
    # WRF check_a_mundo (3406-3415) and OperationalNamelist normalize 17 to18.
    assert namelist.mp_physics == (18 if args.mp == 17 else args.mp)
    report["resolved_mp"] = namelist.mp_physics
    if args.hail_source is not None:
        assert args.mp in (17, 18)
        from gpuwrf.io.netcdf_lock import Dataset
        import jax.numpy as jnp
        with Dataset(args.hail_source) as nc:
            hail = np.asarray(nc["QHAIL"][0])
            number = np.asarray(nc["QNHAIL"][0])
            source_grid = {name: np.asarray(nc[name][:]) for name in ("XLAT", "XLONG", "ZNU")}
        with Dataset(case / "wrfinput_d01") as nc:
            for name, value in source_grid.items():
                np.testing.assert_array_equal(value, np.asarray(nc[name][:]))
        assert hail.shape == carry.state.qh.shape == number.shape
        assert np.isfinite(hail).all() and np.isfinite(number).all() and np.count_nonzero(hail) > 0
        carry = carry.replace(state=carry.state.replace(
            qh=jnp.asarray(hail, carry.state.qh.dtype), Nh=jnp.asarray(number, carry.state.Nh.dtype)))
        report["active_hail_input"] = dict(source=str(args.hail_source.resolve()),
            sha256=hashlib.sha256(args.hail_source.read_bytes()).hexdigest(),
            qh_nonzero=int(np.count_nonzero(hail)), qh_max=float(hail.max()),
            same_grid_XLAT_XLONG_ZNU_BYTE=True,
            scope="Original same-grid hail injected into Swiss initial carry; controlled carry diagnostic, not forecast truth")
    pc._wrf_hydrostatic_pressure_profiles_from_state = profile
    om._acoustic_core_state_from_prep = acoustic
    om._psfc_from_state = psfc
    report["load_s"] = time.monotonic()-start
    report["work"] = {name: getattr(namelist, name) for name in ("mp_physics", "dt_s", "bl_pbl_physics",
        "sf_surface_physics", "sf_sfclay_physics", "ra_lw_physics", "ra_sw_physics", "cu_physics", "radiation_cadence_steps")}
    clock = om.build_clock_base(namelist)
    save()
    clear()
    with jax.disable_jit():
        for step in (1, 2):
            start = time.monotonic()
            carry = om._advance_chunk_fori(carry, namelist, step, clock, n_steps=1,
                                          cadence=int(namelist.radiation_cadence_steps))
            jax.block_until_ready(carry)
            psfc(carry.state, namelist.metrics)
            bad = sum(int(np.count_nonzero(~np.isfinite(np.asarray(x)))) for x in jax.tree.leaves(carry)
                      if np.issubdtype(np.asarray(x).dtype, np.floating))
            from gpuwrf.diagnostics.census import f2_record
            row = dict(step=step, wall_s=time.monotonic()-start, nonfinite=bad,
                       guards=np.asarray(carry.census.guards).tolist(), f2=f2_record(carry.census, carry.state.qv.shape))
            report["steps"].append(row)
            save()
            print("STEP", step, row["wall_s"], "nonfinite", bad, flush=True)
            assert bad == 0
    assert {r["step"] for r in report["sites"] if r["site"] == "calc_cq"} >= {1, 2}
    if args.hail_source is not None:
        producer = [r for r in report["sites"] if r["step"] == 1 and r["site"] == "history_PSFC"][-1]
        consumer = next(r for r in report["sites"] if r["step"] == 2 and r["site"] == "calc_cq")
        assert producer["qh_nonzero"] > 0 and consumer["qh_nonzero"] > 0
        assert producer["qh_interior_sha256"] == consumer["qh_interior_sha256"]
        report["active_post_MP_to_next_RK_hail_interior_BYTE"] = True
    report["peakRSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["status"] = "PASS"
    save()


if __name__ == "__main__":
    main()
