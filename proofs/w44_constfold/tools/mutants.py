"""Deletion-sensitivity mutants for tests/test_w44_constfold.py (sequential, in-place, restored in finally).

Each mutant reverts ONE fix (site back to the Python f64 fold, or the REAL constant rebuilt from the f64 fold).
A mutant counts as KILLED only if pytest reports >= 1 failed test with an AssertionError and 0 errors (E202).
usage: JAX_PLATFORMS=cpu python mutants.py REPO OUT_JSON BASETEMP
"""
import json
import re
import subprocess
import sys
from pathlib import Path

repo, out, basetemp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
MUTANTS = [
    ("qsat_site", "src/gpuwrf/physics/mynn_edmf.py",
     "real_constant(_QSAT_CHI_DEN_REAL, (T0C - 6.0) - TICE, jnp.result_type(t))", "((T0C - 6.0) - TICE)"),
    ("qsat_const", "src/gpuwrf/physics/mynn_edmf.py",
     "_QSAT_CHI_DEN_REAL = (R4(T0C) - R4(6.0)) - R4(TICE)", "_QSAT_CHI_DEN_REAL = R4((T0C - 6.0) - TICE)"),
    ("xl_cice_site", "src/gpuwrf/physics/mynn_sgs_cloud.py",
     "real_constant(_CPV_M_CICE_REAL, CPV - CICE, dtype)", "(CPV - CICE)"),
    ("xl_chi_site", "src/gpuwrf/physics/mynn_sgs_cloud.py",
     "real_constant(_T0C_M_TICE_REAL, T0C - TICE, dtype)", "(T0C - TICE)"),
    ("q1rh_site_i", "src/gpuwrf/physics/mynn_sgs_cloud.py",
     "q1_rh_i = -3.0 + 3.0 * (rh - RHCRIT) / real_constant(_ONE_M_RHCRIT_REAL, 1.0 - RHCRIT, jnp.result_type(rh))",
     "q1_rh_i = -3.0 + 3.0 * (rh - RHCRIT) / (1.0 - RHCRIT)"),
    ("q1rh_site_c", "src/gpuwrf/physics/mynn_sgs_cloud.py",
     "q1_rh_c = -3.0 + 3.0 * (rh - RHCRIT) / real_constant(_ONE_M_RHCRIT_REAL, 1.0 - RHCRIT, jnp.result_type(rh))",
     "q1_rh_c = -3.0 + 3.0 * (rh - RHCRIT) / (1.0 - RHCRIT)"),
    ("level2_site", "src/gpuwrf/physics/mynn_pbl.py",
     "one_m_c2 = real_constant(_ONE_M_C2_REAL, 1.0 - C2, state.theta.dtype)", "one_m_c2 = (1.0 - C2)"),
    ("level2_const", "src/gpuwrf/physics/mynn_pbl.py", "_ONE_M_C2_REAL = R4(1.0) - R4(C2)", "_ONE_M_C2_REAL = R4(1.0 - C2)"),
    ("sl_ep3_site", "src/gpuwrf/physics/surface_layer.py",
     "ep_3 = real_constant(_EP_3_REAL, 1.0 - EP2, F)", "ep_3 = (1.0 - EP2)"),
    ("sl_inv_site", "src/gpuwrf/physics/surface_layer.py",
     "inv_t0 = real_constant(_INV_27315_REAL, 1.0 / 273.15, F)", "inv_t0 = (1.0 / 273.15)"),
    ("sl_ep3_const", "src/gpuwrf/physics/surface_layer.py",
     "_EP_3_REAL = R4(1.0) - R4(R_D) / R4(R_V)", "_EP_3_REAL = R4(1.0 - R_D / R_V)"),
    ("vkc_site", "src/gpuwrf/physics/noahmp/energy.py",
     "vkc2 = real_constant(_VKC2_REAL, VKC * VKC, jnp.result_type(cmfm))", "vkc2 = (VKC * VKC)"),
    ("rsurf_site", "src/gpuwrf/physics/noahmp/energy.py",
     "real_constant(_E_M1_REAL, 2.71828 - 1.0, jnp.result_type(smcmax1))", "(2.71828 - 1.0)"),
    ("snow_lists_site", "src/gpuwrf/physics/noahmp/snow.py",
     "excess = snliq_k - _liq_ratio(I[k]) * I[k]", "excess = snliq_k - MAX_LIQ_MASS_FRACTION / (1.0 - MAX_LIQ_MASS_FRACTION) * I[k]"),
    ("snow_const", "src/gpuwrf/physics/noahmp/snow.py",
     "_LIQ_RATIO_REAL = R4(MAX_LIQ_MASS_FRACTION) / (R4(1.0) - R4(MAX_LIQ_MASS_FRACTION))",
     "_LIQ_RATIO_REAL = R4(MAX_LIQ_MASS_FRACTION / (1.0 - MAX_LIQ_MASS_FRACTION))"),
    ("kf_site", "src/gpuwrf/kernels/phys_kf_column.py",
     "real_constant(_RHH_SLOPE_REAL, 0.2 / 1000.0)", "(0.2 / 1000.0)"),
    ("sw_site", "src/gpuwrf/kernels/rad_sw_band_sums.py",
     "_c(real_constant(_BPADE_REAL, 1.0 / 0.278))", "_c(1.0 / 0.278)"),
    ("dampmag_site", "src/gpuwrf/kernels/dyn_acoustic_fp32.py",
     "dampmag=_dampmag(cfg.dt, cfg.dampcoef),", "dampmag=cfg.dt*cfg.dampcoef,"),
    ("dampmag_const", "src/gpuwrf/kernels/dyn_acoustic_fp32.py",
     "return real_scalar_product(dts, dampcoef)", "return dts * dampcoef"),
    ("albedo_site", "src/gpuwrf/physics/noahmp/energy_radiation.py",
     "jnp.exp(real_scalar(_class_alb_arg(dt)))", "jnp.exp(real_scalar(-0.01 * dt / 3600.0))"),
    ("albedo_helper", "src/gpuwrf/physics/noahmp/energy_radiation.py",
     "return real_constant((R4(-0.01) * R4(dt)) / R4(3600.0), -0.01 * dt / 3600.0,",
     "return real_constant(R4(-0.01 * dt / 3600.0), -0.01 * dt / 3600.0,"),
    ("snow_ret_site", "src/gpuwrf/physics/noahmp/snow.py",
     "qout_k = jnp.maximum((vol_liq - epore[k]) * Z[k], real_scalar_product(SNOW_RET_FAC, dt, jnp.result_type(qout_k)) * qout_k)",
     "qout_k = jnp.maximum((vol_liq - epore[k]) * Z[k], SNOW_RET_FAC * dt * qout_k)"),
    ("snow_ret_helper", "src/gpuwrf/contracts/real_constfold.py",
     "return R4(factor) * R4(value)", "return R4(factor * value)"),
    ("traced_cast", "src/gpuwrf/contracts/real_constfold.py",
     "real = lambda x: x if is_host_scalar(x) else jnp.asarray(x, jnp.float32)  # noqa: E731",
     "real = lambda x: x  # noqa: E731"),
    ("flag_registry", "src/gpuwrf/runtime/aot_cheap_key.py",
     '    ("gpuwrf.contracts.real_constfold", "ENABLED"),  # w44-constfold WRF REAL constant folds\n', ""),
]
results = []
ONLY = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
for name, rel, old, new in MUTANTS:
    if ONLY and name not in ONLY:
        continue
    path = repo / rel
    original = path.read_text()
    assert original.count(old) == 1, (name, original.count(old))
    try:
        path.write_text(original.replace(old, new))
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp",
                               f"{basetemp}/{name}", "tests/test_w44_constfold.py"], cwd=repo, capture_output=True,
                              text=True, timeout=900)
    finally:
        path.write_text(original)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", tail)) else 0
    killed = failed > 0 and errors == 0 and "AssertionError" in proc.stdout + proc.stderr or (
        failed > 0 and errors == 0 and "assert" in proc.stdout)
    results.append({"mutant": name, "file": rel, "summary": tail, "failed": failed, "errors": errors, "killed": bool(killed)})
    print(name, "KILLED" if killed else "SURVIVED", tail, flush=True)
out.write_text(json.dumps({"mutants": results, "killed": sum(r["killed"] for r in results), "total": len(results)}, indent=1))
print("killed", sum(r["killed"] for r in results), "of", len(results))
