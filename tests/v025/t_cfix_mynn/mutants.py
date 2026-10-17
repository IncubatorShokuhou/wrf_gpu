"""F20 genuine mutants (E39/E202): each must make named tests FAIL with an assertion, with 0 setup errors.

Run (CPU, one process at a time, restores every file in ``finally``):
  JAX_PLATFORMS=cpu python tests/v025/t_cfix_mynn/mutants.py <basetemp> <out.json>
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src/gpuwrf"
TEST = "tests/v025/t_cfix_mynn/test_f20_mynn_caller.py"
MUTANTS = {
    "M1_mass_w": ("coupling/physics_couplers.py", "column(_w_bottom_faces(state, dtype=wind_dtype)),",
                  "column(_w_mass(state, dtype=wind_dtype)),",
                  ["test_caller_hands_mynn_the_bottom_w_faces", "test_two_step_operational_mynn_reads"]),
    "M2_upper_faces": ("coupling/physics_couplers.py", "    return w[:-1, :, :]\n", "    return w[1:, :, :]\n",
                       ["test_caller_hands_mynn_the_bottom_w_faces", "test_two_step_operational_mynn_reads"]),
    "M3_kte_face": ("physics/mynn_edmf.py", " & (jnp.arange(nz) < nz - 1)", "",
                    ["test_taper_reads_faces_kts_to_kte_minus_1_only"]),
    "M4_no_limits": ("physics/surface_layer.py", "    if pbl_limits:\n", "    if False:\n",
                     ["test_driver_flux_limits_match_pristine_public_call_bytewise",
                      "test_mynn_view_alone_applies_the_driver_limits"]),
    "M5_producer_limits_leak": ("physics/noahmp_coupler.py", "**_mynn_surface_exner_kwargs(state, r))",
                                "**_mynn_surface_exner_kwargs(state, r), pbl_limits=True)",
                                ["test_extreme_fluxes_reach_cumulus_unlimited",
                                 "test_shared_handle_producers_stay_unlimited"]),
    "M6_view_unlimited": ("coupling/physics_couplers.py", "    return tuple(jnp.where(beyond,",
                          "    return (theta_flux, qv_flux, fltv) or tuple(jnp.where(beyond,",
                          ["test_mynn_view_alone_applies_the_driver_limits"]),
    # Host-transfer gate: no control flow in the limit view (a cond predicate is a per-step host read inside the
    # stepping while loops). E41 PROD d01: the kts operands come from kts slices, never from a 3-D intermediate.
    "M7_host_predicate_cond": ("coupling/physics_couplers.py", "    return tuple(jnp.where(beyond,",
                               "    return jax.lax.cond(jnp.any(beyond), lambda _: (theta_flux, qv_flux, fltv), "
                               "lambda _: (theta_flux, qv_flux, fltv), None) and tuple(jnp.where(beyond,",
                               ["test_limit_view_is_branch_free_and_forms_kts_operands_only"]),
    "M8_full_column_rho": ("coupling/physics_couplers.py",
                           "_wrf_phy_prep_rho_from_state(state, metrics, kts_only=True)",
                           "_wrf_phy_prep_rho_from_state(state, metrics)[0]",
                           ["test_limit_view_is_branch_free_and_forms_kts_operands_only"]),
    "M9_full_column_exner": ("coupling/physics_couplers.py",
                             "_mynn_exner_from_pressure(jnp.asarray(state.p[0], dtype=_mynn_real_dtype()))",
                             "_mynn_exner_from_pressure(jnp.asarray(state.p, dtype=_mynn_real_dtype()))[0]",
                             ["test_limit_view_is_branch_free_and_forms_kts_operands_only"]),
    "M10_kts_wrong_faces": ("coupling/physics_couplers.py", "(lambda field, n: field[:n]) if kts_only",
                            "(lambda field, n: field[1:n + 1]) if kts_only",
                            ["test_mynn_view_alone_applies_the_driver_limits"]),
}


def main(basetemp, out):
    results = {}
    Path(basetemp).mkdir(parents=True, exist_ok=True)  # E202: a missing --basetemp parent is a setup error, not a kill
    for name, (rel, old, new, expect) in MUTANTS.items():
        path = SRC / rel
        original = path.read_text()
        assert original.count(old) == 1, (name, rel)
        try:
            path.write_text(original.replace(old, new))
            sel = " or ".join(expect)
            env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
            proc = subprocess.run([sys.executable, "-m", "pytest", TEST, "-q", "-k", sel, "-p", "no:cacheprovider",
                                   "--basetemp", f"{basetemp}/{name}", "-rf"],
                                  cwd=ROOT, env=env, capture_output=True, text=True, timeout=2400)
        finally:
            path.write_text(original)
        tail = proc.stdout[-6000:]
        failed = re.findall(r"^FAILED \S+::(\S+)", proc.stdout, re.M)
        summary = re.findall(r"^=*\s*(\d+ (?:failed|passed|error).*?) in [0-9.]+s", proc.stdout, re.M)
        errors = re.findall(r"(\d+) error", summary[-1] if summary else "")
        results[name] = {"file": rel, "failed": failed, "summary": summary[-1] if summary else None,
                         "setup_errors": int(errors[0]) if errors else 0,
                         "assertion": "AssertionError" in proc.stdout or "assert " in proc.stdout,
                         "killed": bool(failed) and not errors and all(any(e in f for f in failed) for e in expect),
                         "tail": tail[-1500:]}
        print(name, results[name]["killed"], results[name]["summary"], flush=True)
    Path(out).write_text(json.dumps(results, indent=1) + "\n")
    assert all(r["killed"] for r in results.values()), {k: v["summary"] for k, v in results.items()}


if __name__ == "__main__":
    main(*sys.argv[1:3])
