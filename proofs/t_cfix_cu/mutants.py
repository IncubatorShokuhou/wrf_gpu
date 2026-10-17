"""Genuine mutants for t-cfix-cu F7/F9/F14/F18 (E202): each must be KILLED by an assertion
of its targeted test (no setup/collection errors); the source file is restored afterwards.

usage: JAX_PLATFORMS=cpu python proofs/t_cfix_cu/mutants.py OUT_JSON [BASETEMP]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OA = "src/gpuwrf/runtime/operational_mode.py"
MUTANTS = [
    ("F7 no dry conversion", "src/gpuwrf/coupling/cu_forcing.py",
     "    rthften = th_phy / t_total * (rthften - rvrd * th_phy * rqvften)\n", "",
     "tests/test_t_cfix_cu_g3_forcing.py::test_finish_matches_wrf_phy_prep_part2_literal"),
    ("F7 capture without msfty", "src/gpuwrf/coupling/cu_forcing.py",
     "rthften = held_rthften if theta_advection is None else jnp.asarray(theta_advection, dtype) * m\n",
     "rthften = held_rthften if theta_advection is None else jnp.asarray(theta_advection, dtype)\n",
     "tests/test_t_cfix_cu_tiedtke_forcing.py::test_capture_multiplies_by_msfty"),
    ("F7 Tiedtke RK computes the unused RTHFTEN theta capture", OA,
     "                        if cu_capture and int(namelist.cu_physics) in CU_RTHFTEN_SCHEMES else None)\n",
     "                        if cu_capture else None)\n",
     "tests/test_t_cfix_cu_tiedtke_forcing.py::test_rthften_capture_only_for_schemes_that_read_it[6]"),
    ("F7 Tiedtke phy_prep_part2 dry-converts RTHFTEN", OA,
     "            rthften=int(namelist.cu_physics) in CU_RTHFTEN_SCHEMES))\n", "            rthften=True))\n",
     "tests/test_t_cfix_cu_tiedtke_forcing.py::test_tiedtke_step_keeps_held_rthften_through_phy_prep_part2"),
    ("F7 slot drops RTHRATEN", OA,
     "            rthraten=next_carry.rthraten, gsw=gsw,\n", "            rthraten=None, gsw=gsw,\n",
     "tests/test_t_cfix_cu_g3_forcing.py::test_operational_slot_feeds_held_forcing_and_radiation"),
    ("F7 legacy RTHRATEN one radt stale (late radiation slot)", OA,
     "    if ((int(namelist.bl_pbl_physics) in (1, 12) or int(namelist.cu_physics) in CU_RTHRATEN_SCHEMES)\n",
     "    if ((int(namelist.bl_pbl_physics) in (1, 12))\n",
     "tests/test_t_cfix_cu_g3_forcing.py::test_legacy_radiation_reaches_cumulus_this_step_from_entry_state[5]"),
    ("F7 CPU-WRF restart hydration drops the held forcing", "src/gpuwrf/validation/wrf_restart_hydration.py",
     "    extra.update(_hydrate_cu_forcing(carry, restart.physics['CU_PHYSICS'], source, put))\n", "",
     "tests/test_cu_forcing_restart_schema.py::test_cpu_wrfrst_hydration_seeds_held_cumulus_forcing"),
    ("F7 captures the wrong species", OA,
     'cu_qv_adv = root_advection[q_species.index("qv")]', 'cu_qv_adv = root_advection[q_species.index("qc")]',
     "tests/test_t_cfix_cu_g3_forcing.py::test_capture_is_final_stage_theta_and_qv_advection_times_msfty"),
    ("F7 no phy_prep_part2 finish", OA,
     "        carry = carry.replace(cu_forcing=finish_cu_forcing(\n",
     "        carry = carry.replace(cu_forcing=carry.cu_forcing) or carry.replace(cu_forcing=finish_cu_forcing(\n",
     "tests/test_t_cfix_cu_g3_forcing.py::test_step_finishes_captured_forcing_at_phy_prep_part2"),
    ("F7 Euler-PBL increment left in the G3 input", OA,
     "            cu_state if source_leaf_mode else next_state, float(namelist.dt_s), namelist.grid,\n"
     "            forcing_entry_state=None if pbl_restored or source_leaf_mode else before,\n"
     "            rthblten=cu_rthbl",
     "            next_state, float(namelist.dt_s), namelist.grid,\n"
     "            forcing_entry_state=None if pbl_restored or source_leaf_mode else before,\n"
     "            rthblten=cu_rthbl",
     "tests/test_t_cfix_cu_g3_forcing.py::test_euler_pbl_increment_reaches_g3_as_rate_not_as_state"),
    ("F9 rates not mass-coupled", "src/gpuwrf/coupling/cu_rk.py",
     "coupled = {name: None if rate is None else mass * jnp.asarray(rate, dtype)",
     "coupled = {name: None if rate is None else jnp.asarray(rate, dtype)",
     "tests/test_t_cfix_cu_cu_rk.py::test_stateless_cumulus_feeds_rk_sources_not_state[5]"),
    ("F9 Tiedtke back to Euler", OA,
     "            qvpblten=cu_zero_first(qvpblten),\n            mp_physics=mp_opt,\n            return_rates=source_leaf_mode,\n",
     "            qvpblten=cu_zero_first(qvpblten),\n            mp_physics=mp_opt,\n            return_rates=False,\n",
     "tests/test_t_cfix_cu_cu_rk.py::test_stateless_cumulus_feeds_rk_sources_not_state[6]"),
    ("F9 held rates not decoupled", "src/gpuwrf/coupling/cu_rk.py",
     "    return tuple(rate if name == \"pratec\" else jnp.asarray(rate, dtype) / mass\n",
     "    return tuple(rate if name == \"pratec\" else jnp.asarray(rate, dtype)\n",
     "tests/test_t_cfix_cu_cu_rk.py::test_held_rates_decoupled_by_final_mass_after_the_rk_step"),
    ("F9 KF99 clear dropped", OA,
     "next_state, kf_tendf, cuplus_momentum, kf_rates = _held_cu_rk(next_state, kf_rates, 99, clear=kf_clear)",
     "next_state, kf_tendf, cuplus_momentum, kf_rates = _held_cu_rk(next_state, kf_rates, 99, clear=None)",
     "tests/test_t_cfix_cu_cu_rk.py::test_kf99_rk_route_clears_expired_rates_after_consumption"),
    ("F9 MYJ back to Euler", OA,
     "        if cu_opt==7 or (source_leaf_mode and _phys_tend_rk_wrf_enabled()):\n",
     "        if cu_opt==7:\n",
     "tests/test_t_cfix_cu_cu_rk.py::test_euler_pbls_publish_rk_sources_in_source_leaf_mode[2]"),
    ("F9 generic PBL back to Euler", OA,
     "    elif bl_opt in PBL_SCAN_ADAPTERS and source_leaf_mode and _phys_tend_rk_wrf_enabled():\n",
     "    elif bl_opt in PBL_SCAN_ADAPTERS and False:\n",
     "tests/test_t_cfix_cu_cu_rk.py::test_euler_pbls_publish_rk_sources_in_source_leaf_mode[3]"),
    ("F14 NTiedtke THFTEN without RTHRATEN", OA,
     " + jnp.asarray(next_carry.rthraten, jnp.float64) + rthbl16)", " + rthbl16)",
     "tests/test_t_cfix_cu_tiedtke_forcing.py::test_ntiedtke_gets_wrf_thften_and_qvften[2]"),
    ("F14 no itimestep-1 zeroing", OA,
     "cu_zero_first = lambda forcing: jnp.where(cu_first, jnp.zeros_like(forcing), forcing)",
     "cu_zero_first = lambda forcing: forcing",
     "tests/test_t_cfix_cu_tiedtke_forcing.py::test_tiedtke_gets_held_rqvften_and_step1_zero[1]"),
    ("F14 Tiedtke ZNU pressure proxy", "src/gpuwrf/coupling/scan_adapters.py",
     "    if eta is not None and int(jnp.shape(eta)[0]) == nz + 1:\n",
     "    if False:\n",
     "tests/test_t_cfix_cu_tiedtke_forcing.py::test_tiedtke_gets_wrf_znu_half_levels"),
    ("F18 GF lateral band dropped", "src/gpuwrf/coupling/scan_adapters.py",
     "owned = owned & ((jnp.arange(nx) >= 4) & (jnp.arange(nx) <= nx - 5))[None, :]",
     "owned = owned & ((jnp.arange(nx) >= 0) & (jnp.arange(nx) <= nx - 1))[None, :]",
     "tests/test_t_cfix_cu_gf_caller.py::test_gf_writes_only_wrf_interior_columns"),
    ("F18 GF PSUR from the lowest mass level", "src/gpuwrf/coupling/scan_adapters.py",
     'psur_b = None if prep["psfc"] is None else jnp.asarray(prep["psfc"], jnp.float64).reshape(ny * nx)',
     'psur_b = None if prep["psfc"] is None else jnp.asarray(prep["p"][0], jnp.float64).reshape(ny * nx)',
     "tests/test_t_cfix_cu_gf_caller.py::test_operational_gf_slot_hands_wrf_inputs_to_gfdrv"),
    ("F18 GF ignores the KPBL the GFS PBL published", OA,
     "            kpbl=pbl_published.get('kpbl'),\n", "            kpbl=None,\n",
     "tests/test_t_cfix_cu_gf_caller.py::test_gf_reads_the_kpbl_published_by_the_gfs_pbl"),
    ("F10 GF ishallow=0 still runs CUP_gf_sh (shallow diagnostics + outqcs leak)", "src/gpuwrf/physics/_gf_jax.py",
     "    if ishallow_g3 == 1:\n        sh = cup_gf_sh(", "    if True:\n        sh = cup_gf_sh(",
     "tests/test_t_cfix_cu_gf_ishallow.py::test_ishallow0_port_has_no_shallow_terms"),
    ("F18 GF kernel drops RTHRATEN", "src/gpuwrf/physics/_gf_jax.py",
     "tn0 = t_col + (rthften_col + rthraten_col + rthblten_col) * pi_col * dt",
     "tn0 = t_col + (rthften_col + rthblten_col) * pi_col * dt",
     "tests/test_t_cfix_cu_gf_ishallow.py::test_port_wrf_caller_forcing_matches_pristine_gfdrv"),
]


def main(out, basetemp):
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(ROOT / "src"))
    results = []
    for name, rel, old, new, test in MUTANTS:
        path = ROOT / rel
        original = path.read_text()
        assert original.count(old) == 1, (name, original.count(old))
        try:
            path.write_text(original.replace(old, new))
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                   "--basetemp", basetemp, test], cwd=ROOT, env=env, capture_output=True,
                                  text=True, timeout=1800)
        finally:
            path.write_text(original)
        text = proc.stdout + proc.stderr
        killed = proc.returncode == 1 and ("AssertionError" in text or "\nE   " in text and "assert" in text) \
            and " error" not in text.splitlines()[-1]
        results.append(dict(mutant=name, file=rel, test=test, rc=proc.returncode, killed=bool(killed),
                            tail=text.strip().splitlines()[-1] if text.strip() else ""))
        print(json.dumps(results[-1]), flush=True)
    Path(out).write_text(json.dumps(results, indent=1) + "\n")
    assert all(r["killed"] for r in results), [r["mutant"] for r in results if not r["killed"]]


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "<USER_HOME>/wrf_gpu2_lanes/t-cfix-cu/pytest_mut")
