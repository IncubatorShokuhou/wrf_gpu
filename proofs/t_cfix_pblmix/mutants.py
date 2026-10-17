#!/usr/bin/env python3
"""t-cfix-pblmix genuine mutants (E39/E202): each removes one fix; the named test must
fail with an AssertionError (not a setup/collection error).  Sequential, in-tree,
every mutated file is restored with `git checkout` (run in an otherwise idle tree).
Usage: mutants.py <group> <out.json>   group in {F6, F8}
"""
import json
import os
import subprocess
import sys

T = "tests/test_t_cfix_pblmix_pbl_condensate.py"
T8 = "tests/test_t_cfix_pblmix_mynn_numbers.py"
T23 = "tests/test_t_cfix_pblmix_f23.py"
T17 = "tests/test_t_cfix_pblmix_gbm_rthraten.py"
MUTANTS = {
    "F6": [
        ("M1 MRF UST rebuild removed", "src/gpuwrf/physics/bl_mrf.py",
         "    ust = KARMAN * wspd / (gz1oz0 - psim)\n    ust = jnp.where((xland - 1.5) >= 0.0, ust, jnp.maximum(ust, 0.1))\n", "",
         f"{T}::test_mrf_double_matches_pristine_with_and_without_ice_slot"),
        ("M2 MRF adapter QC3D zeros", "src/gpuwrf/coupling/scan_adapters.py",
         "        # mixed (RQCBLTEN/RQIBLTEN) and they switch MRF2D's in-cloud Ri (:999-1000).\n        cols(state.qc),",
         "        # mixed (RQCBLTEN/RQIBLTEN) and they switch MRF2D's in-cloud Ri (:999-1000).\n        jnp.zeros_like(f[\"qv_cols\"]),",
         f"{T}::test_adapter_passes_state_condensate_and_applies_its_rates"),
        ("M3 MRF qi lane dropped", "src/gpuwrf/physics/bl_mrf.py",
         "    qixx = jnp.zeros(n, dtype=jnp.float64) if qi is None else _flip(qi)",
         "    qixx = jnp.zeros(n, dtype=jnp.float64)", f"{T}::test_mrf_double_matches_pristine_with_and_without_ice_slot"),
        ("M4 GFS PBLFLG off-by-one restored", "src/gpuwrf/physics/bl_gfs.py",
         "    pblflg2 = pblflg & (kpbl2 > 0)", "    pblflg2 = pblflg & (kpbl2 > 1)",
         f"{T}::test_gfs_double_matches_pristine_with_and_without_ice_tracer"),
        ("M5 GFS qi tracer dropped", "src/gpuwrf/physics/bl_gfs.py",
         "    if qi is None:\n        rqiblten = jnp.zeros_like(qctend)", "    if True:\n        rqiblten = jnp.zeros_like(qctend)",
         f"{T}::test_gfs_double_matches_pristine_with_and_without_ice_tracer"),
        ("M6 MYJ QCI species dropped", "src/gpuwrf/physics/bl_myj.py",
         "    species = species.at[4].set(qi_top)  # NSP=4: QCI (myjpbl.F:505-509)\n", "",
         f"{T}::test_myj_double_matches_pristine_with_and_without_qci"),
        ("M7 MYJ qc/qi increments not applied", "src/gpuwrf/physics/myj_adapters.py",
         "qke=qke_new,**condensate,", "qke=qke_new,",
         f"{T}::test_myj_adapter_passes_raw_condensate_and_applies_rates"),
        ("M8 F_QI gate always true", "src/gpuwrf/runtime/operational_mode.py",
         "    return int(mp_physics) not in (0, 1, 3)", "    return True",
         f"{T}::test_pbl_f_qi_follows_wrf_registry_qi_slot"),
    ],
    "F8": [
        ("N1 Nc solve floor dropped", "src/gpuwrf/kernels/phys_mynn_numbers.py",
         "            out[name] = jnp.where(sqc2 > c(1e-12), jnp.maximum(mixed, c(NC_MIN)), mixed)",
         "            out[name] = mixed", f"{T8}::test_number_solves_match_pristine_mynn_tendencies"),
        ("N2 aerosol bounds dropped", "src/gpuwrf/kernels/phys_mynn_numbers.py",
         "        out[name] = jnp.minimum(jnp.maximum(c(low) * decay, jnp.maximum(mixed, c(0.0))), c(high) * decay)",
         "        out[name] = jnp.maximum(mixed, c(0.0))", f"{T8}::test_number_solves_match_pristine_mynn_tendencies"),
        ("N3 passive MF flux dropped", "src/gpuwrf/physics/mynn_edmf.py",
         "            out[mf_flux_key(name)] = call(values)[\"s_awqni\"]",
         "            out[mf_flux_key(name)] = 0.0 * call(values)[\"s_awqni\"]",
         f"{T8}::test_passive_plume_fluxes_match_pristine_dmp_mf"),
        ("N4 runtime number rates not coupled", "src/gpuwrf/runtime/operational_mode.py",
         "                for name in _pbl_mynn_number_scalars(mp_opt):\n                    # pbl_driver.F:1879-1911",
         "                for name in ():\n                    # pbl_driver.F:1879-1911",
         f"{T8}::test_operational_mynn_couples_number_rates_once"),
        ("N6 RK stage drops the number sources", "src/gpuwrf/runtime/operational_mode.py",
         "if (n in physics_sc and n not in skip) else t",
         "if (n in physics_sc and n not in skip and n not in (\"Nc\", \"nwfa\", \"nifa\")) else t",
         f"{T8}::test_swiss_production_step_carries_number_sources_to_every_rk_stage"),
        ("N5 registry set wrong (Nc for mp8)", "src/gpuwrf/runtime/operational_mode.py",
         "    nc = (\"Nc\",) if mp in (9, 11, 14, 16, 26, 28, 38, 40, 51, 52, 53, 56) else ()",
         "    nc = (\"Nc\",) if mp in (8, 9, 11, 14, 16, 26, 28, 38, 40, 51, 52, 53, 56) else ()",
         f"{T8}::test_number_scalar_set_follows_wrf_registry"),
    ],
    "F23": [
        ("P1 opt_sfc=2 seed back to sfclay CH/CM", "src/gpuwrf/physics/noahmp_coupler.py",
         "    own_sfcdif = options is not None and int(getattr(options, \"opt_sfc\", 1)) == 2",
         "    own_sfcdif = False", f"{T23}::test_noahmp_sfcdif_seed_follows_wrf_chxy"),
        ("P2 RUC flat GSW", "src/gpuwrf/runtime/operational_mode.py",
         "ruc_net = RucRadiation(jnp.asarray(rdiag.swnorm) - jnp.asarray(rdiag.swup_topographic), rdiag.glw)",
         "ruc_net = RucRadiation(jnp.asarray(rdiag.swdown) - jnp.asarray(rdiag.swup), rdiag.glw)",
         f"{T23}::test_ruc_receives_topographic_gsw"),
        ("P3 SSiB flat SWDOWN", "src/gpuwrf/runtime/operational_mode.py",
         "for x in (rdiag.swnorm,rdiag.glw,rdiag.coszen))", "for x in (rdiag.swdown,rdiag.glw,rdiag.coszen))",
         f"{T23}::test_ssib_receives_topographic_swdown"),
    ],
    "F23S": [
        ("S1 QNSEPBL ignores SICE in the RLIVWV branch", "src/gpuwrf/physics/pbl_qnse.py",
         "qfc1 = jnp.where((snow > 0.) | (sice > .5), qfc1*c['RLIVWV'], qfc1)",
         "qfc1 = jnp.where((snow > 0.), qfc1*c['RLIVWV'], qfc1)",
         "tests/test_t_cfix_pblmix_qnse_sice.py::test_port_qnsepbl_matches_pristine_real4_bytes"),
    ],
    "F23R": [
        ("R1 nested loader skips the QNSE sea-ice admission", "src/gpuwrf/integration/nested_pipeline.py",
         "        refuse_qnse_sea_ice(admission_run, name, _domain_physics_int(admission_run, \"bl_pbl_physics\", name, 5))",
         "        pass",
         "tests/test_t_cfix_pblmix_qnse_sice.py::test_real_loaders_refuse_qnse_sea_ice_before_model_work[sea_ice-nested]"),
        ("R2 daily loader skips the QNSE sea-ice admission", "src/gpuwrf/integration/daily_pipeline.py",
         "    refuse_qnse_sea_ice(admission_run, config.domain,",
         "    (lambda *a: None)(admission_run, config.domain,",
         "tests/test_t_cfix_pblmix_qnse_sice.py::test_real_loaders_refuse_qnse_sea_ice_before_model_work[sea_ice-daily]"),
        ("R3 fractional sea ice admitted (threshold 0.5)", "src/gpuwrf/io/qnse_admission.py",
         "np.any(value > 0.0)", "np.any(value > 0.5)",
         "tests/test_t_cfix_pblmix_qnse_sice.py::test_qnse_sea_ice_admission_predicate"),
        ("R4 non-finite SEAICE admitted", "src/gpuwrf/io/qnse_admission.py",
         "if not np.isfinite(value).all() or np.any(value > 0.0):", "if np.any(value > 0.0):",
         "tests/test_t_cfix_pblmix_qnse_sice.py::test_qnse_sea_ice_admission_predicate"),
        ("R5 public Noah loader skips the admission", "src/gpuwrf/io/noahmp_land_init.py",
         "    check_qnse_sea_ice_fields(domain, domain_bl_pbl_physics(namelist, domain),",
         "    (lambda *a: None)(domain, domain_bl_pbl_physics(namelist, domain),",
         "tests/test_t_cfix_pblmix_qnse_sice.py::test_public_noah_loader_refuses_qnse_sea_ice[sea_ice]"),
        ("R6 public replay builder skips the admission", "src/gpuwrf/integration/d02_replay.py",
         "    refuse_qnse_sea_ice(run, domain, history_source=",
         "    (lambda *a, **k: None)(run, domain, history_source=",
         "tests/test_t_cfix_pblmix_qnse_sice.py::test_public_replay_builder_refuses_before_model_init[sea_ice]"),
        ("R7 only the first present alias is read", "src/gpuwrf/io/qnse_admission.py",
         "for key in SEA_ICE_ALIASES if key in fields}",
         "for key in [k for k in SEA_ICE_ALIASES if k in fields][:1]}",
         "tests/test_t_cfix_pblmix_qnse_sice.py::test_every_present_alias_is_checked"),
        ("R8 inconsistent aliases admitted", "src/gpuwrf/io/qnse_admission.py",
         "    if len(arrays) > 1:", "    if False:",
         "tests/test_t_cfix_pblmix_qnse_sice.py::test_every_present_alias_is_checked"),
        ("R9 replay initial-state source not checked", "src/gpuwrf/integration/d02_replay.py",
         "history_source=None if native or not load_lateral_boundaries",
         "history_source=None if True or native or not load_lateral_boundaries",
         "tests/test_t_cfix_pblmix_qnse_sice.py::test_public_replay_builder_checks_the_replay_initial_state_source"),
    ],
    "F17": [
        ("G1 RTHRATEN dropped in kernel", "src/gpuwrf/physics/bl_gbm.py",
         "    rttenx = jnp.zeros_like(thlx) if rthraten is None else (rthraten * pii)[::-1]",
         "    rttenx = jnp.zeros_like(thlx)", f"{T17}::test_double_port_matches_pristine_gbmpbl_transcription"),
        ("G2 pblhgt never extends upward", "src/gpuwrf/physics/bl_gbm.py",
         "        done = k < 2\n        ktop = s[\"ktop\"].at[ilay].set(k)",
         "        done = k < 9999\n        ktop = s[\"ktop\"].at[ilay].set(k)",
         f"{T17}::test_double_port_matches_pristine_gbmpbl_transcription"),
        ("G3 multi-layer my top treatment dropped", "src/gpuwrf/physics/bl_gbm.py",
         "    kzm, kth, kethl = jax.lax.fori_loop(1, ktop.shape[0], top, (kzm, kth, kethl))",
         "    kzm, kth, kethl = jax.lax.fori_loop(1, 1, top, (kzm, kth, kethl))",
         f"{T17}::test_double_port_matches_pristine_gbmpbl_transcription"),
        ("G4 operational GBM gets no RTHRATEN", "src/gpuwrf/runtime/operational_mode.py",
         "                **({'rthraten':next_carry.rthraten} if bl_opt==12 else {}),",
         "                **({} if bl_opt==12 else {}),",
         f"{T17}::test_operational_gbm_receives_held_rthraten"),
        ("G5 GBM reads RTHRATEN before this step's legacy refresh", "src/gpuwrf/runtime/operational_mode.py",
         "if int(namelist.bl_pbl_physics) in (1, 12) and next_carry.radiation_diagnostics is None:",
         "if int(namelist.bl_pbl_physics) in (1,) and next_carry.radiation_diagnostics is None:",
         "tests/test_t_cfix_pblmix_gbm_rthraten.py::test_radiation_step_gbm_reads_this_steps_rthraten_from_entry_state"),
    ],
}


def main():
    group, out = sys.argv[1], sys.argv[2]
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    results = []
    for label, path, old, new, test in MUTANTS[group]:
        src = open(path).read()
        assert src.count(old) == 1, (label, "anchor not unique/absent")
        open(path, "w").write(src.replace(old, new))
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                                "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/t-cfix-pblmix/pytest_mut", test],
                               capture_output=True, text=True, env=env, timeout=1200)
        finally:
            subprocess.run(["git", "checkout", "--", path], check=True)
        text = r.stdout + r.stderr
        last = text.strip().split("\n")[-1]
        genuine = (r.returncode == 1 and " failed" in last and " error" not in last
                   and ("AssertionError" in text or "\nE       assert" in text or "E   AssertionError" in text))
        results.append(dict(label=label, file=path, test=test, rc=r.returncode, killed=genuine,
                            tail=text.strip().split("\n")[-1]))
        print(label, "KILLED" if genuine else "SURVIVED/ERROR", results[-1]["tail"], flush=True)
    json.dump(results, open(out, "w"), indent=1)
    assert subprocess.run(["git", "diff", "--quiet", "--", "src"]).returncode == 0, "tree not restored"


if __name__ == "__main__":
    main()
