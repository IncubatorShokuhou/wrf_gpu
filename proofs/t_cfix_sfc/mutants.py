"""Genuine product mutants for the selected-surface series (lane t-cfix-sfc).

usage: JAX_PLATFORMS=cpu python proofs/t_cfix_sfc/mutants.py <repo> <basetemp> <out.json> [myj|sf0|aot]
Each mutant edits ONE product line in place, runs the named test node, and the file
is restored from git in ``finally``. KILLED requires an AssertionError (E202): setup
errors or collection failures count as NOT killed.
"""
import json, subprocess, sys
from pathlib import Path

repo, basetemp, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
T = 'tests/v040/test_t_cfix_sfc_selected_surface.py'
MUTANTS = [
    ('noah_gets_mynn_resolve', 'src/gpuwrf/runtime/operational_mode.py',
     "            **({'surface_diagnostics':selected_noah_surface} if selected_noah_surface is not None else {}),\n", '',
     f'{T}::test_noahmp_receives_the_selected_surface_call'),
    ('moist_theta_operand', 'src/gpuwrf/coupling/sfclay_selected.py',
     "theta = real(_dry_theta_view(state))", "theta = real(state.theta)",
     f'{T}::test_two_step_caller_threads_wrf_memory_and_phy_prep_operands'),
    ('mm5_legacy_constants', 'src/gpuwrf/coupling/sfclay_selected.py',
     "'pblh', 'mavail', 'lakemask', 'dx', 'hfx', 'qfx')), wrf_constants=True)",
     "'pblh', 'mavail', 'lakemask', 'dx', 'hfx', 'qfx')), wrf_constants=False)",
     f'{T}::test_selected_call_inside_pristine_wrf_envelope'),
    ('gfs_cp_1004', 'src/gpuwrf/physics/sfclay_gfs.py',
     "cp_w, r_w = real(1004.5), real(287.0)", "cp_w, r_w = real(1004.0), real(287.0)",
     f'{T}::test_selected_call_inside_pristine_wrf_envelope'),
    ('mm5_f64_table', 'src/gpuwrf/physics/sfclay_old_mm5.py',
     "        PSIMTB = jnp.asarray(_PSIMTB_REAL, ux.dtype)\n", "",
     f'{T}::test_selected_call_inside_pristine_wrf_envelope'),
    ('memory_not_written', 'src/gpuwrf/coupling/sfclay_selected.py',
     "roughness_m=out['znt'], mol=mol, qsfc=out['qsfc'], hfx=out['hfx'], qfx=out['qfx'])",
     "roughness_m=out['znt'])",
     f'{T}::test_two_step_caller_threads_wrf_memory_and_phy_prep_operands'),
    ('gfs_pbl_legacy_resolve', 'src/gpuwrf/runtime/operational_mode.py',
     "(selected_surface_forcing is not None and bl_opt == 3)", "(selected_surface_forcing is not None and bl_opt == -3)",
     f'{T}::test_gfs_pbl_consumes_the_selected_sf_gfs_solve'),
    ('gfs_admission_open', 'src/gpuwrf/coupling/physics_dispatch.py',
     "    if pbl_opt == 3 and sfclay_opt != 3:", "    if pbl_opt == 3 and sfclay_opt not in (1, 3, 5, 7, 91):",
     f'{T}::test_gfs_pbl_admission_is_wrf_isfc2'),
    ('px_constant_f64', 'src/gpuwrf/physics/sfclay_pleim_xiu.py',
     "4648.0 * (jnp.asarray(1.0, tsk.dtype) / jnp.asarray(SVPT0_K, tsk.dtype) - 1.0 / tsk)",
     "4648.0 * (1.0 / SVPT0_K - 1.0 / tsk)",
     f'{T}::test_selected_call_inside_pristine_wrf_envelope'),
    ('history_held_t2', 'src/gpuwrf/coupling/sfclay_selected.py',
     "    if int(sf) == 91:\n        values.update(t2=noah.t2, q2=noah.q2)", "    if int(sf) == -91:\n        values.update(t2=noah.t2, q2=noah.q2)",
     f'{T}::test_no_lsm_history_is_the_selected_call'),
    ('gfs_pbl_f32_solver', 'src/gpuwrf/coupling/scan_adapters.py',
     "        out = gfs_bl_real_double(args, kwargs, float(dt))", "        out = gfs_columns(*args, dt=float(dt), **kwargs)",
     'tests/v040/test_t_cfix_sfc_blgfs.py::test_actual_sf3_bl3_caller_is_the_pristine_bl_gfs_call'),
    ('gfs_pbl_ust_dropped', 'src/gpuwrf/coupling/scan_adapters.py',
     "        written = {'ustar': out['ust'].reshape(shape).astype(state.ustar.dtype)}", "        written = {}",
     'tests/v040/test_t_cfix_sfc_blgfs.py::test_bl_gfs_ust_and_wspd_inout_survive_into_the_next_surface_call'),
    ('gfs_pbl_wspd_dropped', 'src/gpuwrf/coupling/scan_adapters.py',
     "        if getattr(state, 'sfc_wspd', None) is not None:\n            u0 = jnp.asarray(f[\"u_cols\"][:, 0]",
     "        if False:\n            u0 = jnp.asarray(f[\"u_cols\"][:, 0]",
     'tests/v040/test_t_cfix_sfc_blgfs.py::test_bl_gfs_ust_and_wspd_inout_survive_into_the_next_surface_call'),
    ('gfs_pbl_real_column_kernel', 'src/gpuwrf/physics/bl_gfs.py',
     "    dt2 = 2.0 * dt", "    dt2 = dt",
     'tests/v040/test_t_cfix_sfc_blgfs.py::test_wrf_bl_gfs_on_real_columns_fed_by_pristine_sf_gfs'),
    # rv-t-cfix SFC2 series 1: REAL right-hand sides before the kind_phys store; WRF MONINP literals.
    ('blgfs_q1_widened_first', 'src/gpuwrf/physics/bl_gfs.py',
     "    q1 = dbl(qv / (one + qv))\n", "    q1 = dbl(qv) / (1.0 + dbl(qv))\n",
     'tests/v040/test_t_cfix_sfc_blgfs.py::test_bl_gfs_real_setup_is_byte_identical_to_the_pristine_moninp_operands'),
    ('blgfs_prsl_widened_first', 'src/gpuwrf/physics/bl_gfs.py',
     "    prsl = dbl(p * lit(0.001))\n", "    prsl = dbl(p) * 0.001\n",
     'tests/v040/test_t_cfix_sfc_blgfs.py::test_bl_gfs_real_setup_is_byte_identical_to_the_pristine_moninp_operands'),
    ('blgfs_phii_double', 'src/gpuwrf/physics/bl_gfs.py',
     "    phii = dbl((z - z[0]) * WRF_G)\n", "    phii = (dbl(z) - dbl(z[0])) * dbl(WRF_G)\n",
     'tests/v040/test_t_cfix_sfc_blgfs.py::test_bl_gfs_real_setup_is_byte_identical_to_the_pristine_moninp_operands'),
    ('blgfs_del_reciprocal', 'src/gpuwrf/physics/bl_gfs.py',
     "    rovg = jax.lax.optimization_barrier(dbl(WRF_ROVG))\n", "    rovg = dbl(WRF_ROVG)\n",
     'tests/v040/test_t_cfix_sfc_blgfs.py::test_bl_gfs_real_setup_is_byte_identical_to_the_pristine_moninp_operands'),
    ('blgfs_moninp_exact_decimals', 'src/gpuwrf/physics/bl_gfs.py',
     "jnp.asarray(pi, jnp.float64), deltim, WRF_MONINP))", "jnp.asarray(pi, jnp.float64), deltim, LEGACY_MONINP))",
     'tests/v040/test_t_cfix_sfc_blgfs.py::test_double_moninp_matches_pristine_moninp_on_its_captured_operands'),
]
M = 'tests/v040/test_t_cfix_sfc_myj.py'
MUTANTS_MYJ = [
    ('myjpbl_private_resolve', 'src/gpuwrf/runtime/operational_mode.py',
     "        if myj_surface is not None:\n            # CHKLOWQ", "        if myj_surface is None:\n            # CHKLOWQ",
     f'{M}::test_noahmp_and_myjpbl_consume_this_steps_myjsfc'),
    ('elflx_pre_lsm', 'src/gpuwrf/runtime/operational_mode.py',
     "        if qnse_surface is not None or myj_surface is not None:\n            # ELFLX",
     "        if qnse_surface is not None:\n            # ELFLX",
     f'{M}::test_noahmp_and_myjpbl_consume_this_steps_myjsfc'),
    ('chklowq_mavail', 'src/gpuwrf/physics/myj_adapters.py',
     "chklowq = jnp.clip(g[\"mavail\"], 0.0, 1.0) if surface is None else",
     "chklowq = jnp.clip(g[\"mavail\"], 0.0, 1.0) if surface is not None else",
     f'{M}::test_noahmp_and_myjpbl_consume_this_steps_myjsfc'),
    ('no_cold_start', 'src/gpuwrf/coupling/myj_surface.py',
     "itimestep=jnp.where(jnp.asarray(first_timestep), 1, 2)", "itimestep=jnp.where(jnp.asarray(first_timestep), 2, 2)",
     f'{M}::test_two_steps_thread_myjsfc_memory'),
    ('legacy_psi_tables', 'src/gpuwrf/physics/sfclay_janjic.py',
     "return (jnp.asarray(_PSIM_REAL_NP, jnp.float32)", "return (jnp.asarray(_PSIM_NP, jnp.float32)",
     f'{M}::test_myjsfc_three_chained_calls_inside_pristine_wrf_envelope'),
    ('pbl_memory_dropped', 'src/gpuwrf/runtime/operational_mode.py',
     "        if myj_memory:\n", "        if False:\n",
     f'{M}::test_two_steps_thread_myjsfc_memory'),
    ('tke_doubled', 'src/gpuwrf/coupling/myj_surface.py',
     "tke=cols(f32(.5) * real(state.qke))", "tke=cols(real(state.qke))",
     f'{M}::test_two_steps_thread_myjsfc_memory'),
    ('restart_inactive', 'src/gpuwrf/io/restart_schema.py',
     "    myj=physics_option(namelist,'sf_sfclay_physics')==2", "    myj=physics_option(namelist,'sf_sfclay_physics')==-2",
     f'{M}::test_myj_memory_restart_roundtrip_and_refusal'),
    ('noah_diag_dropped', 'src/gpuwrf/runtime/operational_mode.py',
     "            **({'surface_diagnostics':myj_surface} if myj_surface is not None else {}),\n", '',
     f'{M}::test_noahmp_and_myjpbl_consume_this_steps_myjsfc'),
    ('akhs_memory_ignored', 'src/gpuwrf/physics/sfclay_janjic.py',
     "akhs_in=f(0.0 if akhs is None else akhs)", "akhs_in=f(0.0)",
     f'{M}::test_myjsfc_three_chained_calls_inside_pristine_wrf_envelope'),
]
MUTANTS_SF0 = [
    ('sf0_surface_runs', 'src/gpuwrf/runtime/operational_mode.py', "    if sf_opt == 0:\n        # module_surface_driver.F:1490",
     "    if sf_opt == -1:\n        # module_surface_driver.F:1490",
     'tests/v040/test_t_cfix_sfc_sf0.py::test_sf0_runs_no_surface_layer_and_no_land_model'),
    ('sf0_ocean_runs', 'src/gpuwrf/runtime/operational_mode.py',
     "    if namelist.sf_ocean_physics == 1 and sf_opt != 0:", "    if namelist.sf_ocean_physics == 1:",
     'tests/v040/test_t_cfix_sfc_sf0.py::test_sf0_also_skips_ocean_mixed_layer_sst_update_and_sst_skin'),
    ('sf0_sst_update_runs', 'src/gpuwrf/runtime/operational_mode.py',
     "    if lower_fields is not None and sf_opt != 0:", "    if lower_fields is not None:",
     'tests/v040/test_t_cfix_sfc_sf0.py::test_sf0_also_skips_ocean_mixed_layer_sst_update_and_sst_skin'),
    ('sf0_sst_skin_runs', 'src/gpuwrf/runtime/operational_mode.py',
     "    if (namelist.sst_skin or carry.sst_skin is not None) and sf_opt != 0:",
     "    if (namelist.sst_skin or carry.sst_skin is not None):",
     'tests/v040/test_t_cfix_sfc_sf0.py::test_sf0_also_skips_ocean_mixed_layer_sst_update_and_sst_skin'),
]
A = 'tests/v040/test_t_cfix_sfc_aotkey.py'
AOT = 'src/gpuwrf/runtime/aot_baked_tables.py'
MUTANTS_AOT = [  # rv-t-cfix SFC2 series 2: new baked tables must reach the key
    ('aot_myj_unkeyed', AOT, '            modules.update(("sfclay_janjic", "sf_myj", "myj_constants", "bl_myj"))\n',
     '            pass\n', f'{A}::test_changed_table_changes_full_key_hlo_and_output'),
    ('aot_sf91_unkeyed', AOT, '            modules.add("sfclay_old_mm5")', '            pass',
     f'{A}::test_changed_table_changes_full_key_hlo_and_output'),
    ('aot_gfs_unkeyed', AOT, '            modules.update(("sfclay_gfs", "bl_gfs"))', '            pass',
     f'{A}::test_changed_table_changes_full_key_hlo_and_output'),
]
if len(sys.argv) > 4 and sys.argv[4] == 'aot':
    MUTANTS = MUTANTS_AOT
if len(sys.argv) > 4 and sys.argv[4] == 'myj':
    MUTANTS = MUTANTS_MYJ
if len(sys.argv) > 4 and sys.argv[4] == 'sf0':
    MUTANTS = MUTANTS_SF0
results = {}
Path(basetemp).mkdir(parents=True, exist_ok=True)  # E202: a missing basetemp parent is a setup error
for name, path, old, new, node in MUTANTS:
    f = repo / path
    text = f.read_text()
    assert text.count(old) == 1, (name, 'anchor')
    try:
        f.write_text(text.replace(old, new))
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', '--basetemp',
                            f'{basetemp}/{name}', node], cwd=repo, capture_output=True, text=True, timeout=1500)
    finally:
        subprocess.run(['git', 'checkout', '--', path], cwd=repo, check=True)
    tail = r.stdout[-3000:]
    # pytest.raises 'DID NOT RAISE' is the assertion form of a refusal test.
    asserted = 'AssertionError' in r.stdout or 'DID NOT RAISE' in r.stdout
    killed = r.returncode != 0 and asserted and ' error' not in tail.splitlines()[-1]
    results[name] = dict(test=node, rc=r.returncode, killed=killed, summary=tail.splitlines()[-1])
    print(name, results[name])
assert subprocess.run(['git', 'diff', '--quiet', '--', 'src'], cwd=repo).returncode == 0
out.write_text(json.dumps(results, indent=1) + '\n')
print('killed', sum(v['killed'] for v in results.values()), 'of', len(results))
