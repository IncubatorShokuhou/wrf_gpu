"""Apply one mutant at a time to the lane source, run the targeted tests, restore. Report kill + signature."""
import subprocess, sys, json
from pathlib import Path
root = Path(sys.argv[1])
MUTANTS = [
 ("M1 window slot ignores base", "src/gpuwrf/coupling/boundary_records.py",
  "    lower_slot, upper_slot = lower - base, upper - base\n", "    lower_slot, upper_slot = lower, upper\n",
  "tests/test_record_windows.py::test_windowed_consumer_reads_the_full_stack_records"),
 ("M2 kernel slot ignores base", "src/gpuwrf/kernels/dyn_boundary_fp32.py",
  "    slot = lower - base\n", "    slot = lower\n",
  "tests/test_record_windows.py::test_native_interpolation_kernel_window_is_bitwise_and_poisons"),
 ("M3 lateral range misses upper record", "src/gpuwrf/integration/record_windows.py",
  "        hi = math.floor(g_b + _FRACTION_TOLERANCE) + 1\n", "        hi = math.floor(g_b + _FRACTION_TOLERANCE)\n",
  "tests/test_record_windows.py::test_lateral_range_covers_every_bracket_of_the_call"),
 ("M4 aux sentinel never activates", "src/gpuwrf/integration/record_windows.py",
  "        steps.append(int(act[sentinel]) if sentinel < count else _INT32_MAX)\n", "        steps.append(_INT32_MAX)\n",
  "tests/test_record_windows.py::test_aux_window_selects_the_full_record_and_sentinel_is_loud"),
 ("M5 plan digest keeps real.exe keys", "src/gpuwrf/runtime/gpu_allocator.py",
  "                    if not key.startswith((\"start_\", \"end_\", \"run_\")) and key not in _REAL_EXE_ONLY_KEYS}\n",
  "                    if not key.startswith((\"start_\", \"end_\", \"run_\"))}\n",
  "tests/test_record_windows.py::test_memory_plan_digest_ignores_real_exe_only_keys"),
 ("M6 prepare trusts a stale window", "src/gpuwrf/integration/record_windows.py",
  "            if not (known and self._lateral_base <= lo and hi <= self._lateral_base + self.window - 1):\n",
  "            if not (known or self._lateral_base is not None):\n",
  "tests/test_record_windows.py::test_prepare_refreshes_only_when_the_window_moves_and_matches_records"),
 ("M7 out-of-window slot clipped (no NaN)", "src/gpuwrf/coupling/boundary_records.py",
  "    return lower, jnp.where(bad, records, lower_slot), jnp.where(bad, records, upper_slot)\n",
  "    return lower, jnp.clip(lower_slot, 0, records - 1), jnp.clip(upper_slot, 0, records - 1)\n",
  "tests/test_record_windows.py::test_windowed_consumer_reads_the_full_stack_records"),
 ("M8 finish skips the cast verification", "src/gpuwrf/integration/record_windows.py",
  "                if window.shape != full[rows].shape or window.tobytes() != full[rows].tobytes():\n",
  "                if False:\n",
  "tests/test_record_windows.py::test_finish_refuses_a_carry_that_is_not_a_cast_of_the_records"),
 ("M9 host cast ignores FTZ/NaN hazards", "src/gpuwrf/integration/record_windows.py",
  "        if not np.isfinite(host).all() or (target.itemsize < host.dtype.itemsize\n",
  "        if False and (target.itemsize < host.dtype.itemsize\n",
  "tests/test_record_windows.py::test_host_cast_equals_the_device_cast_and_keeps_hazards_on_device"),
 ("M10 prepare ignores the cast hazard", "src/gpuwrf/integration/record_windows.py",
  "            reason = _cast_hazard(raw)\n", "            reason = None\n",
  "tests/test_record_windows.py::test_cast_sensitive_records_keep_the_full_stacks"),
 ("M11 history reads the initial aux window", "src/gpuwrf/integration/record_windows.py",
  "        return lower_boundary_history(self._aux_window(self._aux_base), own_step)\n",
  "        return lower_boundary_history(self._aux_window(0), own_step)\n",
  "tests/test_record_windows.py::test_aux_window_selects_the_full_record_and_sentinel_is_loud"),
 ("M12 DFL prepass loads record windows", "src/gpuwrf/dfi/driver12.py",
  "    _h, bundles, _meta, _start, _dt, carries = pipeline._load_domains(config, (\"d01\",), record_windows=False)\n",
  "    _h, bundles, _meta, _start, _dt, carries = pipeline._load_domains(config, (\"d01\",))\n",
  "tests/test_t_dfidiff.py"),
 ("M13 namelist holder pickles its owner", "src/gpuwrf/runtime/operational_mode.py",
  "        return (_RecordWindowsAux, (None,))\n", "        return (_RecordWindowsAux, (self.value,))\n",
  "tests/test_record_windows.py::test_namelist_treedef_pickles_without_the_owner"),
 ("M14 plan digest filters whole lines (critic gate 5)", "src/gpuwrf/runtime/gpu_allocator.py",
  "    try:\n        groups = parse_namelist(namelist_path)\n",
  "    text = \"\\n\".join(line for line in text.splitlines() if not re.match(\n"
  "        r\"\\s*(?:start_|end_|run_)[A-Za-z_]+\\s*=\", line, re.IGNORECASE) and not any(\n"
  "        re.match(rf\"\\s*{key}\\s*=\", line, re.IGNORECASE) for key in _REAL_EXE_ONLY_KEYS))\n"
  "    return hashlib.sha256(text.encode()).hexdigest()\n"
  "    try:\n        groups = parse_namelist(namelist_path)\n",
  "tests/test_record_windows.py::test_memory_plan_digest_keeps_runtime_keys_sharing_a_line_with_dropped_keys"),
 ("M15 surface-w reset removed from the step body", "src/gpuwrf/runtime/operational_mode.py",
  '    if _W_SURFACE_RESET:\n        # WRF solve_em.F:4818-4834: after all RK stages, the microphysics and the boundary\n        # updates, every step ends with set_w_surface(fill_w_flag=.false.) from the final u/v.\n        if _SHARDED_CARRY_HALO_CONTEXT is not None:\n            raise NotImplementedError("GPUWRF_W_SURFACE_RESET does not support the sharded carry")\n        next_state = next_state.replace(w=_reset_surface_w(\n            next_state, namelist.grid, namelist.metrics,\n            periodic_x=_acoustic_lateral_bc_flags(namelist)[0]))\n',
  '    if _W_SURFACE_RESET:\n        # WRF solve_em.F:4818-4834: after all RK stages, the microphysics and the boundary\n        # updates, every step ends with set_w_surface(fill_w_flag=.false.) from the final u/v.\n        if _SHARDED_CARRY_HALO_CONTEXT is not None:\n            raise NotImplementedError("GPUWRF_W_SURFACE_RESET does not support the sharded carry")\n        pass\n',
  'tests/v025/b_core/test_w_surface_reset.py::test_reset_follows_the_boundary_pass_and_precedes_precision'),
 ("M16 surface-w reset after the precision pass", "src/gpuwrf/runtime/operational_mode.py",
  '    if _W_SURFACE_RESET:\n        # WRF solve_em.F:4818-4834: after all RK stages, the microphysics and the boundary\n        # updates, every step ends with set_w_surface(fill_w_flag=.false.) from the final u/v.\n        if _SHARDED_CARRY_HALO_CONTEXT is not None:\n            raise NotImplementedError("GPUWRF_W_SURFACE_RESET does not support the sharded carry")\n        next_state = next_state.replace(w=_reset_surface_w(\n            next_state, namelist.grid, namelist.metrics,\n            periodic_x=_acoustic_lateral_bc_flags(namelist)[0]))\n    if capture_rca:\n        state_health.append(_rca_state_health(next_state))\n        state_target.append(_rca_state_target(next_state))\n    next_state = _enforce_operational_precision(\n        next_state,\n        force_fp64=bool(namelist.force_fp64),\n        acoustic_precision_mode=namelist.acoustic_precision_mode,\n        base_state=carry.base_state,\n    )\n',
  '    if capture_rca:\n        state_health.append(_rca_state_health(next_state))\n        state_target.append(_rca_state_target(next_state))\n    next_state = _enforce_operational_precision(\n        next_state,\n        force_fp64=bool(namelist.force_fp64),\n        acoustic_precision_mode=namelist.acoustic_precision_mode,\n        base_state=carry.base_state,\n    )\n    if _W_SURFACE_RESET:\n        # WRF solve_em.F:4818-4834: after all RK stages, the microphysics and the boundary\n        # updates, every step ends with set_w_surface(fill_w_flag=.false.) from the final u/v.\n        if _SHARDED_CARRY_HALO_CONTEXT is not None:\n            raise NotImplementedError("GPUWRF_W_SURFACE_RESET does not support the sharded carry")\n        next_state = next_state.replace(w=_reset_surface_w(\n            next_state, namelist.grid, namelist.metrics,\n            periodic_x=_acoustic_lateral_bc_flags(namelist)[0]))\n',
  'tests/v025/b_core/test_w_surface_reset.py::test_reset_follows_the_boundary_pass_and_precedes_precision'),
]
results = []
for name, path, old, new, test in MUTANTS:
    p = root / path; text = p.read_text()
    assert text.count(old) == 1, (name, old)
    p.write_text(text.replace(old, new))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                            "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec/pytest/mut", test],
                           cwd=root, capture_output=True, text=True, timeout=600)
    finally:
        p.write_text(text)
    out = r.stdout + r.stderr
    # E202: only genuine test failures count (assert / pytest.raises DID NOT RAISE), never setup errors.
    genuine = (r.returncode == 1 and ("AssertionError" in out or "DID NOT RAISE" in out)
               and " error" not in out.strip().splitlines()[-1])
    results.append({"mutant": name, "rc": r.returncode, "killed_by_assertion": genuine, "tail": out.strip().splitlines()[-1]})
    print(json.dumps(results[-1]), flush=True)
Path(sys.argv[2] if len(sys.argv) > 2 else "<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec/probe/mutants.json").write_text(json.dumps(results, indent=1))
