"""Genuine mutants for the ens presets (lane w41-stoch): each mutant must turn a named test RED through a
test assertion (assert / AssertionError / pytest.raises DID NOT RAISE), not a setup error or an unexpected
exception from the code under test, and the unmutated tree must be GREEN (E202). A mutant may be several
(old, new) hunks. First run (2026-10-17 13:2xZ) counted M2 (ValueError from the product's own seed-collision
guard) and M5 (KeyError) as kills: those were not test assertions; M2 now also disables the guard (M7 tests
the guard alone) and the M5 test asserts with .get().
Runs in an immutable detached snapshot: python check_mutants.py <snapshot> <basetemp> <out.json> [presets|rng]
"""
import json, subprocess, sys
from pathlib import Path

snap, basetemp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SET = sys.argv[4] if len(sys.argv) > 4 else "presets"
F = snap/"src/gpuwrf/ens/presets.py"
T = "tests/test_ensemble_presets.py"
R = "tests/test_stochastic_rng_parallel.py"
RNG = {
    "R1_lane_start_off_by_one_proposal": ("    for bit in bin(2*block)[3:]:", "    for bit in bin(2*block+2)[3:]:",
                                          [R+"::test_jump_table_equals_sequential_stepping"]),
    "R2_resume_drops_last_proposal": ("2*(last - lane*block + 1)", "2*(last - lane*block)",
                                      [R+"::test_parallel_stream_is_bitwise_serial"]),
    "R3_compaction_ignores_acceptance": ("    accepted = ok.astype(bool)\n", "    accepted = jnp.ones_like(ok, bool)\n",
                                         [R+"::test_parallel_stream_is_bitwise_serial"]),
    "R4_fallback_condition_inverted": ("jnp.sum(ok) >= count", "jnp.sum(ok) < count",
                                       [R+"::test_chained_calls_and_fallback_are_bitwise"]),
}
MUTANTS = {
    "M1_seed_no_int4_wrap": ("    group = [wrap(iseed+nens*1000000), mod(19211), mod(71209), mod(11279)]",
                             "    group = [iseed+nens*1000000, mod(19211), mod(71209), mod(11279)]",
                             [T+"::test_host_seed_copy_matches_model_rand_seed"]),
    "M2_same_nens_every_member": ([("            nens = nens_start + index", "            nens = nens_start"),
                                   ("                    if word in used:", "                    if False:")],
                                  None, [T+"::test_default_schemes_and_nens_start",
                                         T+"::test_stochastic_members_distinct_nens_unique_seeds_shared_inputs"]),
    "M7_no_seed_collision_guard": ("                    if word in used:", "                    if False:",
                                   [T+"::test_seed_collision_is_refused"]),
    "M3_no_model_stochastic_admission": ("    errors = stochastic_refusals(member)\n", "    errors = []\n",
                                         [T+"::test_stochastic_refusals_write_nothing[sppt_ysu]"]),
    "M4_no_spectral_setup_check": ("    check_spectra(member)\n", "",
                                   [T+"::test_default_sppt_scale_resolved_on_swiss_grid_but_spp_pbl_default_refused"]),
    "M5_drop_row_extra_keys": ("            text = _set_keys(text, \"physics\", {k: [v]*max_dom for k, v in physics.items()})",
                               "            text = _set_keys(text, \"physics\", {k: [v]*max_dom for k, v in physics.items() if k != 'use_mp_re'})",
                               [T+"::test_multiphysics_rows_only_physics_changes"]),
    "M6_no_spp_mynn_guard": ("    if \"spp\" in schemes and _first(member.get(\"physics\", {}).get(\"bl_pbl_physics\")) != 5:",
                             "    if False:", [T+"::test_stochastic_refusals_write_nothing[spp_ysu]"]),
}
basetemp.mkdir(parents=True, exist_ok=True)
if SET == "rng":
    F, MUTANTS = snap/"src/gpuwrf/physics/stochastic_rng.py", RNG
env = {"JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0", "PATH": "/usr/bin:/bin:<USER_HOME>/miniconda3/bin",
       "HOME": "<USER_HOME>"}


def run(nodes, tag):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(basetemp/tag), *nodes]
    r = subprocess.run(cmd, cwd=snap, env=env, capture_output=True, text=True, timeout=900)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return r.returncode, tail, r.stdout


original = F.read_text()
result = {"baseline": None, "mutants": {}}
nodes_all = sorted({n for m in MUTANTS.values() for n in m[2]})
rc, tail, _ = run(nodes_all, "baseline")
result["baseline"] = {"rc": rc, "summary": tail}
assert rc == 0, tail
try:
    for name, (old, new, nodes) in MUTANTS.items():
        hunks = old if isinstance(old, list) else [(old, new)]
        text = original
        for a, b in hunks:
            assert text.count(a) == 1, name
            text = text.replace(a, b)
        F.write_text(text)
        rc, tail, stdout = run(nodes, name)
        elines = [l[1:].strip() for l in stdout.splitlines() if l.startswith("E ")]
        assertion = any(l.startswith(("assert ", "AssertionError", "Failed: DID NOT RAISE")) for l in elines)
        unexpected = [l for l in elines if l.split(":")[0].endswith(("Error", "Exception")) and not l.startswith("AssertionError")]
        killed = rc == 1 and " error" not in tail and "failed" in tail and assertion and not unexpected
        result["mutants"][name] = {"rc": rc, "summary": tail, "killed": killed, "assertion": assertion,
                                   "unexpected_exceptions": unexpected[:3], "signature": elines[:3]}
        F.write_text(original)
finally:
    F.write_text(original)
assert subprocess.run(["git", "-C", str(snap), "diff", "--quiet"]).returncode == 0
result["all_killed"] = all(m["killed"] for m in result["mutants"].values())
out.write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps({k: (v["killed"], v["summary"]) for k, v in result["mutants"].items()}, indent=1), result["all_killed"])
