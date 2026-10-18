"""Genuine mutants for topo_wind (lane w41-topowind, E39/E202).

A hard-linked sandbox of src/ + tests/ (the one mutated file is unlinked and rewritten,
never written through a shared inode), one mutant at a time, the targeted tests of
tests/test_v041_topo_wind.py run with cwd=sandbox (pyproject pythonpath=src -> sandbox
src, E93). A mutant counts as KILLED only with an assertion failure in a test call and
no collection/setup error (E202). Writes <out>/MUTANTS.json.
usage: JAX_PLATFORMS=cpu python mutants.py <repo> <sandbox_dir> <out_dir>
"""
import json, os, re, shutil, subprocess, sys
from pathlib import Path

REPO, SBX, OUT = (Path(a).resolve() for a in sys.argv[1:4])
T = "tests/test_v041_topo_wind.py"
MUTANTS = [
    ("M1 start_em alog -> log10", "src/gpuwrf/io/topo_wind_static.py",
     "_alog(np.where(small, F4(1.0), ctopo))", "np.log10(np.where(small, F4(1.0), ctopo)).astype(F4)",
     "test_start_em_ctopo_is_byte_identical"),
    ("M2 start_em CTOPO also over water", "src/gpuwrf/io/topo_wind_static.py",
     "ctopo = np.where(land, np.sqrt(np.where(land, vs, F4(1.0))), ctopo)", "ctopo = np.sqrt(vs)",
     "test_start_em_ctopo_is_byte_identical"),
    ("M3 start_em CTOPO2 ramp dropped", "src/gpuwrf/io/topo_wind_static.py",
     "ctopo2 = np.where(keep | blend, ctopo2, np.where(ramp, ramped, F4(0.0)))", "ctopo2 = ctopo2",
     "test_start_em_ctopo_is_byte_identical"),
    ("M4 YSU adapter drops CTOPO", "src/gpuwrf/coupling/scan_adapters.py",
     "            ysu_topdown_pblmix=int(ysu_topdown_pblmix),\n            **_topo_wind_column_kwargs(topo_wind_ctopo),\n",
     "            ysu_topdown_pblmix=int(ysu_topdown_pblmix),\n", "test_ysu_adapter_passes_flat_real_ctopo_columns"),
    ("M5 step drops CTOPO on the sf1 PBL branch", "src/gpuwrf/runtime/operational_mode.py",
     "else selected_surface_forcing), _with_sources=True, **topo_kw,", "else selected_surface_forcing), _with_sources=True,",
     "test_operational_step_hands_start_em_ctopo"),
    ("M6 step never publishes the CTOPO2 U10/V10 blend", "src/gpuwrf/runtime/operational_mode.py",
     "        next_carry = _publish_topo_wind_u10(next_carry, topo_pbl_rates, topo_kw['topo_wind_ctopo'][1])\n",
     "        pass\n", "test_operational_step_hands_start_em_ctopo"),
    ("M7 Shin-Hong drag (ctopo*ust)*ust", "src/gpuwrf/physics/bl_shinhong.py",
     "fric = ctopo * (ust * ust) / wspd1", "fric = ctopo * ust * ust / wspd1",
     "test_literal_real4_shinhong_with_real_ctopo"),
    ("M8 Shin-Hong CTOPO2 blend toward u10 instead of ux", "src/gpuwrf/physics/bl_shinhong.py",
     '"u10": c2 * u10 + (1.0 - c2) * u[:, 0]', '"u10": c2 * u10 + (1.0 - c2) * u10',
     "test_literal_real4_shinhong_with_real_ctopo"),
    ("M9 feedback ignores the grandchild", "src/gpuwrf/io/topo_wind_static.py",
     "jnp.asarray(fed[child][k])", "jnp.asarray(own[child][k])", "test_two_way_feedback_composes"),
    ("M10 first step uses the fed-back CTOPO", "src/gpuwrf/runtime/operational_mode.py",
     "first = jnp.rint(jnp.asarray(lead_seconds) / float(namelist.dt_s)) == 0",
     "first = jnp.rint(jnp.asarray(lead_seconds) / float(namelist.dt_s)) < 0",
     "test_two_way_feedback_selects_own_ctopo_only_on_the_first_step"),
    ("M11 admission ignores GOT_VAR_SSO", "src/gpuwrf/io/topo_wind_static.py",
     "if var_sso is None or not got_var_sso:", "if var_sso is None:", "test_topo_wind_admission_refuses_like_wrf"),
    ("M12 TOPO_WIND attribute for every option", "src/gpuwrf/io/wrf_history_metadata.py",
     "if int(domain_value('physics', 'topo_wind', 0)) == 1:", "if True:", "test_history_writes_topo_wind_attribute"),
]


def sandbox():
    if SBX.exists():
        shutil.rmtree(SBX)
    SBX.mkdir(parents=True)
    # tests/conftest.py imports scripts/v025/cpu_guard.py; runtime assets live in data/ (data/wrf_pristine
    # stays a symlink); consumer fixtures read proofs/v060 savepoints. Unmutated control must pass (E202).
    for d in ("src", "tests", "scripts", "data", "proofs"):
        subprocess.run(["cp", "-al", str(REPO / d), str(SBX / d)], check=True)
    shutil.copy2(REPO / "pyproject.toml", SBX / "pyproject.toml")


def write_new_inode(path, text):
    path.unlink()
    path.write_text(text)


def main():
    sandbox()
    rows = []
    for name, rel, old, new, test in MUTANTS:
        path = SBX / rel
        original = path.read_text()
        assert original.count(old) == 1, (name, original.count(old))
        write_new_inode(path, original.replace(old, new))
        proc = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", T, "-k", test,
                               "--basetemp", str(OUT / "pytest")], cwd=SBX, capture_output=True, text=True,
                              env=dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0"), timeout=1800)
        write_new_inode(path, original)
        tail = proc.stdout[-3000:]
        failed = re.search(r"(\d+) failed", tail)
        errors = re.search(r"(\d+) error", tail)
        # pytest.raises/pytest.warns report a missing exception/warning as Failed: DID NOT RAISE/WARN
        # (assertion helpers inside the test call, never setup/collection errors).
        assertion = ("AssertionError" in proc.stdout or "assert " in proc.stdout.split("short test summary")[-1]
                     or "DID NOT RAISE" in proc.stdout or "DID NOT WARN" in proc.stdout)
        killed = bool(failed) and not errors and assertion
        rows.append(dict(mutant=name, file=rel, test=test, rc=proc.returncode, killed=killed,
                         summary=tail.strip().splitlines()[-1] if tail.strip() else proc.stderr.strip()[-300:]))
        print(rows[-1], flush=True)
    # The unmutated sandbox must pass the same selection (no false kills).
    sel = " or ".join(sorted({m[4] for m in MUTANTS}))
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", T, "-k", sel,
                           "--basetemp", str(OUT / "pytest")], cwd=SBX, capture_output=True, text=True,
                          env=dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0"), timeout=3600)
    report = dict(mutants=rows, all_killed=all(r["killed"] for r in rows),
                  control_rc=proc.returncode,
                  control_summary=(proc.stdout.strip().splitlines() or proc.stderr.strip().splitlines() or [""])[-1])
    (OUT / "MUTANTS.json").write_text(json.dumps(report, indent=1))
    shutil.rmtree(SBX)
    print(json.dumps({k: v for k, v in report.items() if k != "mutants"}))


if __name__ == "__main__":
    main()
