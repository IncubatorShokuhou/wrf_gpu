"""Genuine mutants for tests/a1_da/test_member_driver.py on a hard-linked copy of the tree (never the worktree).
Each mutant edits ONE file (unlink + write, so the hard-linked original is untouched) and must make at least one
test FAIL with an assertion/raises failure (not a setup error, E202)."""
import os, shutil, subprocess, sys
from pathlib import Path

TREE = Path(sys.argv[1]).resolve()
WORK = Path(sys.argv[2])
MUTANTS = [
    ("process registry not seeded", "src/gpuwrf/runtime/domain_tree.py",
     "        for key, call in _PROCESS_AOT_MEMO.items():", "        for key, call in {}.items():"),
    ("in-process Compiled published", "src/gpuwrf/runtime/domain_tree.py",
     "        if call is not None and not isinstance(call, jax.stages.Compiled):", "        if call is not None:"),
    ("offline-bootstrap process accepted", "src/gpuwrf/assimilation/letkf/member_driver.py",
     '        if getattr(gpuwrf, "_OFFLINE_COMMAND", False):', "        if False:"),
    ("namelist pin dropped", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "        if signature != self.namelist_signature:", "        if False:"),
    ("env pin dropped", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "        if environment != self.environment:", "        if False:"),
    ("time_control kept in signature", "src/gpuwrf/assimilation/letkf/member_driver.py",
     'if name != "time_control"}', "}"),
    ("prefetch never disabled", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "            aot_prefetch = not (self.share_executables and self.runs)", "            aot_prefetch = True"),
    ("zero increment re-rounded", "src/gpuwrf/assimilation/letkf/restart_inject.py",
     "    return np.where(increment == 0.0, old, new)", "    return new"),
    ("theta_m without moisture factor", "src/gpuwrf/assimilation/letkf/restart_inject.py",
     "        return (np.asarray(frame[\"T\"], np.float64) + 300.0) * (\n            1.0 + RVOVRD * np.maximum(np.asarray(frame[\"QVAPOR\"], np.float64), 0.0))",
     "        return np.asarray(frame[\"T\"], np.float64) + 300.0"),
    ("MU change accepted", "src/gpuwrf/assimilation/letkf/restart_inject.py",
     "    if not np.array_equal(np.asarray(an[\"MU\"]), np.asarray(bg[\"MU\"])):", "    if False:"),
    ("window 0 per arm", "src/gpuwrf/assimilation/letkf/cycle_loop.py",
     "            first_fc = out / \"streams\" / restart_arms[0]", "            first_fc = out / \"streams\" / restart_arms[-1] / \"x\""),
    ("forcing hours = window always", "src/gpuwrf/assimilation/letkf/cycle_loop.py",
     "    return int(cfg[\"window_hours\"]) * (int(cfg[\"cycles\"]) if restart else 1)", "    return int(cfg[\"window_hours\"])"),
    ("restart resume not extended", "src/gpuwrf/assimilation/letkf/cycle_loop.py",
     "                            resume_checkpoint=resume, extend_run=resume is not None)",
     "                            resume_checkpoint=resume, extend_run=False)"),
    ("zero restart arm analysed like letkf", "src/gpuwrf/assimilation/letkf/cycle_loop.py",
     "                if _seam(cfg, arm) == \"restart\" and cfg[\"arms\"][arm][\"kind\"] == \"zero_update\":",
     "                if False:"),
    ("relative paths accepted", "src/gpuwrf/assimilation/letkf/cycle_loop.py",
     "    if not isinstance(value, str) or not value or not Path(value).is_absolute():",
     "    if not isinstance(value, str) or not value:"),
    # review cr-w41-da-cycle 15:57Z
    ("analyses matched by label first (reviewed bug)", "src/gpuwrf/assimilation/letkf/cycle_loop.py",
     "        match = by_background.get(str(Path(frame).resolve()))",
     "        match = (label, members[label]) if label in members else by_background.get(str(Path(frame).resolve()))"),
    ("labels may hold several path components", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "    if not isinstance(label, str) or not _LABEL.fullmatch(label):", "    if not isinstance(label, str):"),
    ("protected-input scan disabled", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "    if hits:\n        raise ProtectedInputError", "    if False:\n        raise ProtectedInputError"),
    ("first-window child unguarded", "src/gpuwrf/assimilation/letkf/cycle_loop.py",
     '    refuse_protected_inputs("first-window inputs",', '    (lambda *a, **k: None)("first-window inputs",'),
    ("zero-update child unguarded", "src/gpuwrf/assimilation/letkf/cycle_loop.py",
     '    refuse_protected_inputs("zero-update inputs",', '    (lambda *a, **k: None)("zero-update inputs",'),
    ("loop inputs unguarded", "src/gpuwrf/assimilation/letkf/cycle_loop.py",
     '    refuse_protected_inputs("cycle inputs", [cfg["source_case"]', '    (lambda *a, **k: None)("cycle inputs", [cfg["source_case"]'),
    ("member run unguarded", "src/gpuwrf/assimilation/letkf/member_driver.py",
     '        refuse_protected_inputs(f"member {member}", [case_dir,', '        (lambda *a, **k: None)(f"member {member}", [case_dir,'),
    ("members manifest unguarded", "src/gpuwrf/assimilation/letkf/member_driver.py",
     '    refuse_protected_inputs("members manifest", [path])', "    pass"),
    ("members manifest cases unguarded", "src/gpuwrf/assimilation/letkf/member_driver.py",
     '    refuse_protected_inputs("members manifest cases", cases.values())', "    pass"),
    # review cr-w41-da-cycle delta 16:20Z
    ("namelist file names not resolved", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "                for value in _namelist_strings(path):", "                for value in []:"),
    ("case directory walked top level only", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "            if entry.is_dir():", "            if entry.is_dir() and directory == root:"),
    ("relative names not resolved under the case dir", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "        for root in (base, Path.cwd()):", "        for root in (Path.cwd(),):"),
    ("environment paths not scanned", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "                                        paths.jax_cache_dir)] + _environment_paths()",
     "                                        paths.jax_cache_dir)]"),
    ("output roots not checked", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "                                outputs=[member_root, self.scratch_root, *filter(None, [checkpoint_dir])],",
     "                                outputs=[],"),
    # review cr-w41-da-cycle 16:45Z: runtime roots + open-layer backstop
    ("runtime roots not walked", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "                _walk(Path(root), runtime_check, namelists=False)", "                runtime_check(Path(root))"),
    ("runtime resolver roots dropped", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "    return [(root(), True) for root in (paths.wrf_run_dir, assets.cache_root, paths.data_root,",
     "    return [(root(), True) for root in () if (paths.wrf_run_dir, assets.cache_root, paths.data_root,"),
    ("links not resolved in the walk", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "            if entry.is_symlink() and check(path):", "            if False and check(path):"),
    ("python-open backstop off", "src/gpuwrf/assimilation/letkf/member_driver.py",
     '    _OPEN_SPY["refused"] = refused', '    _OPEN_SPY["refused"] = None'),
    ("netCDF backstop off", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "    netCDF4.Dataset = checked_dataset", "    netCDF4.Dataset = original"),
    ("swallowed refusal ignored", "src/gpuwrf/assimilation/letkf/member_driver.py",
     "            if refused:\n                raise ProtectedInputError(f\"member {member}: the forecast",
     "            if False:\n                raise ProtectedInputError(f\"member {member}: the forecast"),
]
results = []


def run(name, rel=None, old=None, new=None):
    """Copy the tree (hard links), apply at most one edit, run the suite; return the failing test ids."""
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    subprocess.run(["cp", "-al", str(TREE / "src"), str(WORK / "src")], check=True)
    for item in ("tests/a1_da", "tests/conftest.py", "tests/_jax_cache_isolation.py", "tests/__init__.py",
                 "tests/test_aot_executable.py", "pyproject.toml", "examples", "scripts/v025"):
        src = TREE / item
        if src.is_dir():
            shutil.copytree(src, WORK / item, symlinks=True)
        elif src.exists():
            (WORK / item).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, WORK / item)
    (WORK / "data").symlink_to(TREE / "data")   # read-only tables/fixtures
    if rel is not None:
        target = WORK / rel
        text = target.read_text()
        assert text.count(old) == 1, (name, text.count(old))
        os.unlink(target)                      # break the hard link before writing
        target.write_text(text.replace(old, new))
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(WORK / "src"))
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider",
                           "tests/a1_da/test_member_driver.py", "--basetemp", str(WORK.parent / "mut_pytest")],
                          cwd=WORK, env=env, capture_output=True, text=True, timeout=1800)
    failed = sorted({line.split()[1] for line in proc.stdout.splitlines() if line.startswith("FAILED ")})
    errors = sorted({line.split()[1] for line in proc.stdout.splitlines() if line.startswith("ERROR ")})
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return proc.returncode, failed, errors, tail


rc, failed, errors, tail = run("baseline")
print(f"BASELINE rc={rc} failed={failed} errors={errors} {tail}", flush=True)
if rc != 0 or failed or errors:
    raise SystemExit("baseline must pass before mutants mean anything")
for name, rel, old, new in MUTANTS:
    rc, failed, errors, tail = run(name, rel, old, new)
    genuine = rc == 1 and bool(failed) and not errors
    results.append((name, genuine))
    print(f"{'KILLED' if genuine else 'SURVIVED/INVALID'} {name}: failed={failed} errors={errors} | {tail}", flush=True)
shutil.rmtree(WORK)
print("killed", sum(r[1] for r in results), "of", len(results))
