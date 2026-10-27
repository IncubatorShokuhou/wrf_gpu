"""Gate-overlap mutants (o5-port): each must make test_rrtmg_precision_gate_overlap.py FAIL with an AssertionError."""
import hashlib, re, subprocess, sys
from pathlib import Path

tree = Path(sys.argv[1]); gate = tree / "tests/v025/b_phys/rrtmg_precision_gate.py"
orig = gate.read_bytes(); sha = hashlib.sha256(orig).hexdigest()
src = orig.decode()
M = {
 "M1_no_pin": ("            with overlap(False):\n                record['lw'] = T1.run_tier1_lw(args.out / 'lw.json')\n                record['sw'] = T1.run_tier1_sw(args.out / 'sw.json')\n",
               "            record['lw'] = T1.run_tier1_lw(args.out / 'lw.json')\n            record['sw'] = T1.run_tier1_sw(args.out / 'sw.json')\n"),
 "M2_pin_noop": ("    rad_mcica._MAXRAND = bool(maxrand)\n", "    pass\n"),
 "M3_always_arm_c": ("    arm, other = ('C', 'D') if maxrand else ('D', 'C')\n", "    arm, other = ('C', 'D')\n"),
 "M4_no_restore": ("        rad_mcica._MAXRAND = saved\n", "        pass\n"),
 "M5_lw_truth_unpinned": ("    with overlap(False):\n        pinned = jax.block_until_ready(fn(state, with_clear_sky=True))\n",
                          "    if True:\n        pinned = jax.block_until_ready(fn(state, with_clear_sky=True))\n"),
 "M6_no_imported_clear": ("                                     ('imported', result, [f for f in truth['fields'] if f.startswith('clear_')])):\n",
                          "                                     ):\n"),
}
res = {}
try:
    for name, (old, new) in M.items():
        assert src.count(old) == 1, name
        gate.write_text(src.replace(old, new))
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp",
                            f"/home/wz/o5port/pytest_radgate_{name}", "tests/v025/b_phys/test_rrtmg_precision_gate_overlap.py"],
                           cwd=tree, capture_output=True, text=True)
        out = p.stdout + p.stderr
        failed = re.findall(r"^FAILED (\S+)", out, re.M)
        errors = re.findall(r"^ERROR (\S+)", out, re.M)
        res[name] = dict(rc=p.returncode, failed=failed, errors=errors, assertion="AssertionError" in out,
                         summary=(re.findall(r"^=*\s*(\d+ \w+.*) in [\d.]+s", out, re.M) or ["?"])[-1])
        print(name, res[name], flush=True)
finally:
    gate.write_bytes(orig)
    assert hashlib.sha256(gate.read_bytes()).hexdigest() == sha
print("RESTORED", sha)
print("KILLED_ALL", all(r["rc"] != 0 and r["failed"] and not r["errors"] and r["assertion"] for r in res.values()))
