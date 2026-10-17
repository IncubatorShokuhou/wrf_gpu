"""t-wsmdz genuine mutants (E39/E202): each source deletion must fail the caller gates by assertion.

Runs inside an immutable detached snapshot of the candidate; mutates one file, runs the
gate file, restores the file from git. KILLED only if pytest rc==1, every failure is an
AssertionError and there are 0 errors (setup errors are never kills).
Usage: python mutants.py <snapshot_root> <out.json> <basetemp>
"""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

root, out, basetemp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
GATE = "tests/test_t_wsmdz_wsm_callers.py"
MUTANTS = {
    "M1_drop_wsm_wdm_radius_selection": ("src/gpuwrf/physics/rrtmg_mp_re.py",
                                         'config["mp_physics"] in (3, 4, 6, 8, 14, 16, 18, 22, 24, 26, 28)',
                                         'config["mp_physics"] in (8, 18, 22, 24, 28)'),
    "M5_drop_wsm3_rrtmg_phase_view": ("src/gpuwrf/coupling/physics_couplers.py",
                                      "    if int(mp_physics) == 3:\n        # WSM3 carries ice in qc",
                                      "    if False:\n        # WSM3 carries ice in qc"),
    "M6_phase_view_for_wsm5_too": ("src/gpuwrf/coupling/physics_couplers.py",
                                   "    if int(mp_physics) == 3:\n        # WSM3 carries ice in qc",
                                   "    if int(mp_physics) in (3, 4):\n        # WSM3 carries ice in qc"),
    "M4_wdm7_reseeds_disabled_radii": ("src/gpuwrf/coupling/scan_adapters.py",
                                       "        mp_physics=26, use_mp_re=int(state.re_cloud is not None))",
                                       "        mp_physics=26)"),
    "M2_delz_legacy_gravity": ("src/gpuwrf/coupling/scan_adapters.py",
                               "jnp.float32(WRF_PHYSICS_G)", "jnp.float32(9.80665)"),
    "M3_den_legacy_thompson_density": ("src/gpuwrf/coupling/scan_adapters.py",
                                       "    if metrics is None:\n        return _rho_from_state(state), delz",
                                       "    if True:\n        return _rho_from_state(state), delz"),
}
env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
result = {"snapshot": str(root), "gate": GATE,
          "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
          "mutants": {}}
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()


def run(tag):
    xml = basetemp / f"{tag}.xml"
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "pythonpath=src",
                           "--basetemp", str(basetemp / tag), "--junitxml", str(xml), GATE],
                          cwd=root, env=env, capture_output=True, text=True, timeout=1500)
    tree = ET.parse(xml).getroot()
    suite = tree if tree.tag == "testsuite" else tree.find("testsuite")
    fails = [(c.get("name"), (c.find("failure").get("message") or "")[:160])
             for c in suite.iter("testcase") if c.find("failure") is not None]
    errors = [c.get("name") for c in suite.iter("testcase") if c.find("error") is not None]
    return dict(rc=proc.returncode, tests=int(suite.get("tests")), failures=fails, errors=errors,
                tail=proc.stdout.splitlines()[-1:] if proc.stdout else [])


result["control"] = run("control")
assert result["control"]["rc"] == 0 and not result["control"]["failures"], result["control"]
for tag, (rel, old, new) in MUTANTS.items():
    path = root / rel
    text = path.read_text()
    assert text.count(old) == 1, (tag, old)
    path.write_text(text.replace(old, new))
    try:
        r = run(tag)
    finally:
        subprocess.run(["git", "checkout", "--", rel], cwd=root, check=True)
    r["killed"] = (r["rc"] == 1 and bool(r["failures"]) and not r["errors"]
                   and all("AssertionError" in m or "assert" in m.lower() or "Arrays are not equal" in m
                           for _, m in r["failures"]))
    result["mutants"][tag] = r
    print(tag, r["killed"], len(r["failures"]), r["errors"], flush=True)
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
out.write_text(json.dumps(result, indent=1))
print("ALL_KILLED", all(m["killed"] for m in result["mutants"].values()))
