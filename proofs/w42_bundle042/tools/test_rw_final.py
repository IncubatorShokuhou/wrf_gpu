"""CPU self-test of rw_final.py on copies of the real follow-up inputs (no GPU, no JAX).
usage: python3 test_rw_final.py FOLLOWUP_DIR MAILBOX DECISION_LINE WORK_DIR   (WORK_DIR must not exist; caller removes it)
Cases: real inputs -> RW3 PASS / integrity binding PASS; M1 an extra degenerate field (d02 QSNOW) -> integrity binding FAIL,
RW3 FAIL; M2 a second QGRAUP frame flagged -> FAIL; M3 RW3T verdict FAIL -> RW3 FAIL; M4 a non-decision mailbox line -> refused.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F, MB, LINE, WORK = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
assert not WORK.exists(), WORK
NAMES = ("RW_VERDICT.json", "d6_N72/summary.json", "QGRAUP_D03_CENSUS.json", "RW3T_VERDICT.json", "RW3T_ACCEPT.json",
         "FLOOR72_N72.json")


def case(name, mutate=None, line=LINE):
    d = WORK / name
    for n in NAMES:
        (d / n).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(F / n, d / n)
    if mutate:
        mutate(d)
    r = subprocess.run([sys.executable, str(HERE / "rw_final.py"), str(d), str(MB), str(line), str(d / "out")],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    if r.returncode:
        return None
    return json.loads((d / "out" / "RW_TABLE_FINAL.json").read_text())


def edit(rel, fn):
    def m(d):
        p = d / rel
        x = json.loads(p.read_text())
        fn(x)
        p.write_text(json.dumps(x))
    return m


def add_degenerate(dom, field, frame):
    def fn(s):
        s["domains"][dom]["output_integrity"].setdefault("degenerate_fields", {}).setdefault(field, []).append(
            {"file": frame, "gpu_value": 0.0, "cpu_range": [0.0, 1e-3]})
    return fn


checks = []
t = case("real")
checks += [("real RW3 PASS", t and t["RW3"]["verdict"] == "PASS" and t["RW3"]["integrity_binding"] == "PASS")]
t = case("m1", edit("d6_N72/summary.json", add_degenerate("d02", "QSNOW", "wrfout_d02_2026-03-03_00:00:00")))
checks += [("M1 extra field -> FAIL", t and t["RW3"]["integrity_binding"] == "FAIL" and t["RW3"]["verdict"] == "FAIL")]
t = case("m2", edit("d6_N72/summary.json", add_degenerate("d03", "QGRAUP", "wrfout_d03_2026-03-01_12:00:00")))
checks += [("M2 second QGRAUP frame -> FAIL", t and t["RW3"]["integrity_binding"] == "FAIL")]
t = case("m3", edit("RW3T_VERDICT.json", lambda v: v.update(verdict="FAIL")))
checks += [("M3 RW3T FAIL -> RW3 FAIL", t and t["RW3"]["verdict"] == "FAIL")]
checks += [("M4 wrong line refused", case("m4", line=LINE - 1) is None)]
bad = [n for n, ok in checks if not ok]
print(json.dumps({"checks": len(checks), "failed": bad}))
sys.exit(1 if bad else 0)
