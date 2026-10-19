"""CPU self-test of hero_summary.py (no GPU, no JAX): synthetic hero session roots built from a REAL fixed-format 24 h PROD
harness receipt (w41-bdyrec S1 whole24, old harness; allocator/pre_exec/launch fields added as the fixed harness writes
them) -> asserts the pre-registered hero rule: clean-arm selection, DISTURBED fallback, attestation refusal, par3 validity.
usage: python3 test_hero_summary.py WORK_DIR      (WORK_DIR must not exist; removed by the caller after the test)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path("<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec/receipts_b1_bcd156d19a0b/S1/whole24/run/receipt.json")
WORK = Path(sys.argv[1])
assert not WORK.exists(), WORK
BUDGET = 3227730974


def prod_arm(root, name, last_after_t0, await_ms, attested=True, compiled_new=0):
    d = root / "runs" / name
    (d / "run" / "wrfout").mkdir(parents=True)
    r = json.loads(SRC.read_text())
    t0 = r["t0_wall_utc"]
    a = r["cli_argv"]
    a[a.index("--output-dir") + 1] = str(d / "run" / "wrfout")
    if name == "W0":
        a[a.index("--hours") + 1] = "3"
    r["launch_wall_utc"] = t0 - 1.0
    r["allocator"] = {"mode": "c_auto" if attested else "demand", "c_auto_budget_bytes": BUDGET, "mem_fraction": "0.095880"}
    r["pre_exec"] = {"pid": r["pid"], "nvml_probe": {"backend_dark": True}}
    r["jax_memory_stats"]["bytes_limit"] = BUDGET + 55878
    (d / "run" / "receipt.json").write_text(json.dumps(r))
    files = {}
    for i, dom in enumerate(("d01", "d02")):
        p = d / "run" / "wrfout" / f"wrfout_{dom}_2026-07-26_18:00:00"
        p.write_bytes(dom.encode())
        files[p.name] = t0 + last_after_t0 - i
    (d / "endpoint.json").write_text(json.dumps({"valid": True, "t0_wall_utc": t0, "harness_launch_wall_utc": t0 - 1.0,
                                                 "session_launch_unix": t0 - 3.83, "file_write_unix": files}))
    (d / "launch_unix").write_text(f"{t0 - 3.83}\n")
    (d / "t_end_unix").write_text(f"{t0 + last_after_t0 + 5}\n")
    (d / "stdout.log").write_text(f"BUNDLE_CONSUMER rc=0 compiled_new={compiled_new} aliases_new=0 runtime_new=0 other_new=0 layer_compiled=0\n")
    (d / "stderr.log").write_text("gpuwrf: C-auto case pool 3.01 GiB, fraction=0.095880\n")
    (d / "disk_await.json").write_text(json.dumps({"await_ms": await_ms, "disturbed": await_ms > 50}))
    (d / "vram_apps.csv").write_text(f"2026/10/19 15:00:00.000, {r['pid']}, 3898\n2026/10/19 15:00:00.500, {r['pid']}, 3901\n")


def wn3_arm(root, starts, await_ms=19.36, cpool=True):
    d = root / "runs" / "WN3"
    (d / "out").mkdir(parents=True)
    l0 = 1792400000.0
    (d / "launch_unix").write_text(f"{l0}\n")
    (d / "t_end_unix").write_text(f"{l0 + 460}\n")
    mt = {}
    for j, (c, last) in enumerate((("wn3_0227", 450.83), ("wn3_0502", 442.64), ("wn3_0614", 446.12))):
        mt[c] = {f"wrfout_d0{1 + k % 3}_{k:03d}": l0 + last - (74 - k) * 5.0 for k in range(75)}
        (d / "out" / c).mkdir()
        (d / "out" / c / "receipt.json").write_text(json.dumps({
            "name": c, "rc": 0, "sizing": {"source": "c-auto-plan", "plan_key": "b5cf", "budget_bytes": 4510470740},
            "allocator": ["gpuwrf: C-auto case pool 4.20 GiB, fraction=0.134000" if cpool else "gpuwrf: C-auto unavailable (x)"],
            "aot_cache_warm": True, "aot_cold_executables": [], "vram_peak_mib": 5268, "vmhwm_kb": 6217608}))
    (d / "mtimes.json").write_text(json.dumps(mt))
    (d / "out" / "parallel_run.json").write_text(json.dumps({
        "events": [{"t_s": s, "start": c, "running": i + 1} for i, (s, c) in enumerate(zip(starts, mt))],
        "peak_concurrency": 3 if max(starts) < 400 else 2, "wall_s": 451.5}))
    (d / "stdout.log").write_text("BUNDLE_CONSUMER rc=0 compiled_new=0 aliases_new=0 runtime_new=1572 other_new=0 layer_compiled=0\n")
    (d / "disk_await.json").write_text(json.dumps({"await_ms": await_ms, "disturbed": await_ms > 50}))


def run(root):
    (root / "disk.jsonl").write_text("")
    b6 = root / "b6_sha.txt"
    b6.write_text("")
    res = subprocess.run([sys.executable, str(HERE / "hero_summary.py"), str(root), str(root / "sum"), str(b6)],
                         capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    assert res.returncode == 0, res.stderr[-3000:]
    return json.loads((root / "sum" / "HERO_RESULT.json").read_text()), res.stdout


checks = []
# 1: H1 clean, H2 disturbed -> H1 only, CLEAN; par3 valid + CLEAN
r1 = WORK / "c1"
prod_arm(r1, "W0", 20.0, 5.0)
prod_arm(r1, "H1", 105.0, 10.0)
prod_arm(r1, "H2", 220.0, 80.0)
wn3_arm(r1, [0.0, 0.0, 0.1])
res, out = run(r1)
hp, hw = res["hero_prod"], res["hero_par3"]
checks += [("c1 label", hp["label"] == "CLEAN"), ("c1 arms", hp["arms"] == ["H1"]),
           ("c1 s/fc-h", hp["HERO_PROD_S_PER_FCH"] == round(106.0 / 24, 3)), ("c1 seconds", hp["HERO_PROD_SECONDS"] == 106.0),
           ("c1 ratio", hp["ratio_vs_cpu_wrf_92_3"] == round(92.3 / round(106.0 / 24, 3), 2)),
           ("c1 session clock", hp["session_clock_s_per_fch"] == round((105.0 + 3.83) / 24, 3)),
           ("c1 attested", res["arms"]["H1"]["c_auto_attested"] is True),
           ("c1 vram csv", res["arms"]["H1"]["vram_peak_mib_pid_csv"] == 3901),
           ("c1 par3", hw["HERO_PAR3_S_PER_CASE_H"] == round(450.83 / 72, 3) and hw["valid_par3"] and hw["label"] == "CLEAN"),
           ("c1 par3 ratio", hw["ratio_vs_cpu_wrf_3x4core_123_385"] == round(123.385 / round(450.83 / 72, 3), 2)),
           ("c1 line", "HERO_PROD_S_PER_FCH=4.417" in out and "HERO_PAR3_S_PER_CASE_H=6.262" in out)]
# 2: both disturbed -> mean of both, DISTURBED; par3 started apart -> INVALID
r2 = WORK / "c2"
prod_arm(r2, "H1", 105.0, 60.0)
prod_arm(r2, "H2", 220.0, 80.0)
wn3_arm(r2, [0.0, 0.0, 420.0])
res, _ = run(r2)
hp, hw = res["hero_prod"], res["hero_par3"]
checks += [("c2 label", hp["label"] == "DISTURBED"), ("c2 arms", hp["arms"] == ["H1", "H2"]),
           ("c2 value", hp["HERO_PROD_S_PER_FCH"] == round((round(106.0 / 24, 3) + round(221.0 / 24, 3)) / 2, 3)),
           ("c2 par3 invalid", hw["valid_par3"] is False and hw["label"] == "INVALID" and hw["started_together"] is False)]
# 3: H1 off C-auto (demand) + H2 compiled_new 3 -> no timed arm, hero_prod None; par3 without pool line -> INVALID
r3 = WORK / "c3"
prod_arm(r3, "H1", 105.0, 10.0, attested=False)
prod_arm(r3, "H2", 106.0, 10.0, compiled_new=3)
wn3_arm(r3, [0.0, 0.0, 0.0], cpool=False)
res, _ = run(r3)
checks += [("c3 refused", res["hero_prod"] is None), ("c3 attest flag", res["arms"]["H1"]["c_auto_attested"] is False),
           ("c3 par3 invalid", res["hero_par3"]["valid_par3"] is False)]
# 4: H3 clean after H1/H2 disturbed -> H3 only
r4 = WORK / "c4"
prod_arm(r4, "H1", 105.0, 60.0)
prod_arm(r4, "H2", 220.0, 80.0)
prod_arm(r4, "H3", 110.0, 12.0)
res, _ = run(r4)
checks += [("c4 H3 only", res["hero_prod"]["arms"] == ["H3"] and res["hero_prod"]["label"] == "CLEAN"),
           ("c4 no wn3", res["hero_par3"] is None)]
# 5: 3 cases ran concurrently (peak 3) but the 3rd started 30 s late -> not "started together" -> INVALID
r5 = WORK / "c5"
wn3_arm(r5, [0.0, 0.0, 30.0])
res, _ = run(r5)
checks += [("c5 late start", res["hero_par3"]["started_together"] is False and res["hero_par3"]["valid_par3"] is False
            and res["hero_par3"]["peak_concurrency"] == 3)]
bad = [n for n, ok in checks if not ok]
print(json.dumps({"checks": len(checks), "failed": bad}, indent=1))
sys.exit(1 if bad else 0)
