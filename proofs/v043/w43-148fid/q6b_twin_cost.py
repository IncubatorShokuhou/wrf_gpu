"""Q6b: N18 CPU twin cost from the measured CPU reference (a2 + a3), rank-0 rsl timing, inside one GuardedRead session.

usage: python analysis/q6b_twin_cost.py <label>  -> analysis/Q6b_twin_cost.json
WRF 'Timing for main' of a nested domain is inclusive of its children (E30): the d01 lines alone give the wall per
d01 step for the whole 9-domain tree. 'Timing for Writing' lines are reported separately (they are inside the
d01 step time of the step that writes). The JSON run configs are read to record cores/decomposition/pinning.
"""
import json, re, sys, statistics
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gread

R = "<DATA_ROOT>/alisios/scratch/n148-cpuref-20261018_12z"
A = "<DATA_ROOT>/alisios/state/manager/runs/mgr_wave_20261009"
RSL = {"a2": f"{R}/cpu/rsl.error.0000", "a3": f"{R}/cpu_a3/rsl.error.0000"}
CONF = [f"{A}/nightly-engine-148-a3/{n}.json" for n in ("CPU_SEAL", "CPU_CONFIG", "CPU_ONLY_ENV", "RANK_START_PROOF", "WRF_RC",
        "PROGRESS", "PROGRESS_A2_REBOOT", "SYSTEMD_BOUNDS", "TIMERS")] + \
       [f"{A}/nightly-engine-148-cpuref/{n}.json" for n in ("CPU_CONFIG", "CPU_RESOURCE_PREFLIGHT", "RANK_START_PROOF", "MPI_PROCESS",
        "TIMERS", "PROGRESS")]
PAT_MAIN = re.compile(r"Timing for main: time (\S+) on domain\s+(\d+):\s+([\d.]+) elapsed seconds")
PAT_WRITE = re.compile(r"Timing for Writing (\S+) for domain\s+(\d+):\s+([\d.]+) elapsed seconds")
T0 = datetime(2026, 10, 18, 18)


def parse(text):
    main, write = [], []
    for line in text.splitlines():
        m = PAT_MAIN.search(line)
        if m:
            if int(m.group(2)) == 1:
                t = datetime.strptime(m.group(1), "%Y-%m-%d_%H:%M:%S")
                main.append(((t - T0).total_seconds() / 3600.0, float(m.group(3))))
            continue
        m = PAT_WRITE.search(line)
        if m:
            write.append((int(m.group(2)), float(m.group(3))))
    return main, write


def main(label):
    gread.use_add2()
    sess, rdir = gread.session(list(RSL.values()) + CONF, label)
    out = {"label": "[M] rank-0 rsl.error.0000 of the CPU reference arms a2 (tau0-40) and a3 (tau39-72); [I] for extrapolations"}
    with sess:
        conf = {}
        for p in CONF:
            try:
                conf[Path(p).parent.name + "/" + Path(p).name] = json.load(open(p))
            except Exception as e:   # absent/invalid -> recorded, never guessed
                conf[Path(p).parent.name + "/" + Path(p).name] = {"_error": f"{type(e).__name__}"}
        for arm, p in RSL.items():
            main_, write = parse(open(p, errors="replace").read())
            steps = [s for _, s in main_]
            hours = {}
            for tau, s in main_:
                hours.setdefault(int(tau - 1e-9) if tau > 0 else 0, []).append(s)
            per_hour = {h: sum(v) for h, v in sorted(hours.items()) if len(v) >= 60}   # 66-67 d01 steps per hour
            out[arm] = {"n_d01_steps": len(steps), "tau_first": main_[0][0] if main_ else None, "tau_last": main_[-1][0] if main_ else None,
                        "step_s_median": statistics.median(steps) if steps else None,
                        "step_s_p10": sorted(steps)[len(steps) // 10] if steps else None,
                        "step_s_p90": sorted(steps)[9 * len(steps) // 10] if steps else None,
                        "sum_main_s": sum(steps), "per_sim_hour_s": per_hour,
                        "write_s_total": sum(s for _, s in write), "n_write": len(write)}
        out["configs"] = conf
    gread.close_receipt(sess, rdir)
    json.dump(out, open(HERE / "Q6b_twin_cost.json", "w"), indent=1, default=str)
    for arm in RSL:
        a = out[arm]
        ph = list(a["per_sim_hour_s"].values())
        print(arm, a["n_d01_steps"], "steps tau", a["tau_first"], "-", a["tau_last"], "median step", a["step_s_median"],
              "sum", round(a["sum_main_s"]), "per-sim-h median", statistics.median(ph) if ph else None, "write", round(a["write_s_total"]))


if __name__ == "__main__":
    main(sys.argv[1])
