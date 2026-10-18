"""S1AB follow-up attribution (CPU, JSON only; manager 2026-10-18T13:31:44Z): why A1 was slow, what the B pre-step delta is.

usage: python followup.py ROOT        (ROOT = S1AB OUT or its receipt copy; writes ROOT/S1AB_FOLLOWUP.json)
Per timed arm from its own receipts: NetCDF writer thread time (nested_output_perf_timers writer_write; async output, so a
slow writer back-pressures the d01 output call and the stepping), output calls > 1 s, JAX persistent-cache retrieval time
before/after the first step, pre-step JAX compile-path time (trace + lower + cache load, compile_time_saved excluded) and
the traced-function multiset (same init programs?), the harness pre-exec phase (launch -> re-exec'd receipt t0; the harness
calls jax.devices() before the CLI, so its default 75 % pool is on the GPU when the CLI's C-auto probe reads free memory),
and the dmon framebuffer peak in that phase.
"""
import collections, json, sys
from pathlib import Path

ARMS = ("A1", "B1", "A2", "B2")


def arm(root, name):
    r = json.loads((root / name / "run/receipt.json").read_text())
    cli = json.loads((root / name / "run/cli_stdout.json").read_text())
    pre = r["derived"]["segments"][0]["t_start"]
    timers = cli["metadata"]["nested_output_perf_timers"]
    retr = [e for e in r["compile_events"] if e["event"].endswith("cache_retrieval_time_sec")]
    path = [e for e in r["compile_events"] if e["t"] < pre and not e["event"].endswith("compile_time_saved_sec")]
    traced = collections.Counter(e.get("fun") for e in path if e["event"].endswith("jaxpr_trace_duration"))
    launch, t0 = float((root / name / "launch_unix").read_text()), float(r["t0_wall_utc"])
    fb = [int(line.split()[12]) for line in (root / name / "nvidia-smi.dmon.log").read_text().splitlines()
          if line.strip() and not line.startswith("#")][:int(t0 - launch) + 2]
    stderr = (root / name / "stderr.log").read_text()
    return {"writer_write_total_s": round(timers["totals_s"]["writer_write"], 2),
            "writer_write_mean_s": round(timers["mean_s"]["writer_write"], 3),
            "output_calls_over_1s": [(o["d"], o["own_step"], round(o["s"], 2)) for o in r["outputs"] if o["s"] > 1],
            "cache_retrieval_s": {"pre_step": round(sum(e["s"] for e in retr if e["t"] < pre), 2),
                                  "after": round(sum(e["s"] for e in retr if e["t"] >= pre), 2),
                                  "max_single": round(max(e["s"] for e in retr), 2)},
            "pre_step_s": round(pre, 3), "pre_step_compile_path_s": round(sum(e["s"] for e in path), 2),
            "pre_step_traced": traced, "n_compile_events": len(r["compile_events"]),
            "harness_pre_exec_s": round(t0 - launch, 2), "pre_exec_fb_peak_mib": max(fb) if fb else None,
            "c_auto_unavailable": "C-auto unavailable" in stderr,
            "measured_env_preallocate": r["env"].get("XLA_PYTHON_CLIENT_PREALLOCATE")}


root = Path(sys.argv[1]).resolve()
arms = {name: arm(root, name) for name in ARMS}
same = {name: arms[name]["pre_step_traced"] == arms["A2"]["pre_step_traced"] for name in ARMS}
for a in arms.values():
    a["pre_step_traced"] = sum(a["pre_step_traced"].values())
out = {"root": str(root), "arms": arms, "pre_step_traced_multiset_equals_A2": same,
       "B_minus_A2": {k: [round(arms[b][k] - arms["A2"][k], 2) for b in ("B1", "B2")]
                      for k in ("pre_step_s", "pre_step_compile_path_s", "writer_write_total_s")}}
(root / "S1AB_FOLLOWUP.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: out[k] for k in ("pre_step_traced_multiset_equals_A2", "B_minus_A2")}))
for name, a in arms.items():
    print(name, {k: v for k, v in a.items() if k != "output_calls_over_1s"}, "slow outputs", len(a["output_calls_over_1s"]))
