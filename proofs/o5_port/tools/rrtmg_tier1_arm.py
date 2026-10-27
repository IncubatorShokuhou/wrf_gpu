"""One arm of the RRTMG Tier-1 bisection (lane o5-port, manager 23:19Z): measurement only.

Runs gpuwrf.validation.tier1_rrtmg.run_tier1_{lw,sw} (the M5 analytic fixtures, frozen manifest tolerances) in THIS
process under the environment it was started with (release flags are import-time constants -> one process per arm)
and prints one JSON line: arm label, resolved radiation flags, backend, pass + max abs/rel error per field.
usage: python rrtmg_tier1_arm.py LABEL OUT.jsonl
"""
import json
import os
import sys
import time
from pathlib import Path

label, out = sys.argv[1], Path(sys.argv[2])
t0 = time.time()
import jax  # noqa: E402
import gpuwrf  # noqa: E402,F401  (applies the release defaults unless GPUWRF_FAST_DEFAULTS=0)
from gpuwrf.validation import tier1_rrtmg as T1  # noqa: E402

KEYS = ("RRTMG", "MCICA", "SOLAR", "REAL_ALL", "FAST_DEFAULTS")
rec = {"arm": label, "backend": jax.devices()[0].platform,
       "flags": {k: v for k, v in sorted(os.environ.items()) if k.startswith("GPUWRF_") and any(s in k for s in KEYS)}}
tmp = Path(os.environ.get("TMPDIR", "/tmp"))
for name, run in (("lw", T1.run_tier1_lw), ("sw", T1.run_tier1_sw)):
    r = run(tmp / f"o5_t1_{os.getpid()}_{name}.json")
    rec[name] = {"pass": r["pass"], "failed": sorted(k for k, ok in r["field_pass"].items() if not ok),
                 "max_abs": {k: round(v, 6) for k, v in r["per_field_max_abs_err"].items()},
                 "max_rel": {k: round(v, 6) for k, v in r["per_field_max_rel_err"].items()}}
rec["wall_s"] = round(time.time() - t0, 1)
with out.open("a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(label, "LW", rec["lw"]["pass"], "SW", rec["sw"]["pass"], "sw_surface_down",
      rec["sw"]["max_abs"].get("surface_down"), "lw_flux_down", rec["lw"]["max_abs"].get("flux_down"), flush=True)
os._exit(0)
