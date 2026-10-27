"""CPU-only fresh-process import anatomy, not GPU startup/speed evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
assert os.environ["JAX_PLATFORMS"] == "cpu"
assert os.environ["GPUWRF_JAX_CACHE"] == "0"
assert subprocess.run(["git", "-C", str(a.source), "symbolic-ref", "--quiet", "HEAD"], capture_output=True).returncode != 0
sys.path.insert(0, str(a.source / "src"))
t0 = time.monotonic()
import jax
t_jax = time.monotonic()
events = []
jax.monitoring.register_event_duration_secs_listener(
    lambda event, duration, **kw: events.append({"event": event, "s": duration, "fn": kw.get("fun_name")})
)
import gpuwrf.cli
t_cli = time.monotonic()
import gpuwrf.integration.nested_pipeline
import gpuwrf.runtime.domain_tree
t_pipeline = time.monotonic()
assert Path(gpuwrf.__file__).is_relative_to(a.source)
rec = {"schema": "w44-perf.cpu_import.v1", "source": str(a.source),
       "source_commit": subprocess.check_output(["git", "-C", str(a.source), "rev-parse", "HEAD"], text=True).strip(),
       "gpu_executed": False, "platform": "cpu", "scope": __doc__,
       "phase_s": {"jax_import": t_jax - t0, "cli_after_jax": t_cli - t_jax,
                   "pipeline_after_cli": t_pipeline - t_cli, "total": t_pipeline - t0},
       "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       "gpuwrf_modules": sorted(k for k in sys.modules if k.startswith("gpuwrf")),
       "jax_events_during_gpuwrf_import": events,
       "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps({k: rec[k] for k in ("source_commit", "phase_s", "rss_peak_kib")}))
