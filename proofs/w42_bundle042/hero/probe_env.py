"""w42-harness dry probe (CPU, never imports jax/gpuwrf): stands in for the harness at the end of the REAL PROD arm
command chain (consumer -> env -> run_arm.sh -> s0_env.sh + flags env -> env -u ...) and records what the harness
process would start with. usage: probe_env.py <harness args...> (writes <--out>/probe.json)."""
import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "probe.json").write_text(json.dumps({
    "argv": sys.argv, "executable": sys.executable, "cwd": os.getcwd(), "affinity": sorted(os.sched_getaffinity(0)),
    "nice": os.nice(0), "env": dict(os.environ)}, indent=1) + "\n")
