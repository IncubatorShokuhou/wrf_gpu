"""CPU-only launcher admission controls. No official lock/GPU invocation."""
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
caps = {"prime32": 1500, "prime40": 1500, "compare": 1800, "radiation": 180, "wn3sds": 300}
results = []
for phase, cap in caps.items():
    for label, expected, start_delta, end_delta in (("future", 4, 1, cap + 60),
                                                   ("short", 6, -60, cap - 1),
                                                   ("positive", 0, -60, cap + 60)):
        now = int(time.time())
        env = {**os.environ, "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0",
               "W44_DRY_NOW": str(now), "GA_START_UTC": str(now + start_delta),
               "GA_END_UTC": str(now + end_delta)}
        proc = subprocess.run(["bash", str(HERE / "night_launch.sh"), phase, "--cpu-dry"],
                              env=env, capture_output=True, text=True, timeout=15)
        assert proc.returncode == expected, (phase, label, proc.returncode, proc.stderr)
        r = json.loads(proc.stdout.splitlines()[-1])
        assert r["cpu_dry"] and not r["gpu_started"]
        results.append({"phase": phase, "case": label, "rc": proc.returncode, "status": r["status"]})
print(json.dumps({"checks": results, "all_pass": True}, indent=2))
