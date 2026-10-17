"""Smoke-only exit receipt, after the real model initialized its own backend.

No JAX import/backend query occurs on startup. The smoke supervisor also checks
the child's rc/output receipt, so a refused forecast cannot pass on this alone.
"""
import atexit
import sys


def _record():
    if "run" not in sys.argv or "--output-dir" not in sys.argv:
        return
    import json
    from pathlib import Path
    import jax

    device = jax.devices()[0]
    output = Path(sys.argv[sys.argv.index("--output-dir")+1])
    output.mkdir(parents=True, exist_ok=True)
    (output/"ensemble_smoke_backend.json").write_text(json.dumps(
        {"platform": device.platform, "device": str(device), "asserted_gpu": device.platform == "gpu"}))
    assert device.platform == "gpu"


atexit.register(_record)
