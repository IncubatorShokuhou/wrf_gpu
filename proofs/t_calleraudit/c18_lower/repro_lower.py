"""CPU lower-only run of the gpuwrf CLI: stop right after the d01 step program LOWERED (no compile).

usage: python repro_lower.py <marker.json> run --input-dir ... (CLI argv)
Exit 0 + marker {"lowered": true, hlo_sha256} on success; a lower error falls through to the CLI's
plain jitted path, which re-raises it -> full traceback printed, exit 1.
"""
import json
import os
import sys
import time
import traceback

marker = sys.argv[1]
t0 = time.time()
import jax  # noqa: E402

import gpuwrf.cli as cli  # noqa: E402
from gpuwrf.runtime import aot_executable as aotx  # noqa: E402

_orig_fail, _orig_hlo = cli._fail, aotx.hlo_sha256_from_lowered


def _fail(message, *, code=2):
    if sys.exc_info()[0] is not None:
        traceback.print_exc()
    with open(marker, "w") as fh:
        json.dump({"lowered": False, "error": message, "wall_s": time.time() - t0}, fh)
    return _orig_fail(message, code=code)


def _hlo(lowered):
    digest = _orig_hlo(lowered)
    caller = sys._getframe(1).f_code.co_name
    if caller not in ("_lower_advance_variant", "_lower_fused"):
        print(f"[repro_lower] hlo digest from {caller} (not the step program) -> continue", flush=True)
        return digest
    with open(marker, "w") as fh:
        json.dump({"lowered": True, "caller": caller, "hlo_sha256": digest, "platform": jax.devices()[0].platform,
                   "wall_s": time.time() - t0}, fh)
    print(f"LOWERED OK platform={jax.devices()[0].platform} hlo={digest}", flush=True)
    os._exit(0)


cli._fail = _fail
aotx.hlo_sha256_from_lowered = _hlo
sys.exit(cli.main(sys.argv[2:]))
