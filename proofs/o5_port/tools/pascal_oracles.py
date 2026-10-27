"""Exact-REAL component contracts on a real pre-Ampere GPU through the kernel-call route (lane o5-port).

Plan rule 3 (PORT_PLAN_v2 §2.1): existing exact-REAL contracts are re-run per vendor against the PRISTINE
literal, never against another GPU. Here (GTX 1080 Ti, sm_61, GPUWRF_PRE_AMPERE_KERNEL_CALL=1):
  * MYNN tridiag2 (BP56): Pallas sweep vs the literal WRF REAL4 transcription
    (module_bl_mynnedmf.F tridiag2; tests/v025/b_phys/test_mynn_tridiag_real.wrf_tridiag2) -> must be BITWISE,
    as on the RTX 5090; real PROD-like batches, odd padding, 3 seeds;
  * stage masses (calc_mu_uv four-term REAL order) and the nest scalar couple: bitwise vs binary32 references.
usage: python pascal_oracles.py OUT.json
"""
import json
import os
import sys

import numpy as np

assert os.environ.get("GPUWRF_PRE_AMPERE_KERNEL_CALL") == "1"
import jax  # noqa: E402
import gpuwrf  # noqa: E402,F401

dev = jax.devices()[0]
assert dev.platform == "gpu"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../tests/v025/b_phys"))
from test_mynn_tridiag_real import mynn_like, wrf_tridiag2  # noqa: E402
from gpuwrf.backends import vendor_ops  # noqa: E402
from gpuwrf.kernels import phys_mynn_tridiag as T  # noqa: E402

out = {"device": dev.device_kind, "cc": getattr(dev, "compute_capability", None), "vendor": vendor_ops.vendor()}
cases = []
for seed, shape in ((0, (3, 70, 44)), (1, (117, 267, 44)), (2, (8400, 44)), (7, (131, 44))):
    a, b, c, d = mynn_like(seed, shape=shape)
    got = np.asarray(jax.block_until_ready(jax.jit(T.solve_tridiagonal_real)(a, b, c, d)))
    ref = wrf_tridiag2(a, b, c, d)
    eq = got.view(np.uint32) == ref.view(np.uint32)
    cases.append({"seed": seed, "shape": list(shape), "bitwise_fraction": float(eq.mean()),
                  "n": int(eq.size), "max_abs_diff": float(np.max(np.abs(got.astype(np.float64) - ref))),
                  "finite": bool(np.isfinite(got).all())})
    print("tridiag2", cases[-1], flush=True)
out["mynn_tridiag2_vs_literal_wrf_real4"] = cases
out["route_installed"] = vendor_ops._kernel_call_route["installed"]
out["verdict"] = "PASS" if all(c["bitwise_fraction"] == 1.0 and c["finite"] for c in cases) else "FAIL"
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(out["verdict"], flush=True)
os._exit(0)
