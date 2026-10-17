"""Compare the actual hail operators with extracted pristine WRF operators."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics.thompson_hail import entry_distribution, fall_speeds, sediment_hail


def run(out: Path):
    fixture = json.loads((Path(__file__).with_name("hail_operators.json")).read_text())
    v = {k: np.asarray(a, np.float32) for k, a in fixture["values"].items()}
    inputs = tuple(jnp.asarray(v[n]) for n in ("QG_INPUT", "NG_INPUT", "QB_INPUT", "RHO", "VISCO"))

    def operator(qg, ng, qb, rho, visco):
        d = entry_distribution(qg, ng, qb, rho)
        vt, vtn = fall_speeds(d, rho, visco)
        sediment = sediment_hail(d.rg, d.ng, d.rb, rho, jnp.asarray(v['DZ']), vt, vtn, 90.)
        return d.qg, d.Ng, d.qvolg, d.rg, d.ng, d.rb, d.index, d.ilamg.astype(jnp.float32), vt, vtn, *sediment

    result = {"platform": jax.devices()[0].platform, "scope": fixture["scope"], "fields": {}}
    for mode, fn in (("eager", operator), ("compiled", jax.jit(operator))):
        start = time.monotonic()
        with jax.disable_jit(mode == "eager"):
            actual = tuple(np.asarray(a) for a in fn(*inputs))
        result[mode + "_wall_s"] = time.monotonic() - start
        fields = {}
        for name, a in zip(("QG", "NG_IN", "QB", "RG", "NG", "RB", "IDX", "ILAMG", "VTG", "VTNG", "QGTEN", "NGTEN", "QBTEN", "PRECIP"), actual):
            ref = v[name] - 1 if name == "IDX" else v[name]
            ref = ref.reshape(a.shape)
            diff = np.abs(a.astype(np.float64) - ref.astype(np.float64))
            ulp = np.abs(np.spacing(ref)).astype(np.float64)
            fields[name] = {"finite": bool(np.isfinite(a).all()), "byte_equal": bool(np.array_equal(a, ref)),
                            "max_abs": float(np.max(diff)), "max_ulp": float(np.max(diff / np.maximum(ulp, 1e-45)))}
        result["fields"][mode] = fields
    result["source_sha256"] = fixture["source_sha256"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run(Path(sys.argv[1]))
