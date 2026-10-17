"""CPU-only component resource receipt; never a GPU/forecast speed claim.

Run from an immutable commit/archive, with JAX_PLATFORMS=cpu and cache disabled.
All observation, grid and tendency arrays are dynamic executable arguments;
only the options are static. No constant-folded observational work is timed.
"""

from dataclasses import asdict
import json
from pathlib import Path
import re
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.assimilation import obs_nudging as on

assert jax.devices()[0].platform == "cpu"
ROOT = Path(__file__).resolve().parents[2]
SAVE = ROOT/"proofs/s1_obsnudge/savepoints"
KEYS = ("mu","c1h","c2h","msfx","msfy","pbase","pp","kpbl","vih1","vih2","terrain","z")


def main(out):
    manifest = json.loads((SAVE/"MANIFEST.json").read_text())
    name = "regime_3_r4.npz"
    with np.load(SAVE/name) as data:
        obs = on.SingleLevelObservations(*(jnp.asarray(data[k]) for k in on.SingleLevelObservations._fields))
        arrays = tuple(jnp.asarray(data[k]) for k in KEYS)
        aten,saved = jnp.asarray(data["initial_aten"]),jnp.asarray(data["initial_saved"])
    o = on.SingleLevelNudgingOptions(**manifest["cases"][name]["options"])

    def step(aten,saved,obs,arrays,ktau,xtime,tfaci):
        return on.single_level_nudob(o,obs,aten,saved,ktau=ktau,xtime=xtime,tfaci=tfaci,**dict(zip(KEYS,arrays)))

    args = (aten,saved,obs,arrays,jnp.int32(0),jnp.float32(.3),jnp.float32(1))
    t = time.perf_counter()
    executable = jax.jit(step).lower(*args).compile()
    compile_s = time.perf_counter()-t
    for _ in range(3):
        jax.block_until_ready(executable(*args))
    samples = []
    for _ in range(30):
        t = time.perf_counter()
        result = jax.block_until_ready(executable(*args))
        samples.append(time.perf_counter()-t)
    text = executable.as_text()
    mem = executable.memory_analysis()
    receipt = {
        "platform":"cpu", "claim":"component diagnostics only; GPU launches/VRAM/readiness unknown",
        "fixture":name, "fixture_sha256":manifest["cases"][name]["sha256"],"options":asdict(o),
        "work":{"shape":list(aten.shape),"stations":len(obs.rio),"refresh":True,"cache":"NONE","dynamic_inputs":True},
        "compile_s":compile_s,"warm_median_s":statistics.median(samples),"warm_min_s":min(samples),
        "samples":len(samples),"finite":all(np.isfinite(np.asarray(x)).all() for x in result),
        "optimized_hlo_f64_tokens":len(re.findall(r"\bf64\[",text)),
        "memory":{k:getattr(mem,k) for k in ("argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes",
                                            "alias_size_in_bytes","generated_code_size_in_bytes")},
    }
    assert receipt["finite"] and receipt["optimized_hlo_f64_tokens"] == 0, receipt
    out.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
