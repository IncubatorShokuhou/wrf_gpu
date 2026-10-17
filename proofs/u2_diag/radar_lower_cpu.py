"""Deviceless Triton lowering of exact full/bottom radar GPU programs.

CPU backend only; this does not initialize CUDA, compile cubins, execute GPU
work, or replace the real GPU E41 gate. It catches unsupported Pallas rules.
"""
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.triton import lowering

from gpuwrf.diagnostics.thompson_radar import thompson_reflectivity


assert jax.devices()[0].platform == "cpu"
with np.load(Path(__file__).with_name("radar_bottom_oracle.npz")) as data:
    arrays = data["inputs"][:1]
inputs = tuple(jnp.asarray(arrays[:, index]) for index in (0, 2, 3, 4, 5, 6, 8, 9))
results = []
for depth in (1, arrays.shape[-1]):
    fn = lambda *a: thompson_reflectivity(*a, ke_diag=depth, interpret=False)
    program = jax.make_jaxpr(fn)(*inputs).jaxpr
    found = 0
    for equation in program.eqns:
        if equation.primitive.name != "pallas_call":
            continue
        params = equation.params
        result = lowering.lower_jaxpr_to_triton_module(params["jaxpr"], params["grid_mapping"], "cuda", 120)
        asm = str(result.module)
        found += 1
        results.append({"kernel": str(params.get("name_and_src_info")), "ke_diag": depth,
                        "lowering": "PASS", "IR_characters": len(asm),
                        "IR_sha256": hashlib.sha256(asm.encode()).hexdigest()})
    assert found == 1, f"expected exactly one Pallas kernel for ke_diag={depth}, got {found}"
target = Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps({"backend": "cpu", "GPU_execution": False, "target": "cuda/sm120",
                              "results": results, "real_GPU_cubin_E41_gate": "pending"}, indent=2) + "\n")
print(target.read_text())
