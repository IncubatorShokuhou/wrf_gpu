"""Original-WRF literal byte gate and old/new compiled groundwater bytes.

Only the two changed source functions are restored for the regression control.
The scientific reference is the existing pristine-WRF fixtures, not that control.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

root, output = map(Path, sys.argv[1:3])
assert os.environ["JAX_PLATFORMS"] == "cpu"
sys.path[:0] = [str(root / "src"), str(root / "proofs/u2_land")]
import jax
import numpy as np
from literal_real32 import NumpyReal, source_backend
from gpuwrf.physics.noahmp import groundwater_options as gw

assert jax.devices()[0].platform == "cpu"
old, new = "jnp.exp(-timean)", "jnp.exp(jnp.asarray(-0.,jnp.float32)-timean)"
source = Path(gw.__file__).read_text()
assert source.count(new) == 2
names = ("groundwater_step", "_equilibrium_soilwater")
candidate = {n: getattr(gw, n) for n in names}
controls = {}
for node in ast.parse(source.replace(new, old)).body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
        env = dict(vars(gw))
        exec(compile(ast.Module(body=[node], type_ignores=[]), gw.__file__, "exec"), env)
        controls[node.name] = env[node.name]
record = dict(platform="cpu", gpu_executed=False, arms={})
for family in ("groundwater", "equilibrium", "prognostic"):
    path = root / "tests" / f"test_u2_{family}_options.py"
    spec = importlib.util.spec_from_file_location("precision_" + family, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for inf in ((None,) if family == "groundwater" else (1, 2)):
        args = () if inf is None else (inf,)
        with source_backend():
            literal, cases, _ = module.matrix(*args, engine=NumpyReal())
        expected = np.asarray([c["expected"] for c in cases], np.float32)
        assert np.isfinite(literal).all() and np.isfinite(expected).all()
        assert literal.tobytes() == expected.tobytes(), (family, inf, "original-WRF BYTE")
        actual, _, _ = module.matrix(*args)
        for n, fn in controls.items():
            setattr(gw, n, fn)
        jax.clear_caches()
        try:
            before, _, _ = module.matrix(*args)
        finally:
            for n, fn in candidate.items():
                setattr(gw, n, fn)
        assert np.isfinite(actual).all() and np.isfinite(before).all()
        assert actual.dtype == before.dtype == np.float32
        assert actual.tobytes() == before.tobytes(), (family, inf, "compiled regression BYTE")
        arm = f"{family}_inf{inf}"
        record["arms"][arm] = dict(values=int(actual.size), original_wrf_literal_byte=True,
                                   compiled_old_new_byte=True, output_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
        output.write_text(json.dumps(record, indent=2) + "\n")
        print(arm, record["arms"][arm], flush=True)
        jax.clear_caches()
