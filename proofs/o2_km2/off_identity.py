"""Release OFF identity using the E151 production-loop jaxpr census.

CPU-only, FAST_DEFAULTS=1; usage off_identity.py SOURCE_ROOT OUT.json.
Require the independently frozen Swiss release digest 2bf12cb8. Pallas interpret
is used only for loading the real carry; tracing retains production Pallas calls.
The same historical CPU Thomas shim is used as in the frozen baseline receipt.
"""
import dataclasses
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

SRC, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, SRC + "/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'o1_smag3d'))
import cpu_safe  # noqa: E402

cpu_safe.install()
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(SRC), om.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
CASE = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")


def canon(jaxpr) -> str:
    text = str(jaxpr)
    text = re.sub(r"name_and_src_info=[^\]\n]*", "", text)
    text = re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", text)
    return text


out = {"src": SRC, "om": om.__file__}
# Release fast defaults select Triton Pallas kernels that cannot EXECUTE on CPU (E156); the carry is built with
# pallas_call forced to interpret mode, then the real pallas_call is restored for tracing (make_jaxpr never lowers).
from jax.experimental import pallas as pl  # noqa: E402

_real_pallas_call = pl.pallas_call


def _interpret_pallas_call(*a, **k):
    k["interpret"] = True
    k.pop("compiler_params", None)
    return _real_pallas_call(*a, **k)


pl.pallas_call = _interpret_pallas_call
with tempfile.TemporaryDirectory(dir="<USER_HOME>/wrf_gpu2_lanes/o2-km2") as tmp:
    cfg = pipeline.NestedPipelineConfig(CASE, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
    _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
pl.pallas_call = _real_pallas_call
ns, carry = bundles["d01"].namelist, carries["d01"]
out["release_namelist"] = {"diff_opt": int(ns.diff_opt), "km_opt": int(ns.km_opt), "bl_pbl_physics": int(ns.bl_pbl_physics)}
arms = {"release": ns}
for name, nl in arms.items():
    if name == "km3" and not Path(SRC, "src/gpuwrf/runtime/les3d_km3.py").exists():
        continue
    jax.clear_caches()
    cb = om.build_clock_base(nl)
    try:
        # The production per-interval loop (physics + radiation cond + boundaries + RK/acoustics), one step.
        jx = jax.make_jaxpr(lambda c, nl=nl, cb=cb: om._advance_chunk_fori(
            c, nl, 1, cb, n_steps=1, cadence=int(nl.radiation_cadence_steps)))(carry)
    except Exception as exc:  # noqa: BLE001 - record untraceable arms (same in both trees)
        out[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        Path(OUT).write_text(json.dumps(out, indent=1))
        print(name, out[name], flush=True)
        continue
    text = canon(jx)
    out[name] = {"sha": hashlib.sha256(text.encode()).hexdigest(), "eqns": len(jx.jaxpr.eqns), "chars": len(text)}
    Path(OUT).write_text(json.dumps(out, indent=1))
    print(name, out[name], flush=True)

assert out["release"]["sha"] == "2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a", out
