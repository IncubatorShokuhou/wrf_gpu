"""W5P02: can WRF-literal REAL ops (no FMA contraction) be expressed for AMD in Pallas-Triton? CPU-only lowering+compile."""
import json, os, re, sys, tempfile
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt
from jax._src.pallas.triton import lowering as tl
import triton
from triton.backends.compiler import GPUTarget
N = 256
def mk(mode):
    def k(a_ref, b_ref, c_ref, o_ref):
        a, b, c = a_ref[...], b_ref[...], c_ref[...]
        if mode == "plain":
            o_ref[...] = a * b + c
        elif mode == "amdasm":
            m = pt.elementwise_inline_asm("v_mul_f32 $0, $1, $2", args=[a, b], constraints="=v,v,v", pack=1,
                                          result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, jnp.float32)])[0]
            o_ref[...] = pt.elementwise_inline_asm("v_add_f32 $0, $1, $2", args=[m, c], constraints="=v,v,v", pack=1,
                                          result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, jnp.float32)])[0]
        elif mode == "div":
            o_ref[...] = a / b + c
    return k
res = []
spec = jax.ShapeDtypeStruct((N,), jnp.float32)
for mode in ("plain", "amdasm", "div"):
    f = pl.pallas_call(mk(mode), out_shape=spec, grid=(1,), interpret=True)
    jp = jax.make_jaxpr(f)(spec, spec, spec)
    p = next(e.params for e in jp.jaxpr.eqns if e.primitive.name == "pallas_call")
    text = tl.lower_jaxpr_to_triton_module(p["jaxpr"], p["grid_mapping"], "rocm", None).module.operation.get_asm()
    with tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=False) as fh:
        fh.write(text); path = fh.name
    for arch in ("gfx942", "gfx1100"):
        for fusion in (True, False):
            try:
                k = triton.compile(path, target=GPUTarget("hip", arch, 64 if arch == "gfx942" else 32),
                                   options={"num_warps": 4, "num_stages": 1, "enable_fp_fusion": fusion})
                a = k.asm["amdgcn"]
                ins = [l.strip().split()[0] for l in a.splitlines() if re.match(r"\s+v_[a-z_0-9]+", l)]
                res.append({"mode": mode, "arch": arch, "enable_fp_fusion": fusion, "status": "PASS",
                            "v_fma": sum(i.startswith(("v_fma_f32", "v_fmac_f32", "v_pk_fma_f32", "v_fmaak", "v_fmamk")) for i in ins),
                            "v_mul": sum(i.startswith(("v_mul_f32", "v_pk_mul_f32")) for i in ins),
                            "v_add": sum(i.startswith(("v_add_f32", "v_pk_add_f32")) for i in ins),
                            "v_div_fixup": sum(i.startswith("v_div_fixup_f32") for i in ins),
                            "v_rcp": sum(i.startswith("v_rcp_f32") for i in ins)})
            except Exception as e:
                res.append({"mode": mode, "arch": arch, "enable_fp_fusion": fusion, "status": "FAIL", "error": str(e)[-300:]})
    os.unlink(path)
for r in res: print(json.dumps(r), flush=True)
with open(sys.argv[1], "w") as fh: json.dump(res, fh, indent=1)
os._exit(0)
