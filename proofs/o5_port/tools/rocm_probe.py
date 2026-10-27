"""Tiny deviceless XLA-ROCm probe (lane o5-port): ROCm PJRT compile-only topology + jit + Pallas kernel. CPU only."""
import json, os, sys, time, traceback
target = sys.argv[1]
out = {"target": target}
import jax, jax.numpy as jnp
from jax.experimental import topologies, pallas as pl
from jax.experimental.pallas import triton as pt
out["jax"] = jax.__version__
assert all(d.platform == "cpu" for d in jax.devices())
try:
    import jax_plugins.xla_rocm7 as rp
    out["rocm_plugin"] = getattr(rp, "__file__", None)
except Exception as e:
    out["rocm_plugin_error"] = repr(e)[:300]
try:
    topo = topologies.get_topology_desc(platform="rocm", target_config=open(target).read(), topology="1x1x1")
    dev = topo.devices[0]
    out["device"] = str(dev)
    sh = jax.sharding.SingleDeviceSharding(dev)
    spec = jax.ShapeDtypeStruct((1024,), jnp.float32, sharding=sh)

    def kern(a_ref, b_ref, o_ref):
        a, b = a_ref[...], b_ref[...]
        m = pt.elementwise_inline_asm("v_mul_f32 $0, $1, $2", args=[a, b], constraints="=v,v,v", pack=1,
                                      result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, jnp.float32)])[0]
        o_ref[...] = jnp.exp(m) + b

    f = jax.jit(lambda a, b: pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct((1024,), jnp.float32),
                                            grid=(1,), compiler_params=pt.CompilerParams(num_warps=4))(a, b) * 2.0)
    t = time.time()
    low = f.lower(spec, spec)
    out["lower_s"] = round(time.time() - t, 2)
    txt = low.as_text()
    out["triton_custom_call"] = "__gpu$xla.gpu.triton" in txt
    t = time.time()
    c = low.compile()
    out["compile_s"] = round(time.time() - t, 2)
    hlo = c.as_text()
    out["optimized_hlo_chars"] = len(hlo)
    exe = c.runtime_executable()
    if not hasattr(exe, "serialize"):
        exe = c._executable.xla_executable
    blob = exe.serialize()
    out["blob_bytes"] = len(blob)
    elfs, pos = [], 0
    while (i := blob.find(b"\x7fELF", pos)) >= 0:
        em = int.from_bytes(blob[i + 18:i + 20], "little")
        elfs.append({"offset": i, "e_machine": em})
        pos = i + 4
    out["elf_magics"] = elfs[:20]
    dump = os.environ.get("O5_DUMP")
    if dump:
        open(dump, "wb").write(blob)
    out["status"] = "PASS"
except Exception as e:
    out["status"] = "FAIL"
    out["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
    out["tb"] = traceback.format_exc()[-1500:]
print(json.dumps(out, indent=1), flush=True)
os._exit(0)
