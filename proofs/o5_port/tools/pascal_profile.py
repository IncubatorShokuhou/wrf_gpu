"""Per-kernel device time of the release production step on the GTX 1080 Ti (wz1080; lane o5-port).

Product load of the real case on the GPU, GPUWRF_PRE_AMPERE_KERNEL_CALL=1 (stock Triton + triton_kernel_call),
one compile, then N single-step calls under jax.profiler (CUPTI). Device events are grouped by kernel name:
Pallas kernels (Triton function names) by family (tools/families.py via the source module recorded in the
census of the same names), XLA fusions/copies/other separately.
usage: python pascal_profile.py CASE OUT.json [N_STEPS]
"""
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

case_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
assert os.environ.get("GPUWRF_PRE_AMPERE_KERNEL_CALL") == "1"
import jax  # noqa: E402
import gpuwrf  # noqa: E402,F401
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from jax._src.pallas.triton import lowering as tl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from families import family  # noqa: E402

dev = jax.devices()[0]
out = {"device": dev.device_kind, "cc": getattr(dev, "compute_capability", None), "case": str(case_dir), "n_steps": n_steps}
assert dev.platform == "gpu"
names = {}
orig = tl.lower_jaxpr_to_triton_module


def census(jaxpr, gm, platform, cc):
    info = jaxpr.debug_info
    names[getattr(info, "func_name", "?")] = str(getattr(info, "func_src_info", "?"))
    return orig(jaxpr, gm, platform, cc)


tl.lower_jaxpr_to_triton_module = census
t0 = time.time()
with tempfile.TemporaryDirectory() as tmp:
    cfg = pipeline.NestedPipelineConfig(case_dir, Path(tmp) / "o", Path(tmp) / "p", hours=1, max_dom=1)
    _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
ns, carry = bundles["d01"].namelist, carries["d01"]
# The product step donates its carry; give every leaf its own buffer (fresh carries can alias leaves).
carry = jax.tree.map(lambda x: jax.numpy.array(x, copy=True) if isinstance(x, jax.Array) else x, carry)
om._resolve_operational_suite(ns)
cb = om.build_clock_base(ns)
cad = int(ns.radiation_cadence_steps)
out["load_s"] = round(time.time() - t0, 1)
t1 = time.time()
carry = jax.block_until_ready(om._advance_chunk_fori(carry, ns, 1, cb, n_steps=1, cadence=cad))
out["first_step_s"] = round(time.time() - t1, 1)
for i in range(2, 4):  # warm
    carry = jax.block_until_ready(om._advance_chunk_fori(carry, ns, i, cb, n_steps=1, cadence=cad))
trace_dir = Path(tempfile.mkdtemp(prefix="o5prof_"))
t2 = time.time()
with jax.profiler.trace(str(trace_dir)):
    for i in range(4, 4 + n_steps):
        carry = jax.block_until_ready(om._advance_chunk_fori(carry, ns, i, cb, n_steps=1, cadence=cad))
out["profiled_wall_s"] = round(time.time() - t2, 3)
xspace = next(trace_dir.rglob("*.xplane.pb"))
from jax.profiler import ProfileData  # noqa: E402

pd = ProfileData.from_file(str(xspace))
by_name = defaultdict(lambda: [0, 0.0])
for plane in pd.planes:
    if not plane.name.startswith("/device:GPU"):
        continue
    for line in plane.lines:
        if "Stream" not in line.name and "stream" not in line.name:
            continue
        for ev in line.events:
            rec = by_name[ev.name]
            rec[0] += 1
            rec[1] += ev.duration_ns / 1e6
fam = defaultdict(float)
for name, (count, ms) in by_name.items():
    if name in names:
        key = "pallas:" + family(names[name])
    elif "fusion" in name:
        key = "xla:fusion"
    elif name.lower().startswith(("memcpy", "memset")) or "copy" in name.lower():
        key = "xla:copy/memset"
    else:
        key = "xla:other"
    fam[key] += ms
total = sum(fam.values())
out["device_ms_per_step"] = round(total / n_steps, 3)
out["family_ms_per_step"] = {k: round(v / n_steps, 3) for k, v in sorted(fam.items(), key=lambda kv: -kv[1])}
out["top_kernels_ms_per_step"] = [(n, c // n_steps, round(ms / n_steps, 3)) for n, (c, ms) in
                                  sorted(by_name.items(), key=lambda kv: -kv[1][1])[:25]]
out["wall_ms_per_step"] = round(out["profiled_wall_s"] * 1000 / n_steps, 3)
out["pallas_kernels_seen"] = len(names)
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "top_kernels_ms_per_step"}, indent=1), flush=True)
os._exit(0)
