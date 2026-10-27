"""A1-route census: XLA deviceless compile of the REAL production step for (fabricated) NVIDIA targets.

CPU only (CompileOnly client, CUDA hidden). One real-case CPU load (as capture_real.py), then per
target config: topology from the target, committed abstract operands (proofs/s1_camstack/lower_step.py
convention), production trace (facade + platform native CUDA, jax.default_backend 'gpu', E156),
`.lower().compile()` = XLA's own pipeline incl. its bundled Triton and the jax-cuda13 ptxas, then the
serialized executable's cubins through scripts/e41/cubin_scan.py (REG/STACK/LOCAL/SHARED/SASS per
kernel) and the optimized HLO through scripts/e41/hlo_scan.py.
Fabricated targets carry public spec values [I]; the code generation is XLA's real A1 pipeline but no
device executes anything: no runtime, fidelity or speed claim.

usage: JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python a1_deviceless.py --alias A --case C --out DIR --target T.pbtxt ...
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--alias", required=True)
p.add_argument("--case", required=True)
p.add_argument("--out", required=True)
p.add_argument("--target", action="append", required=True)
p.add_argument("--vendor", default="cuda", choices=("cuda", "rocm"))
a = p.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root = Path(a.alias)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for key, name in (("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda")):
    (out / name).mkdir(exist_ok=True)
    os.environ[key] = str(out / name)
sys.path[:0] = [str(root / "src"), str(root / "proofs/o1_nlbind"), str(root / "scripts")]

import jax  # noqa: E402
import gpuwrf  # noqa: E402,F401
import cpu_safe  # noqa: E402

_lapack_tridiagonal_solve = jax.lax.linalg.tridiagonal_solve
cpu_safe.install()  # CPU load only (E105); the original is restored before the native trace
from jax.experimental import pallas as jpl, topologies  # noqa: E402
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.backends import pallas as facade, platform as plat  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm  # noqa: E402
import wave_cubin_gate as gate  # noqa: E402
from e41 import cubin_scan, hlo_scan  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amdgpu_elfs  # noqa: E402

CUOBJDUMP = "/usr/local/cuda-13.1/bin/cuobjdump"
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
_real = jpl.pallas_call


def _interp(*args, **kw):
    kw["interpret"] = True
    kw.pop("compiler_params", None)
    return _real(*args, **kw)


summary = {"alias": str(root), "case": a.case, "jax": jax.__version__, "xla_flags": os.environ.get("XLA_FLAGS", ""),
           "scope": "deviceless A1 code generation on CPU; fabricated targets [I]; no GPU execution", "targets": {}}
t0 = time.time()
scratch = Path(os.environ.get("O5_SCRATCH", "<DATA_ROOT>/wrf_gpu2/scratch/o5-port"))
with tempfile.TemporaryDirectory(dir=scratch) as tmp:
    case = Path(tmp) / "case"
    case.mkdir()
    for f in Path(a.case).iterdir():
        (case / f.name).symlink_to(f)
    cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "o", Path(tmp) / "p", hours=1, max_dom=1)
    jpl.pallas_call = _interp
    facade.cuda_native = lambda: False
    try:
        _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
    finally:
        jpl.pallas_call = _real
ns, carry = bundles["d01"].namelist, carries["d01"]
summary["load_s"] = round(time.time() - t0, 1)
summary["grid"] = list(carry.state.theta.shape)
om._resolve_operational_suite(ns)
cb = om.build_clock_base(ns)
jax.lax.linalg.tridiagonal_solve = _lapack_tridiagonal_solve  # true GPU program: cuSPARSE/hipSPARSE gtsv
if a.vendor == "cuda":
    facade.cuda_native = lambda: True  # production CUDA convention
else:
    from gpuwrf.backends import vendor_ops
    facade.cuda_native = lambda: False
    vendor_ops._override_vendor("rocm")  # admitted-ROCm native bodies (AMDGCN literal RN)
plat.runtime_backend = lambda: a.vendor
jax.default_backend = lambda: "gpu"
summary["vendor"] = a.vendor


def scan_blob_rocm(blob, label):
    rows, stats = [], []
    for i, elf in enumerate(amdgpu_elfs.elfs(blob)):
        path = out / f"{label}_e{i}.hsaco"
        path.write_bytes(elf)
        try:
            krows, st, arch = amdgpu_elfs.kernel_rows(str(path))
            for r in krows:
                r["elf"] = i
            rows += krows
            stats.append(dict(st, elf=i, arch=arch, bytes=len(elf)))
        finally:
            path.unlink(missing_ok=True)
    return "rocm", rows, {"elf_stats": stats}


def scan_blob(blob, label):
    if a.vendor == "rocm":
        return scan_blob_rocm(blob, label)
    fmt = ax._gpu_blob_format(blob)
    native = gate._record(blob, 2, ax, asm) if fmt == ax.GPU_THUNK_FORMAT else blob
    rows = []
    for i, cubin in enumerate(gate._elfs(native)):
        path = out / f"{label}_e{i}.cubin"
        path.write_bytes(cubin)
        try:
            for row in cubin_scan.scan(path, cuobjdump=CUOBJDUMP):
                row = {k: row[k] for k in ("kernel", "regs", "stack", "local", "shared", "sass", "calls") if k in row}
                row["elf"] = i
                rows.append(row)
        finally:
            path.unlink(missing_ok=True)
    try:
        text = gate._optimized_hlo(blob, ax, asm)
        hs = hlo_scan.scan_text(text)
        kin = [t for t in hs["transpose_hero_fusions"] if t["kind"] == "kInput"]
        hlo = {"n_fusions": hs["n_fusions"], "kinput_transposes": len(kin),
               "max_kinput_transpose_instr": max((t["n_instr"] for t in kin), default=0),
               "triton_custom_calls": text.count("__gpu$xla.gpu.triton")}
    except Exception as exc:  # noqa: BLE001
        hlo = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return fmt, rows, hlo


for target in a.target:
    label = Path(target).stem
    rec = {"target": target, "target_sha256": hashlib.sha256(Path(target).read_bytes()).hexdigest()}
    t1 = time.time()
    try:
        topology = topologies.get_topology_desc(platform=a.vendor, target_config=Path(target).read_text(), topology="1x1x1")
        device = topology.devices[0]
        sharding = jax.sharding.SingleDeviceSharding(device)
        rec["virtual_device"] = str(device)

        def abstract(v):
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                return jax.ShapeDtypeStruct(v.shape, v.dtype, weak_type=bool(getattr(v, "weak_type", False)),
                                            sharding=sharding if getattr(v, "committed", False) else None)
            return v

        args = jax.tree.map(abstract, (carry, ns, jax.numpy.asarray(1, jax.numpy.int32), cb))
        jax.clear_caches()
        lowered = om._advance_chunk_fori.lower(*args, n_steps=1, cadence=int(ns.radiation_cadence_steps))
        rec["lower_s"] = round(time.time() - t1, 1)
        t2 = time.time()
        compiled = lowered.compile()
        rec["compile_s"] = round(time.time() - t2, 1)
        try:
            m = compiled.memory_analysis()
            rec["memory"] = {k: getattr(m, k) for k in ("argument_size_in_bytes", "output_size_in_bytes",
                                                         "alias_size_in_bytes", "temp_size_in_bytes")}
        except Exception as exc:  # noqa: BLE001
            rec["memory_error"] = f"{type(exc).__name__}: {exc}"[:200]
        exe = compiled.runtime_executable()
        if not hasattr(exe, "serialize"):
            exe = compiled._executable.xla_executable
        blob = exe.serialize()
        rec["blob_bytes"] = len(blob)
        rec["format"], rows, rec["hlo"] = scan_blob(blob, label)
        (out / f"{label}_kernels.json").write_text(json.dumps(rows, indent=0) + "\n")
        rec["n_kernels"] = len(rows)
        if a.vendor == "cuda":
            rec["max_regs"] = max((r.get("regs", 0) for r in rows), default=None)
            rec["max_stack"] = max((r.get("stack", 0) for r in rows), default=None)
            rec["stack_over256"] = [(r["kernel"][:80], r["stack"]) for r in rows if r.get("stack", 0) > 256]
            rec["local_nonzero"] = sum(1 for r in rows if r.get("local", 0))
        else:
            rec["max_vgpr"] = max((r.get("vgpr") or 0 for r in rows), default=None)
            rec["vgpr_spill_kernels"] = [(r["kernel"][:80], r["vgpr_spill"]) for r in rows if r.get("vgpr_spill")]
            rec["scratch_kernels"] = [(r["kernel"][:80], r["scratch"]) for r in rows if r.get("scratch")]
            rec["sgpr_spill_kernels"] = sum(1 for r in rows if r.get("sgpr_spill"))
        try:
            import gzip
            text = compiled.as_text()
            (out / f"{label}_optimized.hlo.gz").write_bytes(gzip.compress(text.encode(), mtime=0))
            rec["optimized_hlo_bytes"] = len(text)
            rec["optimized_hlo_command_buffers"] = text.count("command_buffer")
            rec["optimized_hlo_while"] = text.count(" while(")
            rec["optimized_hlo_conditional"] = text.count(" conditional(")
            del text
        except Exception as exc:  # noqa: BLE001
            rec["optimized_hlo_error"] = f"{type(exc).__name__}: {exc}"[:200]
        rec["status"] = "PASS"
        del compiled, lowered, blob
    except Exception as exc:  # noqa: BLE001 - a refusal is a result
        rec["status"] = "FAIL"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:1200]}"
        rec["traceback_tail"] = traceback.format_exc()[-1500:]
    rec["wall_s"] = round(time.time() - t1, 1)
    summary["targets"][label] = rec
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(label, json.dumps({k: v for k, v in rec.items() if k not in ("traceback_tail", "stack_over256")}, default=str)[:900],
          flush=True)
import shutil  # noqa: E402
for name in ("tmp", "triton", "cuda"):  # compile scratch: terminal after the receipts above
    shutil.rmtree(out / name, ignore_errors=True)
os._exit(0)
