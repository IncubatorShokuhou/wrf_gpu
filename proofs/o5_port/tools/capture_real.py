"""Real-case kernel capture + OFF identity (lane o5-port, CPU only, no GPU).

Reproduces the established release-identity trace (proofs/o1_nlbind/probe.py via
proofs/s1_port/release_identity.py and w44-params off_raw.py): fresh CPU load of
the real case through nested_pipeline._load_domains (Pallas interpreted, local
facade non-native), then one production step `_advance_chunk_fori`:
  * canonical jaxpr sha (E151 location-stripped; the release identity),
  * raw target HLO of `.trace(...).lower(lowering_platforms=(PLATFORM,))`
    with jax.default_backend()->'gpu' (E156) and the facade native,
  * every Pallas kernel's Triton IR as lowered for that platform
    (lower_jaxpr_to_triton_module intercepted; real shapes, release flags).

--convention identity : facade.cuda_native True at trace (exactly the release probe).
--convention production : additionally gpuwrf.backends.platform.runtime_backend -> VENDOR
    (all backend-dark guards see the vendor, e.g. rad_mcica KISS kernel; E115/E117 trap).
--vendor rocm needs the vendor_ops facade (o5-port); on the base tree use --convention portable
    (facade native but cuda_native False: the W5P01 portable bodies).

usage: JAX_PLATFORMS=cpu python capture_real.py --alias ALIAS --case CASE --out DIR
           [--platform cuda|rocm] [--convention identity|production|portable] [--vendor cuda|rocm]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--alias", required=True, help="stable import alias (never resolved; E58/E94)")
p.add_argument("--case", required=True)
p.add_argument("--out", required=True)
p.add_argument("--platform", default="cuda", choices=("cuda", "rocm"))
p.add_argument("--convention", default="identity", choices=("identity", "production", "portable"))
p.add_argument("--vendor", default="cuda", choices=("cuda", "rocm"))
p.add_argument("--no-canon", action="store_true", help="skip the canonical jaxpr (saves ~20 s)")
p.add_argument("--mlir", action="store_true", help="also write the lowered StableHLO (large)")
p.add_argument("--nested", action="store_true",
               help="max_dom=2: also lower the production d01->d02 force-down and the d02 child step on its output")
a = p.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu", "CPU only (E50)"
root = Path(a.alias)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "ttir").mkdir(exist_ok=True)
sys.path[:0] = [str(root / "src"), str(root / "proofs/o1_nlbind")]

import jax  # noqa: E402
import gpuwrf  # noqa: E402,F401  (release defaults incl. JAX_PALLAS_USE_MOSAIC_GPU=false BEFORE pallas import)
import cpu_safe  # noqa: E402

_lapack_tridiagonal_solve = jax.lax.linalg.tridiagonal_solve
cpu_safe.install()  # CPU load only (E105); the original is restored before the native trace
from jax.experimental import pallas as jpl  # noqa: E402
from jax._src.pallas.triton import lowering as tl  # noqa: E402
from jax._src.pallas.triton import pallas_call_registration as reg  # noqa: E402
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.backends import pallas as facade  # noqa: E402
from gpuwrf.backends import platform as plat  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(str(root)), om.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
try:
    from gpuwrf.backends import vendor_ops as vops  # o5-port facade (absent on base)
except ImportError:
    vops = None

_real_pallas_call = jpl.pallas_call


def _interpret_pallas_call(*args, **kw):
    kw["interpret"] = True
    kw.pop("compiler_params", None)
    return _real_pallas_call(*args, **kw)


def canon(jaxpr) -> str:
    text = str(jaxpr)
    text = re.sub(r"name_and_src_info=[^\]\n]*", "", text)
    return re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", text)


def strip_loc(text: str) -> str:
    text = re.sub(r"\s*loc\((?:[^()]|\([^()]*\))*\)", "", text)
    return "\n".join(line for line in text.splitlines() if not line.startswith("#loc"))


# --- kernel interception -------------------------------------------------------------
_tls = threading.local()
kernels = []
_seen = {}
_orig_reg = reg.pallas_call_lowering
_orig_lower = tl.lower_jaxpr_to_triton_module


def reg_wrap(ctx, *nodes, compiler_params=None, **params):
    _tls.cp = compiler_params
    _tls.name = params.get("name")
    try:
        return _orig_reg(ctx, *nodes, compiler_params=compiler_params, **params)
    finally:
        _tls.cp = None


def lower_wrap(jaxpr, grid_mapping, platform, cc):
    res = _orig_lower(jaxpr, grid_mapping, platform, cc)
    try:
        caller = sys._getframe(1).f_locals  # pallas_call_registration.pallas_call_lowering
        text = res.module.operation.get_asm(enable_debug_info=False)
        text_loc = res.module.operation.get_asm(enable_debug_info=True)
        body = strip_loc(text)
        h = hashlib.sha256(body.encode()).hexdigest()[:16]
        src = jaxpr.debug_info.func_src_info if jaxpr.debug_info else "?"
        fname = jaxpr.debug_info.func_name if jaxpr.debug_info else "kernel"
        nw = caller.get("num_warps")
        ns = caller.get("num_stages")
        key = (h, nw, ns)
        rec = {"name": caller.get("name") or fname, "func": fname, "src": str(src), "platform": platform,
               "cc": cc, "ttir_sha16": h, "ttir_loc_sha16": hashlib.sha256(text_loc.encode()).hexdigest()[:16],
               "ttir_chars": len(body), "grid": [int(g) for g in res.grid] if hasattr(res, "grid") else None,
               "num_warps": nw, "num_stages": ns, "n_in": len(jaxpr.invars), "calls": 1,
               "asm_ops": len(re.findall(r"tt\.elementwise_inline_asm", body)),
               "f64_tokens": len(re.findall(r"\bf64\b", body))}
        if key in _seen:
            _seen[key]["calls"] += 1
        else:
            _seen[key] = rec
            kernels.append(rec)
            (out / "ttir" / f"{platform}_{h}.ttir").write_text(text)
    except Exception as err:  # capture failure is a result, never fatal
        kernels.append({"status": "CAPTURE_FAIL", "error": f"{type(err).__name__}: {err}"[:400]})
    return res


reg.pallas_call_lowering = reg_wrap
tl.lower_jaxpr_to_triton_module = lower_wrap

# --- load real case on CPU (interpreted) -----------------------------------------------
receipt = {"alias": str(root), "case": a.case, "platform": a.platform, "convention": a.convention,
           "vendor": a.vendor, "jax": jax.__version__, "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS", "<unset=1>"),
           "gpu_executed": False, "vendor_ops": vops is not None}
t0 = time.time()
case_src = Path(a.case)
scratch = Path(os.environ.get("O5_SCRATCH", "<DATA_ROOT>/wrf_gpu2/scratch/o5-port"))
scratch.mkdir(parents=True, exist_ok=True)
try:
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        case = Path(tmp) / "case"
        case.mkdir()
        for f in case_src.iterdir():
            (case / f.name).symlink_to(f)
        names = ("d01", "d02") if a.nested else ("d01",)
        cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=len(names))
        jpl.pallas_call = _interpret_pallas_call
        facade.cuda_native = lambda: False
        try:
            _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, names)
            if a.nested:  # production force-down output = the child step's real operand (E126/E156)
                from gpuwrf.runtime.domain_tree import DomainTree, _operational_force
                edge = DomainTree.from_domains(_h, bundles, feedback_enabled=False).edges["d01"][0]
                forced_d02 = _operational_force(edge, carries["d01"], carries["d02"])
        finally:
            jpl.pallas_call = _real_pallas_call
    ns, carry = bundles["d01"].namelist, carries["d01"]
    receipt["load_s"] = round(time.time() - t0, 1)
    receipt["grid"] = list(carry.state.theta.shape)

    # --- trace conventions --------------------------------------------------------------
    if a.convention != "identity":  # identity keeps the established probe's cpu_safe patch
        jax.lax.linalg.tridiagonal_solve = _lapack_tridiagonal_solve
    if a.convention == "identity":
        facade.cuda_native = lambda: True
    elif a.convention == "production":
        facade.cuda_native = lambda: a.vendor == "cuda"
        plat.runtime_backend = lambda: a.vendor
        if vops is not None:
            vops._override_vendor(a.vendor, native=True)
    else:  # portable: native Pallas, non-CUDA bodies (W5P01 semantics)
        facade.cuda_native = lambda: False
        facade._jax_call = _real_pallas_call  # looked up by facade.__getattr__ at call time
    om._resolve_operational_suite(ns)
    jax.clear_caches()
    cb = om.build_clock_base(ns)
    if not a.no_canon:
        t1 = time.time()
        jx = jax.make_jaxpr(lambda c, nl=ns, cb=cb: om._advance_chunk_fori(
            c, nl, 1, cb, n_steps=1, cadence=int(nl.radiation_cadence_steps)))(carry)
        text = canon(jx)
        receipt["canon_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        receipt["canon_chars"] = len(text)
        receipt["canon_s"] = round(time.time() - t1, 1)
        del jx, text
    t2 = time.time()
    original_backend = jax.default_backend
    jax.default_backend = lambda: "gpu"  # native lowering only (E156)
    try:
        native_trace = om._advance_chunk_fori.trace(
            carry, ns, jax.numpy.asarray(1, jax.numpy.int32), cb,
            n_steps=1, cadence=int(ns.radiation_cadence_steps))
        lowered = native_trace.lower(lowering_platforms=(a.platform,))
        hlo = lowered.compiler_ir("hlo").as_hlo_text()
    finally:
        jax.default_backend = original_backend
    receipt["lower_s"] = round(time.time() - t2, 1)
    if a.nested:
        receipt["programs"] = {}
        ns2 = bundles["d02"].namelist
        om._resolve_operational_suite(ns2)
        cb2 = om.build_clock_base(ns2)
        jax.default_backend = lambda: "gpu"
        try:
            for label, fn, args in (
                    ("forcedown_d01_d02", jax.jit(lambda pc, cc: _operational_force(edge, pc, cc)),
                     (carries["d01"], carries["d02"])),
                    ("step_d02", None, None)):
                t3 = time.time()
                try:
                    if fn is None:
                        tr = om._advance_chunk_fori.trace(forced_d02, ns2, jax.numpy.asarray(1, jax.numpy.int32), cb2,
                                                          n_steps=1, cadence=int(ns2.radiation_cadence_steps))
                    else:
                        tr = fn.trace(*args)
                    text2 = tr.lower(lowering_platforms=(a.platform,)).compiler_ir("hlo").as_hlo_text()
                    (out / f"raw_{label}.hlo").write_text(text2)
                    receipt["programs"][label] = {"raw_hlo_sha256": hashlib.sha256(text2.encode()).hexdigest(),
                                                  "raw_hlo_bytes": len(text2.encode()),
                                                  "triton_custom_calls": text2.count("__gpu$xla.gpu.triton"),
                                                  "lower_s": round(time.time() - t3, 1)}
                    del text2, tr
                except Exception as err:  # noqa: BLE001 - recorded per program
                    receipt["programs"][label] = {"error": f"{type(err).__name__}: {str(err)[:600]}",
                                                  "traceback_tail": traceback.format_exc()[-1200:]}
        finally:
            jax.default_backend = original_backend
    (out / "raw.hlo").write_text(hlo)
    receipt["raw_hlo_sha256"] = hashlib.sha256(hlo.encode()).hexdigest()
    receipt["raw_hlo_bytes"] = len(hlo.encode())
    receipt["native_custom_calls"] = hlo.count("custom-call(")
    receipt["triton_custom_calls"] = hlo.count("__gpu$xla.gpu.triton")
    if a.mlir:
        mlir_text = lowered.as_text()
        (out / "step.mlir").write_text(mlir_text)
        receipt["mlir_sha256"] = hashlib.sha256(mlir_text.encode()).hexdigest()
except Exception as exc:  # noqa: BLE001
    receipt["error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
    receipt["traceback_tail"] = traceback.format_exc()[-2500:]
receipt["kernels"] = kernels
receipt["n_kernels_unique"] = sum(1 for k in kernels if "ttir_sha16" in k)
receipt["n_kernel_lowerings"] = sum(k.get("calls", 0) for k in kernels)
receipt["wall_s"] = round(time.time() - t0, 1)
(out / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k not in ("kernels", "traceback_tail")}, default=str), flush=True)
if "traceback_tail" in receipt:
    print(receipt["traceback_tail"], flush=True)
os._exit(0 if "error" not in receipt else 1)
