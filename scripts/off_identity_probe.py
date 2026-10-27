#!/usr/bin/env python3
"""OFF-identity probe of the real production step(s), native CUDA pass FIRST (CPU only, no GPU).

Why (FINDINGS E161 + critic cr-o5-port-probe 2026-10-24): the established probes
(proofs/o1_nlbind/probe.py via proofs/s1_port/release_identity.py and the w44 ``off_raw.py`` copies)
(1) trace the canonical jaxpr FIRST on the CPU default backend, so inner jits keep interpreted traces and the
native CUDA lowering omits the MYNN cloudmix/plume/tridiag2 and Noah snowwater kernels (PROD d01: 138 instead
of 153 Triton calls); (2) leave ``gpuwrf.backends.platform`` on the CPU (drops the McICA KISS kernels);
(3) hash ``as_hlo_text()``, which elides every constant with > 10 elements as ``{...}``: PROD d01 has 393
such constants / 15.4 M elements (RRTMG/MOST tables, baked per-case grid fields) — a changed value stays
IDENTICAL; (4) cover only the root domain.

This probe, in one fresh process:
  1. loads the real case on the CPU like the release probe (Pallas interpreted, local facade non-native,
     pure-JAX tridiagonal for initialization only); with ``--nested`` it loads d01+d02 and runs the product
     force-down (``domain_tree._operational_force``) so the child program sees its real operand (E126/E156);
  2. ``jax.clear_caches()``, then the NATIVE pass, production convention: local facade native,
     ``platform.runtime_backend`` -> 'cuda', ``jax.default_backend`` -> 'gpu' (E156), for every covered program
     (root step; with ``--nested`` also the force-down and the d02 child step): raw HLO sha (legacy, elided
     constants), ``raw_hlo_full_sha256`` over ``as_hlo_text(print_large_constants=True)`` (constant VALUES),
     every Pallas kernel as JAX lowers it to Triton (body sha with/without source lines, launch params);
  3. BLINDNESS self-checks -> verdict BLIND (exit 2): any ``pallas_call`` still ``interpret=True`` inside a native
     program, or the set of native ``pallas_call`` kernel names differs from the set the Triton lowering saw;
  4. ``jax.clear_caches()``, then the canonical release jaxpr of the root step (the published identity value;
     it does not see constant values, so ``canon_consts_sha256`` hashes the TOP-LEVEL jaxpr constants as well;
     constants captured inside inner jits are covered by ``raw_hlo_full_sha256``, which the comparator also requires).
Any failure (incl. case loading) still writes a receipt with verdict ERROR (exit 1).

usage:
  JAX_PLATFORMS=cpu python scripts/off_identity_probe.py --alias ALIAS --case CASE --out OUT.json
      [--nested] [--no-canon] [--order native-first|canon-first]
``--order canon-first`` reproduces the old blind order for diagnosis only. Compare two trees with
scripts/off_identity_compare.py. Source paths are canonicalised by gpuwrf's
``jax_hlo_source_file_canonicalization_regex``; source LINE numbers are part of the Triton IR, so a pure line
shift in a Pallas file is reported as DIFFERENT (check ``ttir_sha16`` equality to recognise it, E109/E128).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


def canon_text(jaxpr) -> str:
    """Release-identity canonicalisation (proofs/o1_nlbind/probe.py, E151)."""
    text = str(jaxpr)
    text = re.sub(r"name_and_src_info=[^\]\n]*", "", text)
    return re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", text)


def strip_loc(text: str) -> str:
    text = re.sub(r"\s*loc\((?:[^()]|\([^()]*\))*\)", "", text)
    return "\n".join(line for line in text.splitlines() if not line.startswith("#loc"))


def consts_sha256(closed_jaxpr) -> str:
    """Hash of the VALUES (dtype, shape, bytes) of a closed jaxpr's constants, in order."""
    import numpy as np

    h = hashlib.sha256()
    for const in getattr(closed_jaxpr, "consts", ()):
        arr = np.asarray(const)
        h.update(f"{arr.dtype}{arr.shape}".encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def hlo_hashes(computation) -> dict:
    """Legacy (constants elided) and full-constant hashes of an XlaComputation."""
    short = computation.as_hlo_text()
    full = computation.as_hlo_text(print_large_constants=True)
    return {"raw_hlo_sha256": hashlib.sha256(short.encode()).hexdigest(), "raw_hlo_bytes": len(short.encode()),
            "raw_hlo_full_sha256": hashlib.sha256(full.encode()).hexdigest(), "raw_hlo_full_bytes": len(full.encode()),
            "elided_constants": short.count("{...}"), "triton_custom_calls": short.count("__gpu$xla.gpu.triton")}


def pallas_stats(closed_jaxpr) -> dict:
    """Count pallas_call equations (recursively), those still interpreted, and per-kernel-name counts."""
    from jax.extend.core import jaxprs_in_params

    stats = {"pallas_calls": 0, "interpreted": 0, "interpreted_names": {}, "native_names": {}}

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "pallas_call":
                stats["pallas_calls"] += 1
                info = getattr(eqn.params.get("jaxpr"), "debug_info", None)
                name = getattr(info, "func_name", "?")
                key = "interpreted_names" if eqn.params.get("interpret") else "native_names"
                stats[key][name] = stats[key].get(name, 0) + 1
                if eqn.params.get("interpret"):
                    stats["interpreted"] += 1
            for sub in jaxprs_in_params(eqn.params):
                walk(sub)

    walk(getattr(closed_jaxpr, "jaxpr", closed_jaxpr))
    return stats


class KernelCensus:
    """Intercept JAX's Pallas->Triton lowering; one record per distinct kernel body."""

    def __init__(self, program: str = "program"):
        from jax._src.pallas.triton import lowering as tl

        self.program = program
        self._tl, self._orig, self.kernels, self._seen = tl, tl.lower_jaxpr_to_triton_module, [], {}

    def __enter__(self):
        orig = self._orig

        def wrap(jaxpr, grid_mapping, platform, cc):
            res = orig(jaxpr, grid_mapping, platform, cc)
            caller = sys._getframe(1).f_locals  # pallas_call_registration.pallas_call_lowering (A3: test-pinned)
            body = strip_loc(res.module.operation.get_asm(enable_debug_info=False))
            located = res.module.operation.get_asm(enable_debug_info=True)
            key = (hashlib.sha256(body.encode()).hexdigest()[:16], caller.get("num_warps"), caller.get("num_stages"))
            if key in self._seen:
                self._seen[key]["calls"] += 1
            else:
                info = jaxpr.debug_info
                rec = {"program": self.program, "func": getattr(info, "func_name", "?"),
                       "name": caller.get("name") or getattr(info, "func_name", "?"),
                       "src": str(getattr(info, "func_src_info", "?")), "platform": platform,
                       "ttir_sha16": key[0], "ttir_loc_sha16": hashlib.sha256(located.encode()).hexdigest()[:16],
                       "num_warps": key[1], "num_stages": key[2], "calls": 1,
                       "grid": [int(g) for g in res.grid] if hasattr(res, "grid") else None}
                self._seen[key] = rec
                self.kernels.append(rec)
            return res

        self._tl.lower_jaxpr_to_triton_module = wrap
        return self

    def __exit__(self, *exc):
        self._tl.lower_jaxpr_to_triton_module = self._orig


def name_selfcheck(stats: dict, kernels: list[dict]) -> list[str]:
    """Problems if the native pallas_call names and the Triton-lowered names differ (F2)."""
    native = set(stats["native_names"])
    lowered = {k["func"] for k in kernels}
    problems = []
    if native - lowered:
        problems.append(f"native pallas_call(s) without a Triton lowering: {sorted(native - lowered)}")
    if lowered - native:
        problems.append(f"Triton lowerings without a native pallas_call: {sorted(lowered - native)}")
    return problems


def provenance(root: Path) -> dict:
    """Which tree was measured (F3): realpath, git HEAD/dirty, content hash of src/gpuwrf, toolchain, env."""
    import jax
    import jaxlib

    out = {"alias": str(root), "realpath": os.path.realpath(root)}
    try:
        out["git_head"] = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                         text=True, timeout=30).stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "src"], capture_output=True,
                               text=True, timeout=60)
        out["git_src_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        out["git_head"] = out["git_src_dirty"] = None
    h = hashlib.sha256()
    src = root / "src" / "gpuwrf"
    for path in sorted(src.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            h.update(str(path.relative_to(src)).encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    out["src_gpuwrf_content_sha256"] = h.hexdigest()
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    out["versions"] = {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "triton": triton_version,
                       "python": sys.version.split()[0]}
    out["program_env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("JAX_", "XLA_"))}
    return out


def _by_name(kernels):
    counts = {}
    for k in kernels:
        counts[k["func"]] = counts.get(k["func"], 0) + k["calls"]
    return dict(sorted(counts.items()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--alias", required=True, help="source tree import path (base and candidate: the same path)")
    p.add_argument("--case", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nested", action="store_true",
                   help="max_dom=2: also the product d01->d02 force-down and the d02 child step on its output")
    p.add_argument("--no-canon", action="store_true")
    p.add_argument("--order", default="native-first", choices=("native-first", "canon-first"))
    p.add_argument("--hlo-dir", help="also write each program's full-constant HLO text here")
    a = p.parse_args(argv)
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise SystemExit("JAX_PLATFORMS=cpu required (E50)")
    root = Path(a.alias)
    operator_env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("GPUWRF_")}
    os.environ.setdefault("GPUWRF_JAX_CACHE", "0")  # a CPU probe never writes the shared cache (E68)
    out = {"tool": "scripts/off_identity_probe.py", "schema": 2, "case": os.path.realpath(a.case),
           "covered_programs": ["root_d01_step"] + (["forcedown_d01_d02", "child_d02_step"] if a.nested else []),
           "max_dom": 2 if a.nested else 1, "order": a.order, "gpu_executed": False, "operator_env": operator_env,
           "programs": {}, "kernels": []}
    t0 = time.time()
    code = 1
    try:
        sys.path[:0] = [str(root / "src"), str(root / "proofs/o1_nlbind")]
        import jax
        import gpuwrf  # noqa: F401  release defaults (incl. JAX_PALLAS_USE_MOSAIC_GPU=false) BEFORE pallas
        from gpuwrf._fast_defaults import FAST_DEFAULTS_STATUS
        import cpu_safe
        from jax.experimental import pallas as jpl
        from gpuwrf.backends import pallas as facade
        from gpuwrf.backends import platform as plat
        from gpuwrf.contracts import state as state_contract
        from gpuwrf.integration import nested_pipeline as pipeline
        from gpuwrf.runtime import operational_mode as om

        out["provenance"] = provenance(root)
        out["release_defaults_active"] = bool(FAST_DEFAULTS_STATUS.get("enabled"))
        assert jax.devices()[0].platform == "cpu"
        assert om.__file__.startswith(str(root)), om.__file__
        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
        lapack_solve = jax.lax.linalg.tridiagonal_solve
        cpu_safe.install()  # CPU initialization only (E105)
        real_call, real_backend, real_runtime = jpl.pallas_call, jax.default_backend, plat.runtime_backend

        def interpret_call(*args, **kw):
            kw["interpret"] = True
            kw.pop("compiler_params", None)
            return real_call(*args, **kw)

        names = ("d01", "d02") if a.nested else ("d01",)
        with tempfile.TemporaryDirectory(dir=os.environ.get("OFF_PROBE_TMP")) as tmp:
            case = Path(tmp) / "case"
            case.mkdir()
            for f in Path(a.case).iterdir():
                (case / f.name).symlink_to(f)
            cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1,
                                                max_dom=len(names))
            jpl.pallas_call = interpret_call
            facade.cuda_native = lambda: False
            try:
                hierarchy, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, names)
                if a.nested:  # the child program's real operand = the product force-down output (E126/E156)
                    from gpuwrf.runtime.domain_tree import DomainTree, _operational_force
                    edge = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False).edges["d01"][0]
                    forced_d02 = _operational_force(edge, carries["d01"], carries["d02"])
            finally:
                jpl.pallas_call = real_call
        out["load_s"] = round(time.time() - t0, 1)
        ns, carry = bundles["d01"].namelist, carries["d01"]
        out["grid_d01"] = list(carry.state.theta.shape)
        om._resolve_operational_suite(ns)
        cb = om.build_clock_base(ns)
        step = jax.numpy.asarray(1, jax.numpy.int32)
        programs = [("root_d01_step", lambda: om._advance_chunk_fori.trace(
            carry, ns, step, cb, n_steps=1, cadence=int(ns.radiation_cadence_steps)))]
        if a.nested:
            ns2 = bundles["d02"].namelist
            om._resolve_operational_suite(ns2)
            cb2 = om.build_clock_base(ns2)
            out["grid_d02"] = list(forced_d02.state.theta.shape)
            programs += [
                ("forcedown_d01_d02", lambda: jax.jit(lambda pc, cc: _operational_force(edge, pc, cc)).trace(
                    carries["d01"], carries["d02"])),
                ("child_d02_step", lambda: om._advance_chunk_fori.trace(
                    forced_d02, ns2, step, cb2, n_steps=1, cadence=int(ns2.radiation_cadence_steps)))]

        def canonical():
            jax.clear_caches()
            facade.cuda_native = lambda: True  # the established release convention
            plat.runtime_backend, jax.default_backend = real_runtime, real_backend
            jax.lax.linalg.tridiagonal_solve = cpu_safe._thomas
            t = time.time()
            jx = jax.make_jaxpr(lambda c, nl=ns, cb=cb: om._advance_chunk_fori(
                c, nl, 1, cb, n_steps=1, cadence=int(nl.radiation_cadence_steps)))(carry)
            text = canon_text(jx)
            out["canon_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            out["canon_consts_sha256"] = consts_sha256(jx)
            out["canon_chars"] = len(text)
            out["canon_s"] = round(time.time() - t, 1)

        def native(clear=True):
            if clear:
                jax.clear_caches()  # no load-phase CPU trace may survive into the native programs (F2)
            facade.cuda_native = lambda: True
            plat.runtime_backend = lambda: "cuda"
            jax.default_backend = lambda: "gpu"
            jax.lax.linalg.tridiagonal_solve = lapack_solve
            problems = []
            try:
                for label, trace in programs:
                    t = time.time()
                    with KernelCensus(label) as census:
                        traced = trace()
                        stats = pallas_stats(traced.jaxpr)
                        comp = traced.lower(lowering_platforms=("cuda",)).compiler_ir("hlo")
                        rec = hlo_hashes(comp)
                        if a.hlo_dir:
                            Path(a.hlo_dir, f"{label}.hlo").write_text(comp.as_hlo_text(print_large_constants=True))
                    del comp, traced
                    rec.update(pallas=stats, n_kernel_bodies=len(census.kernels),
                               n_pallas_lowerings=sum(k["calls"] for k in census.kernels),
                               lowerings_by_name=_by_name(census.kernels), s=round(time.time() - t, 1))
                    out["programs"][label] = rec
                    out["kernels"] += census.kernels
                    if stats["interpreted"]:
                        problems.append(f"{label}: interpreted pallas_call(s) in the native program = stale CPU "
                                        f"traces (E161): {json.dumps(stats['interpreted_names'])}")
                    problems += [f"{label}: {msg}" for msg in name_selfcheck(stats, census.kernels)]
            finally:
                jax.default_backend, plat.runtime_backend = real_backend, real_runtime
            root_rec = out["programs"]["root_d01_step"]
            for key in ("raw_hlo_sha256", "raw_hlo_full_sha256", "raw_hlo_bytes", "triton_custom_calls",
                        "n_kernel_bodies", "n_pallas_lowerings"):
                out[key] = root_rec[key]  # schema-1 compatible root fields
            out["native_pallas"] = root_rec["pallas"]
            return problems

        if a.order == "canon-first":
            if not a.no_canon:
                canonical()
            problems = native(clear=False)  # the old probes never cleared caches between the two passes
        else:
            problems = native()
            if not a.no_canon:
                canonical()
        out["blind_reasons"] = problems
        out["verdict"] = "BLIND" if problems else "COMPLETE"
        code = 2 if problems else 0
    except Exception as exc:  # noqa: BLE001 - a failed probe is a result, never a pass
        out["verdict"] = "ERROR"
        out["error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
        out["traceback_tail"] = traceback.format_exc()[-2000:]
        code = 1
    out["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("kernels", "traceback_tail", "programs")},
                     default=str), flush=True)
    os._exit(code)


if __name__ == "__main__":
    main()
