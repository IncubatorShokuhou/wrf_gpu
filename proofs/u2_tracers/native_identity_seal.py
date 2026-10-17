"""Add recursive constant bytes to the unchanged official native identity probe.

Usage: python native_identity_seal.py SOURCE OUTPUT_JSON [CASE] [--cpu-interpret]
Set FAST/CENSUS and JAX_PLATFORMS=cpu externally. No forecast execution or new
model tracing function is introduced. The official probe owns its canonical
graph and facade policy; this wrapper retains its last outer make_jaxpr result.
Output: the official JSON plus OUTPUT_JSON's stem + '.constants.json'.
--cpu-interpret instead runs the unchanged O1 probe with CPU facade, interpreted
Pallas and layout hints omitted, matching the coupled CPU harness policy. Its
graph is a CPU diagnostic identity, NOT the canonical native release hash.
"""

from __future__ import annotations

from collections import Counter
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile


DEFAULT_CASE = Path("<USER_HOME>/wrf_gpu2_lanes/u2-tracers/TR01_original_cpu_case")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constant_records(closed_jaxpr) -> dict:
    """Walk ClosedJaxpr/Jaxpr and basic parameter containers deterministically.

    Seal every ClosedJaxpr.consts entry and array-valued equation parameter.
    Scalar/static parameter values and literals belong to the official graph.
    Opaque parameter objects are counted by type, never repr/id/address hashed;
    their private attributes are explicitly outside this additional seal.
    """
    import numpy as np

    records, opaque = [], Counter()
    active = set()
    counts = Counter()

    def add_array(value, path):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"object-dtype constant cannot have a stable byte seal: {path}")
        raw = array.tobytes(order="C")
        records.append({"path": path, "dtype": array.dtype.str,
                        "shape": list(array.shape), "nbytes": len(raw),
                        "bytes_sha256": hashlib.sha256(raw).hexdigest()})

    def key_label(key):
        if key is None or isinstance(key, (str, bool, int, float)):
            return type(key).__name__ + ":" + json.dumps(key, allow_nan=False, ensure_ascii=True)
        raise TypeError(f"unsupported parameter dictionary key type: {type(key).__name__}")

    def walk(value, path, *, const=False):
        if const:
            add_array(value, path)
            return
        if (isinstance(value, (np.ndarray, np.generic))
                or (not isinstance(value, type) and hasattr(value, "dtype")
                    and hasattr(value, "shape") and hasattr(value, "__array__"))):
            add_array(value, path)
            return
        if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
            return
        closed = hasattr(value, "jaxpr") and hasattr(value, "consts")
        jaxpr = hasattr(value, "eqns") and hasattr(value, "constvars") and hasattr(value, "invars")
        container = isinstance(value, (dict, list, tuple))
        if closed or jaxpr or container:
            # Identity is used only to reject cycles; it never enters the seal.
            # Repeated acyclic references are traversed per semantic occurrence.
            identity = id(value)
            if identity in active:
                raise ValueError(f"cycle in program parameter containers: {path}")
            active.add(identity)
            try:
                if closed:
                    counts["closed_jaxprs"] += 1
                    for index, constant in enumerate(value.consts):
                        walk(constant, f"{path}.consts[{index}]", const=True)
                    walk(value.jaxpr, path + ".jaxpr")
                elif jaxpr:
                    counts["jaxprs"] += 1
                    for index, equation in enumerate(value.eqns):
                        walk(equation.params, f"{path}.eqns[{index}].params")
                elif isinstance(value, dict):
                    for key in sorted(value, key=key_label):
                        walk(value[key], path + "[" + key_label(key) + "]")
                else:
                    for index, item in enumerate(value):
                        walk(item, f"{path}[{index}]")
            finally:
                active.remove(identity)
        else:
            opaque[type(value).__module__ + "." + type(value).__qualname__] += 1

    walk(closed_jaxpr, "outer")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    dtypes = {}
    for record in records:
        entry = dtypes.setdefault(record["dtype"], {"count": 0, "nbytes": 0})
        entry["count"] += 1
        entry["nbytes"] += record["nbytes"]
    return {"constants_sha256": hashlib.sha256(payload).hexdigest(),
            "constant_count": len(records), "constant_nbytes": sum(r["nbytes"] for r in records),
            "dtypes": dict(sorted(dtypes.items())), "program_counts": dict(sorted(counts.items())),
            "records": records, "opaque_parameter_types_excluded": dict(sorted(opaque.items())),
            "scope": "recursive ClosedJaxpr.consts and array-valued params in dict/list/tuple; "
                     "opaque object attributes excluded; scalar/static params and literals use official graph"}


def main(source: Path, output: Path, case: Path, *, cpu_interpret: bool = False) -> dict:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("set JAX_PLATFORMS=cpu externally for this CPU identity probe")
    if int(os.environ.get("O1_EXEC_STEPS", "0")) != 0:
        raise ValueError("O1_EXEC_STEPS must be 0: identity only, no forecast execution")
    if not cpu_interpret and os.environ.get("O1_INTERPRET_EXEC", "0") != "0":
        raise ValueError("O1_INTERPRET_EXEC must be 0: retain the official native trace policy")
    official = source / "proofs/s1_port/release_identity.py"
    upstream = source / "proofs/o1_nlbind/probe.py"
    hashes = {"wrapper": file_sha256(Path(__file__)), "official_runner": file_sha256(official),
              "upstream_probe": file_sha256(upstream),
              "operational_mode": file_sha256(source / "src/gpuwrf/runtime/operational_mode.py")}
    git_identity = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD", "HEAD^{tree}"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    # Importing this file, syntax checks, and collector tests do not import JAX.
    import jax

    original_make_jaxpr = jax.make_jaxpr
    retained = []
    depth = 0

    @wraps(original_make_jaxpr)
    def retaining_make_jaxpr(*args, **kwargs):
        traced = original_make_jaxpr(*args, **kwargs)

        @wraps(traced)
        def remember_result(*trace_args, **trace_kwargs):
            nonlocal depth
            outer = depth == 0
            depth += 1
            try:
                result = traced(*trace_args, **trace_kwargs)
            finally:
                depth -= 1
            if outer:
                retained[:] = [result]
            return result

        return remember_result

    argv = sys.argv[:]
    restore_env = {}
    cpu_restore = None
    jax.make_jaxpr = retaining_make_jaxpr
    try:
        if cpu_interpret:
            sys.path[:0] = [str(source / "src"), str(source / "proofs/o1_nlbind")]
            from gpuwrf.backends import pallas as facade
            from jax.experimental import layout, pallas as pl

            cpu_restore = (facade, facade.cuda_native, layout, layout.with_layout_constraint,
                           pl, pl.pallas_call, tempfile.TemporaryDirectory)
            facade.cuda_native = lambda: False
            layout.with_layout_constraint = lambda value, *args, **kwargs: value
            original_tempdir = tempfile.TemporaryDirectory

            def lane_tempdir(*args, **kwargs):
                directory = kwargs.get("dir")
                if directory is not None and Path(directory) == Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind"):
                    kwargs["dir"] = "<USER_HOME>/wrf_gpu2_lanes/u2-tracers"
                return original_tempdir(*args, **kwargs)

            tempfile.TemporaryDirectory = lane_tempdir
            for key, value in (("O1_CASE", str(case)), ("O1_INTERPRET_EXEC", "1"), ("O1_EXEC_STEPS", "0")):
                restore_env[key] = os.environ.get(key)
                os.environ[key] = value
            sys.argv = [str(upstream), str(source), str(output), "release"]
            runpy.run_path(str(upstream), run_name="__main__")
        else:
            sys.argv = [str(official), str(source), str(output), str(case)]
            runpy.run_path(str(official), run_name="__main__")
    finally:
        jax.make_jaxpr = original_make_jaxpr
        sys.argv = argv
        if cpu_restore is not None:
            facade, native, layout, constraint, pl, call, tempdir = cpu_restore
            facade.cuda_native = native
            layout.with_layout_constraint = constraint
            pl.pallas_call = call
            tempfile.TemporaryDirectory = tempdir
        for key, value in restore_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    metadata = json.loads(output.read_text())
    arm = metadata["arms"]["release"]
    if "error" in arm or "sha" not in arm or not retained:
        raise RuntimeError(f"official identity probe did not return a release program: {arm.get('error')}")
    # make_jaxpr(return_shape=True) is not used by the official final call.
    if not hasattr(retained[0], "consts") or not hasattr(retained[0], "jaxpr"):
        raise TypeError("last outer make_jaxpr call did not return a ClosedJaxpr")
    result = constant_records(retained[0])
    result.update({"schema": ("u2-tracers-cpu-interpreted-program-constants-v1" if cpu_interpret
                              else "u2-tracers-native-program-constants-v1"), "source": str(source),
                   "case": str(case), "official_output": str(output), "source_hashes": hashes,
                   "source_git_commit": git_identity[0], "source_git_tree": git_identity[1],
                   "official_graph_sha256": arm["sha"], "forecast_executed": False,
                   "identity_mode": "cpu-interpret" if cpu_interpret else "native-release",
                   "selected_runner": str(upstream if cpu_interpret else official),
                   "canonical_native_release_hash": not cpu_interpret,
                   "cpu_layout_constraints_omitted": cpu_interpret,
                   "native_facade_at_trace": not cpu_interpret, "gpu_executed": False,
                   "extra_model_program_created": False,
                   "environment": {key: os.environ.get(key) for key in
                                   ("JAX_PLATFORMS", "GPUWRF_FAST_DEFAULTS", "GPUWRF_CENSUS")}})
    result["graph_and_constants_sha256"] = hashlib.sha256(
        (arm["sha"] + ":" + result["constants_sha256"]).encode()).hexdigest()
    seal = output.with_name(output.stem + ".constants.json")
    seal.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"constant_seal": str(seal), "constant_count": result["constant_count"],
                      "constant_nbytes": result["constant_nbytes"],
                      "official_graph_sha256": arm["sha"], "constants_sha256": result["constants_sha256"]}))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("case", type=Path, nargs="?", default=DEFAULT_CASE)
    parser.add_argument("--cpu-interpret", action="store_true")
    args = parser.parse_args()
    main(args.source.resolve(), args.output.resolve(), args.case.resolve(), cpu_interpret=args.cpu_interpret)
