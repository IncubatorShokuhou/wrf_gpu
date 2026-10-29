"""Private component cache layout and actual native executable/E41 evidence."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import sys

ARM_CAP_BYTES = 384 * 1024**2
RUN_CAP_BYTES = 2 * ARM_CAP_BYTES
CACHE_DIRS = {
    "JAX_COMPILATION_CACHE_DIR": "jax", "GPUWRF_JAX_CACHE_DIR": "jax",
    "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda",
    "XDG_CACHE_HOME": "xdg", "TMPDIR": "tmp",
}


def private_paths(root):
    root = Path(root).resolve()
    paths = {}
    for name, leaf in CACHE_DIRS.items():
        path = root / leaf
        if path.is_symlink():
            raise ValueError(f"private cache must not be a symlink: {path}")
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = str(path)
    return paths


def check_contract(root, backend):
    paths = private_paths(root)
    for key, path in paths.items():
        if os.environ.get(key) != path:
            raise ValueError(f"private cache environment differs: {key}")
    expected = "1" if backend == "gpu" else "0"
    if os.environ.get("GPUWRF_JAX_CACHE") != expected:
        raise ValueError("native persistence must be enabled; CPU smoke persistence must be disabled")
    return paths


def usage(root, limit=ARM_CAP_BYTES):
    total = 0
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise ValueError(f"unowned symlink in private cache: {path}")
        if path.is_file():
            total += path.stat().st_size
    if total > limit:
        raise RuntimeError(f"component exceeds {limit} byte disk cap: {total}")
    return total


def export_native(compiled, lowered, root, device, serializer=None):
    """Retain the executable returned by the measured lower/compile itself."""
    if getattr(device, "platform", None) != "gpu":
        raise ValueError("native export requires the actual GPU device")
    if serializer is None:
        from gpuwrf.runtime.aot_executable import serialize
        serializer = serialize
    target = Path(root) / "native"
    target.mkdir(parents=True, exist_ok=True)
    blob, meta = serializer(compiled, lowered=lowered, dev=device)
    if meta.trace_metadata != "rich" or meta.retained_module != "full":
        raise RuntimeError("E41 export requires rich actual GPU HLO")
    path = target / "operator.xlaexec"
    path.write_bytes(blob)
    (target / "operator.meta").write_bytes(pickle.dumps(meta))
    return path, dict(path=str(path), sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob),
                      trace_metadata=meta.trace_metadata, retained_module=meta.retained_module)


def native_gate(path, source_root, scanner=None):
    """Use the project scanner on this component blob, without labelling it a domain step."""
    module = scanner
    if module is None:
        spec = importlib.util.spec_from_file_location("mynn_component_e41", Path(source_root) / "scripts/wave_cubin_gate.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    out = Path(path).parent / "e41"
    out.mkdir(exist_ok=True)
    entry = module.Executable(Path(path), "aot", "MYNN_PDF_COMPONENT")
    record = module.scan_executable(entry, out, out / "unused.sqlite")
    module.assert_coverage([entry], [record])
    # Same verdict rules as wave_cubin_gate.run_gate; UNKNOWN is never PASS.
    fail = any(elf["stack_over256"] or elf["nonleaf_over1k"] for elf in record["elfs"]) or bool(record.get("kinput_transpose_over500"))
    unknown = any(elf.get("callgraph_unknown") for elf in record["elfs"]) or record.get("kinput_transpose_status") != "MEASURED_RICH_AOT_HLO"
    verdict = "FAIL" if fail else "INCOMPLETE" if unknown else "PASS"
    result = dict(e41_gate=verdict, entries=[record], scope="actual component executable only")
    (out / "component_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    if verdict != "PASS":
        raise RuntimeError(f"actual component E41 {verdict}: {out}")
    return result


def cleanup_caches(root):
    # Exact allowlist only. Keep native executable + E41 evidence for review.
    for leaf in sorted(set(CACHE_DIRS.values())):
        path = Path(root) / leaf
        if path.is_symlink():
            raise ValueError(f"refusing symlink cleanup: {path}")
        if path.exists():
            shutil.rmtree(path)
