"""Run the existing CAM smoke in-process, saving its real compiler operands."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys


def capture_inputs(directory, delegate):
    """Observe compile arguments without changing module, options or execution."""
    def compile_and_load(backend, module, devices, options, callbacks):
        name = str(module.operation.attributes["sym_name"]).strip('"')
        if "advance_chunk_fori" in name:
            raw = str(module).encode()
            digest = hashlib.sha256(raw).hexdigest()
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{digest}.mlir.gz").write_bytes(gzip.compress(raw, mtime=0))
            opts = options.SerializeAsString()
            (directory / f"{digest}.options.pb").write_bytes(opts)
            (directory / f"{digest}.json").write_text(json.dumps({
                "name": name, "platform": backend.platform,
                "mlir_sha256": digest, "options_sha256": hashlib.sha256(opts).hexdigest(),
            }, indent=2) + "\n")
        return delegate(backend, module, devices, options, callbacks)
    return compile_and_load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1":
        parser.error("requires scripts/with_gpu_lock.sh --label s1-camstack")
    if os.environ.get("JAX_PLATFORMS") != "cuda":
        parser.error("capture requires JAX_PLATFORMS=cuda")
    # Bootstrap product flags before importing JAX (E107).
    import gpuwrf  # noqa: F401
    from jax._src import compiler
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import gpu_smoke_v034 as smoke

    original = compiler.backend_compile_and_load
    compiler.backend_compile_and_load = capture_inputs(args.out / "compiler_inputs", original)
    try:
        return smoke.child(argparse.Namespace(child="ra3_cam", out=str(args.out / "run"),
                                              cpu_dry_run=False, trace_only=False, steps=3))
    finally:
        compiler.backend_compile_and_load = original


if __name__ == "__main__":
    raise SystemExit(main())
