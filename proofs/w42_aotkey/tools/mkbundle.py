"""mkbundle.py ROOT SRC [--autotune]: a sealed mini bundle in the release layout (as tests/test_aot_key_cli.py)."""
import importlib.util, json, sys
from pathlib import Path

root, src = Path(sys.argv[1]), Path(sys.argv[2])
repo = src.parent
jax_dir = root / "jax"
(jax_dir / "aot" / "0.4.1-jax0.10.0-jaxlib0.10.0-cuda_sm120" / "d01").mkdir(parents=True)
(jax_dir / ("jit__advance_chunk_fori-" + "ab" * 32 + "-cache")).write_bytes(b"sealed jit entry")
(jax_dir / "aot" / "0.4.1-jax0.10.0-jaxlib0.10.0-cuda_sm120" / "d01" / ("k_" + "cd" * 32 + ".xlaexec")).write_bytes(b"blob")
for name in ("autotune", "triton", "cuda"):
    (root / name).mkdir()
    (root / name / "sealed.bin").write_bytes(name.encode())
autotune = f" GPUWRF_XLA_AUTOTUNE_CACHE_DIR={root / 'autotune'}" if "--autotune" in sys.argv else ""
(root / "bundle.env").write_text(
    f"export PYTHONPATH={src}\n"
    f"export GPUWRF_JAX_CACHE_DIR={jax_dir} JAX_COMPILATION_CACHE_DIR={jax_dir}{autotune}\n"
    f"export TRITON_CACHE_DIR={root / 'triton'} CUDA_CACHE_PATH={root / 'cuda'}\n")
spec = importlib.util.spec_from_file_location("compile_bundle", repo / "scripts/release/compile_bundle.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)
(root / "sealed").mkdir()
manifest = {"schema": cb.SCHEMA, "cache": str(jax_dir.resolve()), "files": cb.scan(jax_dir), "aot_meta": {}}
(root / "sealed" / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest))
print("bundle", root, "autotune" if autotune else "no-autotune-var")
