"""Reuse the c11 actual-composite CPU CUDA compiler, retaining its executable.

Arguments are the upstream dl_dump.py arguments. Only post-compile diagnostics
are added; loading, native tracing, target and compiler options are unchanged.
"""
from pathlib import Path
import hashlib
import sys

source = Path(sys.argv[sys.argv.index("--source") + 1])
upstream = source / "proofs/t_cfix_cu/c11_ptxas/dl_dump.py"
original = upstream.read_text()
needle = "        del lowered, compiled\n"
assert original.count(needle) == 1
diagnostics = '''        executable = compiled.runtime_executable()
        blob = executable.serialize()
        native_dir = a.out / "cache/aot/deviceless"
        native_dir.mkdir(parents=True, exist_ok=True)
        (native_dir / (name + ".xlaexec")).write_bytes(blob)
        row["native_sha256"] = hashlib.sha256(blob).hexdigest()
        row["native_bytes"] = len(blob)
        try:
            memory = executable.get_compiled_memory_stats()
            row["memory"] = {k: getattr(memory, k) for k in (
                "argument_size_in_bytes", "output_size_in_bytes",
                "alias_size_in_bytes", "temp_size_in_bytes")}
        except Exception as exc:
            row["memory_api_error"] = str(exc)
        row["upstream_harness_sha256"] = UPSTREAM_SHA256
        row["effective_harness_sha256"] = EFFECTIVE_SHA256
        save()
        del executable, blob, lowered, compiled
'''
effective = original.replace(needle, diagnostics)
exec(compile(effective, str(upstream), "exec"), {
    "__name__": "__main__", "__file__": str(Path(__file__).resolve()),
    "UPSTREAM_SHA256": hashlib.sha256(original.encode()).hexdigest(),
    "EFFECTIVE_SHA256": hashlib.sha256(effective.encode()).hexdigest(),
})
