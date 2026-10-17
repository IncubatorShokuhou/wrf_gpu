"""CPU inspection of a captured CAM step; retain exact rich HLO before cleanup.

Uses the repository's E188/E201 decoder and v0.3.4 cuobjdump fallback because
the historical opus-a19 scanners were removed. Conservative CALL flags remain
unresolved; this tool does not turn STACK PASS into a full E41 verdict.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        parser.error("inspection requires JAX_PLATFORMS=cpu")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import gpu_smoke_e41 as fallback
    import zstandard
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm

    gate = fallback._gate_module()
    entries = gate.discover(args.cache)
    args.out.mkdir(parents=True, exist_ok=True)
    inventory = []
    for entry in entries:
        raw = entry.path.read_bytes()
        record = {"entry": entry.path.name, "kind": entry.kind,
                  "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        if entry.kind == "jax_cache":
            blob = zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 33)[4:]
            hlo = gate._optimized_hlo(blob, ax, asm).encode()
            name = hashlib.sha256(raw).hexdigest()[:12] + ".optimized.hlo.gz"
            (args.out / name).write_bytes(gzip.compress(hlo, mtime=0))
            record.update(hlo=name, hlo_sha256=hashlib.sha256(hlo).hexdigest())
        inventory.append(record)
    targets = []
    if args.dump is not None:
        for path in sorted(args.dump.rglob("*gpu_target_config.pbtxt")):
            raw = path.read_bytes()
            name = hashlib.sha256(raw).hexdigest()[:12] + ".target.textproto"
            (args.out / name).write_bytes(raw)
            targets.append({"path": str(path), "retained": name,
                            "sha256": hashlib.sha256(raw).hexdigest()})
    (args.out / "capture.json").write_text(json.dumps({
        "entries": inventory, "targets": targets,
        "target_status": "CAPTURED" if targets else "MISSING",
        "scope": "actual serialized executable inspection; no performance/fidelity verdict",
    }, indent=2) + "\n")
    result = fallback.scan(args.cache, args.out)
    print(json.dumps({key: result[key] for key in
                      ("n_entries", "hard_stack_gate", "max_stack_B", "unscanned")}))
    if result["unscanned"] or result["hard_stack_gate"] == "INCOMPLETE":
        return 1
    return 2 if result["hard_stack_gate"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
