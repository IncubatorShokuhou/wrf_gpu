"""Complete a merged native scan using observed pre-compaction AOT HLO.

The generic scanner's UNKNOWN compact metadata remains unchanged. This proof
requires an exact returned-blob SHA binding to retained full optimized HLO;
equal native ELF alone is insufficient.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gate", type=Path)
    p.add_argument("bindings", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    tool = Path(__file__).resolve().parents[2] / "scripts/e41/hlo_scan.py"
    spec = importlib.util.spec_from_file_location("merged_hlo_scan", tool)
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    gate = json.loads(a.gate.read_text())
    bindings = json.loads(a.bindings.read_text())
    by_hash = {b["aot_blob_sha256"]: b for b in bindings}
    assert len(by_hash) == len(bindings), "duplicate AOT binding"
    assert gate["coverage"]["status"] == "PASS"
    assert gate["coverage"]["n_discovered"] == gate["coverage"]["n_scanned"] > 0
    assert gate["hard_stack_gate"] == "PASS" and not gate["conservative_flags"]
    completed = []
    for record in gate["entries"]:
        assert record["elfs"]
        for elf in record["elfs"]:
            assert elf["max_stack_B"] <= 256 and not elf["stack_over256"]
            assert not elf["nonleaf_over1k"] and not elf["callgraph_unknown"]
        status = record["kinput_transpose_status"]
        if status.startswith("MEASURED_RICH_"):
            assert record["kinput_transpose_over500"] == []
            continue
        assert status == "UNKNOWN_AOT_METADATA" and record["source_kind"] == "aot"
        b = by_hash[record["blob_sha256"]]
        # The native scanner hashed the bytes it inspected; independently
        # recheck the still-retained cache file before accepting the binding.
        assert hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() == record["blob_sha256"]
        raw = gzip.decompress(Path(b["optimized_hlo_file"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == b["optimized_hlo_sha256"]
        assert b["delegate_output_unchanged"] is True
        result = scanner.scan_text(raw.decode())
        rows = [r for r in result["transpose_hero_fusions"] if r["kind"] == "kInput"]
        assert rows and max(r["n_instr"] for r in rows) <= 500
        completed.append({"program": record["program"], **b,
                          "max_kinput_transpose_instructions": max(r["n_instr"] for r in rows)})
    assert completed, "no compact AOT metadata completed by this proof"
    result = {"e41_gate": "PASS", "generic_gate": gate["e41_gate"],
              "native_and_coverage_gate": str(a.gate), "bindings": str(a.bindings),
              "scope": "exact observed AOT serialization output plus retained pre-compaction HLO",
              "completed_aot_programs": completed}
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
