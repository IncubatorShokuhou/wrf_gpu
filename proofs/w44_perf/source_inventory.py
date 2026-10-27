"""Static version inventory. No import/trace of any version or GPU speed claim."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {"v032": "v0.3.2", "v033": "v0.3.3", "v040": "v0.4.0", "v043": "v0.4.3"}
HOT = ("src/gpuwrf/_fast_defaults.py", "src/gpuwrf/runtime/operational_mode.py",
       "src/gpuwrf/runtime/domain_tree.py", "src/gpuwrf/integration/nested_pipeline.py",
       "src/gpuwrf/coupling/physics_couplers.py", "src/gpuwrf/coupling/scan_adapters.py",
       "src/gpuwrf/kernels/phys_mynn.py", "src/gpuwrf/kernels/phys_thompson.py")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def defaults(ref):
    data = git("show", ref + ":src/gpuwrf/_fast_defaults.py")
    tree = ast.parse(data)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "FAST_PATH_DEFAULTS":
            return ast.literal_eval(node.value)
    raise AssertionError(ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = {"schema": "w44-perf.static_sources.v1", "gpu_executed": False,
              "scope": "source/default facts only; no native identity, workload count or speed inference", "versions": {}}
    for label, ref in VERSIONS.items():
        hashes = {}
        for file in HOT:
            exists = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", ref + ":" + file],
                                    capture_output=True).returncode == 0
            hashes[file] = hashlib.sha256(git("show", ref + ":" + file)).hexdigest() if exists else None
        result["versions"][label] = {"ref": ref, "commit": git("rev-parse", ref + "^{}").decode().strip(),
                                      "src_tree": git("rev-parse", ref + ":src/gpuwrf").decode().strip(),
                                      "defaults": defaults(ref), "hot_file_sha256": hashes}
    result["default_changes"] = {}
    for a, b in zip(VERSIONS, list(VERSIONS)[1:]):
        aa, bb = result["versions"][a]["defaults"], result["versions"][b]["defaults"]
        result["default_changes"][a + "_to_" + b] = {key: {a: aa.get(key), b: bb.get(key)}
                                                    for key in sorted(aa.keys() | bb.keys()) if aa.get(key) != bb.get(key)}
    # Keep the historical benchmark/private source distinction explicit.
    private = "3e194296c22518ae649506174a3a6e360fcd2c3d"
    result["v032_benchmark_vs_public_tag"] = {
        "benchmark_commit": private, "benchmark_src_tree": git("rev-parse", private + ":src/gpuwrf").decode().strip(),
        "diff_stat": git("diff", "--stat", private, "v0.3.2", "--", "src/gpuwrf").decode(),
        "hot_path_diff": git("diff", "--unified=1", private, "v0.3.2", "--", *HOT).decode(),
        "scope": "tag is the requested new comparison source; historical benchmark native equality not claimed"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print({key: len(value) for key, value in result["default_changes"].items()})


if __name__ == "__main__":
    main()
