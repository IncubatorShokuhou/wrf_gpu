"""Private cache/native-export contract; every simulated native object is test-only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mynn_operator_artifacts", ROOT / "proofs/v45_mynnopts/operator_artifacts.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


@pytest.mark.parametrize("backend,enabled", [("cpu", "0"), ("gpu", "1")])
def test_all_six_private_controls_are_bound(tmp_path, monkeypatch, backend, enabled):
    paths = A.private_paths(tmp_path / "pdf0")
    for key, value in paths.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GPUWRF_JAX_CACHE", enabled)
    assert A.check_contract(tmp_path / "pdf0", backend) == paths
    for key in paths:
        with monkeypatch.context() as patch:
            patch.delenv(key)
            with pytest.raises(ValueError, match=key):
                A.check_contract(tmp_path / "pdf0", backend)
    with monkeypatch.context() as patch:
        patch.setenv("GPUWRF_JAX_CACHE", "0" if enabled == "1" else "1")
        with pytest.raises(ValueError, match="persistence"):
            A.check_contract(tmp_path / "pdf0", backend)


def test_export_retains_the_measured_object_and_rich_metadata(tmp_path):
    compiled, lowered, device = object(), object(), SimpleNamespace(platform="gpu")
    calls = []
    def serialize(value, **kwargs):
        assert value is compiled and kwargs["lowered"] is lowered and kwargs["dev"] is device
        calls.append(1)
        return b"PRIVATE_SIMULATED_EXECUTABLE", SimpleNamespace(trace_metadata="rich", retained_module="full")
    path, receipt = A.export_native(compiled, lowered, tmp_path, device, serialize)
    assert calls == [1] and path.read_bytes() == b"PRIVATE_SIMULATED_EXECUTABLE"
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.with_suffix(".meta").is_file()
    with pytest.raises(ValueError, match="actual GPU"):
        A.export_native(compiled, lowered, tmp_path, SimpleNamespace(platform="cpu"), serialize)


@pytest.mark.parametrize("fault,verdict", [(None, "PASS"), ("stack_over256", "FAIL"),
    ("nonleaf_over1k", "FAIL"), ("callgraph_unknown", "INCOMPLETE"), ("hlo", "FAIL"), ("unknown_hlo", "INCOMPLETE")])
def test_e41_checks_that_exact_export_and_refuses_unknown(tmp_path, fault, verdict):
    path = tmp_path / "operator.xlaexec"
    path.write_bytes(b"PRIVATE_SIMULATED_EXECUTABLE")
    record = dict(path=str(path), elfs=[dict(stack_over256=[], nonleaf_over1k=[], callgraph_unknown=[])],
                  kinput_transpose_over500=[], kinput_transpose_status="MEASURED_RICH_AOT_HLO")
    if fault in record["elfs"][0]:
        record["elfs"][0][fault] = ["fault"]
    if fault == "hlo":
        record["kinput_transpose_over500"] = ["501 instructions"]
    if fault == "unknown_hlo":
        record["kinput_transpose_status"] = "UNKNOWN_AOT_METADATA"
    def scan(entry, *args):
        assert entry.path == path and entry.kind == "aot" and entry.program == "MYNN_PDF_COMPONENT"
        return record
    def coverage(entries, records):
        assert entries[0].path == path and records[0]["path"] == str(path)
    scanner = SimpleNamespace(Executable=lambda path, kind, program: SimpleNamespace(path=path, kind=kind, program=program),
                              scan_executable=scan, assert_coverage=coverage)
    if verdict == "PASS":
        assert A.native_gate(path, ROOT, scanner)["e41_gate"] == "PASS"
    else:
        with pytest.raises(RuntimeError, match=verdict):
            A.native_gate(path, ROOT, scanner)


def test_cache_cleanup_preserves_export_and_disk_cap(tmp_path):
    A.private_paths(tmp_path)
    native = tmp_path / "native"
    native.mkdir()
    (native / "operator.xlaexec").write_bytes(b"reference")
    (tmp_path / "jax" / "cache").write_bytes(b"cache")
    with pytest.raises(RuntimeError, match="disk cap"):
        A.usage(tmp_path, limit=1)
    A.cleanup_caches(tmp_path)
    assert (native / "operator.xlaexec").read_bytes() == b"reference"
    assert all(not (tmp_path / name).exists() for name in set(A.CACHE_DIRS.values()))


def test_exact_cpu_launcher_prefix_rewires_each_pdf_privately(tmp_path):
    """Stop at a private fake flock: no real lock, CUDA or physics process."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    probe = bin_dir / "flock"
    probe.write_text('''#!<USER_HOME>/miniconda3/bin/python
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
out=Path(args[args.index('--out')+1]).with_suffix('.prefix.json')
keys=('JAX_COMPILATION_CACHE_DIR','GPUWRF_JAX_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','XDG_CACHE_HOME','TMPDIR','GPUWRF_JAX_CACHE')
out.write_text(json.dumps({'args':args,'env':{key:os.environ.get(key) for key in keys}}))
''')
    probe.chmod(0o755)
    output = tmp_path / "output"
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               WRF_GPU2_SEQ_SMOKE="1", MO_SOURCE_ROOT=str(ROOT), MO_OUTPUT_ROOT=str(output), JAX_PLATFORMS="cpu")
    for key in A.CACHE_DIRS:
        env[key] = str(tmp_path / "foreign" / key)
    run = subprocess.run(["bash", str(ROOT / "proofs/v45_mynnopts/night_operators.sh"), "operators"],
                         env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    records = []
    for option in (0, 1):
        record = json.loads((output / f"operators_cpu_pdf{option}.prefix.json").read_text())
        args = record["args"]
        arm = Path(args[args.index("--cache-root") + 1])
        assert arm.name == f"pdf{option}" and arm.is_relative_to(output / "cache_runs")
        assert args[args.index("--option") + 1] == str(option)
        assert record["env"]["GPUWRF_JAX_CACHE"] == "0"
        for key, leaf in A.CACHE_DIRS.items():
            assert record["env"][key] == str(arm / leaf)
        records.append(record)
    assert records[0]["env"]["JAX_COMPILATION_CACHE_DIR"] != records[1]["env"]["JAX_COMPILATION_CACHE_DIR"]


def test_native_launcher_prefix_enables_private_persistence_before_lock(tmp_path):
    """Inspect actual env argv and stop before the official wrapper; no GPU/lock."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    probe = bin_dir / "env"
    probe.write_text('''#!<USER_HOME>/miniconda3/bin/python -S
import json,sys
from pathlib import Path
args=sys.argv[1:];assert args.pop(0)=='-i'
values={}
while args and '=' in args[0]:
 key,value=args.pop(0).split('=',1);values[key]=value
out=Path(args[args.index('--out')+1]).with_suffix('.native_prefix.json')
out.write_text(json.dumps({'command':args,'env':values}))
''')
    probe.chmod(0o755)
    output = tmp_path / "output"
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               WRF_GPU2_SEQ_SMOKE="0", MO_SOURCE_ROOT=str(ROOT), MO_OUTPUT_ROOT=str(output), JAX_PLATFORMS="cpu")
    run = subprocess.run(["bash", str(ROOT / "proofs/v45_mynnopts/night_operators.sh"), "operators"],
                         env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    for option in (0, 1):
        record = json.loads((output / f"operators_gpu_pdf{option}.native_prefix.json").read_text())
        command = record["command"]
        assert command[:4] == ["<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh", "--label", "v45-mynnopts", "--"]
        arm = Path(command[command.index("--cache-root") + 1])
        assert record["env"]["GPUWRF_JAX_CACHE"] == "1"
        assert record["env"]["GPUWRF_AOT_KEEP_TRACE_METADATA"] == "1"
        for key, leaf in A.CACHE_DIRS.items():
            assert record["env"][key] == str(arm / leaf)
