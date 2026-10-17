"""Deletion-sensitive mutants and offline timing; run each mode separately."""
from pathlib import Path
import json
import resource
import shutil
import sys
import tempfile
import time
import types

MODE = sys.argv[1]
sys.argv = ["gpuwrf", "cycle"]  # offline benchmark must not import JAX
ROOT = Path(__file__).resolve().parents[2]
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/u2-cycling")
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.cycling import core

if MODE == "mutants":
    sys.path.insert(0, str(ROOT / "tests"))
    import test_cycle as tests
    mutations = {}
    with tempfile.TemporaryDirectory(dir=LANE, prefix="mutants-") as folder:
        temp = Path(folder)
        mutant = types.ModuleType("gpuwrf.cycling.transfer_deleted")
        code = Path(core.__file__).read_text()
        assert code.count("var[0] = np.where(mask, data, before)") == 1
        code = code.replace("var[0] = np.where(mask, data, before)", "var[0] = before")
        exec(compile(code, "transfer_deleted.py", "exec"), mutant.__dict__)
        pair = tests.make_case(temp)
        original = tests.cycle
        tests.cycle = mutant.cycle
        try:
            tests.test_independent_literal_copy_reference(pair, temp)
        except AssertionError as exc:
            mutations["delete_transfer"] = {"killed": True, "assertion": str(exc)[:300]}
        finally:
            tests.cycle = original
        original = core.restore_land_state
        core.restore_land_state = lambda *a: None
        try:
            tests.test_real_swiss_consecutive_cpu_frames(temp)
        except AssertionError as exc:
            mutations["delete_loader_restore"] = {"killed": True, "assertion": str(exc)[:300]}
        finally:
            core.restore_land_state = original
    assert len(mutations) == 2 and all(v["killed"] for v in mutations.values()), mutations
    result = {"mutants": mutations, "setup_errors": 0, "platform": "cpu", "gpu_executed": False}
    (LANE / "CY05_mutants.json").write_text(json.dumps(result, indent=2) + "\n")
else:
    assert MODE == "bench"
    samples = []
    with tempfile.TemporaryDirectory(dir=LANE, prefix="timing-") as folder:
        temp = Path(folder)
        for hour in (1, 2, 3):
            template = temp / f"template{hour}"
            shutil.copyfile(ROOT / "examples/switzerland_d01/wrfinput_d01", template)
            with Dataset(template, "r+") as ds:
                ds["Times"][0] = np.frombuffer(f"2023-01-15_0{hour}:00:00".encode(), dtype="S1")
            source = Path(f"<USER_HOME>/wrf_gpu2_lanes/u2-diag/cpu_reference_3h/wrfout_d01_2023-01-15_0{hour}:00:00")
            start = time.perf_counter()
            report = core.cycle(source, template, temp / f"out{hour}")
            samples.append({"hour": hour, "seconds": time.perf_counter() - start,
                            "source_bytes": source.stat().st_size, "output_bytes": (temp / f"out{hour}").stat().st_size,
                            "fields": len(report["fields"]), "source_sha256": report["source_sha256"]})
    assert "jax" not in sys.modules
    result = {"samples": samples, "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "jax_imported": False, "gpu_executed": False, "cache": "NONE; OS page cache not controlled",
              "scope": "full offline cycle: double SHA256 input pass, land validation, full template copy/write/fsync"}
    (LANE / "CY06_timing.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
