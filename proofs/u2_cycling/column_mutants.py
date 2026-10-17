"""Delete either admission call; require genuine assertion failures, no setup errors."""
from pathlib import Path
import json
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[2]
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/u2-cycling")
sys.path.insert(0, str(ROOT / "tests"))
import pytest
import test_cycle as tests
from gpuwrf.cycling import core

results = {}
with tempfile.TemporaryDirectory(dir=LANE, prefix="column-mutants-") as folder:
    parent = Path(folder)
    text = Path(core.__file__).read_text()
    cases = (("publication", "        validate_columns(src, index, land, noahmp=int(dst.SF_SURFACE_PHYSICS) == 4)\n",
              "cycle", tests.test_inconsistent_column_refused_before_publication),
             ("restore", "    validate_columns(ds, 0, land, noahmp=True)\n",
              "restore_land_state", tests.test_internally_hashed_bad_column_refused_before_restore))
    for name, call, symbol, check in cases:
        healthy = parent / (name + "-healthy")
        healthy.mkdir()
        check(healthy, True, "zero_soil_thickness")
        assert text.count(call) == 1
        module = types.ModuleType("gpuwrf.cycling.mutant_" + name)
        exec(compile(text.replace(call, ""), name + "_deleted.py", "exec"), module.__dict__)
        directory = parent / (name + "-mutant")
        directory.mkdir()
        original = getattr(tests, symbol)
        setattr(tests, symbol, getattr(module, symbol))
        try:
            check(directory, True, "zero_soil_thickness")
        except (AssertionError, pytest.fail.Exception) as exc:
            message = str(exc)
            assert ("DID NOT RAISE" in message if name == "publication" else
                    "invalid column reached model fields before admission" in message), message
            results[name] = {"killed": True, "exception": type(exc).__name__, "assertion": message}
        finally:
            setattr(tests, symbol, original)
assert len(results) == 2 and all(row["killed"] for row in results.values())
report = {"mutants": results, "setup_errors": 0, "scope": "changed host-admission calls; original CPU-WRF private copies",
          "cache": "NONE", "gpu_executed": False}
(LANE / "CY17_column_mutants.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
