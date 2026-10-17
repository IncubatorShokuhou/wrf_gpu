"""Delete only hail from the actual writer helper; retain healthy controls."""
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v025"))
import cpu_guard  # Must precede the writer's JAX imports, as in tests/conftest.
import pytest
from gpuwrf.io import wrfout_writer as writer

original = writer.total_moist_mass


def drop_hail(state):
    fields = dict(vars(state))
    fields["qh"] = None
    return original(SimpleNamespace(**fields))


if __name__ == "__main__":
    writer.total_moist_mass = drop_hail
    try:
        raise SystemExit(pytest.main([
            "-q", "tests/test_writer_psfc_active_moist.py::test_direct_writer_all_masses_match_original",
            "-k", "optional0 or optional1", *sys.argv[1:]]))
    finally:
        writer.total_moist_mass = original
