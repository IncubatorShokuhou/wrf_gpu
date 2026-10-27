"""rrtmg_precision_gate --mode oracle: McICA overlap of its two WRF references (o5-port W09, CPU).

The M5 analytic Tier-1 fixtures encode WRF icld=1 RANDOM overlap, so the gate pins random overlap for that step only;
the overlap the process imported (release default GPUWRF_RRTMG_MAXRAND=1 = cldovrlp=2) is checked against pristine
WRF of the same cldovrlp on fid-q2 RE01. CPU pytest runs the legacy column solvers (tests/conftest); the overlap is a
trace-time module constant, so the release overlap is simulated by pinning rad_mcica._MAXRAND.
"""
import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import jax
import pytest

from gpuwrf.kernels import rad_mcica
from gpuwrf.validation import tier1_rrtmg as T1

HERE = Path(__file__).resolve().parent


def _gate():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("rrtmg_precision_gate", HERE / "rrtmg_precision_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _gate()


@pytest.fixture(autouse=True)
def _fresh_traces():
    jax.clear_caches()  # the overlap is read at trace time (E140/E161)
    yield
    jax.clear_caches()


def test_m5_tier1_needs_random_overlap_and_the_pin_restores(monkeypatch, tmp_path):
    monkeypatch.setattr(rad_mcica, "_MAXRAND", True)  # the release overlap
    unpinned = T1.run_tier1_sw(tmp_path / "unpinned.json")
    assert not unpinned["pass"], "M5 fixtures no longer random-overlap: drop the pin"
    assert unpinned["per_field_max_abs_err"]["surface_down"] > 100.0
    with gate.overlap(False):
        assert rad_mcica._MAXRAND is False
        pinned = T1.run_tier1_sw(tmp_path / "pinned.json")
        lw = T1.run_tier1_lw(tmp_path / "pinned_lw.json")
    assert pinned["pass"] and lw["pass"], (pinned["per_field_max_abs_err"], lw["per_field_max_abs_err"])
    assert rad_mcica._MAXRAND is True


def test_oracle_mode_runs_tier1_inside_the_pin_then_checks_re01():
    tree = ast.parse(inspect.getsource(gate.main))
    pinned = [node for node in ast.walk(tree) if isinstance(node, ast.With)
              and any(ast.unparse(item.context_expr) == "overlap(False)" for item in node.items)]
    assert len(pinned) == 1, "Tier-1 must run inside exactly one overlap(False) block"
    calls = {ast.unparse(n.func) for n in ast.walk(pinned[0]) if isinstance(n, ast.Call)}
    assert {"T1.run_tier1_lw", "T1.run_tier1_sw"} <= calls
    outside = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)} - calls
    assert {"re01_overlap_gate", "lw_driver_truth"} <= outside and not {"T1.run_tier1_lw", "T1.run_tier1_sw"} & outside


def test_lw_driver_truth_pins_random_for_all_sky(monkeypatch):
    monkeypatch.setattr(rad_mcica, "_MAXRAND", True)  # the release overlap
    state, _ = T1.load_lw_fixture_state()
    fn = gate.LW.solve_rrtmg_lw_column
    result = jax.block_until_ready(fn(state, with_clear_sky=True))
    record = gate.lw_driver_truth(fn, state, result)
    assert set(record) == {"random:flux_down", "random:flux_up", "random:clear_flux_down", "random:clear_flux_up",
                           "imported:clear_flux_down", "imported:clear_flux_up"}
    # The overlap-dependent all-sky fields; the clear-sky fields pass on the release radiation path only (the legacy
    # CPU path still carries B22's 50 W/m2 clear-up miss) and are asserted by the direct gate run.
    for key in ("random:flux_down", "random:flux_up"):
        assert record[key]["finite"] and record[key]["passed"], record
    truth = gate.json.loads((HERE.parent / "b_rad/data/lw_driver_clear_flux.json").read_text())
    down = gate.np.asarray(truth["fluxes"])[..., truth["fields"].index("flux_down")]
    assert gate.np.abs(gate.np.asarray(result.flux_down) - down).max() > 10.0  # all-sky truth is random overlap
    assert rad_mcica._MAXRAND is True


@pytest.mark.parametrize("maxrand, arm", ((True, "C"), (False, "D")))
def test_re01_gate_follows_the_imported_overlap(monkeypatch, maxrand, arm):
    monkeypatch.setattr(rad_mcica, "_MAXRAND", maxrand)
    result = gate.re01_overlap_gate()
    assert result["reference_arm"] == arm and result["columns"] == 186
    assert result["columns_outside"] == 0, result
    assert result["columns_outside_other_arm"] >= 20, result
