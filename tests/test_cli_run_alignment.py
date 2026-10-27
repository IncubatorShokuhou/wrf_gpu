"""Run-length alignment pre-flight (FINDINGS E187/E225): ``gpuwrf run --dry-run`` refuses a forecast length whose
seconds are not an exact multiple of every domain time step -- previously only the nested pipeline refused it, after
model init, and the dry-run returned rc 0 (measured: --hours 1 on WN3, root dt 54 s).

Unit tests cover the shared JAX-free helper (integration/run_alignment.py, also used by the pipeline); CLI tests run
the real ``main([... "--dry-run"])`` on the shipped Swiss example, once with its own dt 18 s and once WN3-like with the
root time_step set to 54 s in a temporary namelist copy.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwrf.cli import main
from gpuwrf.integration import run_alignment as ra

REPO = Path(__file__).resolve().parents[1]
SWISS = REPO / "examples" / "switzerland_d01"
WN3_NAMELIST = REPO / "tests" / "fixtures" / "wn3_20260227_cadence.namelist"
WN3_DT = {"d01": 54.0, "d02": 18.0, "d03": 6.0}
NAMES = ("d01", "d02", "d03")


def test_steps_by_domain_wn3_like_chain() -> None:
    assert ra.steps_by_domain(3, WN3_DT, NAMES) == {"d01": 200, "d02": 600, "d03": 1800}
    assert ra.steps_by_domain(72, WN3_DT, NAMES) == {"d01": 4800, "d02": 14400, "d03": 43200}
    assert ra.steps_by_domain(1.5, WN3_DT, NAMES) == {"d01": 100, "d02": 300, "d03": 900}
    for hours in (1, 11, 4):
        with pytest.raises(ValueError, match=r"does not align with root dt=54\.0s"):
            ra.steps_by_domain(hours, WN3_DT, NAMES)


def test_steps_by_domain_reports_a_child_that_does_not_align() -> None:
    with pytest.raises(ValueError, match=r"does not align with d02 dt=7\.0s"):
        ra.steps_by_domain(1, {"d01": 36.0, "d02": 7.0}, ("d01", "d02"))


def test_dt_by_domain_ratio_chain_and_pipeline_alias() -> None:
    grids = {"d02": SimpleNamespace(parent_id=1, parent_grid_ratio=3),
             "d03": SimpleNamespace(parent_id=2, parent_grid_ratio=3)}
    run = SimpleNamespace(namelist={"domains": {"time_step": [54]}}, grid=grids.__getitem__)
    assert ra.dt_by_domain(run, NAMES) == WN3_DT
    from gpuwrf.integration import nested_pipeline as npl

    assert npl._dt_by_domain is ra.dt_by_domain and npl._steps_by_domain is ra.steps_by_domain


def _fake_run(time_step, grids):
    def grid(name):
        if isinstance(grids[name], Exception):
            raise grids[name]
        return grids[name]
    return SimpleNamespace(namelist={"domains": {"time_step": time_step}}, grid=grid)


def test_preflight_dt_falls_back_to_root_only_and_skips_without_time_step() -> None:
    ok = {"d02": SimpleNamespace(parent_id=1, parent_grid_ratio=3),
          "d03": SimpleNamespace(parent_id=2, parent_grid_ratio=3)}
    assert ra.preflight_dt_by_domain(_fake_run([54], ok), NAMES) == WN3_DT
    unreadable = dict(ok, d03=FileNotFoundError("wrfinput_d03"))
    assert ra.preflight_dt_by_domain(_fake_run([54], unreadable), NAMES) == {"d01": 54.0}
    bad_ratio = dict(ok, d02=SimpleNamespace(parent_id=1, parent_grid_ratio=1))  # dt chain ValueError
    assert ra.preflight_dt_by_domain(_fake_run([54], bad_ratio), NAMES) == {"d01": 54.0}
    assert ra.preflight_dt_by_domain(_fake_run(None, ok), NAMES) is None
    assert ra.preflight_dt_by_domain(_fake_run([18], {}), ("d01",)) == {"d01": 18.0}


def test_pipeline_runs_the_shared_check_before_cadence_planning() -> None:
    """Deletion-sensitive: _execute_nested_pipeline must call _steps_by_domain(config.hours, dt_by_domain, names)
    and take root_steps from it before the output-cadence planning (the post-init check stays in place)."""

    from gpuwrf.integration import nested_pipeline as npl

    tree = ast.parse(Path(npl.__file__).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_execute_nested_pipeline")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    check = [c for c in calls if c.func.id == "_steps_by_domain"
             and [ast.unparse(a) for a in c.args] == ["config.hours", "dt_by_domain", "names"]]
    cadence = [c for c in calls if c.func.id == "_output_cadence_steps_by_domain"]
    assert len(check) == 1 and cadence and check[0].lineno < min(c.lineno for c in cadence)
    assigns = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in ast.walk(fn)
               if isinstance(n, ast.Assign) and len(n.targets) == 1}
    assert assigns.get("root_steps") == "total_steps_by_domain[root]"
    assert assigns.get("total_steps_by_domain", "").startswith("_steps_by_domain(")


@pytest.mark.parametrize("hours, rc", [("1", 2), ("11", 2), ("3", 0)])
def test_cli_nested_wn3_dry_run_refuses_misaligned_hours(tmp_path, capsys, hours, rc) -> None:
    # Namelist-only 3-nest WN3 case (dt 54/18/6): child grids are unreadable, so this also covers the
    # root-only fallback of the pre-flight.
    (tmp_path / "namelist.input").write_bytes(WN3_NAMELIST.read_bytes())
    code = main(["run", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                 "--domains-from-namelist", "--hours", hours, "--dry-run"])
    captured = capsys.readouterr()
    assert code == rc
    if rc:
        assert "run-length alignment" in captured.err and "root dt=54.0s" in captured.err
    else:
        assert json.loads(captured.out)["run_type"] == "nested_live"


def _case(tmp_path: Path, time_step: int | None = None) -> Path:
    case = tmp_path / "case"
    case.mkdir()
    for name in ("wrfinput_d01", "wrfbdy_d01"):
        (case / name).symlink_to(SWISS / name)
    text = (SWISS / "namelist.input").read_text()
    if time_step is not None:
        text, n = re.subn(r"(?mi)^(\s*time_step\s*=\s*)\d+", rf"\g<1>{time_step}", text)
        assert n == 1
    (case / "namelist.input").write_text(text)
    return case


def _dry(case: Path, out: Path, *extra: str) -> int:
    return main(["run", "--input-dir", str(case), "--output-dir", str(out), "--domains-from-namelist",
                 "--dry-run", *extra])


@pytest.mark.skipif(not (SWISS / "wrfbdy_d01").exists(), reason="shipped Swiss example missing")
def test_cli_dry_run_refuses_misaligned_wn3_like_root_dt(tmp_path, capsys) -> None:
    case = _case(tmp_path, time_step=54)
    assert _dry(case, tmp_path / "o1", "--hours", "1") == 2
    err = capsys.readouterr().err
    assert "run-length alignment" in err and "root dt=54.0s" in err
    assert _dry(case, tmp_path / "o3", "--hours", "3") == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["effective_hours"] == 3 and plan["dry_run"] is True


@pytest.mark.skipif(not (SWISS / "wrfbdy_d01").exists(), reason="shipped Swiss example missing")
def test_cli_dry_run_swiss_native_dt18(tmp_path, capsys) -> None:
    case = _case(tmp_path)
    assert _dry(case, tmp_path / "ok", "--hours", "1") == 0
    capsys.readouterr()
    # Explicit windows were already root-checked by run_window.explicit_window (same rc, older wording).
    assert _dry(case, tmp_path / "bad", "--minutes", "1") == 2  # 60 s is not a multiple of 18 s
    assert "root dt=18" in capsys.readouterr().err
