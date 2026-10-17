"""v0.3.4: every honest per-scheme qualification note is reachable through classify_scheme.

Regression for a duplicate module-level ``_IMPLEMENTED_NOTES`` in io/scheme_catalog.py: the second
definition silently replaced the first, so km_opt=2 (an unqualified v022 scaffold) and km_opt=3
reported only the generic "Operationally wired into the GPU scan." reason.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from gpuwrf.io import scheme_catalog
from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme

GENERIC = "Operationally wired into the GPU scan."

# (key, code) -> a phrase the scheme's honest caveat must carry
EXPECTED = {
    ("sf_urban_physics", 2): "BEP",
    ("sf_urban_physics", 3): "BEM",
    ("km_opt", 2): "3-D TKE",
    ("km_opt", 5): "SMS-3DTKE",
    # km_opt=2/5: refused in v0.3.4 (unqualified, NaN under the release carry; manager on rel034
    # 18:08Z) -> RECOGNIZED_FAIL_CLOSED with that reason (tests/test_namelist_binding.py).
    ("km_opt", 3): "3-D Smagorinsky",
    ("cu_physics", 5): "Grell-3D",
    ("cu_physics", 93): "Grell-Devenyi",
    ("bl_pbl_physics", 9): "CAM-UW",
    # mp=40: demoted to REFERENCE_ONLY by o1-nlbind in v0.3.4, IMPLEMENTED again in v0.4.0 (below).
    ("sf_surface_physics", 3): "RUC LSM",
    # v0.4.0 o2-rootbdy: root extra species take WRF flow_dep_bdy; single domains only.
    ("mp_physics", 24): "WSM7",
    ("mp_physics", 28): "QNWFA/QNIFA",
    ("mp_physics", 40): "Morrison-aerosol",
}


def test_implemented_notes_defined_once():
    tree = ast.parse(Path(scheme_catalog.__file__).read_text(encoding="utf-8"))
    names = []
    for node in tree.body:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
        names += [t.id for t in targets if isinstance(t, ast.Name)]
    assert names.count("_IMPLEMENTED_NOTES") == 1, names.count("_IMPLEMENTED_NOTES")


@pytest.mark.parametrize("key,code", sorted(EXPECTED))
def test_note_reachable_through_classify_scheme(key, code):
    support = classify_scheme(key, code)
    assert support.status is SupportStatus.IMPLEMENTED
    assert support.reason != GENERIC
    assert EXPECTED[(key, code)] in support.reason


QUALIFICATION_CONFIGS = {
    ("km_opt", 2): "km2", ("km_opt", 3): "km3",
    ("cu_physics", 5): "cu5", ("cu_physics", 93): "cu93",
    ("bl_pbl_physics", 9): "bl9", ("sf_surface_physics", 3): "sf3",
    ("mp_physics", 24): "mp24", ("mp_physics", 28): "mp28",
    ("mp_physics", 40): "mp40", ("ra_lw_physics", 3): "ra3",
    ("ra_sw_physics", 3): "ra3", ("sf_urban_physics", 1): "urban1",
    ("dfi_opt", 3): "dfi3", ("moist_adv_opt", 0): "moist0",
}


@pytest.mark.parametrize("key,code", sorted(QUALIFICATION_CONFIGS))
def test_notes_match_recorded_gpu_qualification(key, code):
    config = QUALIFICATION_CONFIGS[(key, code)]
    receipt = Path(__file__).resolve().parents[1] / "proofs/v040/gpuqual" / config / "D6.json"
    result = json.loads(receipt.read_text())
    support = (scheme_catalog.classify_control(key, code)
               if key in ("dfi_opt", "moist_adv_opt") else classify_scheme(key, code))
    assert support.status is SupportStatus.IMPLEMENTED
    reason = support.reason
    if result["verdict"] == "PASS":
        if config == "urban1":
            # The receipt qualifies the default SLUCM setup. LCZ/NUDAPT options
            # carry their own pending qualification in the same catalog note.
            reason, separator, options = reason.partition(" Additional opt-in ")
            assert separator
            assert "GPU qualification pending" in options
            assert "experimental one-cell zero-init parity only" in options
        assert result["coverage_complete"] and result["paired_frames"] == 25
        assert "GPU-qualified vs CPU-WRF" in reason
        assert "E41 robustness flag:" in reason
        assert f"proofs/v040/gpuqual/{config}" in reason
        assert "qualification pending" not in reason
    else:
        assert "GPU-qualified" not in reason
        assert "qualification pending" in reason
