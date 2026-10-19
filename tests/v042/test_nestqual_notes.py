"""v0.4.2 nested boundary-layer qualification (lane w41-nestqual): the catalog note of each scheme states exactly the
verdict recorded in proofs/w41_nestqual/NESTQUAL_SUMMARY.json (D6 24 h d01-d03 vs an original CPU-WRF twin on the
ALISIOS G2 Tenerife 3-nest arms). A PASS scheme is called nested-qualified and cites the receipt; any other verdict
keeps the scheme out of the nested-qualified wording. Deleting or weakening a note, or flipping a scheme whose
receipt does not pass, fails here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "proofs/w41_nestqual/NESTQUAL_SUMMARY.json"
ARMS = {"C2": 1, "C3": 11, "C4": 7, "C7": 16}   # G2 arms -> bl_pbl_physics (all with sf_sfclay_physics=1)
SUMMARY = json.loads(RECEIPT.read_text())


def test_receipt_covers_the_four_arms():
    assert SUMMARY["schema"] == "w41-nestqual.summary.v1"
    assert SUMMARY["issue"] == "20260707_18z"
    assert set(SUMMARY["arms"]) == set(ARMS)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_note_matches_recorded_nested_verdict(arm):
    result = SUMMARY["arms"][arm]
    support = classify_scheme("bl_pbl_physics", ARMS[arm])
    assert support.status is SupportStatus.IMPLEMENTED
    reason = support.reason
    if result["verdict"] == "PASS":
        assert result["input_identity"] is True
        for d in ("d01", "d02", "d03"):
            dom = result["domains"][d]
            assert dom["verdict"] == "PASS" and dom["paired_frames"] == 25
            assert dom["failures"] == 0 and dom["nonfinite"] == 0 and dom["coverage_complete"]
        gate = ROOT / "proofs/w41_nestqual" / arm / "NEST_GATE.json"
        assert hashlib.sha256(gate.read_bytes()).hexdigest() == result["nest_gate_sha256"]
        assert "Nested-qualified" in reason
        assert "proofs/w41_nestqual/NESTQUAL_SUMMARY.json" in reason
        assert "3-nest" in reason and "20260707_18z" in reason
    else:
        assert "Nested-qualified" not in reason
        assert "not validated inside nests" in reason
