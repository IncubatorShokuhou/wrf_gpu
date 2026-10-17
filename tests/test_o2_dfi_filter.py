"""WRF DFI filter weights (share/dfi.F dfcoef/optfil_driver) vs the pristine-Fortran oracle, bitwise.

The fixture ``proofs/o2_dfi/dfcoef_oracle.json`` holds every ``hcoeff`` of 64 configurations as
binary32 bit patterns, produced by the VERBATIM pristine WRF 4.7.1 routines compiled like the
CPU-WRF reference build (proofs/o2_dfi/build_dfcoef_oracle.py).  Pure NumPy (no JAX).
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.dfi.filter import SUPPORTED_NFILTER, dfcoef, optfil_hcoeff, optfil_nstep2

FIXTURE = Path(__file__).resolve().parents[1] / "proofs" / "o2_dfi" / "dfcoef_oracle.json"


def _cases():
    payload = json.loads(FIXTURE.read_text())
    assert payload["source_sha256"] == "c7b94989ccdf637ebe756b9dcdff26d5b9ba798852ec74c3415a853a32a4ae0f"
    return payload["cases"]


def _bits(case):
    return np.array([struct.unpack(">f", bytes.fromhex(b))[0] for b in case["hcoeff_bits"]], dtype=np.float32)


def test_fixture_covers_every_supported_filter_and_both_stage_signs():
    cases = _cases()
    assert {c["nfilter"] for c in cases} == set(SUPPORTED_NFILTER)
    assert {c["dt_s"] > 0 for c in cases} == {True, False}
    parities = {int(1 + c["rundfi_s"] / abs(c["dt_s"])) % 2 for c in cases}
    assert parities == {0, 1}


@pytest.mark.parametrize("index", range(64))
def test_optfil_hcoeff_bitwise_against_pristine_wrf(index):
    case = _cases()[index]
    ref = _bits(case)
    got = optfil_hcoeff(case["rundfi_s"], case["dt_s"], case["tauc_s"], case["nfilter"])
    assert got.dtype == np.float32
    assert got.shape == ref.shape
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), (
        case["nfilter"], case["rundfi_s"], case["dt_s"],
        int(np.abs(got.view(np.int32).astype(np.int64) - ref.view(np.int32).astype(np.int64)).max()))


def test_swiss_dolph_window_matches_the_cpu_wrf_run_log():
    # rsl.out of the CPU-WRF TDFI oracle run: "In dolph: TIME, W = 0.0 0.1880077E-01 ... 30.0 0.6295491E-01".
    h = optfil_hcoeff(1080, -18.0, 1200.0, 7)
    assert h.shape == (62,) and h[61] == 0.0
    assert float(h[30]) == pytest.approx(0.1880077e-01, abs=5e-9)   # centre weight W(0)
    assert float(h[0]) == pytest.approx(0.6295491e-01, abs=5e-9)    # edge weight W(M)
    assert np.array_equal(h[:30], h[31:61][::-1])                     # mirrored about the centre
    assert float(np.sum(h, dtype=np.float64)) == pytest.approx(1.0, abs=1e-6)


def test_nstep2_and_parity_follow_optfil_driver():
    assert optfil_nstep2(1080, -18.0) == 31          # ceiling((1+60)/2)
    assert optfil_nstep2(1200, 18.0) == 34           # ceiling((1+66.67)/2)
    even = optfil_hcoeff(1200, 18.0, 1200.0, 7)       # int(1+66.67)=67 odd -> 67 weights + one zero
    assert even.shape == (68,) and even[67] == 0.0 and even[66] == even[0]


@pytest.mark.parametrize("nfilter", [-1, 8, 9])
def test_unreproduced_filters_are_refused(nfilter):
    with pytest.raises(ValueError, match="dfi_nfilter"):
        dfcoef(10, 18.0, 1200.0, nfilter)
