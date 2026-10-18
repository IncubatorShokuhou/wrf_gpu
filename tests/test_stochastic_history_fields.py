"""SPP history fields FIELD_CONV/FIELD_PBL/FIELD_SF with cu_physics=0 (lane w41-stoch).

Original CPU-WRF 24 h Swiss references st3 (spp=1) and st4 (all switches) write all three as +0 in every
cell of all 25 frames; the GPU arms of the same namelists lacked them (writer gap). The writer now emits
WRF's +0 for cu_physics 0/1 with MYNN + Noah-MP, never for other cumulus or land schemes.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.producer_history import STOCH_FIELDS, inactive_names
from gpuwrf.io.wrfout_writer import _lookup, prepare_wrfout_payload, write_prepared_wrfout
from test_m7_netcdf_writer import synthetic_case, writer_authority

PROOF = json.loads((Path(__file__).resolve().parents[1]/"proofs/w41_stoch/FIELD_SPP_CPU_ALL_FRAMES.json").read_text())
SPP = SimpleNamespace(patterns=(None,)*4 + (object(),)*3)


@pytest.mark.parametrize("case_id", ("st3", "st4"))
@pytest.mark.parametrize("full", (False, True))
def test_cu0_spp_fields_are_original_cpu_zero_bits(tmp_path, case_id, full):
    proof = PROOF["cases"][case_id]
    assert len(proof["frames"]) == 25 and set(proof["fields"]) == set(STOCH_FIELDS)
    state, grid, nl = synthetic_case()
    nl.update(mp_physics=8, cu_physics=0, bl_pbl_physics=5, sf_surface_physics=4, stochastic_config=SPP)
    authority = writer_authority(grid, "d01")
    p = prepare_wrfout_payload(state, grid, nl, tmp_path/"wrfout", domain="d01", domain_authority=authority,
                               valid_time="2023-01-15T00:00:00", run_start="2023-01-15T00:00:00", lead_hours=0.,
                               variable_subset=STOCH_FIELDS, full_variable_set=full, diagnostics={})
    write_prepared_wrfout(p, expected_domain="d01", expected_domain_authority=authority, variable_subset=STOCH_FIELDS)
    with Dataset(p.target) as ds:
        for name, ref in proof["fields"].items():
            assert ref["constant"] and ref["bits"] == 0 and ref["dtype"] == "<f4"
            assert all(f["fields"][name]["all_equal_bits"] and f["fields"][name]["bits"] == 0 for f in proof["frames"])
            assert name in ds.variables, f"missing WRF SPP history field {name}"
            assert list(ds[name].dimensions) == ref["dimensions"] and ds[name].dtype == np.float32
            assert np.all(np.asarray(ds[name][:]).view(np.uint32) == 0)


def test_spp_zero_fields_only_on_proven_branches():
    base = {"stochastic_config": SPP, "bl_pbl_physics": 5, "sf_surface_physics": 4}
    for cu in (0, 1):
        assert set(STOCH_FIELDS) <= set(inactive_names({**base, "cu_physics": cu}, _lookup))
    for change in ({"cu_physics": 3}, {"cu_physics": 0, "sf_surface_physics": 3}, {"cu_physics": 0, "bl_pbl_physics": 1},
                   {"cu_physics": 0, "stochastic_config": None},
                   {"cu_physics": 0, "stochastic_config": SimpleNamespace(patterns=(None,)*7)}):
        assert not set(STOCH_FIELDS) & set(inactive_names({**base, **change}, _lookup)), change
