from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.contracts import cammg_fields
from gpuwrf.contracts import precision as precision_contract
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import FP32_GATED, FP64, INT32, PRECISION_MATRIX, SURFACE_LAYER_CARRY_LEAVES
from gpuwrf.contracts.state import State


def test_precision_matrix_covers_every_state_leaf_dtype():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)

    for field in state.active_field_names():
        assert field in PRECISION_MATRIX
        expected_dtype, _gate_required = PRECISION_MATRIX[field]
        assert getattr(state, field).dtype == expected_dtype


def test_precision_matrix_gate_flags_match_adr007_boundary_classes():
    gated = {
        "u",
        "v",
        "theta",
        "qv",
        "qc",
        "qr",
        "qi",
        "qs",
        "qg",
        "Ni",
        "Nr",
        "Ns",
        "Ng",
        # qke moved gated->FP64-locked (qke-fp64-fix sprint 2026-06-04): the
        # MYNN TKE budget overflows fp32 at 1km (d03 NONFINITE after hour 1).
        # qke is OUTSIDE the conserved mass/pressure path, so it joins the
        # FP64-locked class. See contracts/precision.py for the rationale.
        "xland",
        "lakemask",
        "mavail",
        "u_bdy",
        "v_bdy",
        "theta_bdy",
        "qv_bdy",
        "qc_bdy",
        "qr_bdy",
        "qi_bdy",
        "qs_bdy",
        "qg_bdy",
        "Ni_bdy",
        "Nr_bdy",
        # v0.6.0 S0 additive WDM6 number concentrations (FP32_GATED like the
        # other hydrometeor number species). Added here to keep this contract
        # test in sync with the matrix (they were appended to PRECISION_MATRIX
        # but never to this set).
        "Nc",
        "Nn",
        # v0.15 MYNN SGS-cloud diagnostics (FP32_GATED like qc); the prognostic
        # total-water variance qsq stays FP64-locked with the qke TKE family.
        "qc_bl",
        "qi_bl",
        "cldfra_bl",
        # v0.17 ADR-032 graupel/hail substrate (FP32_GATED, same class as the
        # qg/Ng hydrometeor + number species they extend).
        "qh",
        "Nh",
        "qvolg",
        "qvolh",
        # v0.16 additive aerosol-aware Thompson (mp=28) aerosol numbers
        # (FP32_GATED, same class as Nc/Nn).
        "nwfa",
        "nifa",
        "tracers",
        "tracer_bdy",
        "tracer_budget",
    }
    # cam_fatal is the WRF CAM-MG INTEGER error flag (all other CAM-MG state is WRF REAL).
    integer_static = {"lu_index", "epsilon_pbl_step", "cam_fatal"}
    # B39: WRF-REAL surface-layer carry, float32 in every configuration (not gated).
    mynn_history = getattr(precision_contract, "MYNN_DIAGNOSTIC_LEAVES", ())
    assert mynn_history in ((), ("el_pbl", "maxmf", "maxwidth", "ztop_plume"))
    gwdo_history = getattr(precision_contract, "GWDO_DIAGNOSTIC_LEAVES", ())
    assert gwdo_history in ((), ("dtaux3d", "dtauy3d", "dusfcg", "dvsfcg"))
    real_locked = set(SURFACE_LAYER_CARRY_LEAVES) | set(mynn_history) | set(gwdo_history)
    real_locked.update(precision_contract.MP_RE_DIAGNOSTIC_LEAVES)
    real_locked.update(precision_contract.LES_TKE_LEAVES)
    real_locked.update(precision_contract.EXTRA_SCALAR_BOUNDARY_LEAVES)
    # The v0.4 schemes retain WRF REAL carries and an INTEGER call counter.
    # Later v0.4 groups (writer profiles rimi/exch_tke, NTU mp state, P3 ice diagnostics, CAM-MG state)
    # are WRF REAL state too; each group is registered here explicitly, so an unregistered new leaf
    # still falls into the FP64-locked class below and must be justified.
    for group in ("SMS_CARRY_LEAVES", "P3_STATE_LEAVES", "JENSEN_STATE_LEAVES", "EPSILON_PBL_LEAVES",
                  "FERRIER_CARRY_LEAVES", "GODDARD4ICE_CARRY_LEAVES",
                  "GODDARD4ICE_HISTORY_LEAVES", "THOMPSON_HAIL_SURFACE_LEAVES",
                  "WRITER_PROFILE_LEAVES", "NTU_STATE_LEAVES", "ICE_DIAGNOSTIC_LEAVES",
                  "MYNN_CLOUDPDF_LEAVES"):
        real_locked.update(getattr(precision_contract, group))
    real_locked.update(cammg_fields.STATE_FIELDS)
    real_locked.difference_update(integer_static)
    # The matrix also covers derived/legacy p/ph/mu/pgeop aliases.
    assert set(State.__slots__) <= set(PRECISION_MATRIX)
    locked = set(State.__slots__) - gated - integer_static - real_locked

    for field in real_locked:
        dtype, gate_required = PRECISION_MATRIX[field]
        assert dtype == jnp.float32
        assert gate_required is False

    for field in gated:
        dtype, gate_required = PRECISION_MATRIX[field]
        assert dtype == FP32_GATED
        assert gate_required is True

    for field in locked:
        dtype, gate_required = PRECISION_MATRIX[field]
        assert dtype == FP64, field
        assert dtype == jnp.float64
        assert gate_required is False

    for field in integer_static:
        dtype, gate_required = PRECISION_MATRIX[field]
        assert dtype == INT32
        assert dtype == jnp.int32
        assert gate_required is False
