"""O2 root lateral boundary for the extra moist/number species of mp=24/26/28/40 (lane o2-rootbdy).

Pristine solve_em.F on a SPECIFIED root (real cases: have_bcs_moist = have_bcs_scalar = .false., Registry default):
  * moist array members other than QV (QHAIL of WSM7/WDM7) -> flow_dep_bdy after every stage's rk_update_scalar
    (:2426-2438);
  * scalar array members (QNSNOW/QNGRAUPEL/QNCLOUD of Morrison-aero, QNCLOUD of Thompson-aero) -> flow_dep_bdy
    (:2993-3021, truth-table rows 6/8);
  * Thompson-aero QNWFA/QNIFA -> flow_dep_bdy whenever aer_init_opt == 0 (rows 3/4, whatever have_bcs_scalar);
    aer_init_opt > 0 (use_aero_icbc / use_rap_aero_icbc) is refused pre-JAX by the catalog.
The port routes these through root_scalar_rk1_split(flow_only=...) -> the per-stage + post-physics flow_dep_bdy of
GPUWRF_ROOT_SCALAR_BDY_RK1 (B43), the same path qc..qg/Ni/Nr already take.

Tests: (1) the static map covers exactly the transported extras of each code; (2) routing + fail-closed rules;
(3) one large step on REAL PROD d01 operands with the unchanged pristine routines (rk_update_scalar + flow_dep_bdy,
tests/v025/b_diff/pristine_boundary.py) vs the port helpers for qv + every extra; (4) the real Swiss production step
(CLI loader, mp edited in namelist.input) dispatches the per-stage flow_dep for the extras.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_B_DIFF = Path(__file__).resolve().parents[2] / "v025" / "b_diff"
sys.path[:0] = [str(_B_DIFF), str(_B_DIFF.parent / "b_core")]

from gpuwrf.coupling import boundary_apply as bd  # noqa: E402
from gpuwrf.runtime import operational_mode as op  # noqa: E402

CORE = ("qv", "qc", "qr", "qi", "qs", "qg")
EXTRAS = {24: ("qh",), 26: ("qh",), 28: ("nwfa", "nifa", "Nc"), 40: ("Ns", "Ng", "Nc")}
TRANSPORT_EXTRAS = {**EXTRAS, 26: ("qh", "Nc", "Nr", "Nn")}
CFG_FALSE = types.SimpleNamespace(have_bcs_moist=False, have_bcs_scalar=False)


def _nml(mp):
    return types.SimpleNamespace(mp_physics=mp, moist_adv_opt=1, scalar_adv_opt=1)


@pytest.mark.parametrize("mp", sorted(EXTRAS))
def test_flow_only_map_covers_every_transported_extra(mp):
    """Every species the root transports beyond the core moist set has a WRF boundary route (no silent omission)."""
    transported = set(op._advected_scalar_species(_nml(mp))) - set(CORE)
    assert set(op._root_flow_only(_nml(mp))["flow_only"]) == transported == set(TRANSPORT_EXTRAS[mp])


def test_thompson_aero_transports_droplet_number():
    """Registry thompsonaero scalar:qni,qnr,qnc,qnwfa,qnifa -- qnc rides WRF's scalar loop (O2: was untransported)."""
    assert "Nc" in op._advected_scalar_species(_nml(28))
    assert op._advected_scalar_species(_nml(8)) == CORE  # release Thompson unchanged


@pytest.mark.parametrize("mp", sorted(EXTRAS))
def test_split_routes_extras_to_flow_dep(mp):
    species = CORE + EXTRAS[mp] + (("Ni", "Nr") if mp in (28, 40) else ())
    relaxed, flow = bd.root_scalar_rk1_split(CFG_FALSE, species, **op._root_flow_only(_nml(mp)))
    assert relaxed == ("qv",)
    assert set(flow) == set(species) - {"qv"}
    with pytest.raises(NotImplementedError):  # without the route the extras stay unrepresented (pre-O2 behaviour)
        bd.root_scalar_rk1_split(CFG_FALSE, species)


def test_have_bcs_true_fails_closed_per_wrf_array():
    """A wrfbdy that claims extra records fails closed (no *_bdy leaves); qh keys on have_bcs_moist, numbers on
    have_bcs_scalar, QNWFA/QNIFA on aer_init_opt only (solve_em.F:2993-3012 rows 3/4)."""
    moist_true = types.SimpleNamespace(have_bcs_moist=True, have_bcs_scalar=False)
    scalar_true = types.SimpleNamespace(have_bcs_moist=False, have_bcs_scalar=True)
    with pytest.raises(NotImplementedError):
        bd.root_scalar_rk1_split(moist_true, ("qv", "qh"), **op._root_flow_only(_nml(24)))
    assert bd.root_scalar_rk1_split(scalar_true, ("qv", "qh"), **op._root_flow_only(_nml(24))) == (("qv",), ("qh",))
    for mp, name in ((40, "Ns"), (40, "Ng"), (40, "Nc"), (28, "Nc")):
        with pytest.raises(NotImplementedError):
            bd.root_scalar_rk1_split(scalar_true, ("qv", name), **op._root_flow_only(_nml(mp)))
    for name in ("nwfa", "nifa"):
        assert bd.root_scalar_rk1_split(scalar_true, ("qv", name), **op._root_flow_only(_nml(28))) == (("qv",), (name,))
        with pytest.raises(NotImplementedError):  # not a flow-only extra of this code -> unrepresented
            bd.root_scalar_rk1_split(CFG_FALSE, ("qv", name), **op._root_flow_only(_nml(40)))


def test_release_and_other_codes_keep_no_route():
    for mp in (8, 0):
        assert op._root_flow_only(_nml(mp)) == {}
    # Real OperationalNamelist always carries its declared microphysics default.
    default_mp = op.OperationalNamelist.__dataclass_fields__["mp_physics"].default
    assert op._root_flow_only(_nml(default_mp)) == {}
    assert set(op._root_flow_only(_nml(18))["flow_only"]) == {"qh", "qvolg", "qvolh"} | set(op._NSSL_NUMBER_SCALARS)


# --------------------------------------------------------------------------------------------------------------
# (3) one large step vs the unchanged pristine routines on REAL PROD d01 operands
# --------------------------------------------------------------------------------------------------------------

def _wrf_route(name):
    """Independent encoding of solve_em's branch for a specified root with have_bcs_* = .false., aer_init_opt = 0."""
    return "relax" if name == "qv" else "flow"


# Real-structured magnitudes (kg/kg, #/kg) so every extra exercises its own REAL rounding scale.
_SCALE = {"qh": 3e-4, "Ns": 2e5, "Ng": 4e3, "Nc": 1.5e8, "nwfa": 3e9, "nifa": 2e6}


@pytest.fixture(scope="module")
def prod_operands():
    import test_root_scalar_bdy_rk1 as b43
    if not (b43.PROD / "wrfbdy_d01").exists():
        pytest.skip(f"PROD inputs unavailable: {b43.PROD}")
    return b43._load_prod_operands()


@pytest.fixture(scope="module")
def pristine_lib(tmp_path_factory):
    import shutil
    fc = os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    if not shutil.which(fc):
        pytest.skip(f"gfortran for the pristine WRF oracle not found: {fc}")
    from pristine_boundary import build
    return build(tmp_path_factory.mktemp("pristine_o2"))


def _chain(op_, lib, mp, *, drop_flow=()):
    """qv + the extras of ``mp`` through one large step: (pristine finals, port finals)."""
    import test_root_scalar_bdy_rk1 as b43
    from pristine_boundary import Oracle
    F32, DT = b43.F32, b43.DT
    qv, mu_t = op_["qv"], op_["mu_t"]
    nz, ny, nx = qv.shape
    cfg = bd.BoundaryConfig(update_cadence_s=b43.CADENCE, have_bcs_moist=False, have_bcs_scalar=False)
    species = ("qv",) + EXTRAS[mp]
    relaxed, flow = bd.root_scalar_rk1_split(cfg, species, **op._root_flow_only(_nml(mp)))
    assert {n: ("relax" if n in relaxed else "flow") for n in species} == {n: _wrf_route(n) for n in species}
    pattern = (1.0 + np.sin(np.arange(nx) / 7.0))[None, None, :] * (1.0 + 0.5 * np.cos(np.arange(ny) / 5.0))[None, :, None]
    fields = {"qv": qv}
    for i, n in enumerate(EXTRAS[mp]):
        prof = (qv / qv.max()) ** (1 + i)
        fields[n] = (_SCALE[n] * prof * pattern).astype(F32)
    mass_t = op_["c1h"][:, None, None] * mu_t.astype(np.float64)[None] + op_["c2h"][:, None, None]
    D = {n: b43._shaped(10 + i, fields[n], mass_t, 2e-4) for i, n in enumerate(fields)}
    P = {n: b43._shaped(40 + i, fields[n], mass_t, 5e-4) for i, n in enumerate(fields)}
    kap = {n: 1e-3 * np.abs(fields[n]).max() * mass_t.mean() / DT / np.abs(b43._lap(mass_t * fields[n])).max()
           for n in fields}
    advect = lambda n, q: (kap[n] * b43._lap(mass_t * np.asarray(q, np.float64))).astype(F32)  # noqa: E731
    dmu = b43._lap(mu_t[None].astype(np.float64))[0]
    dmu = dmu / np.abs(dmu).max() * 30.0
    mu_s = [(mu_t + (s + 1) / 3.0 * dmu).astype(F32) for s in range(3)]
    ru_s = [(op_["u"] * (1 + 0.1 * s) - (0.3 if s == 1 else 0.0)).astype(F32) for s in range(3)]
    rv_s = [(op_["v"] * (1 - 0.1 * s) + (0.3 if s == 2 else 0.0)).astype(F32) for s in range(3)]
    dt_rk = (DT / 3.0, DT / 2.0, DT)
    mass3 = op_["c1h"][:, None, None] * mu_s[2].astype(np.float64)[None] + op_["c2h"][:, None, None]
    val, ten = op_["val_ten"]
    lead = DT
    wrf = {}
    for n in fields:
        o = Oracle(lib, (nz, ny, nx), dt=DT, dtbc=lead, dts=DT / 3.0)
        o.set("c1h", op_["c1h"]); o.set("c2h", op_["c2h"]); o.set("msftx", op_["msfty"]); o.set("msfty", op_["msfty"])
        o.set("mu", mu_t); o.set("mut", mu_t); o.set("scalar", fields[n])
        o.set("scalar_tend", (D[n].astype(np.float64) + P[n]).astype(F32))
        if _wrf_route(n) == "relax":
            o.boundary("scalar", val[0], ten[0]); o.run("relax_bdy_scalar"); o.run("spec_bdy_scalar")
        o.set("sc_tend", o.field("scalar_tend", (nz, ny, nx)))
        o.set("scalar_2", fields[n]); o.set("scalar_1", fields[n]); o.set("mu_old", mu_t)
        o.set("mu_base", np.zeros_like(mu_t)); o.set("h_tendency", np.zeros_like(qv)); o.set("z_tendency", np.zeros_like(qv))
        q = fields[n]
        for s in range(3):
            o.set("advect_tend", advect(n, q)); o.set("mu_new", mu_s[s]); o.s[2] = F32(dt_rk[s])
            o.run("rk_update_scalar", tag=s + 1)
            q = o.field("scalar_2", (nz, ny, nx))
            if _wrf_route(n) == "flow":
                o.set("field", q); o.set("ru", ru_s[s]); o.set("rv", rv_s[s]); o.run("flow_dep_bdy")
                q = o.field("field", (nz, ny, nx)); o.set("scalar_2", q)
        if _wrf_route(n) == "relax":
            o.set("field", q); o.set("mu", mu_s[2]); o.run("spec_bdy_final", tag=4)
            q = o.field("field", (nz, ny, nx))
        wrf[n] = q

    real = lambda a: jnp.asarray(a, jnp.float32)  # noqa: E731
    metrics = types.SimpleNamespace(c1h=real(op_["c1h"]), c2h=real(op_["c2h"]), msfty=real(op_["msfty"]))
    ref = b43._St({**{n: real(f) for n, f in fields.items()}, "mu_total": real(mu_t),
                   **{k: real(a) for k, a in op_["leaves"].items()}})
    sc = {n: real(D[n]) for n in fields}
    sc = op._root_sc_with_boundary(sc, bd.root_scalar_boundary_tendencies(ref, lead, metrics, DT, cfg, relaxed), 1)
    port_flow = tuple(n for n in flow if n not in drop_flow)
    state = ref
    for s in range(3):
        q_t = op._root_scalar_stage_tendencies(tuple(real(advect(n, np.asarray(getattr(state, n)))) for n in species),
                                               species, state, cfg, metrics.msfty, bounded=True, sc_tendencies=sc)
        state = op._apply_moisture_large_step(state.replace(mu_total=real(mu_s[s])), ref, q_tendencies=q_t,
                                              dt_rk=dt_rk[s], metrics=metrics, species=species)
        state = op._root_flow_dep(state, port_flow, real(ru_s[s]), real(rv_s[s]), cfg)
    state = state.replace(**{n: getattr(state, n) + real(DT * P[n] / mass3) for n in species})
    state = op._root_flow_dep(state, port_flow, real(ru_s[2]), real(rv_s[2]), cfg)
    state = bd.root_scalar_spec_final(state, lead, metrics, cfg, relaxed)
    return wrf, {n: np.asarray(getattr(state, n)) for n in species}


@pytest.mark.parametrize("mp", sorted(EXTRAS))
def test_root_extras_chain_matches_pristine_wrf(prod_operands, pristine_lib, mp):
    import test_root_scalar_bdy_rk1 as b43
    wrf, port = _chain(prod_operands, pristine_lib, mp)
    spec, relax, interior = b43._zones(wrf["qv"].shape)
    err = b43._ulp_of_max(port["qv"], wrf["qv"])
    assert err[:, spec].max() <= b43.SPEC_ULP_OF_MAX and err[:, relax].max() <= b43.RELAX_ULP_OF_MAX
    for n in EXTRAS[mp]:
        e = b43._ulp_of_max(port[n], wrf[n])
        noise = e[:, interior].max()
        assert e[:, spec | relax].max() <= b43.FLOW_NOISE_FACTOR * max(noise, 1.0), (n, e[:, spec | relax].max(), noise)


@pytest.mark.parametrize("mp", (24, 40))
def test_gate_sees_a_missing_flow_dep(prod_operands, pristine_lib, mp):
    """E39: dropping the per-stage/post-physics flow_dep of the extras is visible to the same gate."""
    import test_root_scalar_bdy_rk1 as b43
    wrf, port = _chain(prod_operands, pristine_lib, mp, drop_flow=EXTRAS[mp])
    spec, _relax, interior = b43._zones(wrf["qv"].shape)
    for n in EXTRAS[mp]:
        e = b43._ulp_of_max(port[n], wrf[n])
        assert e[:, spec].max() > 10 * b43.FLOW_NOISE_FACTOR * max(e[:, interior].max(), 1.0), n


# --------------------------------------------------------------------------------------------------------------
# (4) the real production step (Swiss RD11 single root through the CLI loader) dispatches the extras' flow_dep
# --------------------------------------------------------------------------------------------------------------

SWISS = Path(os.environ.get("O2_SWISS_CASE", "<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu"))


def _swiss_root(mp, tmp_path, monkeypatch):
    import re
    from jax.experimental import pallas as pl
    from gpuwrf.contracts import state as state_contract
    from gpuwrf.integration import nested_pipeline as pipeline
    if not (SWISS / "namelist.input").exists():
        pytest.skip(f"Swiss case unavailable: {SWISS}")
    monkeypatch.setattr(state_contract, "_gpu_device", lambda: jax.devices("cpu")[0])
    case = tmp_path / "case"
    case.mkdir()
    for f in SWISS.iterdir():
        if f.name != "namelist.input":
            (case / f.name).symlink_to(f)
    text = re.sub(r"^\s*mp_physics\s*=.*$", f" mp_physics = {mp},", (SWISS / "namelist.input").read_text(),
                  flags=re.M | re.I)
    (case / "namelist.input").write_text(text)
    real_call = pl.pallas_call

    def interpret_call(*a, **k):
        k["interpret"] = True
        k.pop("compiler_params", None)
        return real_call(*a, **k)

    cfg = pipeline.NestedPipelineConfig(case, tmp_path / "out", tmp_path / "proof", hours=1, max_dom=1)
    pl.pallas_call = interpret_call
    try:
        _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
    finally:
        pl.pallas_call = real_call
    return bundles["d01"].namelist, carries["d01"]


@pytest.mark.parametrize("mp", (24, 28, 40))
def test_swiss_production_step_dispatches_extra_flow_dep(monkeypatch, tmp_path, mp):
    monkeypatch.setenv("GPUWRF_ROOT_SCALAR_BDY_RK1", "1")
    nml, carry = _swiss_root(mp, tmp_path, monkeypatch)
    assert int(nml.mp_physics) == mp and op._root_scalar_bdy_rk1_active(nml)
    calls: list[tuple[str, ...]] = []
    real = op._root_flow_dep

    def spy(state, names, *a, **k):
        calls.append(tuple(names))
        return real(state, names, *a, **k)

    monkeypatch.setattr(op, "_root_flow_dep", spy)
    shapes = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carry)
    clock = op.build_clock_base(nml)
    try:
        jax.clear_caches()
        jax.eval_shape(lambda c: op._advance_chunk_fori(
            c, nml, jnp.asarray(1, jnp.int32), clock, n_steps=1, cadence=int(nml.radiation_cadence_steps)), shapes)
    finally:
        jax.clear_caches()
    assert jax.devices()[0].platform == "cpu"
    # E114: three RK stages + the post-physics repeat, each carrying every extra
    assert len(calls) == 4, calls
    for names in calls:
        assert set(EXTRAS[mp]) <= set(names) and "qv" not in names, names


# --------------------------------------------------------------------------------------------------------------
# (5) nested runs stay refused: the extras have no parent->child forcing (WRF bdy_interp 'f' species)
# --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mp", (18, 24, 26, 28, 40))
def test_nested_run_refused_pre_jax(mp):
    from gpuwrf.io.namelist_check import NotOperationallyWiredError, refuse_single_domain_mp_nested
    cfg = {"physics": {"mp_physics": [mp, mp, mp]}, "domains": {"max_dom": 3}}
    refuse_single_domain_mp_nested(cfg, 1)
    with pytest.raises(NotOperationallyWiredError, match="single domains only"):
        refuse_single_domain_mp_nested(cfg, 3)


def test_nested_release_mp_not_refused():
    from gpuwrf.io.namelist_check import refuse_single_domain_mp_nested
    refuse_single_domain_mp_nested({"physics": {"mp_physics": [8, 8]}}, 2)
    refuse_single_domain_mp_nested({"physics": {"mp_physics": [8, 8, 40]}}, 2)  # d03 not run


@pytest.mark.parametrize("mp", (24, 28, 40))
def test_catalog_runs_the_routed_codes(mp):
    from gpuwrf.io.namelist_check import validate_operational_namelist
    from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme
    support = classify_scheme("mp_physics", mp)
    assert support.status is SupportStatus.IMPLEMENTED
    assert "flow_dep_bdy" in support.reason and "qualification pending" in support.reason
    validate_operational_namelist({"physics": {"mp_physics": [mp], "cu_physics": [0]}})


def test_wdm7_runs_after_number_transport_fix():
    from gpuwrf.io.namelist_check import validate_operational_namelist
    validate_operational_namelist({"physics": {"mp_physics": [26], "cu_physics": [0]}})


@pytest.mark.parametrize("mp", (24, 28, 40))
def test_cli_refuses_nested_before_jax(tmp_path, capsys, mp):
    """Real ``gpuwrf run`` entry: --max-dom 2 with a single-domain-only mp fails closed (rc 2) at the CLI gate."""
    from gpuwrf.cli import main
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text(f"&physics\n mp_physics = {mp}, {mp},\n cu_physics = 0, 0,\n/\n")
    rc = main(["run", "--namelist", str(case / "namelist.input"), "--input-dir", str(case),
               "--output-dir", str(tmp_path / "out"), "--max-dom", "2", "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 2 and "single domains only" in err, (rc, err[-800:])


def test_wsm7_runs_in_the_fp64_island():
    """mp=24 reached the WSM6 _nislfv carry-dtype TypeError under the REAL32 carry once its qh route existed
    (Swiss probe); it must stay wrapped like WSM6 (tests/test_mp_fp64_island.py exercises every wrapped code)."""
    from gpuwrf.coupling import scan_adapters as sa
    assert 24 in sa.FP64_ISLAND_MP_CODES
    assert sa.MP_SCAN_ADAPTERS[24].__name__ == "fp64_island_wsm7_adapter"
    assert sa.MP_SCAN_ADAPTERS[24].__wrapped__ is sa.wsm7_adapter
