"""CAM-UW (bl_pbl_physics=9) operational wiring (v0.3.4).

The v023 F3 verdict (reference-only, scaffold RED) compared against a defective oracle (no
WRF esinti -> zero saturation table). The faithful r8 port is parity-gated in
tests/test_v034_camuw_oracle_parity.py; these tests lock the operational wiring:
namelist acceptance, the surface-flux pairing rule, routing through the PBL slot with the
OperationalCarry.camuw_pbl (WRF KVM3D/KVH3D/TAURES) carry, and the PBLH hand-off.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.grid import (  # noqa: E402
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes  # noqa: E402
from gpuwrf.coupling.physics_dispatch import UnsupportedSchemeSelection  # noqa: E402
from gpuwrf.io.namelist_check import validate_namelist, validate_operational_namelist  # noqa: E402
from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    OperationalNamelist,
    _initial_carry_for_run,
    _physics_step_forcing,
    _resolve_operational_suite,
)

TIME_UTC = "2024-06-01T12:00:00Z"


def _grid(ny: int = 3, nx: int = 3, nz: int = 10) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="camuw-wire-test", sha256="camuw-wire-test", shape=(ny, nx), units="m",
        projection_transform="native-wrf-lambert", max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0,
                                 provenance="camuw-wire-flat")
    return GridSpec(projection, terrain_meta, vertical, bc, eta, jnp.zeros((ny, nx)), metrics=metrics)


def _state(grid: GridSpec) -> State:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    fields = {n: jnp.zeros(s, dtype=jnp.float64) for n, s in _state_field_shapes(grid).items()}
    p = jnp.broadcast_to(jnp.linspace(97000.0, 25000.0, nz)[:, None, None], (nz, ny, nx))
    ph = jnp.broadcast_to(jnp.linspace(0.0, 10000.0 * 9.81, nz + 1)[:, None, None], (nz + 1, ny, nx))
    zc = jnp.linspace(0.0, 10000.0, nz)[:, None, None]
    theta = jnp.broadcast_to(300.0 + 0.004 * zc - 0.6 * (zc < 900.0), (nz, ny, nx))  # unstable ML
    fields.update(
        theta=theta, p_total=p, ph_total=ph, mu_total=jnp.full((ny, nx), 90000.0),
        qv=jnp.full((nz, ny, nx), 6.0e-3), qc=jnp.full((nz, ny, nx), 1.0e-5),
        u=jnp.broadcast_to(3.0 + 0.002 * zc, (nz, ny, nx + 1)) * 1.0,
        v=jnp.full((nz, ny + 1, nx), 1.0),
        t_skin=jnp.full((ny, nx), 305.0), xland=jnp.full((ny, nx), 1.0),
        mavail=jnp.full((ny, nx), 0.5), roughness_m=jnp.full((ny, nx), 0.1),
        ustar=jnp.full((ny, nx), 0.3), lu_index=jnp.zeros((ny, nx), dtype=jnp.int32),
    )
    return State(**fields)


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)  # noqa: E731
    return Tendencies(z((nz, ny, nx + 1)), z((nz, ny + 1, nx)), z((nz + 1, ny, nx)), z((nz, ny, nx)),
                      z((nz, ny, nx)), z((nz, ny, nx)), z((nz + 1, ny, nx)), z((ny, nx)))


def _namelist(grid: GridSpec, **over) -> OperationalNamelist:
    base = OperationalNamelist.from_grid(grid, dt_s=10.0, tendencies=_cpu_tendencies(grid))
    return dataclasses.replace(base, time_utc=TIME_UTC, run_physics=True, **over)


def test_camuw_is_implemented_with_honest_qualification_note() -> None:
    support = classify_scheme("bl_pbl_physics", 9)
    assert support.status is SupportStatus.IMPLEMENTED
    assert "CPU-oracle-qualified" in support.reason and "GPU" in support.reason
    cfg = {"physics": {"bl_pbl_physics": [9], "sf_sfclay_physics": [5]}}
    validate_namelist(cfg)
    validate_operational_namelist(cfg)


def test_camuw_suite_requires_wrf_surface_flux_producer() -> None:
    grid = _grid()
    ok = _resolve_operational_suite(_namelist(grid, mp_physics=0, bl_pbl_physics=9, sf_sfclay_physics=5,
                                              cu_physics=0, use_noahmp=False))
    assert ok.pbl.option == 9 and ok.pbl.gpu_runnable is True
    with pytest.raises(UnsupportedSchemeSelection, match="CAM-UW"):
        _resolve_operational_suite(_namelist(grid, mp_physics=0, bl_pbl_physics=9, sf_sfclay_physics=1,
                                             cu_physics=0, use_noahmp=False))


def test_operational_step_routes_camuw_threads_carry_and_pblh() -> None:
    grid = _grid()
    state = _state(grid)
    nml = _namelist(grid, mp_physics=0, bl_pbl_physics=9, sf_sfclay_physics=5, cu_physics=0, use_noahmp=False)
    carry = _initial_carry_for_run(state, nml)
    assert carry.camuw_pbl is not None
    assert float(jnp.max(jnp.abs(carry.camuw_pbl.kvh3d))) == 0.0  # itimestep=1 == zero carry
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    after = forcing.state
    nxt = forcing.carry if hasattr(forcing, "carry") else forcing
    camuw = getattr(nxt, "camuw_pbl", None)
    assert camuw is not None
    for leaf in ("theta", "u", "v", "qv", "qc"):
        assert np.all(np.isfinite(np.asarray(getattr(after, leaf)))), leaf
    assert not np.allclose(np.asarray(after.theta), np.asarray(state.theta))
    assert not np.allclose(np.asarray(after.u), np.asarray(state.u))
    kvh = np.asarray(camuw.kvh3d)
    assert kvh.shape == (grid.nz + 1, grid.ny, grid.nx) and kvh.dtype == np.float32
    assert np.max(kvh) > 1.0  # convective mixed layer -> UW CL eddy diffusivity
    assert np.all(kvh[0] == 0.0) and np.all(kvh[-1] == 0.0)  # WRF: surface/top interfaces never set
    assert np.all(np.asarray(after.pblh) > 0.0)  # WRF PBLH2D hand-off (read by SFCLAY_mynn)
    assert np.any(np.asarray(camuw.tauresx2d) != 0.0)


@pytest.mark.parametrize("real_glue", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_camuw_pristine_rates_reach_rk_without_post_rk_increment(monkeypatch, real_glue, bounded):
    """Original CAM rates must follow update_phy_ten CAMUWPBLSCHEME, not Euler.

    WRF module_physics_addtendc.F:903–969 folds heat, A-grid momentum,
    QV/QC/QI/Ni into the RK sources. Feed actual pristine-driver outputs
    through the production column/adapter/forcing seam (the column's own
    equations have the separate 78-record original-WRF parity gate).
    """
    import gpuwrf.coupling.scan_adapters as adapters
    import gpuwrf.runtime.operational_mode as om

    oracle = Path(__file__).resolve().parents[1] / "proofs/v034/camuw_oracle/camuw_oracle_v034.json"
    records = json.loads(oracle.read_text())["records"]
    record = next(r for r in records if r["name"] == "ice_cloud_cold" and r["step"] == 1)
    original = record["out"]
    grid = _grid(nz=len(original["RUBLTEN"]))
    if bounded:
        grid = dataclasses.replace(grid, bc=dataclasses.replace(grid.bc, source="wrfbdy"))
    state = _state(grid)
    nml = _namelist(grid, mp_physics=8, bl_pbl_physics=9, sf_sfclay_physics=5,
                    cu_physics=0, use_noahmp=False, rad_rk_tendf=1)
    monkeypatch.setenv("GPUWRF_PHYS_TEND_RK_WRF", "1")
    monkeypatch.setenv("GPUWRF_CARRY_REAL_ALL", str(int(real_glue)))
    monkeypatch.setattr(om, "surface_adapter", lambda s, *a, **kw: s)
    monkeypatch.setattr(adapters, "camuw_column", lambda *a, **kw: {
        key: jnp.asarray(original[name], jnp.float32) for key, name in (
            ("rublten", "RUBLTEN"), ("rvblten", "RVBLTEN"), ("rthblten", "RTHBLTEN"),
            ("rqvblten", "RQVBLTEN"), ("rqcblten", "RQCBLTEN"), ("rqiblten", "RQIBLTEN"),
            ("rqniblten", "RQNIBLTEN"), ("kvm3d", "KVM"), ("kvh3d", "KVH"),
            ("tauresx2d", "TAURESX"), ("tauresy2d", "TAURESY"), ("tke_pbl", "TKE_PBL"),
            ("pblh", "PBLH"))})
    carry = _initial_carry_for_run(state, nml)
    carry = carry.replace(radiation_diagnostics=None)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    for name in ("theta", "u", "v", "qv", "qc", "qi", "Ni"):
        np.testing.assert_array_equal(getattr(forcing.state, name), getattr(carry.state, name),
                                      err_msg=f"{name}: CAM must not add a post-RK Euler increment")
    dry = forcing.dry_tendencies
    assert dry.ru_tendf is not None and dry.rv_tendf is not None
    assert dry.moist_tendf is not None
    assert set(dry.moist_tendf) == {"qv", "qc", "qi", "Ni"}
    dtype = np.float32 if real_glue else np.float64
    metrics = nml.metrics
    mass = (np.asarray(metrics.c1h, dtype)[:, None, None]
            * np.asarray(carry.state.mu_total, dtype)[None]
            + np.asarray(metrics.c2h, dtype)[:, None, None])
    for key, wrf in (("qv", "RQVBLTEN"), ("qc", "RQCBLTEN"), ("qi", "RQIBLTEN"), ("Ni", "RQNIBLTEN")):
        expected = mass * np.asarray(original[wrf], np.float32).astype(dtype)[:, None, None]
        # WRF add_a2a excludes the specified boundary ring.
        np.testing.assert_allclose(np.asarray(dry.moist_tendf[key])[:, 1:-1, 1:-1],
                                   expected[:, 1:-1, 1:-1], rtol=2e-7, atol=1e-20)
    assert om._acoustic_lateral_bc_flags(nml) == ((False, True, False) if bounded else (True, False, False))
    ni_source = np.asarray(dry.moist_tendf["Ni"])
    if bounded:
        # Pristine CAMUWPBLSCHEME -> add_a2a: the four outer edges receive
        # no number source. MYNN's direct scalar assignment is a different caller.
        np.testing.assert_array_equal(ni_source[:, (0, -1), :], 0)
        np.testing.assert_array_equal(ni_source[:, :, (0, -1)], 0)
    else:
        assert np.any(ni_source[:, 0, :] != 0)
    for key, wrf in (("ru_tendf", "RUBLTEN"), ("rv_tendf", "RVBLTEN")):
        expected = mass * np.asarray(original[wrf], np.float32).astype(dtype)[:, None, None]
        np.testing.assert_allclose(np.asarray(getattr(dry, key))[:, 1, 1], expected[:, 1, 1],
                                   rtol=2e-7, atol=1e-20)
    # Literal WRF conv_t_tendf_to_moist, independently evaluated with NumPy.
    rvrd = dtype(dtype(461.6) / dtype(287.0))
    qv = np.asarray(carry.state.qv, dtype)
    theta = np.asarray(carry.state.theta, dtype)
    factor = dtype(1) + rvrd * qv
    rth = np.asarray(original["RTHBLTEN"], np.float32).astype(dtype)[:, None, None]
    rqv = np.asarray(original["RQVBLTEN"], np.float32).astype(dtype)[:, None, None]
    heat = factor * (mass * rth)
    vapor_heat = rvrd * theta / factor * (mass * rqv)
    # _source_leaf_dry_tendencies stores this in the prognostic theta dtype,
    # which remains REAL32 even in the wide-glue arm.
    expected_heat = (heat + vapor_heat).astype(np.asarray(dry.t_tendf).dtype)
    # Eight rounded operations bound XLA contraction vs literal NumPy; use
    # the two input term scales because this cold column has cancellation.
    bound = 8 * np.finfo(dtype).eps * (np.abs(heat) + np.abs(vapor_heat)) + 1e-12
    assert np.all(np.abs(np.asarray(dry.t_tendf) - expected_heat) <= bound)
    assert all(np.any(np.asarray(original[k]) != 0) for k in
               ("RTHBLTEN", "RUBLTEN", "RVBLTEN", "RQVBLTEN", "RQCBLTEN", "RQIBLTEN", "RQNIBLTEN"))
    np.testing.assert_array_equal(forcing.state.pblh, np.full((grid.ny, grid.nx), original["PBLH"]))


def test_camuw_caller_passes_hydrostatic_pressure_but_nonhydrostatic_exner(monkeypatch):
    """E11: first_rk_step_part1.F:1130–1132, phy_prep:4940–4974."""
    import gpuwrf.coupling.scan_adapters as adapters
    grid = _grid(nz=4)
    state = _state(grid)

    def expose_inputs(*args, **kwargs):
        p, pw, t, pi = args[10], args[11], args[14], args[17]
        zero = jnp.zeros_like(p)
        return dict(rublten=zero, rvblten=zero, rthblten=p, rqvblten=pw[:-1], rqcblten=t,
            rqiblten=pi, rqniblten=zero, kvm3d=args[23], kvh3d=args[24],
            tauresx2d=args[25], tauresy2d=args[26], tke_pbl=jnp.zeros_like(pw), pblh=jnp.sum(p))

    monkeypatch.setattr(adapters, "camuw_column", expose_inputs)
    _, _, rates = adapters.camuw_pbl_adapter(state, 18., grid, return_tendencies=True)
    f = np.float32
    qtot = sum(np.asarray(getattr(state, key), f) for key in ("qv", "qc", "qr", "qi", "qs", "qg"))
    m = grid.metrics
    face = np.full((grid.ny, grid.nx), np.asarray(m.p_top, f), f)
    faces = [face]
    for k in range(grid.nz - 1, -1, -1):
        mass = np.asarray(m.c1h, f)[k] * np.asarray(state.mu_total, f) + np.asarray(m.c2h, f)[k]
        face = face - (f(1) + qtot[k]) * mass * np.asarray(m.dnw, f)[k]
        faces.append(face)
    pw = np.stack(faces[::-1])
    expected_p = f(.5) * (pw[:-1] + pw[1:])
    np.testing.assert_array_equal(rates["rthblten"], expected_p)
    np.testing.assert_array_equal(rates["rqvblten"], pw[:-1])
    assert not np.allclose(expected_p, np.asarray(state.p))  # deletion distinguishes the two views
    expected_pi = (np.asarray(state.p) / 1.e5) ** (287. / 1004.5)
    np.testing.assert_allclose(rates["rqiblten"], expected_pi.astype(f), rtol=2e-7)


@pytest.mark.parametrize("raw_rates", [False, True])
def test_camuw_forwards_original_cumulus_inputs_without_carry_change(monkeypatch, raw_rates):
    """Original CAM driver TP/KP feed cumulus, alongside either adapter mode."""
    import gpuwrf.coupling.scan_adapters as adapters
    from gpuwrf.physics.bl_camuw import CamUwCarry
    oracle = Path(__file__).resolve().parents[1] / "proofs/v034/camuw_oracle/camuw_oracle_v034.json"
    record = next(r for r in json.loads(oracle.read_text())["records"]
                  if r["name"] == "convective_clear" and r["step"] == 1)
    original = record["out"]
    grid = _grid(nz=len(original["RTHBLTEN"]))
    state = _state(grid)
    mapping = {"rublten": "RUBLTEN", "rvblten": "RVBLTEN", "rthblten": "RTHBLTEN",
        "rqvblten": "RQVBLTEN", "rqcblten": "RQCBLTEN", "rqiblten": "RQIBLTEN", "rqniblten": "RQNIBLTEN",
        "kvm3d": "KVM", "kvh3d": "KVH", "tauresx2d": "TAURESX", "tauresy2d": "TAURESY",
        "tke_pbl": "TKE_PBL", "pblh": "PBLH", "tpert": "TPERT", "kpbl": "KPBL"}
    monkeypatch.setattr(adapters, "camuw_column", lambda *a, **kw: {
        key: jnp.asarray(original[name], jnp.int32 if key == "kpbl" else jnp.float32)
        for key, name in mapping.items()})
    control = adapters.camuw_pbl_adapter(state, 10., grid, return_tendencies=raw_rates)
    result = adapters.camuw_pbl_adapter(state, 10., grid, return_tendencies=raw_rates,
                                        return_cumulus_inputs=True)
    assert len(result) == len(control) + 1
    for got, expected in zip(jax.tree.leaves(result[:-1]), jax.tree.leaves(control), strict=True):
        np.testing.assert_array_equal(got, expected)
    tp, kp = map(np.asarray, result[-1])
    assert tp.shape == kp.shape == (grid.ny, grid.nx)
    assert tp.dtype == np.float32 and kp.dtype == np.int32
    np.testing.assert_array_equal(tp, np.full(tp.shape, original["TPERT"], np.float32))
    np.testing.assert_array_equal(kp, np.full(kp.shape, original["KPBL"], np.int32))
    assert np.all(tp > 0) and np.all(kp >= 1)  # active original regime, one-based index
    assert result[1]._fields == CamUwCarry._fields
    assert CamUwCarry._fields == ("kvm3d", "kvh3d", "tauresx2d", "tauresy2d", "tke_pbl")
