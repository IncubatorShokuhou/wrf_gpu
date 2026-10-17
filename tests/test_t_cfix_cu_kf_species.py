"""t-cfix-cu F11 (KF): microphysics-species flags of the WRF KF caller.

WRF sets ``warm_rain=.true.`` for Kessler (module_physics_init.F:4479-4481) and passes
F_QR/F_QI/F_QS = moist-array membership to KF_eta_CPS (module_cu_kfeta.F:211-216),
which forwards them to KF_eta_PARA (melting/freezing branches) and writes
RQICUTEN/RQSCUTEN only if F_QI/F_QS, folding DQRDT into RQCCUTEN when .NOT.F_QR
(:493-526).  The port called the kernel with the mixed-phase defaults for every MP.

Oracle: the p0_4 KF_ETA_CPS column driver against UNMODIFIED pristine WRF 4.7.1
module_cu_kfeta.F with a flag mode from argv (proofs/t_cfix_cu/kf_species/oracle);
mode 0 reproduces the committed p0_4 savepoints byte-for-byte.  Tolerances are the
predeclared p0_4 gate.  CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "proofs" / "t_cfix_cu" / "kf_species" / "savepoints"
P04 = ROOT / "proofs" / "p0_4" / "savepoints"
CASES = (1, 2, 3, 4)
FIELDS = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQRCUTEN", "RQICUTEN", "RQSCUTEN")
TEND_REL, TEND_ABS, RAIN_REL, RAIN_ABS = 2.0e-3, 1.0e-9, 3.0e-3, 1.0e-4  # p0_4 predeclared
# mode -> (warm_rain, F_QR, F_QI, F_QS): 1 = Kessler (mp1), 2 = no-ice with rain (WSM3 mp3)
MODES = {1: (True, True, False, False), 2: (False, True, False, False)}


def _load(case, mode):
    return json.loads((SAVE / f"kf_case_{case}_m{mode}.json").read_text())


def _port(case, flags, native, monkeypatch):
    from gpuwrf.coupling.scan_adapters import kf_cps_species_gate
    from gpuwrf.physics import cumulus_kf as J

    monkeypatch.setenv("GPUWRF_KF_COLUMN_FP32", "1" if native else "0")
    if native:  # Pallas kernel on CPU: interpret mode, same kernel body
        from gpuwrf.kernels import phys_kf_column as K
        if not getattr(K.kf_column, "_interpret", False):
            kernel = K.kf_column

            def interpreted(*a, **kw):
                return kernel(*a, **kw, interpret=True)

            interpreted._interpret = True
            monkeypatch.setattr(K, "kf_column", interpreted)
    jax.clear_caches()
    warm_rain, f_qr, f_qi, f_qs = flags
    d = _load(case, 0)  # identical inputs in every mode
    s, c = d["scalars"], d["columns"]
    dtype = jnp.float32 if native else jnp.float64
    a = lambda n: jnp.asarray(c[n], dtype)  # noqa: E731
    out = J.kf_eta_para(a("T"), a("QV"), a("P"), a("DZ"), a("RHO"), a("W0AVG"), a("U"), a("V"),
                        s["DT"], s["DX"], s["KX"], warm_rain, f_qi, f_qs)
    out = {k: np.asarray(v, np.float64) for k, v in out.items()}
    out["RQCCUTEN"], out["RQRCUTEN"], out["RQICUTEN"], out["RQSCUTEN"] = (
        np.asarray(x, np.float64) for x in kf_cps_species_gate(
            out["RQCCUTEN"], out["RQRCUTEN"], out["RQICUTEN"], out["RQSCUTEN"],
            f_qr=f_qr, f_qi=f_qi, f_qs=f_qs))
    return out


def _failures(out, ref):
    s, c = ref["scalars"], ref["columns"]
    bad = []
    for f in FIELDS:
        b = np.asarray(c[f], np.float64)
        err = float(np.max(np.abs(out[f] - b)))
        if err > TEND_ABS and err / max(float(np.max(np.abs(b))), TEND_ABS) > TEND_REL:
            bad.append((f, err))
    if abs(float(out["RAINCV"]) - s["RAINCV"]) > max(RAIN_REL * abs(s["RAINCV"]), RAIN_ABS):
        bad.append(("RAINCV", float(out["RAINCV"]), s["RAINCV"]))
    if int(out["ISHALL"]) != int(round(s["SHALL"])):
        bad.append(("ISHALL",))
    return bad


def test_flag_oracle_reproduces_committed_mixed_phase_savepoints():
    """E83: mode 0 of the argv-flag driver equals the original p0_4 driver output."""
    for c in CASES:
        new, old = _load(c, 0), json.loads((P04 / f"kf_case_{c}.json").read_text())
        assert new["columns"] == old["columns"]
        assert {k: v for k, v in new["scalars"].items() if k != "FLAGMODE"} == old["scalars"]


def test_flag_modes_change_wrf_output():
    """E154 census: the Kessler/no-ice flags move WRF's own tendencies (heating, qi -> qc)."""
    for mode in MODES:
        moved = [c for c in CASES if any(
            _load(c, mode)["columns"][f] != _load(c, 0)["columns"][f] for f in FIELDS)]
        assert len(moved) >= 3, (mode, moved)
        assert all(not np.any(_load(c, mode)["columns"]["RQICUTEN"]) for c in CASES)
    th = lambda c, m: np.asarray(_load(c, m)["columns"]["RTHCUTEN"])  # noqa: E731
    assert np.max(np.abs(th(1, 1) - th(1, 2))) > 1e-6  # warm_rain itself acts (freezing heat)


@pytest.mark.parametrize("native", (False, True), ids=("fp64", "native_fp32"))
@pytest.mark.parametrize("mode", sorted(MODES))
def test_port_matches_pristine_kf_with_wrf_flags(mode, native, monkeypatch):
    for c in CASES:
        assert _failures(_port(c, MODES[mode], native, monkeypatch), _load(c, mode)) == [], (mode, c)


@pytest.mark.parametrize("mode", sorted(MODES))
def test_mixed_phase_profile_fails_the_kessler_oracles(mode, monkeypatch):
    """Deletion sensitivity: the old hard-wired mixed-phase call is caught."""
    bad = [c for c in CASES if _failures(_port(c, (False, True, True, True), True, monkeypatch), _load(c, mode))]
    assert len(bad) >= 2


@pytest.mark.parametrize("mp, expected", ((1, (True, False, False)), (3, (False, False, False)),
                                          (8, (False, True, True))))
def test_operational_kf_cadence_hands_wrf_flags(mp, expected, monkeypatch):
    """Caller path: _kf_cadence_step -> kf_adapter -> step_kf_column carries WRF's flags."""
    import importlib.util

    from gpuwrf.coupling import scan_adapters as A
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_state import initial_operational_carry

    spec = importlib.util.spec_from_file_location("kf_fixture", ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    grid = f._grid(ny=3, nx=4, nz=8)
    state = f._state(grid)
    seen = []
    real = A.step_kf_column

    def spy(*args, **kw):
        seen.append((kw.get("warm_rain", False), kw.get("f_qi", True), kw.get("f_qs", True)))
        return real(*args, **kw)

    monkeypatch.setattr(A, "step_kf_column", spy)
    rates = tuple(jnp.zeros_like(state.theta) for _ in range(6)) + (jnp.zeros_like(state.t_skin),)
    carry = initial_operational_carry(state).replace(
        cumulus_carry=(jnp.zeros_like(state.theta), jnp.full_like(state.t_skin, -100.0)),
        cumulus_tendencies=rates)
    nml = SimpleNamespace(dt_s=54.0, cumulus_cadence_steps=1, cudt_minutes=0.0, grid=grid, mp_physics=mp)
    op._kf_cadence_step(state, carry, nml, jnp.asarray(1))
    assert seen and set(seen) == {expected}
