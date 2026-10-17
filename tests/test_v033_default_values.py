"""Independent literal contract for the v0.3.3 FINAL RC selection (manager 2026-10-06T07:12:53Z; QNI 08:27:52Z)."""
import os
import pytest
from gpuwrf import _fast_defaults as defaults

EXPECTED = {
    'GPUWRF_RRTMG_MAXRAND': '1',
    'GPUWRF_RRTMG_MP_RE': '1',
    'GPUWRF_MYNN_SGS_MIXING_RATIO': '1',
    'GPUWRF_MYNN_SCALE_AWARE': '1',
    'GPUWRF_W_SURFACE_RESET': '1',
    'GPUWRF_MYNN_SFC_WSPD': '1',
    'GPUWRF_SPEC_W_WORK_COPY': '1',
    'GPUWRF_MYNN_FLTV_WRF': '1',
    'GPUWRF_MYNN_DHEAT': '1',
    'GPUWRF_MYNN_PSIQ_FLUX_WRF': '1',
    'GPUWRF_W_DAMP_STAGE': '1',
    'GPUWRF_ACOUSTIC_NO_MU_FLOOR': '1',
    'GPUWRF_NOAHMP_JULIAN_ADVANCE': '1',
    'GPUWRF_NEST_O3_FROM_PARENT': '1',
    'GPUWRF_ROOT_SCALAR_BDY_RK1': '1',
    'GPUWRF_THOMPSON_MIXED_PHASE_WRF': '1',
    'GPUWRF_MYNN_DMP_KTOP_BOUND': '1',
    'GPUWRF_MYNN_ELB_MF': '1',
    'GPUWRF_MYNN_PHY_EXNER': '1',
    'GPUWRF_MYNN_PSIG_CLAMP': '1',
    'GPUWRF_MYNN_QNI_MIXING': '1',
    # GPUWRF_MYNN_PLUME_CLOUD_BASE (BP91) was EXCLUDED at the v0.3.3 RC (manager 2026-10-06T07:12:53Z)
    # under an open night-sea Sc candidate-phase fallback (CANDIDATE_HISTORY F0x). The shipped release
    # RESTORES WRF plume cloud-base geometry (docs/release/V0.3.3.md: "the release restores WRF plume
    # cloud-base geometry"; the smallest sufficient matched set was plume geometry + Sm/Sh floors +
    # condensation) and registers the flag ON with the closure set (6e38cbf17 v0.3.3 release DELTA;
    # GATES_FROZEN_FINAL33V 18-key DELTA; MYNNSET72 72 h guard PASS manager 01:03:20Z). Pin tracks the
    # CURRENT registered selection; the exclusion is resolved.
    'GPUWRF_MYNN_PLUME_CLOUD_BASE': '1',
}
# No open-issue default remains excluded from the shipped v0.3.3 registry: the sole historical exclusion
# (GPUWRF_MYNN_PLUME_CLOUD_BASE) is resolved and is now a registered default in EXPECTED above.
EXCLUDED: tuple[str, ...] = ()


@pytest.fixture(autouse=True)
def _restore_release_default_environment():
    # apply_fast_path_defaults writes every unset release key, beyond EXPECTED.
    # Preserve the caller's full flag set and status for the following tests.
    keys = tuple(defaults.FAST_PATH_DEFAULTS) + ('GPUWRF_FAST_DEFAULTS',)
    before = {key: os.environ.get(key) for key in keys}
    status = dict(defaults.FAST_DEFAULTS_STATUS)
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        defaults.FAST_DEFAULTS_STATUS.clear()
        defaults.FAST_DEFAULTS_STATUS.update(status)


def test_v033_default_values_are_the_registered_selections():
    assert {key: defaults.FAST_PATH_DEFAULTS.get(key) for key in EXPECTED} == EXPECTED


def test_v033_excluded_flags_are_not_defaults(monkeypatch):
    # Fail-closed companion: any flag listed in EXCLUDED must stay entirely absent from the defaults
    # registry and must never be force-set by apply_fast_path_defaults. (EXCLUDED is empty now that the
    # sole historical exclusion, GPUWRF_MYNN_PLUME_CLOUD_BASE, is resolved and registered in EXPECTED.)
    assert not set(EXCLUDED) & set(defaults.FAST_PATH_DEFAULTS)
    for key in EXCLUDED:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('GPUWRF_FAST_DEFAULTS', '1')
    before = set(os.environ)
    defaults.apply_fast_path_defaults()
    assert all(key not in os.environ for key in EXCLUDED)
    # Fail-closed: apply writes only registered release keys, never a foreign/excluded one.
    written = set(os.environ) - before
    assert written <= set(defaults.FAST_PATH_DEFAULTS)


@pytest.mark.parametrize('override', list(EXPECTED))
def test_v033_defaults_keep_each_explicit_optout(monkeypatch, override):
    for key in EXPECTED:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('GPUWRF_FAST_DEFAULTS', '1')
    monkeypatch.setenv(override, '0')
    defaults.apply_fast_path_defaults()
    expected = dict(EXPECTED, **{override: '0'})
    assert {key: os.environ.get(key) for key in EXPECTED} == expected


def test_v033_master_optout_leaves_new_selections_unset(monkeypatch):
    for key in EXPECTED:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('GPUWRF_FAST_DEFAULTS', '0')
    defaults.apply_fast_path_defaults()
    assert all(key not in os.environ for key in EXPECTED)
