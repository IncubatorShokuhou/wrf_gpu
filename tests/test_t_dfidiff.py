"""DFL's real host prepass binds the diffusion used by the forecast loader.

Numerical integration is stubbed; this tests orchestration, not WRF fidelity.
The independent original-WRF solver-entry proof is proofs/t_dfidiff/oracle.py.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('dfi_diffusion', [None, (1, 4)])
def test_dfl_prepass_forecast_uses_restored_diffusion(tmp_path, monkeypatch, dfi_diffusion):
    from gpuwrf.dfi import driver, driver12
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.runtime import operational_mode as om

    @dataclass
    class State:
        qv: object
        w: object
        def replace(self, **edits):
            return replace(self, **edits)

    @dataclass
    class Carry:
        state: object
        def replace(self, **edits):
            return replace(self, **edits)

    @dataclass
    class Config:
        input_dir: Path
        output_dir: Path
        proof_dir: Path
        scratch_dir: Path | None = None
        max_dom: int = 1
        hours: float = 24
        resume_checkpoint: Path | None = None

    src = tmp_path / 'input'
    src.mkdir()
    text = (Path(__file__).parent / 'fixtures/t_dfidiff_c49.namelist').read_text()
    if dfi_diffusion is not None:
        text = text.replace('&dynamics', '&dynamics\n diff_opt_dfi = 1,\n km_opt_dfi = 4,', 1)
    (src / 'namelist.input').write_text(text)
    with Dataset(src / 'wrfinput_d01', 'w') as ds:
        ds.createDimension('Time', 1)
        ds.createDimension('DateStrLen', 19)
        ds.createVariable('Times', 'S1', ('Time', 'DateStrLen'))[0] = np.array(list('2023-01-15_00:00:00'), 'S1')
        for name in ('bottom_top', 'south_north', 'west_east'):
            ds.createDimension(name, 1)
        ds.createVariable('QVAPOR', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east'))[:] = .001
    (src / 'wrfbdy_d01').write_bytes(b'host-only boundary fixture')
    state = State(qv=np.ones((1, 1, 1), np.float32), w=np.zeros((2, 1, 1), np.float32))
    carry = Carry(state)
    loaded = []

    def load(config, names):
        assert names == ('d01',)
        run = Gen2Run(config.input_dir)
        # Exact native loader group/domain resolver, including the original 1/4 input.
        diff, km = [pipeline._domain_int(run, 'dynamics', key, 'd01', 0)
                    for key in ('diff_opt', 'km_opt')]
        loaded.append((diff, km))
        nl = SimpleNamespace(dt_s=18, diff_opt=diff, km_opt=km,
                             boundary_config=SimpleNamespace(update_cadence_s=10800))
        return None, {'d01': SimpleNamespace(namelist=nl)}, None, None, None, {'d01': carry}

    monkeypatch.setattr(pipeline, '_load_domains', load)
    monkeypatch.setattr(driver, '_land', lambda carry: None)
    monkeypatch.setattr(driver, '_shifted_boundary', lambda *args, **kwargs: {})
    monkeypatch.setattr(driver, '_prognostic_fields', lambda *args, **kwargs: {})
    monkeypatch.setattr(driver, '_stage_loop', lambda *args, **kwargs: (carry, {'w': state.w}, None))
    monkeypatch.setattr(om, '_commit_to_operational_device', lambda carry: carry)
    monkeypatch.setattr(driver, '_batch_size', lambda: 1)
    # Boundary rebasing is independently tested in test_s1_dfi12; retain host routing here.
    monkeypatch.setattr(driver12, 'rebase_dfl_boundary', lambda *args, **kwargs: 540)
    cfg = Config(src, tmp_path / 'out', tmp_path / 'proof')
    forecast, report = driver.apply_dfi_prepass(cfg)
    load(forecast, ('d01',))
    expected = dfi_diffusion or (0, 1)
    assert loaded == [(1, 4), expected, expected], 'DFL forward/forecast did not restore WRF diffusion'
    assert report['dfl_diffusion'] == dict(zip(('diff_opt', 'km_opt'), expected))
    assert forecast.hours == 23.85
    # Resume must reuse the persisted post-DFI namelist, not the original 1/4 input.
    resumed, resumed_report = driver.apply_dfi_prepass(replace(cfg, resume_checkpoint=tmp_path / 'checkpoint'))
    assert resumed_report is None
    load(resumed, ('d01',))
    assert loaded[-1] == expected and resumed.input_dir == forecast.input_dir
    assert (src / 'namelist.input').read_text() == text
