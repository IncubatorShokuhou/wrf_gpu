"""Pristine Goddard deposition at the real c64 saturation threshold."""
from pathlib import Path
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ('TH_IN','QV_IN','QC_IN','QR_IN','QI_IN','QS_IN','QG_IN','QH_IN',
         'DEN','PII','P','DELZ','Z','W')
OUTPUT = dict(theta='TH_OUT', qv='QV_OUT', qc='QC_OUT', qr='QR_OUT',
              qi='QI_OUT', qs='QS_OUT', qg='QG_OUT', qh='QH_OUT')
OUTPUT.update({'re_'+k+'_gsfc':'RE_'+k.upper() for k in
               ('cloud','rain','ice','snow','graupel','hail')})
OUTPUT.update({k:k.upper() for k in
               ['phys'+x for x in 'cedsmf']+['acphys'+x for x in 'cedsmf']+
               ['preci3d','precs3d','precg3d','prech3d','precr3d']})


def case():
    path = ROOT/'proofs/v040/mpmisc/goddard4ice/probe.py'
    spec = importlib.util.spec_from_file_location('tcn_goddard_literal', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = module.literal()
    data = json.loads((ROOT/'proofs/t_c02nan/savepoints/goddard_c64_ssi.json').read_text())
    args = [np.asarray(data['columns'][key], np.float32) for key in INPUT]
    return fn, args, data


def test_saturated_deposition_matches_pristine_real_column():
    fn, args, data = case()
    with np.errstate(all='ignore'):
        cfg = data['caller']
        out = fn(*args, *(np.float32(cfg[k]) for k in ('dt','dx','xland','topo')), cfg['itimestep'])
    for name, original in OUTPUT.items():
        actual = np.asarray(out[name], np.float32)
        expected = np.asarray(data['columns'][original], np.float32)
        assert np.isfinite(actual).all(), name
        np.testing.assert_array_equal(actual, expected, err_msg=name)


def test_quotient_error_exposes_original_zero_over_zero_branch():
    fn, args, data = case()
    env = fn.__globals__['saticel_s'].__globals__
    def approximate(numerator, denominator):
        quotient = np.asarray(numerator / denominator, np.float32)
        error = np.nextafter(np.nextafter(quotient, np.float32(np.inf)), np.float32(np.inf))
        return np.where(np.arange(44)==26, error, quotient)
    # Numerical-error control only. This is not a CPU deletion of the CUDA
    # instruction, whose actual native comparison remains a GPU gate.
    env['_saturation_ratio'] = approximate
    with np.errstate(all='ignore'):
        cfg = data['caller']
        out = fn(*args, *(np.float32(cfg[k]) for k in ('dt','dx','xland','topo')), cfg['itimestep'])
    for name in ('theta','qv','qs','qg','qh','acphysd'):
        assert not np.isfinite(np.asarray(out[name])[26]), name
        assert np.isfinite(np.asarray(data['columns'][OUTPUT[name]], np.float32)).all()
    assert np.isfinite(np.asarray(out['qi'])).all()
