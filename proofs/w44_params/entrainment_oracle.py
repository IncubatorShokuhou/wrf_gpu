"""Reuse the existing real-column original-WRF DMP caller at two entrainment values.

Only DMP flux/mean outputs are used: this older caller supplies an analytic Kh
profile for its later tendency demonstration, which is outside this gate.
Pristine files are read-only; each compiler output belongs to the supplied arm.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
DMP_FIELDS = ('edmf_s_aw', 'edmf_s_awqv', 'edmf_s_awqt', 'edmf_s_awqc',
              'edmf_edmf_a', 'edmf_edmf_w')


def build_and_run(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    phys = WRF / 'phys'
    source = (phys / 'module_bl_mynnedmf.F').read_text()
    marker = 'ENT(k,ip) = 0.33_kind_phys/'
    assert source.count(marker) == 1
    case = ROOT / 'proofs/mynn_edmf/column_d03_12z.json'
    emitter = ROOT / 'proofs/mynn_edmf/emit_flat.py'
    emit_source = emitter.read_text()
    for old, new in (("SRC = HERE / \"column_d03_12z.json\"", 'SRC = Path(' + repr(str(case)) + ')'),
                     ('DST = HERE / "fortran_oracle" / "column_d03_12z.flat"',
                      'DST = Path(' + repr(str(directory / 'column_d03_12z.flat')) + ')')):
        assert emit_source.count(old) == 1
        emit_source = emit_source.replace(old, new)
    # Reuse exactly the existing emitter's formatting and input-field mapping.
    exec(compile(emit_source, str(emitter), 'exec'), {'__file__': str(emitter)})
    wrapper = ROOT / 'proofs/mynn_edmf/fortran_oracle/oracle.f90'
    compiler = os.environ.get('FC', '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran')
    flags = ['-cpp', '-O2', '-ffree-line-length-none', '-ffp-contract=off']
    commands, results, source_hashes = [], {}, {}
    for label, factor in (('original', None), ('default', .33), ('changed', .363)):
        arm = directory / label
        arm.mkdir()
        if factor is None:
            core = phys / 'module_bl_mynnedmf.o'
        else:
            code = source.replace(marker, 'ENT(k,ip) = ' + str(factor) + '_kind_phys/')
            source_hashes[label] = hashlib.sha256(code.encode()).hexdigest()
            core = arm / 'module_bl_mynnedmf.F90'
            core.write_text(code)
        exe = arm / 'oracle'
        command = [compiler, *flags, '-I' + str(phys), '-J' + str(arm), str(core), str(wrapper),
                   str(phys / 'module_bl_mynnedmf_common.o'), str(WRF / 'share/module_model_constants.o'),
                   str(phys / 'ccpp_kind_types.o'), '-o', str(exe)]
        subprocess.run(command, cwd=arm, check=True, timeout=90, capture_output=True, text=True)
        commands.append(command)
        (arm / 'column_d03_12z.flat').symlink_to(directory / 'column_d03_12z.flat')
        subprocess.run([str(exe)], cwd=arm, check=True, timeout=30, capture_output=True, text=True)
        raw = dict(line.strip().split('=', 1) for line in (arm / 'oracle_out.txt').read_text().splitlines() if '=' in line)
        results[label] = {name: np.fromstring(raw[name], sep=',', dtype=np.float32) for name in DMP_FIELDS}
        assert all(value.size and np.isfinite(value).all() for value in results[label].values())
    assert source_hashes['default'] == hashlib.sha256(source.encode()).hexdigest()
    (directory / 'manifest.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'source_variants': source_hashes,
        'only_source_change': marker + ' -> ENT(k,ip) = 0.363_kind_phys/',
        'input_sha256': hashlib.sha256(case.read_bytes()).hexdigest(),
        'wrapper_sha256': hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        'original_object_sha256': hashlib.sha256((phys / 'module_bl_mynnedmf.o').read_bytes()).hexdigest(),
        'scope': 'Only DMP arrays; no downstream analytic-Kh tendency qualification',
        'fields': DMP_FIELDS, 'commands': commands,
    }, indent=2))
    return results
