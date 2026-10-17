"""Print-only WRF solver-entry probe, with a paired numeric identity check.

Only solve_interface's diagnostic statements differ between the executables.
All model objects/modules/libraries are read from the retained pristine build.
The private executables relocate the two legacy dump paths (E104).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
CASE = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c49')
FC = Path('<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = '<USER_HOME>/miniconda3/envs/wrfbuild/lib:' + env.get('LD_LIBRARY_PATH', '')
    flags = ['-O2', '-ftree-vectorize', '-funroll-loops', '-w', '-ffree-form',
             '-ffree-line-length-none', '-fconvert=big-endian', '-frecord-marker=4',
             '-fallow-argument-mismatch', '-fallow-invalid-boz']
    includes = [f'-I{WRF / p}' for p in ('main', 'frame', 'share', 'phys', 'external/esmf_time_f90')]
    libraries = [str(WRF / p) for p in (
        'main/libwrflib.a', 'external/fftpack/fftpack5/libfftpack.a',
        'external/io_grib1/libio_grib1.a', 'external/io_grib_share/libio_grib_share.a',
        'external/io_int/libwrfio_int.a')]
    libraries += [f'-L{WRF}/external/esmf_time_f90', '-lesmf_time',
                  str(WRF / 'frame/module_internal_header_util.o'), str(WRF / 'frame/pack_utils.o'),
                  f'-L{WRF}/external/io_netcdf', '-lwrfio_nf',
                  '-L<USER_HOME>/miniconda3/envs/wrfbuild/lib', '-lnetcdff', '-lnetcdf']
    source = (WRF / 'share/solve_interface.f90').read_text()
    seam = '   CALL model_to_grid_config_rec ( grid%id , model_config_rec , config_flags )'
    assert source.count(seam) == 1
    diagnostic = '''
   IF (grid%itimestep .LE. 2) THEN
      WRITE(*,*) 'TD_DFI_CFG', grid%dfi_stage, grid%itimestep, &
          grid%diff_opt, grid%km_opt, config_flags%diff_opt, config_flags%km_opt
   ENDIF
'''
    manifest = {'wrf_source': str(WRF), 'case': str(CASE), 'cache': 'NONE',
                'compiler': str(FC), 'compiler_version': subprocess.check_output([FC, '--version'], text=True).splitlines()[0],
                'source_sha256': sha(WRF / 'share/solve_interface.f90'),
                'original_binary_sha256': sha(WRF / 'main/wrf.exe'),
                'original_case_inputs': {n: sha(CASE / 'input' / n) for n in ('namelist.input', 'wrfinput_d01', 'wrfbdy_d01')},
                'commands': [], 'executables': {}}
    for arm in ('control', 'dump'):
        build = root / arm
        build.mkdir()
        text = source if arm == 'control' else source.replace(seam, seam + diagnostic)
        (build / 'solve_interface.f90').write_text(text)
        compile_cmd = [str(FC), *flags, *includes, '-c', 'solve_interface.f90', '-o', 'solve_interface.o']
        link_cmd = [str(FC), *flags, '-o', 'wrf.exe', str(WRF / 'main/wrf.o'),
                    str(WRF / 'main/module_wrf_top.o'), 'solve_interface.o', *libraries]
        for cmd in (compile_cmd, link_cmd):
            manifest['commands'].append(cmd)
            with (build / ('compile.log' if cmd == compile_cmd else 'link.log')).open('w') as out:
                subprocess.run(cmd, cwd=build, env=env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=120)
        exe = build / 'wrf.exe'
        data = exe.read_bytes()
        replacements = []
        for old_name, short_name in (('em_quarter_ss_center_dump.txt', 'd0'), ('em_grav2d_x_touchdown_dump.txt', 'd1')):
            old = f'<DATA_ROOT>/wrf_gpu2/wrf_truth/{old_name}'.encode()
            new = str(build / short_name).encode()
            assert len(new) <= len(old), (new, old)
            assert data.count(old) == 1
            replacements.append({'offset': data.index(old), 'old': old.decode(), 'new': new.decode()})
            data = data.replace(old, new.ljust(len(old), b' '))
        exe.write_bytes(data)
        manifest['executables'][arm] = {'sha256': sha(exe), 'wrapper_sha256': sha(build / 'solve_interface.f90'),
                                       'private_dump_paths': replacements}
        run = build / 'run'
        run.mkdir()
        # Link existing tables read-only; never link reference outputs or diagnostic paths.
        for entry in (CASE / 'cpu').iterdir():
            if entry.is_file() and not (entry.name.startswith(('wrf', 'rsl.', 'namelist', 'log', 'd0', 'd1')) or entry.suffix == '.log'):
                (run / entry.name).symlink_to(entry.resolve())
        for name in ('wrfinput_d01', 'wrfbdy_d01'):
            (run / name).symlink_to((CASE / 'input' / name).resolve())
        nml = (CASE / 'input/namelist.input').read_text()
        edits = dict(run_days=0, run_hours=0, run_minutes=10, run_seconds=0,
                     end_hour=0, end_minute=10, end_second=0, history_interval=0)
        for key, value in edits.items():
            nml, count = re.subn(rf'(?m)^(\s*){key}\s*=.*$', rf'\g<1>{key} = {value},', nml)
            assert count == 1, key
        nml = nml.replace('&time_control', '&time_control\n history_interval_s = 18,', 1)
        (run / 'namelist.input').write_text(nml)
        with (build / 'run.log').open('w') as out:
            subprocess.run([str(exe)], cwd=run, env=env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=300)
        assert 'SUCCESS COMPLETE WRF' in (build / 'run.log').read_text()
    files = sorted((root / 'control/run').glob('wrfout_d01*'))
    assert len(files) >= 3
    compared = []
    for first in files:
        second = root / 'dump/run' / first.name
        with Dataset(first) as a, Dataset(second) as b:
            assert set(a.variables) == set(b.variables)
            for name in a.variables:
                x, y = np.asarray(a[name][:]), np.asarray(b[name][:])
                assert x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes(), (first.name, name)
                if np.issubdtype(x.dtype, np.number):
                    assert np.isfinite(x).all(), (first.name, name)
            compared.append({'file': first.name, 'variables_byte_identical': len(a.variables),
                             'DIFF_OPT': int(a.DIFF_OPT), 'KM_OPT': int(a.KM_OPT)})
    rows = [[int(x) for x in line.split('TD_DFI_CFG')[1].split()]
            for line in (root / 'dump/run.log').read_text().splitlines() if 'TD_DFI_CFG' in line]
    assert rows and len({row[0] for row in rows}) == 2, rows
    assert all(row[2:] == [0, 1, 0, 1] for row in rows), rows
    # Original c49 full forecast: independent all-frame evidence, reference left untouched.
    headers = []
    for path in sorted((CASE / 'cpu').glob('wrfout_d01*')):
        with Dataset(path) as ds:
            headers.append({'path': str(path), 'DIFF_OPT': int(ds.DIFF_OPT), 'KM_OPT': int(ds.KM_OPT)})
    assert len(headers) == 24 and all((r['DIFF_OPT'], r['KM_OPT']) == (0, 1) for r in headers)
    manifest.update(verdict='PASS', diagnostic_rows=rows,
                    diagnostic_columns=['dfi_stage', 'itimestep', 'grid_diff_opt', 'grid_km_opt', 'solver_diff_opt', 'solver_km_opt'],
                    paired_output_byte_gate=compared, original_c49_headers=headers,
                    scope='CPU original WRF solver configuration and dump-only byte identity; no GPU or performance claim')
    (root / 'receipt.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'verdict': manifest['verdict'], 'rows': rows, 'paired_files': len(compared), 'original_frames': len(headers)}))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
