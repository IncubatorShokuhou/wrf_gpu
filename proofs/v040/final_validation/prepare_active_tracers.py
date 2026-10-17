"""Separate positive-tracer reference using the pristine real.exe initializer."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import numpy as np
from prepare import ROOT, PRISTINE, SLOTS, sha, write_json
from extra_cases import stage
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.namelist_parser import parse_namelist

assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert shutil.disk_usage('<DATA_ROOT>').free >= 120 * 1024**3
old = ROOT / 'c46'
case = ROOT / 'c90'
data = Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep/arms/c90')
assert not case.exists() and not data.exists()
(data / 'cpu').mkdir(parents=True)
(data / 'input').mkdir()
case.mkdir()
(case / 'cpu').symlink_to(data / 'cpu', target_is_directory=True)
groups = parse_namelist(old / 'input/namelist.input')
files = {p.name: p.read_text() for p in (old / 'input').iterdir()
         if p.is_file() and p.name not in ('namelist.input', 'wrfinput_d01', 'wrfbdy_d01')}
row = dict(id='c90', geography='Swiss', hours=24, domains=['d01'], extra={},
           options={k: groups['physics'][k] for k in SLOTS},
           gpu_eligible=False, reason_excluded='Pending active-tracer original reference and reuse proof',
           features=['K1-matched-original-real-tracer-template', 'positive-passive-tracer-inputs'])
row = stage(row, old / 'input', groups, files)
assert not row['admission_errors']
# Preserve the exact resolved K1 controls, including their original text bytes.
(case / 'input/namelist.input').write_bytes((old / 'input/namelist.input').read_bytes())
control = case / 'cpu/namelist.input'
assert control.is_symlink()
control.unlink()
control.write_bytes((case / 'input/namelist.input').read_bytes())
for p in list((data / 'cpu').iterdir()):
    if p.is_symlink() and (p.name == 'namelist.output' or p.name.startswith(('rsl.', 'wrfout_', 'wrfrst_'))):
        p.unlink()  # Owned links only; no foreign targets are written.
target = data / 'input/wrfinput_d01'
shutil.copyfile(old / 'input/wrfinput_d01', target)
link = case / 'input/wrfinput_d01'
assert link.is_symlink()
link.unlink()
link.symlink_to(target)
cpu_link = case / 'cpu/wrfinput_d01'
assert cpu_link.is_symlink()
cpu_link.unlink()
cpu_link.symlink_to(target)
assert not target.is_symlink() and target.resolve().is_relative_to(data)
with Dataset(target) as ds:
    nx, ny, nz = (len(ds.dimensions[n]) for n in ('west_east', 'south_north', 'bottom_top'))
    assert ds.getncattr('WEST-EAST_GRID_DIMENSION') == nx + 1
    assert ds.getncattr('SOUTH-NORTH_GRID_DIMENSION') == ny + 1
    assert not any(f'tr17_{i}' in ds.variables for i in range(1, 9))
source = PRISTINE / 'dyn_em/module_initialize_real.F'
text = source.read_text()
start = text.index('      IF (config_flags%tracer_opt .eq. 2) THEN')
end = text.index('\n      END IF', start) + len('\n      END IF')
body = text[start:end]
oracle = data / 'original_initializer'
oracle.mkdir()
program = f'''program real_tracer_initializer
implicit none
integer,parameter :: nx={nx},ny={ny},nz={nz}
integer,parameter :: P_tr17_1=1,P_tr17_2=2,P_tr17_3=3,P_tr17_4=4
integer :: i,j,u,ids,ide,jds,jde,its,ite,jts,jte
type config_type
integer :: tracer_opt
end type
type(config_type) :: config_flags
real(4) :: tracer(nx,nz,ny,8)
ids=1; ide=nx+1; jds=1; jde=ny+1
its=1; ite=nx; jts=1; jte=ny
config_flags%tracer_opt=2
tracer=0.
{body}
open(newunit=u,file='tr17_init.bin',access='stream',form='unformatted',status='replace')
write(u) tracer
close(u)
end program
'''
(oracle / 'initializer.f90').write_text(program)
subprocess.run(['<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran', '-O0', '-ffree-form', '-fno-fast-math', 'initializer.f90', '-o', 'initializer'],
               cwd=oracle, check=True, timeout=30)
subprocess.run(['./initializer'], cwd=oracle, check=True, timeout=10)
values = np.fromfile(oracle / 'tr17_init.bin', dtype=np.float32).reshape(8, ny, nz, nx).transpose(0, 2, 1, 3)
published = sorted((old / 'gpu').glob('wrfout_d01*'))[0]
with Dataset(published) as ds:
    for i in range(8):
        assert np.array_equal(values[i], ds[f'tr17_{i+1}'][0]), 'Original real initializer differs from paid GPU t0'
with Dataset(target, 'a') as dst, Dataset(sorted((old / 'cpu').glob('wrfout_d01*'))[0]) as template:
    for i in range(8):
        name = f'tr17_{i+1}'
        ref = template[name]
        v = dst.createVariable(name, 'f4', ref.dimensions)
        for attr in ref.ncattrs():
            if attr != '_FillValue':
                v.setncattr(attr, ref.getncattr(attr))
        v[0] = values[i]
with Dataset(old / 'input/wrfinput_d01') as original, Dataset(target) as new:
    original.set_auto_mask(False)
    new.set_auto_mask(False)
    for name, v in original.variables.items():
        assert np.asarray(v[:]).tobytes() == np.asarray(new[name][:]).tobytes(), name
receipt = dict(source=str(source), source_sha256=sha(source),
               exact_original_body_sha256=hashlib.sha256(body.encode()).hexdigest(),
               initializer=str(oracle / 'initializer.f90'), initializer_sha256=sha(oracle / 'initializer.f90'),
               original_weather_variables_BYTE_unchanged=True, paid_GPU_t0_tracers_BYTE=True,
               active_cells_by_species=[int(np.count_nonzero(v)) for v in values],
               scope='Pristine real.exe initialization on unchanged authentic weather; not invented trajectory truth. '
                     'Original wrf.exe must read these inputs and independently forecast all24h.')
write_json(case / 'TRACER_INITIALIZATION.json', receipt)
for p in (oracle / 'initializer', oracle / 'tr17_init.bin'):
    p.unlink()
binary = data / 'wrf.exe'
shutil.move(case / 'wrf.exe', binary)
(case / 'wrf.exe').symlink_to(binary)
row['input_sha256'] = {p.name: sha(p) for p in (case / 'input').iterdir() if p.is_file()}
row['reference_semantics'] = 'Original52f executable; unchanged original-real tracer initializer; active reference pending'
write_json(case / 'inputs.json', row)
matrix = json.loads((ROOT / 'MATRIX.json').read_text())
assert not any(r['id'] == 'c90' for r in matrix['cases'])
matrix['cases'].append(row)
write_json(ROOT / 'MATRIX.json', matrix)
print(json.dumps(receipt, indent=2))
