"""Pristine WRF Jensen ISHMAEL (mp_physics=55) REFL_10CM oracle on real c87 columns.

usage: python proofs/t_classd/jensen_refl_oracle/build.py <scratch_dir>
Columns come READ-ONLY from the original CPU-WRF c87 history (v-prep/c87/cpu): every Jensen
prognostic of the frame, P=P+PB, DZ8W from (PH+PHB)/9.81, dry TH=(T+300)/(1+Rv/Rd*QV)
(use_theta_m=1, moist_physics_prep_em). The UNCHANGED module_mp_jensen_ishmael.F is compiled
with the build flags of proofs/s2_mpmore/jensen55/build_reference.py (REAL4) and called through
its public mp_jensen_ishmael; DIAG_DBZ3D (refl_10cm, :411) is the reference. Inputs and every
public output are frozen into fixtures/ (npz + manifest with SHA-256s).
"""
from pathlib import Path
import hashlib, json, subprocess, sys

import numpy as np
import netCDF4

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 's2_mpmore/jensen55'))
from build_reference import SOURCE, TABLES, FC, NAMES, TABLE_FILES  # noqa: E402

CASE = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c87/cpu')
HOURS = (3, 9, 15, 21)
ACTIVE, QUIET = 24, 8
HERE = Path(__file__).resolve().parent
FLAGS = ['-O2', '-ffree-form', '-ffree-line-length-none', '-cpp', '-finit-real=zero', '-finit-integer=0',
         '-finit-logical=false', '-ffp-contract=off', '-fconvert=big-endian', '-frecord-marker=4']
RV_RD = np.float32(461.6 / 287.)


def caller(ncol):
    decl = ','.join(name + '(1,nz,1)' for name in NAMES)
    prognostic = ','.join(NAMES[:19])
    return f'''program jensen_refl_oracle
 use module_mp_jensen_ishmael, only: jensen_ishmael_init,mp_jensen_ishmael
 implicit none
 integer :: nz,it,unit,c
 character(64) :: filename
 real :: dt
 real,allocatable :: {','.join(name + '(:,:,:)' for name in NAMES)}
 real :: rain(1,1),rainstep(1,1),snow(1,1),snowstep(1,1)
 call jensen_ishmael_init()
 do c=1,{ncol}
 write(filename,'("input_",I0,".bin")') c
 open(newunit=unit,file=filename,access='stream',form='unformatted',status='old')
 read(unit) nz,it
 read(unit) dt
 allocate({decl})
 read(unit) {prognostic}
 close(unit)
 {chr(10).join(name + '=0.' for name in NAMES[19:])}
 rain=0.; rainstep=0.; snow=0.; snowstep=0.
 call mp_jensen_ishmael(it,dt,{prognostic}, &
  1,1,1,1,1,nz,1,1,1,1,1,nz,1,1,1,1,1,nz, &
  rain,rainstep,snow,snowstep, &
  effc,effi,dbz,vm1,di1,rho1,phi1, &
  vm2,di2,rho2,phi2,vm3,di3,rho3,phi3,type1,type2,type3)
 write(filename,'("output_",I0,".bin")') c
 open(newunit=unit,file=filename,access='stream',form='unformatted',status='replace')
 write(unit) {','.join(NAMES)},rain,rainstep,snow,snowstep
 close(unit)
 deallocate({','.join(NAMES)})
 enddo
end program
'''


def frame_columns(path):
    with netCDF4.Dataset(path) as ds:
        g = lambda n: np.asarray(ds[n][0], np.float32)  # noqa: E731
        qv = g('QVAPOR')
        z = (g('PH') + g('PHB')) / np.float32(9.81)
        f = dict(p=g('P') + g('PB'), dz=z[1:] - z[:-1], th=(g('T') + np.float32(300.)) / (1 + RV_RD * qv),
                 qv=qv, qc=g('QCLOUD'), qr=g('QRAIN'), nr=g('QNRAIN'))
        for i, suffix in ((1, ''), (2, '2'), (3, '3')):
            f[f'qi{i}'] = g('QICE' + suffix); f[f'ni{i}'] = g('QNICE' + suffix)
            f[f'ai{i}'] = g('QVOLI' + suffix); f[f'ci{i}'] = g('QAOLI' + suffix)
        refl = g('REFL_10CM')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return f, refl, sha


def select(refl):
    """Interior columns: ACTIVE quantile-spaced by column-max CPU echo, QUIET without echo."""
    peak = refl[:, 1:-1, 1:-1].max(axis=0)
    flat = np.argsort(peak, axis=None, kind='stable')
    live = flat[peak.flat[flat] > -35]
    dead = flat[peak.flat[flat] <= -35]
    take = list(live[np.linspace(0, len(live) - 1, ACTIVE).round().astype(int)]) if len(live) >= ACTIVE else list(live)
    take += list(dead[np.linspace(0, len(dead) - 1, QUIET).round().astype(int)])
    ny = peak.shape[0]
    return [(int(k // peak.shape[1]) + 1, int(k % peak.shape[1]) + 1) for k in take]


def main(scratch):
    scratch.mkdir(parents=True, exist_ok=False)
    out = HERE / 'fixtures'
    out.mkdir(exist_ok=False)
    columns, cpu_refl, frames = [], [], []
    for hour in HOURS:
        path = CASE / f'wrfout_d01_2023-01-15_{hour:02d}:00:00'
        f, refl, sha = frame_columns(path)
        picks = select(refl)
        frames.append(dict(file=path.name, sha256=sha, hour=hour, columns=picks))
        for j, i in picks:
            columns.append({n: f[n][:, j, i] for n in NAMES[:19]})
            cpu_refl.append(refl[:, j, i])
    ncol, nz = len(columns), len(columns[0]['qv'])
    dt, it = np.float32(18.0), 2001  # c87 DT; a non-first itimestep
    (scratch / 'module.F').write_bytes(SOURCE.read_bytes())
    (scratch / 'stubs.f90').write_text('module module_wrf_error\nend module\n')
    (scratch / 'driver.F90').write_text(caller(ncol))
    for name in TABLE_FILES:
        (scratch / name).symlink_to(TABLES / name)
    for name in ('stubs.f90', 'module.F', 'driver.F90'):
        subprocess.run([FC, *FLAGS, '-c', name], cwd=scratch, check=True, timeout=300)
    subprocess.run([FC, *FLAGS, 'stubs.o', 'module.o', 'driver.o', '-o', 'oracle'], cwd=scratch, check=True, timeout=300)
    big = np.dtype('>f4')
    for c, col in enumerate(columns, 1):
        raw = np.asarray([nz, it], '>i4').tobytes() + np.asarray(dt, big).tobytes()
        raw += b''.join(np.asarray(col[n], big).tobytes() for n in NAMES[:19])
        (scratch / f'input_{c}.bin').write_bytes(raw)
    run = subprocess.run([str(scratch / 'oracle')], cwd=scratch, timeout=1200, capture_output=True)
    run.check_returncode()
    outputs = np.stack([np.frombuffer((scratch / f'output_{c}.bin').read_bytes(), big).astype(np.float32)
                        for c in range(1, ncol + 1)])
    assert outputs.shape == (ncol, len(NAMES) * nz + 4) and np.isfinite(outputs).all()
    inputs = np.stack([np.concatenate([col[n] for n in NAMES[:19]]) for col in columns]).astype(np.float32)
    np.savez_compressed(out / 'oracle.npz', inputs=inputs, outputs=outputs, cpu_refl=np.stack(cpu_refl))
    dbz = outputs[:, NAMES.index('dbz') * nz:(NAMES.index('dbz') + 1) * nz]
    manifest = dict(
        scope='Unchanged original Jensen public caller (REAL4) on real c87 CPU-history columns; '
              'cpu_refl is the same frame REFL_10CM (attribution only: WRF forms dbz inside the step)',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        caller_sha256=hashlib.sha256(caller(ncol).encode()).hexdigest(),
        tables={n: hashlib.sha256((TABLES / n).read_bytes()).hexdigest() for n in TABLE_FILES},
        flags=FLAGS, compiler=FC, dt=float(dt), it=it, nz=nz, input_names=list(NAMES[:19]), output_names=list(NAMES),
        precipitation_names=['rain', 'rainstep', 'snow', 'snowstep'], frames=frames, columns=ncol,
        live_dbz_values=int((dbz > -35).sum()), dbz_max=float(dbz.max()),
        npz_sha256=hashlib.sha256((out / 'oracle.npz').read_bytes()).hexdigest(),
        stdout_sha256=hashlib.sha256(run.stdout).hexdigest())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({k: manifest[k] for k in ('columns', 'live_dbz_values', 'dbz_max')}))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
