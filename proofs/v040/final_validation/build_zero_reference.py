"""Private full WRF build with the wave's compiler-defined zero semantics."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from prepare import ROOT, PRISTINE, sha, write_json

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
build = ROOT / 'pristine-zero'
repair = '--resume-morrison-exception' in sys.argv
if not build.exists():
    subprocess.run(['cp', '--reflink=auto', '-a', str(PRISTINE), str(build)], check=True)
else:
    if (ROOT / 'ZERO_BUILD.json').exists():
        assert repair
        old = json.loads((ROOT / 'ZERO_BUILD.json').read_text())
        assert old['executable_sha256'] is None and old['source_bodies_unchanged']
        (ROOT / 'ZERO_BUILD.json').rename(ROOT / 'ZERO_BUILD_FAILED_GLOBAL_INIT.json')
source_files = [p for sub in ('phys', 'dyn_em', 'share', 'frame', 'Registry')
                for p in (PRISTINE / sub).rglob('*') if p.is_file() and p.suffix in ('.F', '.F90', '.h')]
source_hashes = {str(p.relative_to(PRISTINE)): sha(p) for p in source_files}
assert all(sha(build / p) == digest for p, digest in source_hashes.items())
config = (build / 'configure.wrf').read_text()
if '-finit-real=zero' not in config:
    config += '\n# v-prep compiler-defined upstream-uninitialized-read reference; no FP traps.\n'
    config += 'FCBASEOPTS += -finit-real=zero -finit-integer=0 -finit-logical=false\n'
(build / 'configure.wrf').write_text(config)
env = dict(os.environ)
env['PATH'] = '<USER_HOME>/miniconda3/envs/wrfbuild/bin:' + env['PATH']
env['NETCDF'] = '<USER_HOME>/miniconda3/envs/wrfbuild'
env['OMP_NUM_THREADS'] = '1'
log = (ROOT / ('VP05-zero-build-repair.log' if repair else 'VP05-zero-build.log')).open('w')
# clean without -a preserves the exact copied configure/Registry and tables.
tcsh = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/tcsh'
exceptions = []
if repair:
    # Upstream bare SAVE + automatic arrays in inactive Morrison-aero cannot
    # compile with -finit-* (GNU14). Retain exactly the canonical compiled
    # object/interface. No source changes, and none of our zero-init-dependent
    # configurations selects MP40. MP40 references use the original binary.
    for name in ('module_mp_morr_two_moment_aero.o', 'module_mp_morr_two_moment_aero.mod'):
        original = PRISTINE / 'phys' / name
        shutil.copyfile(original, build / 'phys' / name)
        exceptions.append(dict(file=name, original=str(original), sha256=sha(original),
                               reason='upstream SAVE automatic arrays reject compiler initialization; inactive in zero-dependent references'))
else:
    subprocess.run([tcsh, '-f', './clean'], cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
start = time.monotonic()
run = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=15', '3600',
    tcsh, '-f', './compile', '-j', '4', 'em_real'], cwd=build, env=env,
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
paused, pauses = False, []
def terminate(signum, frame):
    if run.poll() is None:
        os.killpg(run.pid, signal.SIGCONT)
        os.killpg(run.pid, signal.SIGTERM)
    raise SystemExit(128 + signum)
signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
while run.poll() is None:
    quiet = Path('/tmp/wrf_gpu2_quiet').exists()
    if quiet != paused:
        os.killpg(run.pid, signal.SIGSTOP if quiet else signal.SIGCONT)
        paused = quiet
        if quiet:
            pauses.append({'start_s': time.monotonic() - start})
        else:
            pauses[-1]['end_s'] = time.monotonic() - start
    time.sleep(.5)
log.close()
equal = all(sha(build / p) == digest for p, digest in source_hashes.items())
exe = build / 'main/wrf.exe'
receipt = dict(schema='v040-private-wrf-zero-build-v1', rc=run.returncode,
    wall_s=time.monotonic() - start, executable=str(exe),
    executable_sha256=sha(exe) if exe.exists() else None,
    source_bodies_unchanged=equal, source_sha256=source_hashes,
    original_configure_sha256=sha(PRISTINE / 'configure.wrf'),
    configure_sha256=sha(build / 'configure.wrf'),
    added_flags=['-finit-real=zero', '-finit-integer=0', '-finit-logical=false'],
    canonical_object_exceptions=exceptions,
    quiet_pauses=pauses, compiler=subprocess.check_output(['gfortran', '--version'], env=env, text=True).splitlines()[0])
write_json(ROOT / 'ZERO_BUILD.json', receipt)
assert run.returncode == 0 and exe.exists() and equal, {k: receipt[k] for k in ('rc','wall_s','executable_sha256','source_bodies_unchanged')}
print(json.dumps({k: receipt[k] for k in ('rc', 'wall_s', 'executable_sha256', 'source_bodies_unchanged')}))
