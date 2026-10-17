"""Build pristine WRF 4.7.1 surface-layer public-call oracles (lane t-cfix-sfc, no JAX).

Copies the UNMODIFIED phys sources into <lane>/oracle/src (E104) and builds one shared
object per scheme and variant:
  zero : -O0 -ffp-contract=off -finit-real=zero      (policy reference, literal REAL4)
  inf  : -O0 -ffp-contract=off -finit-real=inf       (uninitialised-read sensitivity)
  fma  : -O2 -mfma -ffp-contract=fast -finit-real=zero (optimiser/FMA spread)
Schemes: 91 SFCLAY (module_sf_sfclay.F), 7 PXSFCLAY (module_sf_pxsfclay.F),
3 SF_GFS (module_sf_gfs.F + GFS machine/physcons/funcphys/progtm), 2 MYJSFC
(module_sf_myjsfc.F + share/module_model_constants.F), pbl3 BL_GFS (module_bl_gfs.F). REAL4 like production
(no -fdefault-real-8; SF_GFS keeps its own kind_phys DOUBLE locals). -DEM_CORE=1 as ARW.
usage: python build_oracle.py <lane_dir>   -> <lane>/oracle/manifest.json
"""
from pathlib import Path
import hashlib, json, shutil, subprocess, sys

HERE = Path(__file__).resolve().parent
LANE = Path(sys.argv[1])
PHYS = Path('<USER_HOME>/src/wrf_pristine/WRF/phys')
SHARE = PHYS.parent / 'share'
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
SCHEMES = {
    91: (['module_sf_sfclay.F'], 'sfclay91_oracle.f90'),
    7: (['module_sf_pxsfclay.F'], 'sfclay7_oracle.f90'),
    3: (['module_gfs_machine.F', 'module_gfs_physcons.F', 'module_gfs_funcphys.F', 'module_progtm.F',
         'module_sf_gfs.F'], 'sfclay3_oracle.f90'),
    2: (['module_model_constants.F', 'module_sf_myjsfc.F'], 'myjsfc_oracle.f90'),
    'pbl3': (['module_model_constants.F', 'module_gfs_machine.F', 'module_gfs_physcons.F', 'module_bl_gfs.F'],
             'blgfs_oracle.f90'),
    # BL_GFS with two COPY-ONLY calls around CALL MONINP (capture_source below; BYTE-checked
    # against 'pbl3' by make_oracle_blgfs_capture.py).
    'pbl3cap': (['module_model_constants.F', 'module_gfs_machine.F', 'module_gfs_physcons.F',
                 'module_bl_gfs_capture.F'], 'blgfs_capture.f90'),
}
GENERATED = {'module_bl_gfs_capture.F'}
CAPTURE_IN = ('      CALL BLGFS_CAPTURE_IN(IM,KX,NTRAC,U1,V1,T1,Q1,PSK,RBSOIL,FM,FH,TSEA,HEAT,EVAP,STRESS,SPD1,'
              'PRSI,DEL,PRSL,PRSLK,PHII,PHIL,DELTIM)\n')
CAPTURE_OUT = '      CALL BLGFS_CAPTURE_OUT(IM,KX,NTRAC,DV,DU,TAU,RTG,KPBL,HPBL)\n'


def capture_source(src):
    """module_bl_gfs.F + exactly two inserted copy-only CALL statements around CALL MONINP."""
    lines = (src / 'module_bl_gfs.F').read_text().splitlines(keepends=True)
    start = [i for i, line in enumerate(lines) if line.startswith('      CALL MONINP(')]
    assert len(start) == 1, start
    # first statement end after the CALL (the MONINP SUBROUTINE header repeats the text)
    end = [i for i, line in enumerate(lines) if i > start[0] and line.strip() == 'RBCR,HGAMQ,ALPHA)'][:1]
    assert end and end[0] - start[0] == 8, (start, end)
    out = lines[:start[0]] + [CAPTURE_IN] + lines[start[0]:end[0] + 1] + [CAPTURE_OUT] + lines[end[0] + 1:]
    (src / 'module_bl_gfs_capture.F').write_text(''.join(out))
    assert len(out) == len(lines) + 2

VARIANTS = {
    'zero': ['-O0', '-ffp-contract=off', '-finit-real=zero'],
    'inf': ['-O0', '-ffp-contract=off', '-finit-real=inf'],
    'fma': ['-O2', '-mfma', '-ffp-contract=fast', '-finit-real=zero'],
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    src = LANE / 'oracle' / 'src'
    src.mkdir(parents=True, exist_ok=True)
    manifest = {'compiler': subprocess.run([FC, '--version'], capture_output=True, text=True).stdout.splitlines()[0],
                'sources': {}, 'wrappers': {}, 'builds': {}}
    for name in sorted({f for files, _ in SCHEMES.values() for f in files} - GENERATED):
        origin = (SHARE if name == 'module_model_constants.F' else PHYS) / name
        shutil.copyfile(origin, src / name)
        assert sha(src / name) == sha(origin)
        manifest['sources'][name] = sha(origin)
    shutil.copyfile(HERE / 'wrf_constants.inc', src / 'wrf_constants.inc')
    manifest['wrappers']['wrf_constants.inc'] = sha(HERE / 'wrf_constants.inc')
    only = set(sys.argv[2:]) or {str(k) for k in SCHEMES}
    if 'pbl3cap' in only:
        capture_source(src)
        manifest['generated'] = {'module_bl_gfs_capture.F': dict(
            sha256=sha(src / 'module_bl_gfs_capture.F'), base_sha256=sha(src / 'module_bl_gfs.F'),
            inserted=[CAPTURE_IN.strip(), CAPTURE_OUT.strip()])}
    for scheme, (files, wrapper) in SCHEMES.items():
        if str(scheme) not in only:
            continue
        manifest['wrappers'][wrapper] = sha(HERE / wrapper)
        for variant, flags in VARIANTS.items():
            out = LANE / 'oracle' / (scheme if isinstance(scheme, str) else f'sf{scheme}') / variant
            out.mkdir(parents=True, exist_ok=True)
            cmd = [FC, '-shared', '-fPIC', *flags, '-cpp', '-DEM_CORE=1', '-ffree-form', '-ffree-line-length-none',
                   '-ffpe-summary=none', f'-I{src}', *(str(src / f) for f in files), str(HERE / wrapper),
                   '-o', str(out / 'oracle.so')]
            subprocess.run(cmd, cwd=out, check=True, timeout=300)
            manifest['builds'][f'{out.parent.name}/{variant}'] = dict(flags=flags, sha256=sha(out / 'oracle.so'))
    path = LANE / 'oracle' / 'manifest.json'
    if path.exists():  # keep earlier scheme builds when rebuilding a subset
        old = json.loads(path.read_text())
        for key in ('sources', 'wrappers', 'builds', 'generated'):
            manifest[key] = {**old.get(key, {}), **manifest[key]}
    path.write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest['builds'], indent=1))


if __name__ == '__main__':
    main()
