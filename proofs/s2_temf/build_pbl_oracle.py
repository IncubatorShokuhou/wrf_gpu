"""Regenerate private pristine temf2d columns with the pbl_driver operands.

Unmodified WRF modules and constants are compiled in a lane artifact folder.
Generated Fortran is a caller only; no translations are used as truth.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
PROFILES = ('ux vx tx thx qvx qcx qix p2d p2di pi2d rho z2d te_temfx '
            'shf_temfx qf_temfx uw_temfx vw_temfx wupd_temfx mf_temfx thup_temfx '
            'qtup_temfx qlup_temfx cf3d_temfx kh_temfx km_temfx '
            'rubltenx rvbltenx rthbltenx rqvbltenx rqcbltenx rqibltenx').split()
SURFACE = ('flhc flqc exch_temfx hd_temfx lcl_temfx hct_temfx cfm_temfx fCor '
           'znt zsrf ust zol hol hpbl psim psih xland hfx qfx tsk qsfc gz1oz0 wspd br psfcpa u10 v10 t2').split()
IN = dict(u='ux', v='vx', temperature='tx', theta='thx', qv='qvx', qc='qcx', pressure='p2d',
          p8w='p2di', exner='pi2d', rho='rho', z='z2d', terrain='zsrf', roughness='znt',
          tskold='tsk', coriolis='fCor', hfx='hfx', qfx='qfx', dt='dt')
OUT = {k.removesuffix('x'):k for k in PROFILES[12:25]}
OUT.update(dict(rublten='rubltenx', rvblten='rvbltenx', rthblten='rthbltenx', rqvblten='rqvbltenx',
                rqcblten='rqcbltenx', hd_temf='hd_temfx', lcl_temf='lcl_temfx', hct_temf='hct_temfx',
                cfm_temf='cfm_temfx', pblh='hpbl', kpbl='kpbl1d', ust='ust', u10='u10', v10='v10', t2='t2'))


def columns(n=44):
    """Structured shallow cloudy/dry convection and stable retained state."""
    out = []
    for regime in ('stable', 'neutral', 'dry_convection', 'moist_convection'):
        for wind in (1., 8.):
            for moisture in (.004, .014, .020):
                zz = np.arange(n+1, dtype=np.float32)*np.float32(90.)+np.float32(30.)
                height = zz[:-1]
                th = np.float32(294.)+np.float32(.006)*height
                if 'convection' in regime:
                    th = np.float32(299.)-np.float32(.003)*np.minimum(height, np.float32(900.))+np.float32(.012)*np.maximum(height-np.float32(900.), np.float32(0.))
                if regime == 'neutral':
                    th[:] = 295.
                p = np.float32(100000.)*np.exp(-height/np.float32(8000.))
                ex = (p/np.float32(100000.))**(np.float32(287.)/np.float32(1004.5))
                q = np.float32(moisture)*np.exp(-height/np.float32(2000.))
                qc = np.zeros(n, np.float32)
                if regime == 'moist_convection':
                    qc[8:12] = np.float32(1e-4)
                sk = np.float32((th[0]+(3. if 'convection' in regime else -3. if regime == 'stable' else 0.))*ex[0])
                kw = dict(u=np.full(n, wind, np.float32)+height*np.float32(.001),
                          v=np.full(n, wind*.2, np.float32), temperature=th*ex,
                          theta=th, qv=q, qc=qc, pressure=p, p8w=p+np.float32(350.), exner=ex,
                          rho=p/(np.float32(287.)*th*ex), z=zz, terrain=0., roughness=.1,
                          tskold=sk, coriolis=8e-5, hfx=150. if 'convection' in regime else -20., qfx=5e-5, dt=18.)
                carry = {k:np.zeros(n, np.float32) for k in OUT if OUT[k] in PROFILES}
                carry['te_temf'][:] = np.float32(.2)
                carry.update(pblh=np.float32(123.), kpbl=2)
                kw['carry'] = carry
                out.append(dict(name=f'{regime}_wind{wind}_q{moisture}', inputs=kw))
    return out


def fortran_driver(source):
    sig = re.search(r'subroutine temf2d\((.*?)\)', source, re.S|re.I).group(1)
    args = [s.strip().replace('&','').strip() for s in sig.replace('\n','').split(',')]
    aliases = {k:k for k in args}
    aliases.update(j='1', ep1='ep_1', ep2='ep_2', dtmin='dt/60.', ids='1', ide='2', jds='1', jde='2', kds='1', kde='n+1',
                   ims='1', ime='1', jms='1', jme='1', kms='1', kme='n+1',
                   its='1', ite='1', jts='1', jte='1', kts='1', kte='n')
    code = ['program oracle', 'use module_bl_temf', 'use module_model_constants', 'implicit none',
            'integer,parameter :: n=44', 'integer :: ic,ncase,s', 'integer :: kpbl1d(1)', 'real :: dt',
            'real :: '+','.join(k+'(1,n+1)' for k in PROFILES),
            'real :: '+','.join(k+'(1)' for k in SURFACE), 'read(*,*) ncase', 'do ic=1,ncase']
    code += [k+'=0.' for k in PROFILES+SURFACE]
    code += ['kpbl1d=2', 'hpbl=123.']
    for k, fk in IN.items():
        dest = fk+('(1,1:n+1)' if k=='z' else '(1,1:n)' if fk in PROFILES else '')
        code += ['read(*,*) '+dest]
    code += ['te_temfx(1,1:n)=.2', 'do s=1,3', "write(*,*) 'RECORD',ic,s"]
    for prefix, mapping in [('IN',IN), ('CARRY', {k:OUT[k] for k in OUT if k not in ('rublten','rvblten','rthblten','rqvblten','rqcblten','u10','v10','t2','ust')})]:
        for k,fk in mapping.items():
            value = fk+('(1,1:n+1)' if k=='z' else '(1,1:n)' if fk in PROFILES else '')
            code += [f"write(*,*) '{prefix}_{k}', {value}"]
    code += ['call temf2d('+', &\n  '.join(k+'='+aliases[k] for k in args)+')']
    for k,fk in OUT.items():
        code += [f"write(*,*) 'OUT_{k}', "+fk+('(1,1:n)' if fk in PROFILES else '')]
    # Euler-replay of the isolated PBL tendencies: regression of three calls,
    # not a coupled WRF forecast or a production RK ordering claim.
    code += [k+'(1,1:n)='+k+'(1,1:n)+dt*'+t+'(1,1:n)' for k,t in
             [('ux','rubltenx'),('vx','rvbltenx'),('thx','rthbltenx'),('qvx','rqvbltenx'),('qcx','rqcbltenx')]]
    code += ['tx=thx*pi2d', 'enddo', 'enddo', 'end program',
             'subroutine wrf_debug(level,message)', 'integer :: level', 'character(*) :: message', 'end subroutine']
    return '\n'.join(code)+'\n'


def main():
    assert os.environ.get('JAX_PLATFORMS') == 'cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    build = Path(sys.argv[1]).resolve(); build.mkdir(parents=True, exist_ok=False)
    sources = [WRF/'phys/module_bl_temf.F', WRF/'share/module_model_constants.F']
    for path in sources:
        shutil.copy2(path, build/path.name)
    (build/'pbl_driver.f90').write_text(fortran_driver(sources[0].read_text()))
    cmd = [FC,'-O2','-ffp-contract=off','-ffree-form','-ffree-line-length-none','-cpp','-DEM_CORE=1',
           '-fcheck=all','-finit-real=snan','module_model_constants.F','module_bl_temf.F','pbl_driver.f90','-o','oracle.exe']
    subprocess.run(cmd, cwd=build, check=True, timeout=60)
    cases = columns()
    inp = str(len(cases))+'\n'
    for rec in cases:
        inp += '\n'.join(' '.join(map(str,np.atleast_1d(np.asarray(rec['inputs'][k],np.float32)))) for k in IN)+'\n'
    (build/'columns.txt').write_text(inp)
    result = subprocess.run([str(build/'oracle.exe')], input=inp, text=True, capture_output=True, timeout=60)
    (build/'raw.txt').write_text(result.stdout); (build/'stderr.txt').write_text(result.stderr)
    result.check_returncode()
    records=[]; rec=None; current=None
    for line in result.stdout.splitlines():
        parts=line.split()
        if not parts:
            continue
        if parts[0]=='RECORD':
            ic,s=map(int,parts[1:]); rec=dict(name=cases[ic-1]['name'], step=s, inputs={'carry':{}}, outputs={}); records.append(rec)
        elif parts[0].startswith(('IN_','CARRY_','OUT_')):
            prefix,k=parts[0].split('_',1)
            values=[float(v.replace('D','E')) for v in parts[1:]]
            obj=rec['inputs'] if prefix=='IN' else rec['inputs']['carry'] if prefix=='CARRY' else rec['outputs']
            obj[k]=values if len(values)>1 else int(values[0]) if k=='kpbl' else values[0]
    assert len(records)==len(cases)*3
    d=dict(source='pristine WRF v4.7.1 module_bl_temf.F, unmodified; TEMF2D direct pbl_driver operand map',
           caller='pbl_driver.F:1172-1201; TSK=tskold; te/qf/updraft fields retained across calls; Euler isolated PBL replay, not forecast',
           source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
           compile_command=cmd, records=records)
    (build/'pbl_oracle.json').write_text(json.dumps(d,indent=2)+'\n')
    nf={k:sum(int(np.count_nonzero(~np.isfinite(np.asarray(rec['outputs'][k])))) for rec in records) for k in OUT}
    print(json.dumps(dict(records=len(records),nonfinite=nf),indent=2))


if __name__=='__main__':
    main()
