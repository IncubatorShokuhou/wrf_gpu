"""Build original legacy19/21 columns. Bodies stay pristine; caller is specialized."""
from pathlib import Path
import hashlib,json,re,subprocess,sys


def make_driver(proof):
    driver=(proof/'nssl2mom_oracle_driver_v2.F90').read_text()
    driver=driver.replace('  INTEGER :: thermo_id, hail_seed, cn_mode','  INTEGER :: thermo_id, hail_seed, cn_mode\n  INTEGER :: variant')
    anchor='  ! ---- init: mirrors module_physics_init.F'
    assert driver.count(anchor)==1
    driver=driver.replace(anchor,"  CALL GET_COMMAND_ARGUMENT(3,arg)\n  READ(arg,*) variant\n  IF (variant /= 19 .and. variant /= 21) STOP 'badvariant'\n"+anchor)
    driver=driver.replace('ipctmp=5, mixphase=0','ipctmp=0, mixphase=0')
    driver=driver.replace('nssl_density_on=.TRUE., nssl_hail_on=.TRUE., nssl_ccn_on=.TRUE.',
                          'nssl_density_on=(variant==19), nssl_hail_on=(variant==19), nssl_ccn_on=(variant==19)')
    driver=driver.replace('VHW=VHW, F_VHW=.TRUE., VHL=VHL, F_VHL=.TRUE.',
                          'VHW=VHW, F_VHW=(variant==19), VHL=VHL, F_VHL=.FALSE.')
    driver=driver.replace('CN=CN, F_CN=.TRUE.','CN=CN, F_CN=(variant==19)')
    driver=driver.replace('HAS_REQC=1, HAS_REQI=1, HAS_REQS=1','HAS_REQC=0, HAS_REQI=0, HAS_REQS=0')
    driver=driver.replace('  ! save inputs','''  ! Registry: no hydrometeor numbers or hail volume in one-moment.
  CCW=0.0; CRW=0.0; CCI=0.0; CSW=0.0; CHW=0.0; CHL=0.0; VHL=0.0
  IF (variant==21) THEN
    QHL=0.0; VHW=0.0; CN=0.0
  ENDIF
  ! save inputs''')
    return driver.replace('ES23.15','ES26.17E3')


def main(tree,artifact):
    work=artifact/'build';out=artifact/'fixtures'
    if artifact.exists():raise SystemExit('Use a fresh artifact root')
    work.mkdir(parents=True);out.mkdir()
    proof=tree/'proofs/v034/oracle/nssl2mom'
    source=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_nssl_2mom.F')
    fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    run=lambda args,**kw:subprocess.run(list(map(str,args)),check=True,timeout=180,**kw)
    for name,args in [('module_vars.txt',[source,1226]),('gs_vars.txt',[source,13714,12638])]:
        with (work/name).open('w') as f:run([sys.executable,proof/'list_module_vars.py',*args],stdout=f)
    instr=work/'instrumented.F'
    run([sys.executable,proof/'make_instrumented.py',source,instr,work/'module_vars.txt',work/'gs_vars.txt'])
    driver=make_driver(proof);rows=[]
    for mode in ('fp32','fp64'):
        flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-finit-real=zero','-finit-integer=0','-finit-logical=false','-ffp-contract=off']
        if mode=='fp64':flags+=['-fdefault-real-8','-fdefault-double-8']
        for kind in ('pristine','instr'):
            b=work/(mode+'_'+kind);b.mkdir()
            (b/'module.F').write_bytes((source if kind=='pristine' else instr).read_bytes())
            (b/'stubs.f90').write_bytes((proof/'nssl2mom_stub_modules.f90').read_bytes())
            (b/'driver.F90').write_text(driver);(b/'namelist.input').write_text('')
            for file in ('stubs.f90','module.F','driver.F90'):
                run([fc,*flags,*(['-DO1_INSTR'] if kind=='instr' and file=='driver.F90' else []),'-c',file],cwd=b)
            run([fc,*flags,'stubs.o','module.o','driver.o','-o','oracle'],cwd=b)
            for variant in (19,21):
                for case in range(1,18):
                    with (b/f'v{variant}_case{case}.txt').open('w') as f:run([b/'oracle',case,'dumpmv' if kind=='instr' and case==1 else 'none',variant],cwd=b,stdout=f)
        for variant in (19,21):
            for case in range(1,18):
                a=(work/(mode+'_pristine')/f'v{variant}_case{case}.txt').read_text()
                path=work/(mode+'_instr')/f'v{variant}_case{case}.txt';text=path.read_text()
                filtered=''.join(x+'\n' for x in text.splitlines() if not re.match(r'^(S[0-9]:|MV:|MVS:|G[0-9]:)',x))
                assert a==filtered,(mode,variant,case,'E26')
                target=out/f'v{variant}_{mode}_{case}.json'
                run([sys.executable,proof/'dump_to_json_v2.py',path,target,mode])
                rows.append(dict(variant=variant,mode=mode,case=case,E26=True,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),caller_sha256=hashlib.sha256(driver.encode()).hexdigest(),rows=rows),indent=2)+'\n')


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
