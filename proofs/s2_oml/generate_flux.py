"""Private builds of unmodified WRF hurricane SFCLAY1D bodies, REAL4.

Only thin existing drivers are adapted: named isftcflx and active high-wind
columns; WRF source bodies are copied byte-for-byte. FP traps off as production.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BUILD=Path('<USER_HOME>/wrf_gpu2_lanes/s2-oml/flux_build')
FC='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
WRF=ROOT/'data/wrf_pristine/WRF'
FLAGS=['-O0','-ffp-contract=off','-ffree-form','-ffree-line-length-none','-cpp','-DEM_CORE=1',
       '-finit-real=zero','-finit-integer=0','-finit-logical=false']


def run(args,**kw):
    return subprocess.run(args,cwd=BUILD,check=True,capture_output=True,text=True,**kw)


def main(*, sensitivity=False, fma=False):
    global BUILD,FLAGS
    if sensitivity:
        BUILD=BUILD.with_name('flux_sensitivity')
        FLAGS=[x for x in FLAGS if not x.startswith('-finit-')]+['-finit-real=snan','-finit-integer=-999','-finit-logical=true']
    if fma:
        BUILD=BUILD.with_name('flux_fma')
        FLAGS=FLAGS+['-O2','-march=native','-ffp-contract=fast']
    if Path('/tmp/wrf_gpu2_quiet').exists():
        raise SystemExit('QUIET')
    BUILD.mkdir(parents=True,exist_ok=True)
    names=['phys/ccpp_kind_types.f90','phys/physics_mmm/sf_sfclayrev.F90',
           'phys/module_sf_sfclayrev.F','phys/module_sf_sfclay.F','phys/module_sf_mynn.F']
    for name in names:
        shutil.copy2(WRF/name,BUILD/Path(name).name)
    shutil.copy2(ROOT/'proofs/v090/module_model_constants.f90',BUILD/'module_model_constants.f90')
    for name in names[:4]:
        run([FC,*FLAGS,'-c',Path(name).name])
    run([FC,*FLAGS,'-c','module_model_constants.f90'])
    run([FC,*FLAGS,'-c','module_sf_mynn.F'])
    cases=[]
    # Warm SFCLAY1D calls, tropical cyclone stable/unstable, dry land and calm.
    for i,u in enumerate((0.1,10,30,55,70,45,20,5)):
        cases.append([u,-u*.3,300.,.018,99000.,60.,1.14,u*.9,-u*.2,120.,
                      1.,1000.,1. if i==7 else 2.,302. if i%2 else 298.,100000.,0.,0.,
                      .0015, max(.05,u*.04),0.,-1.,150.,.0002])
    header=lambda opt:f'{len(cases)} 2 1 {opt} 0 0 9000\n'
    driver=(ROOT/'proofs/v090/mynn_oracle_driver.f90').read_text()
    driver=driver.replace('ustm=0.;','ustm=0.35;').replace('30(ES22.14','31(ES22.14').replace('ZNT(i)\n','ZNT(i),ustm(i)\n')
    (BUILD/'mynn_driver.f90').write_text(driver)
    run([FC,*FLAGS,'mynn_driver.f90','module_model_constants.o','module_sf_mynn.o','-o','mynn'])
    outputs={}
    for opt in (1,2):
        raw=run([str(BUILD/'mynn')],input=header(opt)+'\n'.join(' '.join(map(str,x)) for x in cases)+'\n').stdout
        rows=[list(map(float,l.split()[1:])) for l in raw.splitlines() if l and not l.startswith('#')]
        outputs[f'mynn_{opt}']={'inputs':cases,'output':rows,'ustm_in':.35}
        differences=[]
        for operand in (0,1,2,3,4,5,6,10,11,13,14,17,18,19,20,21,22):
            deltas=[]
            for direction in (-np.inf,np.inf):
                varied=np.asarray(cases,np.float32).copy()
                varied[:,operand]=np.nextafter(varied[:,operand],np.float32(direction))
                raw=run([str(BUILD/'mynn')],input=header(opt)+'\n'.join(' '.join(map(str,x)) for x in varied)+'\n').stdout
                value=np.asarray([list(map(float,l.split()[1:])) for l in raw.splitlines() if l and not l.startswith('#')],np.float32)
                deltas.append(np.abs(value-np.asarray(rows,np.float32)))
            differences.append(np.max(deltas,axis=0))
        outputs[f'mynn_{opt}']['input_ulp_bound']=np.sum(differences,axis=0).tolist()
    for scheme,driver_path,objects in (
        ('revised','proofs/v060/oracle/sfclayrev_oracle_driver.f90',['ccpp_kind_types.o','sf_sfclayrev.o','module_sf_sfclayrev.o']),
        ('old','proofs/v013/oracle/surface_lsm/sfclay_old_mm5_oracle_driver.f90',['module_sf_sfclay.o'])):
        driver=(ROOT/driver_path).read_text()
        if scheme=='old':
            driver=driver.replace('REAL, PARAMETER :: CP = 1004.0','REAL, PARAMETER :: CP = 1004.5').replace('EP1 = 0.608','EP1 = 461.6/287.-1.').replace('EP2 = 0.622','EP2 = 287./461.6')
            driver=driver.replace('INTEGER :: case_id, i, itimestep, isfflx','INTEGER :: case_id, i, itimestep, isfflx, fluxoption')
            driver=driver.replace('  REAL :: DX','  REAL, DIMENSION(N,1) :: USTM,CK,CKA,CD,CDA\n  REAL :: DX')
            driver=driver.replace('its,ite, jts,jte, kts,kte)',
                'its,ite, jts,jte, kts,kte, USTM,CK,CKA,CD,CDA,fluxoption,0,0)')
            marker='  DO i = its, ite\n     DXCOL(i) = DX'
        else:
            marker='\n  CALL sfclayrev('
        # Driver-only overrides exercise genuine high-wind branches. Inputs are
        # emitted by the driver's normal dumps and consumed verbatim by tests.
        patch='''
  DO i=1,N
    U3D(i,1,1)=5.+9.*i; V3D(i,1,1)=-0.3*U3D(i,1,1)
    UST(i,1)=0.04*U3D(i,1,1); ZNT(i,1)=0.0015
    T3D(i,1,1)=300.; QV3D(i,1,1)=0.018; P3D(i,1,1)=99000.
    TSK(i,1)=302.; PSFC(i,1)=100000.; XLAND(i,1)=2.; LAKEMASK(i,1)=0.
    IF(MOD(i,2)==0) TSK(i,1)=298.
    IF(i==8) XLAND(i,1)=1.
  END DO
'''
        patch+='  USTM=0.35; CK=0.; CKA=0.; CD=0.; CDA=0.\n'
        marker_dump="     CALL dump('UST', i, UST(i,1))"
        assert marker_dump in driver
        driver=driver.replace(marker_dump,marker_dump+"\n     CALL dump('USTM', i, USTM(i,1))")
        if scheme=='revised':
            patch+='  UST_IN_SAVE=UST(:,1); ZNT_IN_SAVE=ZNT(:,1)\n'
        # Driver-only perturbations for a pristine-derived conditioning bound;
        # no port outputs participate in its construction.
        operands=('U3D(:,1,1)','V3D(:,1,1)','T3D(:,1,1)','QV3D(:,1,1)','P3D(:,1,1)','DZ8W(:,1,1)',
                  'PSFC(:,1)','TSK(:,1)','ZNT(:,1)','UST(:,1)','MOL(:,1)','QSFC(:,1)','HFX(:,1)','QFX(:,1)',
                  'PBLH(:,1)','MAVAIL(:,1)')
        driver=driver.replace('  INTEGER :: case_id,','  INTEGER :: perturbation\n  INTEGER :: case_id,',1)
        pulse='''
  perturbation=0
  IF(COMMAND_ARGUMENT_COUNT()>=3) THEN
    CALL GET_COMMAND_ARGUMENT(3,arg)
    READ(arg,*) perturbation
  ENDIF
  SELECT CASE(ABS(perturbation))
'''
        for i,name in enumerate(operands,1):
            pulse+=f'  CASE({i})\n    {name}=NEAREST({name},SIGN(1.,REAL(perturbation)))\n'
        pulse+='  END SELECT\n'
        for opt in (1,2):
            current=driver.replace(marker,patch+pulse+f'  {"isftcflx" if scheme=="revised" else "fluxoption"}={opt}\n'+marker,1)
            (BUILD/'driver.f90').write_text(current)
            run([FC,*FLAGS,'driver.f90',*objects,'-o',scheme])
            raw=run([str(BUILD/scheme),'1']).stdout
            def parse(raw):
                values={}
                for line in raw.splitlines():
                    match=re.fullmatch(r'([A-Z0-9_]+)\[(\d+)\]=\s*(\S+)\s*',line)
                    if match:
                        key=match[1];values.setdefault(key,[]).append(float(match[3]))
                return values
            values=parse(raw)
            assert len(values['U'])==8, (scheme,'driver dump parser failed')
            bounds={key:np.zeros(8,np.float32) for key in values}
            for operand in range(1,len(operands)+1):
                pairs=[parse(run([str(BUILD/scheme),'1','0',str(direction*operand)]).stdout) for direction in (-1,1)]
                for key in bounds:
                    bounds[key]+=np.maximum(*(np.abs(np.asarray(p[key],np.float32)-np.asarray(values[key],np.float32)) for p in pairs))
            values['input_ulp_bound']={key:value.tolist() for key,value in bounds.items()}
            outputs[f'{scheme}_{opt}']=values
    out=ROOT/'proofs/s2_oml/fixtures'
    out.mkdir(exist_ok=True)
    if fma:
        (out/'flux_fma.json').write_text(json.dumps(outputs,indent=2)+'\n')
        manifest=json.loads((out/'flux_manifest.json').read_text())
        manifest['fma_flags']=FLAGS
        (out/'flux_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Pristine FMA comparison',list(outputs))
        return
    if sensitivity:
        primary=json.loads((out/'flux.json').read_text())
        equal={}
        for key,value in outputs.items():
            fields=('output',) if key.startswith('mynn') else tuple(k for k in value if k!='input_ulp_bound')
            equal[key]=all(np.array_equal(np.asarray(value[k]),np.asarray(primary[key][k])) for k in fields)
        manifest=json.loads((out/'flux_manifest.json').read_text())
        manifest['compiler_init_sensitivity_byte_equal']=equal
        manifest['sensitivity_flags']=FLAGS
        (out/'flux_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Compiler initialization sensitivity',equal)
        assert all(equal.values()),'upstream initialization-sensitive outputs require disclosure'
        return
    (out/'flux.json').write_text(json.dumps(outputs,indent=2)+'\n')
    (out/'flux_manifest.json').write_text(json.dumps({'compiler':FC,'flags':FLAGS,'source_sha256':{
        name:hashlib.sha256((WRF/name).read_bytes()).hexdigest() for name in names},
        'active_columns':{'mynn':14,'revised':14,'old':14},'regimes':['high-wind stable water','high-wind unstable water','land','calm MYNN'],
        'wrf_constants':'CP=1004.5, EP1=REAL(461.6)/287-1, EP2=287/REAL(461.6)',
        'pristine_bodies_modified':False,
        'conditioning':'sum of pristine-only +/-1 REAL ULP per-operand envelopes; no port data used'},indent=2)+'\n')
    print('Hurricane fixtures',list(outputs))


if __name__=='__main__':
    import sys
    main(sensitivity='--sensitivity' in sys.argv,fma='--fma' in sys.argv)
