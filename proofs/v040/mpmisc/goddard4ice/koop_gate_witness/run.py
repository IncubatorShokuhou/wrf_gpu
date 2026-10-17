"""Original mp38 near-999e3 Koop gate witness; no JAX or physics-body edits."""
from pathlib import Path
import argparse,hashlib,json,re,shutil,subprocess
import numpy as np

CACHE=Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmisc/thompson38/H38-03')
TEMPLATE=Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmisc/thompson38/H38-18/fresh_driver.f90')
FC='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
FLAGS=['-O2','-g','-cpp','-ffree-form','-ffree-line-length-none','-fallow-argument-mismatch','-fconvert=big-endian','-ffp-contract=off','-finit-real=zero','-finit-integer=0','-finit-logical=false']


def gdb(directory,case,ni=None):
    target=directory/('pilot' if ni is None else 'case'+str(case));target.mkdir(exist_ok=True)
    run=f'run {case}'+(' '+ni.name if ni else '')
    fields={'ns':160,'smo0':160,'rho':160,'ni':160,'temp':160,'ssati':160,'pni_iha':320,'pri_iha':320,'pni_inu':320,'pni_rfz':320,'pni_wfz':320}
    commands=['set pagination off','set confirm off','break module_mp_thompson.F:2635','commands','silent',
        'printf "KOOP_GATE k=%d ns=%.9g smo0=%.9g ni=%.9g xni=%.9g T=%.9g ssati=%.9g\\n", k,ns(k),smo0(k),ni(k),xni,temp(k),ssati(k)',
        'continue','end','break module_mp_thompson.F:2960',run,'print csg(15)','print cse(15)','set language c']
    commands += [f'dump binary memory {target}/{name}.bin (char*)&{name} (char*)&{name}+{size}' for name,size in fields.items()]
    commands += ['disable 1','disable 2','continue','quit']
    script=target/'commands.gdb';script.write_text('\n'.join(commands)+'\n')
    log=subprocess.check_output(['gdb','-q','-batch','-x',str(script),str(directory/'oracle_debug')],cwd=directory,text=True,stderr=subprocess.STDOUT,timeout=180)
    (target/'gdb.log').write_text(log)
    gates=[]
    for line in log.splitlines():
        match=re.search(r'KOOP_GATE k=(\d+) ns=([^ ]+) smo0=([^ ]+) ni=([^ ]+) xni=([^ ]+) T=([^ ]+) ssati=([^ ]+)',line)
        if match:gates.append(dict(zip(('k','ns','smo0','ni','xni','T','ssati'),[int(match[1]),*map(float,match.groups()[1:])])))
    assert len(gates)==40,log[-2000:]
    arrays={name:np.fromfile(target/(name+'.bin'),dtype=np.float64 if size==320 else np.float32) for name,size in fields.items()}
    assert all(len(value)==40 for value in arrays.values())
    return target,gates,arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    table=CACHE/'qr_acr_qg_mp38V1.dat';stat0=table.stat()
    source=CACHE/'module_mp_thompson.F'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802'
    for name in ('qr_acr_qg_mp38V1.dat','qr_acr_qsV2.dat','freezeH2O.dat','CCN_ACTIVATE.BIN'):
        dest=out/name
        if not dest.exists():dest.symlink_to(CACHE/name)
    driver=TEMPLATE.read_text().replace('IF (case_id == 7) THEN','IF (case_id == 7 .OR. case_id >= 10) THEN')
    driver=driver.replace('        w(1,k,1)=1.', '        w(1,k,1)=1.\n        IF(case_id >= 10) THEN\n          qi(1,k,1)=1.e-5;qs(1,k,1)=1.e-4\n          ni(1,k,1)=998000./(0.622*p(1,k,1)/(287.04*(th(1,k,1)*pii(1,k,1))*(qv(1,k,1)+0.622)))\n          nwfa(1,k,1)=1.e9;nifa(1,k,1)=1.e6\n        ENDIF')
    driver=driver.replace('  hgt3d = 0.0', "  IF (COMMAND_ARGUMENT_COUNT() >= 2) THEN\n    CALL GET_COMMAND_ARGUMENT(2,arg)\n    OPEN(89,file=trim(arg),status='old',action='read')\n    DO k=kts,kte\n      READ(89,*) ni(1,k,1)\n    ENDDO\n    CLOSE(89)\n  ENDIF\n\n  hgt3d = 0.0")
    (out/'driver.f90').write_text(driver)
    subprocess.run([FC,*FLAGS,'-I'+str(CACHE),'-c','driver.f90','-o','driver.o'],cwd=out,check=True,timeout=60)
    deps=[CACHE/name for name in ('stubs.o','module_model_constants.o','module_mp_radar.o','libmassv.o')]
    for kind,module in (('debug','thompson_debug.o'),('o2','module_mp_thompson.o')):
        subprocess.run([FC,*FLAGS,'-o','oracle_'+kind,'driver.o',str(CACHE/module),*map(str,deps)],cwd=out,check=True,timeout=60)
    _,pilot,arrays=gdb(out,10)
    result={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(driver.encode()).hexdigest(),'flags':FLAGS,'pilot_gate':pilot,'cases':{},'cached_table_preserved':None,'JAX_execution':False}
    for case,delta in ((10,-2.),(11,2.)):
        F=np.float32
        ni=(F(999000)-arrays['ns']+F(delta))/arrays['rho']
        path=out/f'ni{case}.txt';path.write_text('\n'.join(format(float(x),'.9e') for x in ni)+'\n')
        target,gates,observed=gdb(out,case,path)
        dump=subprocess.check_output([str(out/'oracle_o2'),str(case),path.name],cwd=out,text=True,stderr=subprocess.STDOUT,timeout=180)
        (out/f'case{case}.txt').write_text(dump)
        # Match the original mixed REAL+DOUBLE expression then assignment.
        true_xni=(observed['ns']+observed['ni']+(observed['pni_rfz']+observed['pni_wfz']+observed['pni_inu'])*F(18)).astype(np.float32)
        legacy_xni=(observed['smo0']+observed['ni']+(observed['pni_rfz']+observed['pni_wfz']+observed['pni_inu'])*F(18)).astype(np.float32)
        base_gate=(observed['temp']<F(238))&(observed['ssati']>=F(.4))
        true_gate=base_gate&(true_xni<=F(999000));legacy_gate=base_gate&(legacy_xni<=F(999000))
        result['cases'][str(case)]={'delta_number_per_m3':delta,'gate':gates,'true_gate_count':int(np.sum(true_gate)),'legacy_smo0_gate_count':int(np.sum(legacy_gate)),'different_gate_count':int(np.sum(true_gate!=legacy_gate)),'pni_iha_nonzero':int(np.count_nonzero(observed['pni_iha'])),'max_pni_iha':float(np.max(observed['pni_iha'])),'target_input_ni':ni.tolist(),'original_ns':observed['ns'].tolist(),'original_smo0':observed['smo0'].tolist(),'original_ni':observed['ni'].tolist(),'original_rho':observed['rho'].tolist()}
    stat1=table.stat();result['cached_table_preserved']=(stat0.st_ino,stat0.st_size,stat0.st_mtime_ns)==(stat1.st_ino,stat1.st_size,stat1.st_mtime_ns)
    assert result['cached_table_preserved']
    (out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:{x:v[x] for x in ('true_gate_count','legacy_smo0_gate_count','different_gate_count','pni_iha_nonzero','max_pni_iha')} for k,v in result['cases'].items()},indent=2))
if __name__=='__main__':main()
