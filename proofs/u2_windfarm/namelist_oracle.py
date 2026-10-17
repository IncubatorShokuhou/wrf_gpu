"""Fortran NAMELIST array defaults and unchanged measured single-domain binding."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import types
import shutil
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.windfarm import windfarm_option
from gpuwrf.io.namelist_binding import bound_options


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);build=args.out/'build';build.mkdir()
    source='''program probe
implicit none
integer :: windfarm_opt(3)
namelist /physics/ windfarm_opt
windfarm_opt=0
read(*,nml=physics)
write(*,'(3I5)') windfarm_opt
end program probe
'''
    (build/'probe.f90').write_text(source)
    fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    subprocess.run([fc,'-O0',str(build/'probe.f90'),'-o',str(build/'probe')],check=True,timeout=60)
    cases=['','windfarm_opt=1','windfarm_opt=1,0,1','windfarm_opt=2*1','windfarm_opt=1,,1','windfarm_opt(2)=1']
    rows=[]
    for i,case in enumerate(cases):
        text=f'&physics\n {case}\n/\n';path=args.out/f'case{i}.nml';path.write_text(text)
        truth=[int(x) for x in subprocess.check_output([str(build/'probe')],input=text,text=True).split()]
        try:
            nl=parse_namelist(path)
        except ValueError as exc:
            # The existing shared parser explicitly refuses indexed assignment
            # syntax. It must fail closed; callers use an explicit full vector.
            assert case=='windfarm_opt(2)=1'
            rows.append(dict(namelist=case,fortran=truth,port_refused=True,reason=str(exc)))
            continue
        actual=[windfarm_option(nl,f'd{d:02d}') for d in (1,2,3)]
        assert actual==truth,(case,actual,truth)
        assert [bound_options(nl,f'd{d:02d}')['windfarm_opt'] for d in (1,2,3)]==truth
        rows.append(dict(namelist=case,fortran=truth,port=actual))
    root=Path(__file__).resolve().parents[2]
    old_source=subprocess.check_output(['git','show','8b65c4a3a:src/gpuwrf/io/namelist_binding.py'],cwd=root,text=True)
    old=types.ModuleType('binding_before_windfarm_array_fix');exec(old_source,old.__dict__)
    reuse=[]
    for name,path,wind in (('Swiss',Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/namelist.input'),False),
                           ('PROD',Path('<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725/namelist.input'),False),
                           ('Swiss-Windfarm',Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/namelist.input'),True)):
        nl=parse_namelist(path)
        if wind:nl['physics'].update(windfarm_opt=1,windfarm_ij=1,windfarm_tke_factor=.25)
        before,after=old.bound_options(nl,'d01'),bound_options(nl,'d01')
        assert before==after,(name,before,after)
        reuse.append(dict(case=name,binding_identical=True,source_namelist_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bound=after))
    result=dict(compiler=fc,fortran_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                registry_default=0,domains=3,cases=rows,measured_binding_reuse=reuse,
                binding_baseline_commit='8b65c4a3a',binding_baseline_sha256=hashlib.sha256(old_source.encode()).hexdigest())
    (args.out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    shutil.rmtree(build)


if __name__=='__main__':main()
