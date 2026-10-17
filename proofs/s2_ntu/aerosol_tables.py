"""Extract original initialized aerosol tables without editing any WRF body."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from build_oracle import FC
from port_scalar import SOURCE

REAL_FIELDS={'dnc0':(),'dnc1':(),'dnc2':(),'dnc3':(),
             **{x:(2,) for x in 'sens ash cmas dnas avan beta1 rxmin'.split()},
             **{x:(3,2) for x in 'zccn cnmod cnstd wmas rfact'.split()}}
DOUBLE_FIELDS={'tblrc':(25,),'tblxf':(25,2)}
INT_FIELDS={'ibaer':(2,),'naern':(2,)}


def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--kind',choices=('r4','r8'),default='r4');args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    extra="\nsubroutine observe_aerosol_tables()\ninteger::u\nopen(newunit=u,file='tables.bin',access='stream',form='unformatted',status='replace')\n"
    extra+='\n'.join('write(u) '+name for name in list(REAL_FIELDS)+list(DOUBLE_FIELDS)+list(INT_FIELDS))
    extra+='\nclose(u)\nend subroutine observe_aerosol_tables\n'
    original=SOURCE.read_text();needle='      END MODULE MODULE_MP_NTU'
    assert original.count(needle)==1
    (out/'original.F').write_text(original.replace(needle,extra+needle))
    (out/'error.F90').write_text('module module_wrf_error\ncontains\nsubroutine wrf_error_fatal(s)\ncharacter(*)::s\nprint *,s\nerror stop 1\nend subroutine\nend module\n')
    (out/'driver.F90').write_text("program tables\nuse module_mp_ntu\ninteger::ccnty\ncharacter(10)::arg\ncall get_command_argument(1,arg)\nread(arg,*)ccnty\ncall aero_const(ccnty)\ncall observe_aerosol_tables()\nend program\n")
    cmd=[FC,'-cpp','-ffree-form','-ffree-line-length-none','-O0','-ffp-contract=off','-fno-range-check','-finit-real=zero','-finit-integer=0','-finit-logical=false','error.F90','original.F','driver.F90','-o','oracle']
    if args.kind=='r8':cmd[1:1]=['-fdefault-real-8','-fdefault-double-8']
    subprocess.run(cmd,cwd=out,check=True)
    real=np.dtype('float32' if args.kind=='r4' else 'float64');fields={}
    with (out/'stdout.log').open('w') as log:
        for ccnty in range(1,5):
            subprocess.run([str(out/'oracle'),str(ccnty)],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=20)
            raw=(out/'tables.bin').read_bytes();offset=0
            for bank,dtype in ((REAL_FIELDS,real),(DOUBLE_FIELDS,np.dtype('float64')),(INT_FIELDS,np.dtype('int32'))):
                for name,shape in bank.items():
                    size=int(np.prod(shape))*dtype.itemsize
                    value=np.frombuffer(raw[offset:offset+size],dtype=dtype).reshape(shape,order='F').copy();offset+=size
                    assert np.isfinite(value).all(),(ccnty,name)
                    fields[f'ccnty{ccnty}_{name}']=value
            assert offset==len(raw)
    np.savez_compressed(out/'tables.npz',**fields)
    result={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'kind':args.kind,'compile':cmd,
            'module_change':'append read-only accessor only; original bodies unchanged',
            'tables':{k:{'shape':v.shape,'dtype':str(v.dtype)} for k,v in fields.items()},
            'npz_sha256':hashlib.sha256((out/'tables.npz').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(result['npz_sha256'])
    for name in ('original.F','error.F90','driver.F90','oracle','tables.bin','module_mp_ntu.mod','module_wrf_error.mod'):(out/name).unlink(missing_ok=True)


if __name__=='__main__':main()
