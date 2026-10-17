"""Diagnostic-only saved-constant dump; pristine scientific oracle is untouched."""
from pathlib import Path
import subprocess
import json
import argparse
import numpy as np
from gpuwrf.physics.milbrandt2mom.constants import constants


def main():
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('input');a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    c=constants('float32')
    names=[k for k,v in c.items() if np.asarray(v).dtype.kind!='b']
    src=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_milbrandt2mom.F').read_text()
    src=src[:src.index('END MODULE my2_mod')+len('END MODULE my2_mod')]
    marker='  endif  !if (KOUNT=0)'
    assert src.count(marker)==1
    dst=root/'constants.bin'
    dump=f"\n open(99,file='{dst}',form='unformatted',access='stream',status='replace')\n"
    dump+=' write(99) '+', &\n'.join('real('+k+')' for k in names)+'\n close(99)\n'
    (root/'diagnostic.f90').write_text(src.replace(marker,marker+dump))
    fc='<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran'
    subprocess.run([fc,'-O2','-ffree-form','-ffree-line-length-none','-c','diagnostic.f90'],cwd=root,check=True)
    driver=Path(__file__).with_name('oracle.F90')
    subprocess.run([fc,'-O2','-ffree-form','-ffree-line-length-none',str(driver),'diagnostic.o','-o','constants_diag'],cwd=root,check=True)
    subprocess.run([str(root/'constants_diag'),a.input,str(root/'unused_output.bin')],check=True)
    reference=np.fromfile(dst,np.float32)
    actual=np.asarray([c[k] for k in names],np.float32)
    diffs={names[i]:{'jax_host':float(actual[i]),'WRF':float(reference[i]),
                    'ulps':int(abs(int(actual.view(np.uint32)[i])-int(reference.view(np.uint32)[i])))} for i in np.flatnonzero(actual!=reference)}
    receipt={'diagnostic_only':True,'count':len(names),'differences':diffs,
             'body_change':'print saved constants only; not scientific qualification executable'}
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (root/'wrf_constants.json').write_text(json.dumps(dict(zip(names,map(float,reference))),indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
