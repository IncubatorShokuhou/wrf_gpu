"""Independent original caller sum, with already qualified Fortran operands.

No microphysics is rerun. The original PHYS_TOT assignment is copied verbatim
into a tiny REAL4 driver, whose six inputs are checked BYTE after decimal I/O.
"""
from pathlib import Path
import argparse,ast,hashlib,json,re,shutil,subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
FIELDS=('physc','physe','physd','physs','physm','physf')
FC='<USER_HOME>/miniconda3/envs/wrfbuild/bin/x86_64-conda-linux-gnu-gfortran'


def main():
 p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);build=a.output/'build';build.mkdir(exist_ok=True)
 wrf=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_microphysics_driver.F')
 statement=re.search(r'phys_tot\(i,k,j\)\s*=\s*physc\(i,k,j\).*?physf\(i,k,j\)',wrf.read_text(),re.S).group()
 variables=','.join((*FIELDS,'phys_tot'))
 source='program original_sum\nimplicit none\ninteger i,j,k,row\nreal,dimension(1,1,1)::'+variables+'\ni=1;j=1;k=1\ndo row=1,720\n read(*,*) '+','.join(n+'(i,k,j)' for n in FIELDS)+'\n'+statement+'\n write(*,\'(7(Z8.8,1X))\') '+','.join('transfer('+n+'(i,k,j),0)' for n in (*FIELDS,'phys_tot'))+'\nenddo\nend program\n'
 (build/'reference.f90').write_text(source);(a.output/'reference.f90').write_text(source)
 flags=['-O2','-ffree-line-length-none','-ffp-contract=off','-finit-real=zero','-finit-integer=0']
 subprocess.run([FC,*flags,'reference.f90','-o','reference'],cwd=build,check=True,timeout=30)
 original=ROOT/'tests/v040/mpmisc/goddard4ice_columns.npz'
 payload={};rows=[];groups=[]
 with np.load(original,allow_pickle=False) as data:
  for group in ('base','water18','land54'):
   for case in range(1,7):
    prefix=f'{group}_{case}__';values=np.stack([data[prefix+n] for n in FIELDS],axis=1)
    assert values.shape==(40,6) and values.dtype==np.float32
    rows.extend(values);groups.append(prefix)
 inputs=np.asarray(rows,np.float32)
 text='\n'.join(' '.join(format(float(v),'.9e') for v in row) for row in inputs)+'\n'
 out=subprocess.check_output([str(build/'reference')],input=text,text=True,cwd=build,timeout=30)
 bits=np.asarray([[int(v,16) for v in row.split()] for row in out.splitlines()],np.uint32)
 assert bits.shape==(720,7)
 assert bits[:,:6].copy().view(np.float32).tobytes()==inputs.tobytes(),'Fortran driver input roundtrip changed operands'
 total=bits[:,6].copy().view(np.float32)
 # Test only the new expression's AST against the independent Fortran result.
 module=ast;tree=module.parse((ROOT/'src/gpuwrf/physics/goddard4ice/column.py').read_text())
 for node in module.walk(tree):
  if isinstance(node,module.Assign) and any(isinstance(t,module.Subscript) and isinstance(t.slice,module.Constant) and t.slice.value=='phys_tot' for t in node.targets):
   expression=module.Expression(node.value);break
 else:raise AssertionError('PHYS_TOT output assignment missing')
 value=eval(compile(expression,'original_phys_tot','eval'),{},dict(outputs={name:inputs[:,index] for index,name in enumerate(FIELDS)}))
 assert value.dtype==np.float32 and value.tobytes()==total.tobytes()
 for index,prefix in enumerate(groups):payload[prefix+'phys_tot']=total[index*40:(index+1)*40]
 target=ROOT/'tests/v040/mpmisc/goddard4ice_phys_tot.npz';np.savez_compressed(target,**payload)
 meta={'source':'original module_microphysics_driver.F PHYS_TOT assignment, verbatim','caller_sha256':hashlib.sha256(wrf.read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(source.encode()).hexdigest(),'original_statement':statement,'original_operand_fixture_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'flags':flags,'byte_operands':4320,'byte_phys_tot_values':720,'full_microphysics_rerun':False,'JAX_execution':False}
 target.with_suffix('.json').write_text(json.dumps(meta,indent=2));(a.output/'receipt.json').write_text(json.dumps(meta,indent=2))
 shutil.rmtree(build);print('4320originaloperandsBYTE;720/720newPHYS_TOTvaluesBYTEvsverbatimoriginalFortran;noJAX')
if __name__=='__main__':main()
