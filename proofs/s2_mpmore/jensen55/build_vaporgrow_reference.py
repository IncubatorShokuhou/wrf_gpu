"""Observe actual original vaporgrow calls, with whole-column E26 preservation."""
from pathlib import Path
import ast,hashlib,importlib.util,json,re,sys
import numpy as np

tree,artifact=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
component=sys.argv[3] if len(sys.argv)>3 else 'vaporgrow'
assert component in ('vaporgrow','aggregation')
tag='vapor' if component=='vaporgrow' else 'aggregation'
assert (tree/'.git').is_file() and not artifact.exists()
artifact.mkdir()
spec=importlib.util.spec_from_file_location('original_jensen',tree/'proofs/s2_mpmore/jensen55/build_reference.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
pristine=original.SOURCE.read_text()
pattern=rf'(?im)^\s*call {component}\((.*?)\)\s*$'
match=re.search(pattern,pristine,re.S)
assert match is not None and len(re.findall(rf'(?im)^\s*call {component}\(',pristine))==1
actuals=[ast.unparse(a) for a in ast.parse('v('+re.sub(r'[\s&]+','',match[1])+')',mode='eval').body.args]
signature=re.search(rf'(?im)^\s*subroutine {component}\((.*?)\)',pristine,re.S)
names=re.sub(r'[\s&]+','',signature[1]).lower().split(',')
assert len(names)==len(actuals)==(45 if component=='vaporgrow' else 24)
if component=='vaporgrow':
    input_names=[n for n in names if n not in ('anf','cnf','rnf','iwcf')]
    output_names='vtbarb vtbarbm vtbarbz anf cnf rnf iwcf fvdum fhdum rbdum rdout dsdumout'.split()
else:
    input_names=names[:16]
    output_names='qagg1 qagg2 qagg3 nagg1 nagg2 nagg3 ddum3'.split()
def values(selected):return ','.join(actuals[names.index(n)] for n in selected)
replaced=('\n call s2_vapor_record('+str(len(input_names))+',['+values(input_names)+'])\n'+match[0]+
          '\n call s2_vapor_record('+str(len(output_names))+',['+values(output_names)+'])\n')
source=pristine[:match.start()]+replaced+pristine[match.end():]
collection_names='eff efdum tmax denfac dn0 enmx enmy w1mx w2mx w1my w2my colamt colamtn wght deltan rmx dmx loglo dict logdiameter'.split()
if component=='aggregation':
    anchor='          enxfer(k,mx,mz) = enxfer(k,mx,mz) + deltan'
    collection_values='eff(k),efdum,tmax,denfac(k),dn0(k),en(k,mx),en(k,my),wct1(k,mx),wct2(k,mx),wct1(k,my),wct2(k,my),colamt,colamtn,wght,deltan,r(k,mx),dn(k,mx),tabloln(mx),dict(mx),log(max(1.e-10,dn(k,mx)))'
    assert source.count(anchor)==1
    # TABLOLN/DICT belong to aggregation, not COL1. Pass their actual
    # values through observer-only module slots for this E26 capture.
    source=source.replace('  !.. Inherent growth ratio data', '  real :: s2_loglo(7),s2_dict(7)\n\n  !.. Inherent growth ratio data',1)
    initializer='    rictmin=1.0001'
    assert source.count(initializer)==1
    source=source.replace(initializer,'    s2_loglo=tabloln\n    s2_dict=dict\n'+initializer)
    collection_values=collection_values.replace('tabloln(mx)','s2_loglo(mx)').replace('dict(mx)','s2_dict(mx)')
    source=source.replace(anchor,anchor+'\n call s2_collect_record(mx,my,ip,['+collection_values+'])')
observer='''
subroutine s2_vapor_record(n,values)
integer,intent(in) :: n
real,intent(in) :: values(n)
integer :: unit
open(newunit=unit,file='COMPONENT_calls.bin',access='stream',form='unformatted',status='unknown',position='append')
write(unit) n
write(unit) values
close(unit)
end subroutine
'''.replace('COMPONENT',tag)
if component=='aggregation':
    observer+='''
subroutine s2_collect_record(mx,my,ip,values)
integer,intent(in) :: mx,my,ip
real,intent(in) :: values(20)
integer :: unit
open(newunit=unit,file='collect_calls.bin',access='stream',form='unformatted',status='unknown',position='append')
write(unit) mx,my,ip
write(unit) values
close(unit)
end subroutine
'''
source,count=re.subn(r'(?im)^(\s*end\s+module\b.*)$',lambda m:observer+m[0],source);assert count==1
path=artifact/'observed.F';path.write_text(source);original.SOURCE=path
original.main(tree,artifact/'capture')
out=artifact/'fixtures';out.mkdir();rows=[]
for mode,R in [('fp32',np.float32),('fp64',np.float64)]:
    for case in range(1,18):
        assert (tree/f'proofs/s2_mpmore/jensen55/fixtures/{mode}_{case}.json').read_bytes()==(artifact/f'capture/fixtures/{mode}_{case}.json').read_bytes()
    raw=(artifact/f'capture/build/{mode}/{tag}_calls.bin').read_bytes()
    incoming=[];outgoing=[];offset=0;dtype=np.dtype(R).newbyteorder('>')
    while offset<len(raw):
        count=int(np.frombuffer(raw,'>i4',count=1,offset=offset)[0]);offset+=4
        assert count in (len(input_names),len(output_names))
        a=np.frombuffer(raw,dtype,count=count,offset=offset).astype(R);offset+=count*dtype.itemsize
        (incoming if count==len(input_names) else outgoing).append(a)
    assert offset==len(raw) and len(incoming)==len(outgoing)>0
    incoming,outgoing=np.stack(incoming),np.stack(outgoing)
    assert np.isfinite(incoming).all() and np.isfinite(outgoing).all()
    extra={}
    if component=='aggregation':
        raw=(artifact/f'capture/build/{mode}/collect_calls.bin').read_bytes();offset=0;ids=[];values=[]
        while offset<len(raw):
            ids.append(np.frombuffer(raw,'>i4',count=3,offset=offset).astype(np.int32));offset+=12
            values.append(np.frombuffer(raw,dtype,count=20,offset=offset).astype(R));offset+=20*dtype.itemsize
        assert offset==len(raw) and len(ids)==7*len(incoming)
        extra=dict(collection_indices=np.stack(ids).reshape(len(incoming),7,3),collection_values=np.stack(values).reshape(len(incoming),7,20))
    path=out/(mode+'.npz');np.savez_compressed(path,inputs=incoming,outputs=outgoing,**extra)
    regimes={}
    if component=='vaporgrow':
        regimes=dict(cold=int(np.count_nonzero(incoming[:,input_names.index('temp')]<=R(273.15))),
                     warm=int(np.count_nonzero(incoming[:,input_names.index('temp')]>R(273.15))),
                     sublimation=int(np.count_nonzero(incoming[:,input_names.index('sui')]<0)),
                     deposition=int(np.count_nonzero(incoming[:,input_names.index('sui')]>=0)))
    else:
        regimes={n:int(np.count_nonzero(outgoing[:,i])) for i,n in enumerate(output_names)}
    rows.append(dict(mode=mode,E26_BYTE_columns=17,calls=len(incoming),nonfinite=0,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest(),regimes=regimes))
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(pristine.encode()).hexdigest(),
    observed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),input_names=input_names,output_names=output_names,
    scope='Original actual caller before/after print-only; all34fullpubliccolumnsBYTE; warm RDOUT caller hold recorded explicitly',component=component,collection_names=collection_names,rows=rows),indent=2)+'\n')
print(rows,flush=True)
