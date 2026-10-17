"""Locate compiled ZENITH divergence with unchanged-source diagnostic writes.

The Python diagnostic changes only the return value of the SAME port function;
the Fortran diagnostic adds one write after the corresponding source stages.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

FORTRAN=('DAY','DAYI','HOUR','YFCTR','STARTYR','DATJUL','DIFJD','SLONM',
         'ANOM','SLON','DEC','RA','DATJ0','TU','STIM0','SIDTIM','HRANG')
PYTHON=('elapsed','dayi','hour','yfctr','start','datjul','difjd','slonm',
        'anomaly','slon','dec','ra','datj0','tu','stim0','sidtim','hrang')


def instrument(source,driver):
    original=source.read_text()
    marker='      HRANG=SIDTIM-RA\n'
    dump='      WRITE(94) '+','.join(FORTRAN)+'\n'
    assert original.count(marker)==1
    modified=original.replace(marker,marker+dump)
    assert modified.replace(dump,'')==original
    source.write_text(modified)
    text=driver.read_text()
    marker="  do i=1,16\n"
    assert text.count(marker)==1
    text=text.replace(marker,"  open(unit=94,file=trim(output)//'.stages',access='stream',form='unformatted',status='replace')\n"+marker)
    driver.write_text(text)
    source.with_suffix('.instrumentation.json').write_text(json.dumps(dict(
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        instrumented_sha256=hashlib.sha256(modified.encode()).hexdigest(),
        stripping_write_restores_original=True,scope='one diagnostic write; verify final outputs BYTE FG19'),indent=2)+'\n')


def qualify(native,stages,output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf.physics import ra_gfdleta_support as support
    assert jax.devices()[0].platform=='cpu'
    tree=ast.parse(inspect.getsource(support.zenith))
    # HRANG exists in native source; the port expression is currently inlined.
    function=tree.body[0]
    function.body.insert(-1,ast.parse('hrang = sidtim-ra').body[0])
    function.body[-1]=ast.Return(ast.Tuple([ast.Name(x,ast.Load()) for x in PYTHON],ast.Load()))
    ast.fix_missing_locations(tree)
    namespace=dict(vars(support));exec(compile(tree,'<same-port-with-stage-return>','exec'),namespace)
    run=jax.jit(namespace['zenith'],static_argnames=('year','month','day','start_hour'))
    values=[]
    with native.open('rb') as f:
        np.fromfile(f,np.float32,633)
        for _ in range(112):
            month,day,year,hour=np.fromfile(f,np.int32,4)
            times,lat,lon,*_=np.fromfile(f,np.float32,6)
            values.append(run(jnp.asarray(times),jnp.asarray(lon),jnp.asarray(lat),
                year=int(year),month=int(month),day=int(day),start_hour=int(hour)))
        assert not f.read(1)
    candidate=np.asarray(values,np.float32)
    reference=np.fromfile(stages,np.float32).reshape(112,len(FORTRAN))
    assert np.isfinite(candidate).all() and np.isfinite(reference).all()
    rows={}
    for i,key in enumerate(FORTRAN):
        err=np.abs(candidate[:,i].astype(np.float64)-reference[:,i].astype(np.float64))
        rows[key]=dict(max_abs=float(err.max()),mismatches=int(np.count_nonzero(err)))
    output.write_text(json.dumps(dict(stages=rows,scope='compiled CPU stage-return diagnostic; no final-physics acceptance'),indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'),native=reference,compiled=candidate)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True)
    i=s.add_parser('instrument');i.add_argument('source',type=Path);i.add_argument('driver',type=Path)
    q=s.add_parser('qualify');q.add_argument('native',type=Path);q.add_argument('stages',type=Path);q.add_argument('output',type=Path)
    a=p.parse_args()
    if a.mode=='instrument':instrument(a.source,a.driver)
    else:qualify(a.native,a.stages,a.output)
