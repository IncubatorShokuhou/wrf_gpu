"""Original Jensen component gates; mp55 remains operationally refused."""
from pathlib import Path
import importlib.util,json
import hashlib,os,time

import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64',True)
DATA=Path(__file__).resolve().parents[2]/'proofs/s2_mpmore/jensen55/math_fixtures'
_spec=importlib.util.spec_from_file_location('jensen_compiler_libm',DATA.parent/'libm_numpy.py')
_libm=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_libm)


@pytest.mark.parametrize('mode',['fp32','fp64'])
def test_original_math_literal(mode):
    from gpuwrf.physics.jensen_ishmael import math as m
    d=json.loads((DATA/(mode+'.json')).read_text());R=np.float32 if mode=='fp32' else np.float64
    rows=np.asarray(d['inputs'],R);ref=np.asarray(d['outputs'],R)
    t,x,ds,a,rime=rows.T
    alph=ref[:,5]  # Actual original caller boundary; not a copied calculation.
    xp=_libm.backend
    got=[m.polysvp(t,0,xp=xp),m.polysvp(t,1,xp=xp),m.gamma(x,xp=xp),
         m.gammln(np.abs(x)+R(.5),xp=xp),m.get_igr(t,xp=xp),alph,
         m.capacitance_gamma(a,ds,R(4.),alph,R(1.)/R(6.),xp=xp),
         m.wet_growth_check(R(4.),t,R(1.),R(2.5e6),R(3.34e5),R(.005),R(2.e-5),R(.025),R(.0038),
                            R(1.4),R(1.2),rime,R(3.e-4),R(1.e4),np.ones_like(t,bool),xp=xp).astype(R)]
    failures=[]
    for index,value in enumerate(got):
        if index==5:
            continue  # Original caller boundary input to capacitance, not a port output.
        assert value.dtype==ref.dtype,(mode,d['output_names'][index],value.dtype,ref.dtype)
        if value.tobytes()!=ref[:,index].tobytes():
            bad=value!=ref[:,index]
            failures.append((d['output_names'][index],np.flatnonzero(bad)[:20],value[bad],ref[:,index][bad]))
    assert not failures,(mode,failures)
    assert set(got[-1])=={R(0.),R(1.)},'Wet/dry growth oracle must be nonvacuous'


@pytest.mark.parametrize('mode',['fp32','fp64'])
def test_original_math_compiled_source_enclosure(mode):
    from gpuwrf.physics.jensen_ishmael import math as m
    import jax.numpy as jnp
    folder=DATA.parent/'math_controls'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['source_sha256']==json.loads((DATA/'manifest.json').read_text())['source_sha256']
    R=np.float32 if mode=='fp32' else np.float64
    controls=[];input_hashes=set()
    for row in manifest['rows']:
        if row['mode']!=mode:continue
        file=folder/f"{mode}_{row['contraction']}.json"
        assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256']
        assert (row['FMA_opcode_rows']==0)==(row['contraction']=='noFMA')
        c=json.loads(file.read_text());controls.append(np.asarray(c['outputs'],R))
        inputs=np.asarray(c['baseline'],R)
        raw_inputs=np.asarray(c['inputs'],R)
        raw=np.asarray([len(raw_inputs)],dtype='>i4').tobytes()+raw_inputs.astype(np.dtype(R).newbyteorder('>')).tobytes()
        assert hashlib.sha256(raw).hexdigest()==row['input_sha256']
        blocks=raw_inputs.reshape(15,len(inputs),7)
        assert blocks[0].tobytes()==inputs.tobytes()
        for index in range(7):
            for direction,offset in ((-np.inf,1),(np.inf,2)):
                expected=inputs.copy();expected[:,index]=np.nextafter(expected[:,index],R(direction))
                assert blocks[2*index+offset].tobytes()==expected.tobytes()
        input_hashes.add(row['input_sha256'])
        if row['contraction']=='noFMA':
            golden=np.asarray(json.loads((DATA/(mode+'.json')).read_text())['outputs'],R)
            assert controls[-1][0].tobytes()==golden.tobytes()
    assert len(controls)==2 and len(input_hashes)==1
    def call(fields):
        t,x,ds,a,rime,alph,xx=[fields[:,i] for i in range(7)]
        return (m.polysvp(t,0),m.polysvp(t,1),m.gamma(x),m.gammln(xx),m.get_igr(t),
                m.capacitance_gamma(a,ds,R(4.),alph,R(1.)/R(6.)),
                m.wet_growth_check(R(4.),t,R(1.),R(2.5e6),R(3.34e5),R(.005),R(2.e-5),R(.025),R(.0038),
                                   R(1.4),R(1.2),rime,R(3.e-4),R(1.e4),jnp.ones_like(t,bool)).astype(R))
    start=time.monotonic();executable=jax.jit(call).lower(inputs).compile();seconds=time.monotonic()-start
    got=jax.tree.map(np.asarray,executable(inputs))
    envelope=np.stack(controls);lo=envelope.min(axis=(0,1));hi=envelope.max(axis=(0,1))
    for index,value in zip((0,1,2,3,4,6,7),got):
        assert value.dtype==inputs.dtype and np.isfinite(value).all()
        bad=(value<lo[:,index])|(value>hi[:,index])
        assert not bad.any(),(mode,index,np.flatnonzero(bad),value[bad],lo[bad,index],hi[bad,index])
    if os.environ.get('GPUWRF_JENSEN_COST_JSON'):
        p=Path(os.environ['GPUWRF_JENSEN_COST_JSON']);cost=json.loads(p.read_text()) if p.exists() else {}
        memory=executable.memory_analysis()
        cost[mode]=dict(CPU_compile_s=seconds,arguments_B=memory.argument_size_in_bytes,
                        outputs_B=memory.output_size_in_bytes,temps_B=memory.temp_size_in_bytes)
        p.write_text(json.dumps(cost,indent=2)+'\n')
