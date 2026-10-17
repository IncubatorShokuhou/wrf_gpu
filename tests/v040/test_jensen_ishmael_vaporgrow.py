"""Literal vapor growth on actual pristine WRF call boundaries."""
from pathlib import Path
import hashlib,importlib.util,json
import jax
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'proofs/s2_mpmore/jensen55'
spec=importlib.util.spec_from_file_location('vapor_libm',DATA/'libm_numpy.py')
libm=importlib.util.module_from_spec(spec);spec.loader.exec_module(libm)
jax.config.update('jax_enable_x64',True)


def load(mode):
    manifest=json.loads((DATA/'vapor/manifest.json').read_text())
    row=next(r for r in manifest['rows'] if r['mode']==mode)
    path=DATA/'vapor'/(mode+'.npz')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    assert row['E26_BYTE_columns']==17 and row['nonfinite']==0
    assert all(row['regimes'][k]>0 for k in ('warm','cold','sublimation','deposition'))
    with np.load(path) as data:d={k:data[k].copy() for k in data.files}
    with np.load(DATA/'stages'/(mode+'.npz')) as data:gamma=data['gamma_table'].copy()
    return manifest,d,gamma


@pytest.mark.parametrize('mode',['fp32','fp64'])
def test_original_actual_vaporgrow_literal_BYTE(mode):
    from gpuwrf.physics.jensen_ishmael.vaporgrow import vaporgrow
    m,d,gamma=load(mode)
    fields={k:d['inputs'][:,i] for i,k in enumerate(m['input_names'])}
    with np.errstate(all='ignore'):got=vaporgrow(fields,gamma,xp=libm.backend)
    failures=[]
    for name,value in got.items():
        ref=d['outputs'][:,m['output_names'].index(name)]
        assert np.isfinite(value).all() and value.dtype==ref.dtype,(name,value.dtype,ref.dtype)
        if value.tobytes()!=ref.tobytes():
            bad=np.flatnonzero(value!=ref)
            failures.append((name,len(bad),bad[:10],value[bad[:10]],ref[bad[:10]]))
    assert not failures,failures
    warm=fields['temp']>fields['temp'].dtype.type(273.15)
    assert got['rdout'][warm].tobytes()==fields['rdout'][warm].tobytes(),'Original warm branch leaves caller RDOUT held'


@pytest.mark.parametrize('mode',['fp32','fp64'])
def test_actual_vaporgrow_original_enclosure_and_disclosed_compiled_deviations(mode):
    from gpuwrf.physics.jensen_ishmael.vaporgrow import vaporgrow
    m,d,gamma=load(mode);controls=DATA/'vapor_controls';cm=json.loads((controls/'manifest.json').read_text())
    assert cm['input_names']==m['input_names'] and cm['output_names']==m['output_names']
    rows=[r for r in cm['rows'] if r['mode']==mode]
    assert len({r['input_sha256'] for r in rows})==1
    all_outputs=[]
    for contraction in ('noFMA','FMA'):
        row=next(r for r in rows if r['contraction']==contraction);p=controls/(mode+'_'+contraction+'.npz')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        assert (row['FMA_opcode_rows']==0)==(contraction=='noFMA')
        with np.load(p) as data:x=data['inputs'];y=data['outputs']
        assert x[0].tobytes()==d['inputs'].tobytes()
        expected=[d['inputs']]
        for name in cm['perturbed_arguments']:
            n=m['input_names'].index(name)
            for direction in (-np.inf,np.inf):
                changed=d['inputs'].copy();changed[:,n]=np.nextafter(changed[:,n],x.dtype.type(direction));expected.append(changed)
        expected=np.stack(expected)
        assert expected.tobytes()==x.tobytes(),'Each perturbation is one actual original argument ULP'
        raw=np.asarray([x.shape[0]*x.shape[1]],'>i4').tobytes()+x.astype(x.dtype.newbyteorder('>')).tobytes()
        assert hashlib.sha256(raw).hexdigest()==row['input_sha256']
        assert np.isfinite(y).all()
        if contraction=='noFMA':assert y[0].tobytes()==d['outputs'].tobytes()
        all_outputs.append(y)
    low=np.minimum.reduce([y.min(axis=0) for y in all_outputs]);high=np.maximum.reduce([y.max(axis=0) for y in all_outputs])
    fields={k:d['inputs'][:,i] for i,k in enumerate(m['input_names'])}
    got=jax.tree.map(np.asarray,jax.jit(vaporgrow)(fields,gamma))
    # The standing compiler policy permits small, explicitly recorded excess.
    # This block is a regression seal of those deviations, NOT an independent
    # fidelity bound. Every other value must still fit the original enclosure.
    disclosed=json.loads((DATA/'VAPOR_COMPILED_DEVIATIONS.json').read_text())['modes'][mode]
    observed={}
    for name,value in got.items():
        n=m['output_names'].index(name);lo=low[:,n];hi=high[:,n]
        assert np.isfinite(value).all() and value.dtype==lo.dtype
        bad=(value<lo)|(value>hi)
        if bad.any():
            observed[name]=dict(rows=np.flatnonzero(bad).tolist(),got=value[bad].tolist(),low=lo[bad].tolist(),high=hi[bad].tolist())
    assert set(observed)==set(disclosed),'Unexpected original-bound deviation fields'
    for name,values in observed.items():
        for key,value in values.items():
            assert value==disclosed[name][key],(name,key,'Compiled deviation changed; remeasure against pristine WRF')
