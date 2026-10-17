"""Reuse measured native observations only with byte-identical compiled source.

Known excesses are explicitly a regression receipt, not a WRF-enclosure pass.
This test never expands the independent original bound to fit the port.
"""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'proofs/s2_mpmore/jensen55'


TYPED=b'from gpuwrf.kernels._column_memory import load as _load, store as _store\n'


def _without_typed_memory(source):
    # t-ntucompile: Triton-only typed global memory (tests/v040/
    # test_column_typed_memory.py proves identical interpretation). Accept ONLY
    # this exact, reversible substitution of the measured source.
    if TYPED not in source:return source
    assert source.count(TYPED)==1 and b'plt.load(ref.at[col,level]' not in source
    return source.replace(TYPED,b'').replace(b'_load(',b'plt.load(').replace(b'_store(',b'plt.store(')


def test_exact_source_native_observations_and_independent_bound_inventory():
    receipt=json.loads((DATA/'NATIVE_COMPILED_DEVIATIONS.json').read_text())
    assert receipt['CPU'] and not receipt['GPU'] and receipt['nonfinite_values']==0
    assert 'not a GPU/D6 fidelity pass' in receipt['classification']
    for path,sha in receipt['source_sha256'].items():
        source=(ROOT/path).read_bytes()
        if path=='src/gpuwrf/kernels/phys_jensen.py':
            # t-mainreds portability migration: retain the immutable measured
            # receipt and accept ONLY this import substitution, with no other
            # source/line changes. CUDA dispatch is the original Pallas object.
            old=b'from jax.experimental import pallas as pl\n'
            new=b'from gpuwrf.backends import pallas as pl\n'
            if new in source:
                assert source.count(new)==1 and old not in source
                from unittest.mock import patch
                from gpuwrf.backends import pallas as portable
                from jax.experimental import pallas
                with patch.object(portable,'cuda_native',return_value=True):
                    assert portable.pallas_call is pallas.pallas_call
                source=source.replace(new,old)
            source=_without_typed_memory(source)
        assert hashlib.sha256(source).hexdigest()==sha,path
    for path,h in receipt['source_sha256_history'].items():
        # r-pkg040: a re-pinned source must carry its identity proof (same table bytes, same lowered program)
        proof=json.loads((ROOT/h['proof']).read_text())
        assert receipt['source_sha256'][path]==h['current_sha256']==proof['source']['current_sha256']
        assert proof['verdict']=='IDENTICAL' and proof['source']['measured_sha256']==h['measured_sha256']
    pins=json.loads((ROOT/'src/gpuwrf/assets_manifest.json').read_text())['assets']
    for path,sha in receipt['table_sha256'].items():
        # > 10 MiB table: may be absent (public tree, wheel); the manifest pin is what gpuwrf.assets enforces on
        # fetched copies, and a repo copy must hash to it too
        assert pins[path]['sha256']==sha,path
        if (ROOT/path).is_file():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    observations=DATA/'native_operational_observations.npz'
    assert hashlib.sha256(observations.read_bytes()).hexdigest()==receipt['observations_sha256']
    with np.load(observations) as data:actual=data['outputs']
    manifest=json.loads((DATA/'column_controls/manifest.json').read_text());nz=40
    indices=np.concatenate([np.arange(manifest['output_names'].index(n)*nz,(manifest['output_names'].index(n)+1)*nz)
                            for n in receipt['output_names']]+
                           [np.arange(len(manifest['output_names'])*nz,len(manifest['output_names'])*nz+4)])
    refs=[]
    for mode in ('noFMA','FMA'):
        with np.load(DATA/'column_controls'/('fp32_'+mode+'.npz')) as d:refs.append(d['outputs'][:,:,indices])
    low=np.minimum.reduce([a.min(axis=0) for a in refs]);high=np.maximum.reduce([a.max(axis=0) for a in refs])
    bad=(actual<low)|(actual>high)
    expected={(d['case']-1,d['index']):d for d in receipt['deviations']}
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    assert set(zip(*np.where(bad)))==set(expected)
    assert int(bad.sum())==receipt['outside_original_enclosure']==335
    for (c,i),d in expected.items():
        assert float(actual[c,i])==d['value'] and float(low[c,i])==d['low'] and float(high[c,i])==d['high']
