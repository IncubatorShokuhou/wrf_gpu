"""Rescore saved native diagnostics against unchanged original WRF controls.

No JAX execution, new compiler arm, tuned bound or product arithmetic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    report=json.loads(a.report.read_text())
    assert report['existing_native_outputs_BYTE'] and report['mp']==55
    folder=root/'proofs/s2_mpmore/jensen55/column_controls'
    manifest=json.loads((folder/'manifest.json').read_text())
    controls=[];hashes={}
    for mode in ('noFMA','FMA'):
        path=folder/f'fp32_{mode}.npz'
        hashes[mode]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as data:controls.append(data['outputs'])
    result=dict(source=report['source'],status='DISCLOSED_EXCESS_REQUIRES_SAME_CRITIC',
        original_controls_sha256=hashes,fields={},deviations=[],
        classification='Strict unchanged original input-ULP/FMA enclosure; no bound widening. '
        'Large absolute nominal differences can lie INSIDE the original interval; '
        'the listed excesses and bit distances are outside that interval.')
    with np.load(a.report.with_suffix('.npz')) as actual:
        for name,score in report['fields'].items():
            value=actual[name];nz=value.shape[-1]
            j=manifest['output_names'].index(name);sl=slice(j*nz,(j+1)*nz)
            lo=np.minimum.reduce([x.min(axis=0)[:,sl] for x in controls])
            hi=np.maximum.reduce([x.max(axis=0)[:,sl] for x in controls])
            assert all(np.isfinite(x).all() for x in (value,lo,hi)) and np.all(lo<=hi)
            bad=(value<lo)|(value>hi)
            assert int(bad.sum())==score['outside_original_enclosure']
            nearest=np.where(value<lo,lo,hi)[bad].astype(np.float32)
            got=value[bad].astype(np.float32)
            assert np.all(nearest>=0) and np.all(got>=0)
            bits=np.abs(got.view(np.uint32).astype(np.int64)-nearest.view(np.uint32).astype(np.int64))
            excess=np.abs(got.astype(np.float64)-nearest.astype(np.float64))
            relative=excess[nearest>0]/nearest[nearest>0].astype(np.float64)
            result['fields'][name]=dict(score,max_boundary_bit_distance=int(bits.max(initial=0)),
                max_relative_excess=float(relative.max(initial=0)),
                zero_boundary_excesses=int(np.count_nonzero(nearest==0)))
            for index,((case,k),distance) in enumerate(zip(np.argwhere(bad),bits,strict=True)):
                result['deviations'].append(dict(field=name,case=int(case+1),WRF_level=int(k+1),
                    actual=float(value[case,k]),low=float(lo[case,k]),high=float(hi[case,k]),
                    excess=float(excess[index]),boundary_bit_distance=int(distance)))
    result['outside_values']=len(result['deviations'])
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['outside_values'],{n:(v['max_boundary_bit_distance'],v['max_relative_excess'])
                                   for n,v in result['fields'].items()},flush=True)


if __name__=='__main__':main()
