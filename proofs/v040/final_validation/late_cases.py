"""One primary Swiss reference per newly admitted MP, packing Noah-MP choices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import numpy as np
from prepare import ROOT, RETAINED, SLOTS, private_executable, sha, write_json
from extra_cases import stage
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.scheme_catalog import _IMPLEMENTED


def hail_table(source):
    """Restore original Fortran record bytes; no quadrature or table alteration."""
    target = ROOT / 'original_tables/qr_acr_qg_mp38V1.dat'
    target.parent.mkdir(exist_ok=True)
    proof = source / 'proofs/v040/mpmisc/thompson38/collision_tables.json'
    expected = json.loads(proof.read_text())['source_sha256']
    if not target.exists():
        shape = (37,37,9,37,37)
        length = int(np.prod(shape))*8
        with target.open('wb') as stream:
            for name in ('tcg_racg','tmr_racg','tcr_gacr','tnr_racg','tnr_gacr'):
                values = np.empty(shape,dtype='>f8',order='F')
                for density in range(9):
                    with np.load(source/f'src/gpuwrf/physics/thompson_hail/collision_tables_{density}.npz',allow_pickle=False) as packed:
                        values[:,:,density,:,:] = packed[name]
                stream.write(struct.pack('>i',length))
                stream.write(values.ravel(order='F').view(np.uint8))
                stream.write(struct.pack('>i',length))
                del values
    assert sha(target) == expected
    write_json(target.with_suffix('.json'),dict(original_raw_sha256=expected,bytes=target.stat().st_size,
        restoration='lossless original-WRF collision table records; no numerical table edits',
        source_proof_sha256=sha(proof)))
    return target


def prepare(source):
    matrix = json.loads((ROOT/'MATRIX.json').read_text())
    swiss = RETAINED/'source_inputs/Swiss'
    rows = [
        ('c62',5,dict(dveg=1,opt_run=1,opt_crs=2,opt_btr=2,opt_sfc=2,opt_frz=2,opt_inf=2,
                     opt_rad=1,opt_alb=1,opt_snf=2,opt_tbot=1,opt_stc=2,opt_rsf=2,opt_gla=2)),
        ('c63',15,dict(dveg=2,opt_run=2,opt_btr=3,opt_rad=2,opt_snf=3,opt_stc=3,opt_rsf=3)),
        ('c64',7,dict(dveg=3,opt_run=4,opt_snf=4,opt_rsf=4)),
        ('c65',38,dict(dveg=5,opt_run=6)),
    ]
    for ident,mp,options in rows:
        if any(r['id']==ident for r in matrix['cases']): continue
        assert mp in _IMPLEMENTED['mp_physics'], 'Wait for manager-admitted MP merge'
        groups = parse_namelist(source/'examples/switzerland_d01/namelist.input')
        groups['physics']['mp_physics'] = mp
        groups['noah_mp'] = options
        row = dict(id=ident,geography='Swiss',hours=24,domains=['d01'],
            options={k:groups['physics'][k] for k in SLOTS},extra={},
            features=[f'primary-MP{mp}','Noah-MP:'+json.dumps(options,separators=(',',':'))])
        row = stage(row,swiss,groups)
        private_executable(ROOT/ident,ROOT/'reference_zero')
        if mp==38:
            for name,target in (('qr_acr_qg_mp38V1.dat',hail_table(source)),
                                ('qr_acr_qsV2.dat',RETAINED/'bl9/cpu/qr_acr_qsV2.dat'),
                                ('freezeH2O.dat',RETAINED/'bl9/cpu/freezeH2O.dat')):
                (ROOT/ident/'cpu'/name).symlink_to(target)
            row['original_table_sha256']={p.name:sha(p) for p in (ROOT/ident/'cpu').glob('*.dat')}
            row['gpu_schedule_class']='standalone-heavy-mp38'
        row['reference_semantics']='compiler-defined zero, inactive canonical Morrison-aero object'
        write_json(ROOT/ident/'inputs.json',row)
        matrix['cases'].append(row);write_json(ROOT/'MATRIX.json',matrix)
        print(json.dumps(dict(id=ident,errors=row['admission_errors'])),flush=True)


if __name__=='__main__':
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path)
    prepare(parser.parse_args().source.resolve())
