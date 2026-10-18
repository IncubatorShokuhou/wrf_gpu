"""Retain scientific observations only while numerical source is identical."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
PROOF=ROOT/'proofs/s2_ntu'
sys.path.insert(0,str(PROOF))
from compare_compiled import compare, code_hash


TYPED='from gpuwrf.kernels._column_memory import load as _load, store as _store\n'


FACADE='from gpuwrf.backends import pallas as pl\n'
STOCK='from jax.experimental import pallas as pl\n'


def _measured_source(path,tmp_path):
    # t-ntucompile: Triton-only typed global memory (tests/v040/
    # test_column_typed_memory.py proves identical interpretation). Accept ONLY
    # this exact, reversible substitution of the measured kernel source.
    text=(ROOT/path).read_text()
    if path!='src/gpuwrf/kernels/phys_ntu.py' or TYPED not in text:return ROOT/path
    # w41-redfix: the portability facade import. On CUDA gpuwrf.backends.pallas returns the
    # stock pallas_call object; the mp56 program lowered for CUDA is byte-identical with
    # either import (jaxpr + StableHLO, proofs/w41_redfix/FACADE_IMPORT_CUDA_LOWERING.json).
    assert text.count(FACADE)==1 and STOCK not in text
    text=text.replace(FACADE,STOCK)
    assert text.count(TYPED)==1 and 'plt.load(ref.at[col,index]' not in text
    # t-ntucompile G05: 2 warps (no replicated level lanes) gives BITWISE outputs on
    # real c88 GPU columns and is faster; Triton-only, interpretation ignores it.
    WARPS='plt.CompilerParams(num_warps=2)'
    assert text.count(WARPS)==1
    original=tmp_path/'phys_ntu.py'
    original.write_text(text.replace(TYPED,'').replace('_load(','plt.load(').replace('_store(','plt.store(')
                        .replace(WARPS,'plt.CompilerParams(num_warps=4)'))
    return original


def _byte_proven(path,live,receipt):
    # t-ntucompile NaN shortcut in find_rc0: the rerun of the frozen observations
    # with the changed math.py is bitwise identical (912/912, same NPZ SHA).
    proof=json.loads((ROOT/'proofs/t_ntucompile/NAN_SHORTCUT_BYTE.json').read_text())
    if (path=='src/gpuwrf/physics/ntu/math.py' and live==proof['candidate_math_code_hash']
            and proof['measured_source_math_sha256']==receipt['source_code_sha256'][path]
            and proof['candidate_observations_sha256']==proof['frozen_observations_sha256']==receipt['observations_sha256']
            and proof['bitwise']['bitwise_equal'] and proof['bitwise']['arrays']==912):
        return receipt['source_code_sha256'][path]
    # t-ntucompile small_dt skip (support.masked_call + regenerated advance_small call):
    # same frozen-observation rerun, bitwise 912/912 with the same NPZ SHA.
    skip=json.loads((ROOT/'proofs/t_ntucompile/SKIP_SMALL_DT_BYTE.json').read_text())
    if (live==skip['candidate_code_hash'].get(path)
            and skip['measured_source_code_hash'][path]==receipt['source_code_sha256'][path]
            and skip['candidate_observations_sha256']==skip['frozen_observations_sha256']==receipt['observations_sha256']
            and skip['bitwise']['bitwise_equal'] and skip['bitwise']['arrays']==912):
        return receipt['source_code_sha256'][path]
    return live


def test_fixed_original_enclosure_and_all_compiled_excesses_reproduce(tmp_path):
    receipt=json.loads((PROOF/'COMPILED_DEVIATIONS.json').read_text())
    for path,sha in receipt['source_code_sha256'].items():
        assert _byte_proven(path,code_hash(_measured_source(path,tmp_path)),receipt)==sha,path
    actual=compare(PROOF/'compiler_controls',PROOF/'native_public_observations.npz')
    live=actual.pop('source_code_sha256');expected=dict(receipt);pinned=expected.pop('source_code_sha256')
    assert {p:_byte_proven(p,code_hash(_measured_source(p,tmp_path)),receipt) for p in live}==pinned
    assert all(_byte_proven(p,live[p],receipt)==pinned[p] for p in live if p!='src/gpuwrf/kernels/phys_ntu.py')
    assert actual==expected
    assert receipt['CPU'] and not receipt['GPU'] and receipt['nonfinite_values']==0
    assert receipt['outside_original_enclosure']==313
    assert 'NOT an original-enclosure or GPU/D6 pass' in receipt['classification']


def test_production_native_shape_and_original_double_attribution():
    r=json.loads((PROOF/'NATIVE_PREFLIGHT.json').read_text())
    assert r['backend']=='cpu' and r['target']=='cuda/sm120'
    assert r['levels']==44 and r['register_tile_levels']==64
    assert r['operational_projection'] and r['ambient_x64_restored']
    assert r['fp64_equations']>0 and not r['fp64_outside_original_DOUBLE']
    assert r['float64_tables']=={'tblrc':[25],'tblxf':[25,2]}
