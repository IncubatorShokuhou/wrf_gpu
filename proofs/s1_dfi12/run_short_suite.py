"""Run the three short WRF phase witnesses sequentially in one CPU JAX process.

Usage: DIAG_INTERPRET=1 ... python run_short_suite.py IMMUTABLE_SOURCE
No flag changes; reuse compiled executables only through normal JAX keys.
"""
import runpy
import sys
from pathlib import Path
import jax.experimental.pallas as pl
import jax.experimental.pallas.triton as pt

source = Path(sys.argv[1])
lane = Path('<USER_HOME>/wrf_gpu2_lanes/s1-dfi12')
original_pc, original_asm = pl.pallas_call, pt.elementwise_inline_asm
for kind in ('dfl-default','dfl-endonly','ddfi'):
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    tag = 'DF12-short-' + kind
    oracle = lane/'oracle'/f'{kind}-short'
    print('START',kind,flush=True)
    sys.argv = [str(source/'proofs/s1_dfi12/run_e2e.py'), str(source), tag, str(oracle)]
    runpy.run_path(sys.argv[0],run_name='__main__')
    pl.pallas_call, pt.elementwise_inline_asm = original_pc, original_asm
    sys.argv = [str(source/'proofs/s1_dfi12/compare_filtered.py'),
                str(lane/tag/'work/fst_input/wrfinput_d01'), str(oracle/'wrfinput_initialized_d01'),
                str(oracle/'wrfinput_d01'), str(lane/tag/'compare.json')]
    try:
        runpy.run_path(sys.argv[0],run_name='__main__')
    except SystemExit as exc:
        if exc.code:
            raise
    print('PASS',kind,flush=True)
