import sys
from pathlib import Path
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/fix_review/snapshot')
sys.path[:0]=[str(ROOT/'scripts/v025'),str(ROOT/'src'),str(ROOT/'tests')]
import cpu_guard
import pytest
tag=sys.argv[1]
if tag=='F1':
    import gpuwrf.runtime.restart_store as store
    store._validate_urban_bep_snapshot=lambda *a,**kw:None
    selection=[str(ROOT/'tests/v040/test_classd_urban_green_roof.py')+'::test_native_checkpoint_refuses_bad_qgr_on_save_and_authenticated_read[missing]']
else:
    import jax.numpy as jnp
    import gpuwrf.physics.nssl2mom.diagnostics as Dm
    Dm._log10_real=lambda x,prec:jnp.log10(x)
    selection=[str(ROOT/'tests/v040/test_nssl_one_moment_radar.py'),'-k','matches_pristine_s6 and fp32']
sys.exit(pytest.main(['-q','-p','no:cacheprovider',*selection,
    '--basetemp','<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/pytest',
    '--junitxml',str(ROOT.parent/f'RV18_{tag}_mutant.xml')]))
