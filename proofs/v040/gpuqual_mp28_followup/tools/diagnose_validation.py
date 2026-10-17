import importlib.util,json,sys,traceback
print('queue_origin',importlib.util.find_spec('queue').origin,flush=True)
assert '/v-run/tools/' not in importlib.util.find_spec('queue').origin
import gpuwrf
import jax
assert jax.devices()[0].platform=='cpu'
import gpuwrf.io.namelist_check as checker
original=checker.validate_operational_namelist
def traced(*a,**k):
 try:return original(*a,**k)
 except BaseException:
  traceback.print_exc();raise
checker.validate_operational_namelist=traced
checker.validate_operational_namelist(sys.argv[1])
checker.apply_cldovrlp(sys.argv[1])
from gpuwrf.cli import main
rc=main(['run','--input-dir',sys.argv[2],'--output-dir',sys.argv[3],'--hours','24','--dry-run'])
assert rc==0,rc
print('VALIDATE+CLI_DRY_RUN_PASS',sys.argv[2],flush=True)
