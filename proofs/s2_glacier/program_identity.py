"""Official native-facade graph identity plus initial-leaf/constant bytes."""
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve()
if len(sys.argv)>3:
    import os
    os.environ['O1_CASE']=sys.argv[3]
sys.path[:0]=[str(root/'src'),str(root/'proofs/o1_nlbind')]
from gpuwrf.backends import pallas as facade
from gpuwrf.integration import nested_pipeline as pipeline
original=pipeline._load_domains
def load(*args,**kwargs):
    facade.cuda_native=lambda:False
    try:return original(*args,**kwargs)
    finally:facade.cuda_native=lambda:True
pipeline._load_domains=load
probe=root/'proofs/o1_nlbind/probe.py';code=probe.read_text()
code=code.replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")','LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s2-glacier")')
needle='        rec["sha"] = hashlib.sha256(text.encode()).hexdigest()'
assert code.count(needle)==1
extra='''        def tree_digest(tree):
            h=hashlib.sha256()
            leaves=jax.tree.leaves(tree)
            for a in leaves:
                a=np.asarray(a)
                h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
            return {"leaves":len(leaves),"sha256":h.hexdigest()}
        rec["initial_carry_bytes"]=tree_digest(carry)
        rec["closed_constants_bytes"]=tree_digest(jx.consts)
'''
code=code.replace(needle,extra+needle)
sys.argv=[str(probe),str(root),str(output),'release']
exec(compile(code,str(probe),'exec'),{'__file__':str(probe),'__name__':'__main__'})
