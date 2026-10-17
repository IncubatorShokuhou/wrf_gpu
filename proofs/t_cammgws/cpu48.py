"""CPU48 gate for a structural variant: proofs/s2_cammg/probe_native.py (24 frozen
pristine cases x 2 steps, interpret-mode native column, UNCHANGED enclosure) run
after variants.install(). usage: cpu48.py TREE OUT.json VARIANT"""
from pathlib import Path
import sys

tree=Path(sys.argv[1]);here=Path(__file__).resolve().parent
sys.path[:0]=[str(tree/'src'),str(tree/'proofs/s2_cammg'),str(tree/'proofs/t_cammgws')]
from variants import install
print('VARIANT',install(sys.argv[3]),flush=True)
import probe_native
sys.argv=[sys.argv[0],str(tree),sys.argv[2]]
probe_native.main()
