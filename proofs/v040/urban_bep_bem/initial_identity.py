"""Hash every initial dynamic input to the smoke's exact production chunk."""
from pathlib import Path

p=Path(__file__).with_name('coupled_smoke.py')
source=p.read_text()
begin=source.index('  carry=om._advance_chunk(')
end=source.index("  record['status']='PASS_CPU_SMOKE_ONLY'",begin)
source=source[:begin]+'''  digest=hashlib.sha256();leaves=[]
  for index,value in enumerate(jax.tree.leaves(carry)):
    value=np.asarray(value)
    payload=value.tobytes(order='C')
    metadata=(str(value.dtype),value.shape)
    digest.update(repr(metadata).encode());digest.update(payload)
    leaves.append(dict(index=index,dtype=str(value.dtype),shape=value.shape,sha256=hashlib.sha256(payload).hexdigest()))
  record['initial_carry_sha256']=digest.hexdigest()
  record['initial_leaves']=leaves
  record['status']='PASS_INITIAL_ONLY'
'''+source[end+len("  record['status']='PASS_CPU_SMOKE_ONLY'\n"):]
exec(compile(source,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
