"""Trace the smoke's exact production chunk to bind a prior execution by identity.

All arguments/CPU emulation/case loading come from coupled_smoke.py. This only
replaces execution/finite assertions with a location-stripped jaxpr checksum.
"""
from pathlib import Path

p=Path(__file__).with_name('coupled_smoke.py')
source=p.read_text()
begin=source.index('  carry=om._advance_chunk(')
end=source.index("  record['status']='PASS_CPU_SMOKE_ONLY'",begin)
source=source[:begin]+'''  jx=jax.make_jaxpr(lambda c:om._advance_chunk(c,ns,1,cb,n_steps=a.steps,cadence=int(ns.radiation_cadence_steps)))(carry)
  text=str(jx)
  text=re.sub(r'name_and_src_info=[^\\]\\n]*','',text)
  text=re.sub(r'/[^ \\n:]*\\.py:\\d+(:\\d+)?','',text)
  record['program_sha256']=hashlib.sha256(text.encode()).hexdigest()
  record['program_chars']=len(text)
  record['status']='PASS_TRACE_ONLY'
'''+source[end+len("  record['status']='PASS_CPU_SMOKE_ONLY'\n"):]
exec(compile(source,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
