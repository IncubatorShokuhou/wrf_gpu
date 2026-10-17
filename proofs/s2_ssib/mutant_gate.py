"""Genuine E202 deletion mutant: replace product SSIB equations by identity.

Run on CPU with outer QUIET guard/core pin/timeout. The validation backend has
no JAX runtime. Only _ssib changes; every other production-source byte stays.
"""
import argparse,ast,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
assert args.output.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/s2-ssib'))
assert not Path('/tmp/wrf_gpu2_quiet').exists()
private=args.output.parent/'SS51_mutant_tree'
private.mkdir();(private/'src/gpuwrf/physics').mkdir(parents=True)
(private/'proofs').mkdir();(private/'proofs/s2_ssib').symlink_to((args.tree/'proofs/s2_ssib').resolve())
source=args.tree/'src/gpuwrf/physics/_ssib_generated.py'
original=source.read_text();tree=ast.parse(original)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ssib')
lines=original.splitlines(keepends=True)
mutated=''.join(lines[:fn.lineno])+'    return ('+', '.join(a.arg for a in fn.args.args)+',)\n'+''.join(lines[fn.end_lineno:])
(private/'src/gpuwrf/physics/_ssib_generated.py').write_text(mutated)
new=ast.parse(mutated)
assert all(ast.dump(a)==ast.dump(b) for a,b in zip(tree.body,new.body) if getattr(a,'name',None)!='_ssib')
try:
 result=subprocess.run([sys.executable,str(args.tree/'proofs/s2_ssib/literal_gate.py'),str(private),str(args.output.parent/'SS51_mutant_comparison.json')],capture_output=True,text=True,timeout=120)
 (args.output.parent/'SS51_mutant.stdout').write_text(result.stdout)
 (args.output.parent/'SS51_mutant.stderr').write_text(result.stderr)
 comparison=json.loads((args.output.parent/'SS51_mutant_comparison.json').read_text())
 receipt={'mutation':'actual product _ssib equations deleted; identity return with same signature',
     'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
     'mutated_sha256':hashlib.sha256(mutated.encode()).hexdigest(),
     'gate_exit':result.returncode,'calls':len(comparison['rows']),
     'different_values':sum(r['different'] for r in comparison['rows']),
     'scientific_assertion':'original-WRF literal BYTE gate failed' in result.stderr,
     'setup_errors':any(s in result.stderr for s in ('ImportError','ModuleNotFoundError','FileNotFoundError'))}
 receipt['killed']=receipt['gate_exit']==1 and receipt['scientific_assertion'] and receipt['different_values']>0 and not receipt['setup_errors']
 args.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
 assert receipt['killed'],receipt
finally:
 shutil.rmtree(private)
