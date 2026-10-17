import ast,json,re
from pathlib import Path
import sys,hashlib
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
files=[];slice_rows=[];shapes=[];calls=[]
paths=list((root/'src/gpuwrf').rglob('*.py'))
selected={p for p in paths if 'pallas_call' in p.read_text()}
# Include imported in-product helpers as well as direct kernel files. This
# conservative module closure includes host/legacy branches; it is a census,
# not proof that a candidate branch executes in a particular Pallas program.
while True:
 before=len(selected)
 for path in tuple(selected):
  for n in ast.walk(ast.parse(path.read_text())):
   if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('gpuwrf.'):
    imported=root/'src'/Path(*n.module.split('.')).with_suffix('.py')
    if imported.is_file():selected.add(imported)
 if len(selected)==before:break
for p in sorted(selected):
 text=p.read_text()
 files.append(str(p.relative_to(root)));tree=ast.parse(text)
 parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
 def owner(node):
  while node in parents:
   node=parents[node]
   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):return node.name
  return '<module>'
 def row(node):return dict(file=str(p.relative_to(root)),line=node.lineno,function=owner(node),source=ast.get_source_segment(text,node))
 for node in ast.walk(tree):
  if isinstance(node,ast.Call):
   name=ast.unparse(node.func)
   if name.endswith('pallas_call'):
    calls.append(row(node)|{'kernel':ast.unparse(node.args[0]) if node.args else '<keyword>'})
   if name in ['jnp.roll','jax.lax.slice','lax.slice','jax.lax.dynamic_slice','lax.dynamic_slice']:
    slice_rows.append(row(node)|{'kind':name})
   if name in ['jnp.array','jnp.asarray','np.array','np.asarray'] and node.args and isinstance(node.args[0],(ast.List,ast.Tuple)):
    n=len(node.args[0].elts)
    if n and n&(n-1):shapes.append(row(node)|{'kind':'literal vector','length':n})
   if name in ['jnp.zeros','jnp.ones','jnp.full','jax.ShapeDtypeStruct','pl.BlockSpec'] and node.args:
    try:
     shape=ast.literal_eval(node.args[0]);shape=shape if isinstance(shape,tuple) else (shape,)
     product=1
     for dim in shape:product*=dim
     if product and product&(product-1):shapes.append(row(node)|{'kind':'literal shape','shape':shape,'size':product})
    except (ValueError,TypeError,SyntaxError):pass
  if isinstance(node,ast.Subscript) and any(isinstance(n,ast.Slice) for n in ast.walk(node.slice)):
   slice_rows.append(row(node)|{'kind':'subscript slice','base':ast.unparse(node.value)})
# Resolve direct kernels and their product helper function references. Names
# used as loop/conditional callbacks count too. This intentionally over-matches
# duplicate nested names and cannot decide static branches from source alone.
funcs={};imports={};kernel_roots=set()
for path in sorted(selected):
 rel=str(path.relative_to(root));tree=ast.parse(path.read_text())
 funcs[rel]={};imports[rel]={}
 for node in ast.walk(tree):
  if isinstance(node,ast.FunctionDef):funcs[rel].setdefault(node.name,[]).append(node)
  if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('gpuwrf.'):
   base='src/'+node.module.replace('.','/')
   for name in node.names:
    imports[rel][name.asname or name.name]=(base+'.py',name.name)
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and ast.unparse(node.func).endswith('pallas_call') and node.args:
   for name in ast.walk(node.args[0]):
    if isinstance(name,ast.Name):kernel_roots.add((rel,name.id))
reachable=set();visited=set();todo=list(kernel_roots)
while todo:
 rel,name=todo.pop()
 if (rel,name) in visited:continue
 visited.add((rel,name))
 if name not in funcs.get(rel,{}):
  target=imports.get(rel,{}).get(name)
  if target:todo.append(target)
  continue
 reachable.add((rel,name))
 for fn in funcs[rel][name]:
  for n in ast.walk(fn):
   if isinstance(n,ast.Name):
    if n.id in funcs[rel]:todo.append((rel,n.id))
    elif n.id in imports[rel]:todo.append(imports[rel][n.id])
reachable_slices=[r for r in slice_rows if (r['file'],r['function']) in reachable]
direct_files=sorted({r['file'] for r in calls})
report=dict(scope='Conservative textual/AST census of every product pallas_call file and recursive imported product modules. Includes host/legacy/tuple slices and dynamic shapes; candidates are NOT asserted Triton failures. Runtime branches and data-dependent shapes require actual native lowering.',files=files,direct_pallas_files=direct_files,reachable_functions=sorted(reachable),reachable_slice_candidates=reachable_slices,direct_file_non_power_two_literals=[r for r in shapes if r['file'] in direct_files],source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(selected)},pallas_call_sites=calls,non_power_two_static_literals=shapes,slice_candidates=slice_rows)
(out/'PALLAS_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'PALLAS_SLICES.txt').write_text('\n'.join(f"{r['file']}:{r['line']} [{r['function']}] {r['source']}" for r in slice_rows)+'\n')
(out/'PALLAS_SHAPES.txt').write_text('\n'.join(f"{r['file']}:{r['line']} [{r['function']}] {r.get('length',r.get('shape'))}: {r['source']}" for r in shapes)+'\n')
print(len(reachable),'reachable functions',len(reachable_slices),'reachable raw slices')
print(len(files),'files',len(calls),'calls',len(shapes),'nonpow2 static literals',len(slice_rows),'slices')
print((out/'PALLAS_SHAPES.txt').read_text())
