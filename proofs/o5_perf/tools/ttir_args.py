"""Per Pallas kernel (one HLO custom call): which Triton function arguments are actually loaded / stored (pointer provenance),
logical load/store element volume per launch, and the unique-byte floor = sum of sizes of loaded + stored arguments."""
import re, sys, json
from pathlib import Path
from collections import Counter
from jax._src.lib import triton as jt
from jax._src.lib.mlir import ir
tt = jt.dialect
DT = {'f32': 4, 'f64': 8, 'i32': 4, 's32': 4, 'pred': 1, 'i1': 1, 'f16': 2, 'bf16': 2, 'i64': 8, 's64': 8, 'u8': 1, 's8': 1}
def unescape(s):
    out = bytearray(); i = 0
    while i < len(s):
        c = s[i]
        if c == '\\':
            n = s[i+1]
            if re.match(r'[0-9A-Fa-f]', n) and i + 2 < len(s) and re.match(r'[0-9A-Fa-f]', s[i+2]):
                out.append(int(s[i+1:i+3], 16)); i += 3; continue
            out.append(ord({'n': '\n', 't': '\t', '"': '"', '\\': '\\'}.get(n, n))); i += 2; continue
        out.append(ord(c)); i += 1
    return bytes(out)
def shape_bytes(sh):
    m = re.match(r'([a-z0-9]+)\[([0-9,]*)\]', sh)
    n = 1
    for d in m.group(2).split(','):
        if d: n *= int(d)
    return n * DT.get(m.group(1), 4)
def nel(t):
    m = re.match(r'tensor<([0-9x]+)x', str(t))
    if not m: return 1
    n = 1
    for d in m.group(1).split('x'): n *= int(d)
    return n
def analyze(line, ctx, symtab):
    raw = re.search(r'ir = "((?:[^"\\]|\\.)*)"', line).group(1)
    grid = 1
    for x in re.findall(r'grid_[xyz] = (\d+)', line): grid *= int(x)
    import roofline_join as R
    inst = R.parse_inst(line)
    in_shapes = [re.findall(r'[a-z0-9]+\[[0-9,]*\]', symtab[o])[0] for o in inst['operands'] if o in symtab]
    out_shapes = re.findall(r'[a-z0-9]+\[[0-9,]*\]', inst['shape'])
    shapes = in_shapes + out_shapes
    with ctx:
        mod = ir.Module.parse(unescape(raw))
    func = None
    for o in mod.body.operations:
        if o.operation.name == 'tt.func':
            func = o; break
    args = list(func.regions[0].blocks[0].arguments)
    arg_index = {str(a.get_name()) if hasattr(a, 'get_name') else id(a): i for i, a in enumerate(args)}
    def root_arg(v, depth=0):
        if depth > 50: return None
        if isinstance(v, ir.BlockArgument):
            owner = v.owner
            try:
                if owner == func.regions[0].blocks[0]:
                    return v.arg_number
            except Exception:
                pass
            return None
        op = v.owner
        for operand in op.operands:
            if 'ptr' in str(operand.type):
                r = root_arg(operand, depth + 1)
                if r is not None: return r
        return None
    loaded, stored = Counter(), Counter(); le = se = 0
    def walk(op):
        nonlocal le, se
        for region in op.regions:
            for block in region.blocks:
                for o in block.operations:
                    n = o.operation.name
                    if n == 'tt.load':
                        a = root_arg(o.operands[0]); loaded[a] += 1; le += nel(o.results[0].type)
                    elif n == 'tt.store':
                        a = root_arg(o.operands[0]); stored[a] += 1; se += nel(o.operands[1].type)
                    walk(o.operation)
    walk(func.operation)
    nargs = len(args)
    def size(i):
        return shape_bytes(shapes[i]) if i is not None and i < len(shapes) else 0
    uniq = sum(size(i) for i in set(loaded) | set(stored))
    return {'grid': grid, 'nargs': nargs, 'n_shapes': len(shapes), 'loaded_args': sorted(k for k in loaded if k is not None),
            'stored_args': sorted(k for k in stored if k is not None), 'unresolved_loads': loaded.get(None, 0),
            'logical_load_MB': le * 4 * grid / 1e6, 'logical_store_MB': se * 4 * grid / 1e6,
            'unique_bytes_MB': uniq / 1e6, 'all_operands_MB': sum(shape_bytes(s) for s in shapes) / 1e6}
if __name__ == '__main__':
    H = Path(sys.argv[1]); names = sys.argv[2:]
    sys.path.insert(0, '<DATA_ROOT>/wrf_gpu2/worktrees/o5-perf/proofs/o5_perf')
    import roofline_join as R
    comps = R.parse_hlo(H)
    symtab = {n: i['shape'] for c in comps.values() for n, i in c.items()}
    ctx = ir.Context(); tt.register_dialect(ctx); ctx.allow_unregistered_dialects = True
    res = {}
    for line in H.open():
        if 'custom-call(' not in line or '__gpu$xla.gpu.triton' not in line: continue
        m = re.search(r', name = "([^"]+)"', line)
        if not m or (names and m.group(1) not in names) or m.group(1) in res: continue
        try:
            res[m.group(1)] = analyze(line, ctx, symtab)
        except Exception as exc:
            res[m.group(1)] = {'error': repr(exc)[:200]}
    print(json.dumps(res, indent=1))
