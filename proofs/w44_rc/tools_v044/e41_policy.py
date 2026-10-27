"""RC ruling 2026-10-26T01:58:22Z; retain raw scanner results and all diagnostics."""
import json,sys
from pathlib import Path
AUTHORITY='MAIN MAILBOX manager RULING 2026-10-26T01:58:22Z'
def assess(g):
    result=dict(authority=AUTHORITY,raw_e41_gate=g.get('e41_gate'),hard_gate='BLOCKED',diagnostics=[],unknown_hlo=[])
    coverage=g.get('coverage') or {};entries=g.get('entries') or []
    if coverage.get('status')!='PASS' or not entries or len(entries)!=g.get('n_entries') or coverage.get('n_discovered')!=len(entries) or coverage.get('n_scanned')!=len(entries):
        return result|dict(reason='missing/inconsistent executable coverage')
    hard=[];unknown=[]
    for entry in entries:
        elfs=entry.get('elfs') or []
        if not elfs:unknown.append(dict(path=entry.get('path'),reason='missing ELF'))
        for elf in elfs:
            stack=elf.get('max_stack_B')
            if type(stack) is not int or stack<0:unknown.append(dict(path=entry.get('path'),reason='unknown stack'))
            elif stack>256 or elf.get('stack_over256'):hard.append(dict(path=entry.get('path'),stack_B=stack,reason='STACK>256'))
            if elf.get('nonleaf_over1k'):hard.append(dict(path=entry.get('path'),reason='non-leaf callee>1000 SASS',callees=elf['nonleaf_over1k']))
            if elf.get('callgraph_unknown'):unknown.append(dict(path=entry.get('path'),reason='unresolved native call graph',count=len(elf['callgraph_unknown'])))
        for name,size in entry.get('kinput_transpose_over500') or []:
            if type(size) is not int or size<=0:return result|dict(reason='invalid diagnostic instruction count')
            result['diagnostics'].append(dict(path=entry.get('path'),fusion=name,optimized_HLO_instructions=size,
                note='diagnostic under explicit RC ruling; source/DAG baseline and warm single-launch profile must be disclosed'))
        if str(entry.get('kinput_transpose_status','')).startswith('UNKNOWN'):
            result['unknown_hlo'].append(dict(path=entry.get('path'),status=entry['kinput_transpose_status']))
    if hard:return result|dict(hard_gate='FAIL',hard_failures=hard)
    if unknown:return result|dict(reason='native gate incomplete',native_unknowns=unknown)
    if g.get('hard_stack_gate')!='PASS':return result|dict(reason='raw stack verdict inconsistent')
    if g.get('e41_gate')=='FAIL' and not result['diagnostics']:
        return result|dict(reason='unexplained raw FAIL cannot pass')
    result['hard_gate']='PASS'
    result['status']='HARD_PASS_WITH_DIAGNOSTICS' if result['diagnostics'] else 'HARD_PASS_HLO_UNKNOWN' if result['unknown_hlo'] else 'PASS'
    return result
if __name__=='__main__':
    inp,out=map(Path,sys.argv[1:3]);result=assess(json.loads(inp.read_text()));out.write_text(json.dumps(result,indent=2)+'\n')
    print(result['hard_gate'],result.get('status',result.get('reason')),'kInput diagnostics',len(result['diagnostics']))
    raise SystemExit(0 if result['hard_gate']=='PASS' else 1)
