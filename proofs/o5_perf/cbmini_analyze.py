"""Post-run analysis of a cbmini session (CPU, stdlib): per arm AOT source/cheap key per domain (patched blob really loaded?),
VLOG lines of the conversion pass, CBMINI_SUMMARY numbers. usage: cbmini_analyze.py RUN_DIR [OUT_JSON]"""
import json, re, sys
from pathlib import Path
R = Path(sys.argv[1]); out = {}
summary = json.loads((R / 'CBMINI_SUMMARY.json').read_text()) if (R / 'CBMINI_SUMMARY.json').exists() else {}
for a in ('M_OFF', 'M_V1P'):
    d = R / a; rec = dict(summary.get(a, {}))
    proofs = list((d / 'run').rglob('nested_pipeline_run.json'))
    if proofs:
        meta = json.loads(proofs[0].read_text()).get('metadata', {})
        aot = meta.get('nested_aot', {}).get('domains', {})
        rec['aot'] = {k: {x: v.get(x) for x in ('source', 'loaded', 'cheap_key', 'real_load')} for k, v in aot.items()}
        pre = meta.get('aot_prefetch', {}).get('domains', {})
        rec['prefetch'] = {k: {x: v.get(x) for x in ('source', 'loaded', 'error', 'cheap_key')} for k, v in pre.items()}
    log = d / 'run.log'
    if log.exists():
        lines = [l for l in log.read_text(errors='replace').splitlines() if 'command_buffer' in l]
        rec['vlog_lines'] = len(lines)
        rec['vlog_sample'] = [re.sub(r'\s+', ' ', l)[:240] for l in lines[:25]]
    out[a] = rec
txt = json.dumps(out, indent=1)
if len(sys.argv) > 2: Path(sys.argv[2]).write_text(txt + '\n')
print(txt[:6000])
