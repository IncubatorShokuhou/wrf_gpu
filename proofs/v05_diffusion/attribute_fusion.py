"""Resolve a stack-failing fusion using ORIGINAL proto debug tables (E224/TP75B)."""
import collections
import json
from pathlib import Path
import re
import sys

root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'src'), str(root/'proofs/w44_e84fix')]
from lowering_audit import debug_from_proto, paths

out=Path(sys.argv[1]); fusion=sys.argv[2]
receipt=json.loads((out/'receipt.json').read_text())
scan=json.loads(next((out/'e41').glob('hlo_scan_*.json')).read_text())
row=next(r for r in scan['transpose_hero_fusions'] if r['fusion']==fusion)
lines=(out/'optimized.hlo').read_text().splitlines()
start,end=row['comp_lines']; block=lines[start-1:end]
comp=re.search(r'%([^ (]+)',block[0])[1]
debug=debug_from_proto((out/'optimized.hlo.pb').read_bytes())
records=[]; sites=collections.Counter()
for line in block[1:]:
 m=re.match(r'\s+(?:ROOT )?%([^ =]+) = ',line)
 if not m:continue
 frames=paths(debug,debug['metadata'].get((comp,m[1]),{}))
 unique={(f['file'],f['function'],f['line']) for f in frames}
 sites.update(unique)
 records.append(dict(instruction=m[1],frames=frames,hlo=line.strip()))
rec=dict(scope='deviceless original-protobuf provenance, no GPU',fusion=row,
         callsite=lines[row['call_line']-1],computation=comp,
         sites=[dict(file=f,function=fn,line=n,instructions=k) for (f,fn,n),k in sites.most_common()],
         instructions=records)
(out/'fusion_attribution.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(dict(fusion=row,sites=rec['sites'][:25]),indent=2))
