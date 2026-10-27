"""Join node activity to same-artifact HLO source metadata (E82/E182).

Receipt/CUDA correlation assigns the owning domain. Normalized exact HLO
instruction names identify fusions; Pallas names come from that HLO's backend
config. Ambiguous source families remain mixed, never split by instruction
count. Unmatched launch names remain unknown. Node-profile time is diagnostic,
not an end-to-end speed verdict or WRF-fidelity check.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re

TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")
KV = re.compile(r"(\w+)=(-?\d+)")
L = Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf')


def family(path):
    p = path.lower()
    for name, needles in [
        ('Thompson', ('thompson',)), ('MYNN', ('mynn', 'boulac')),
        ('KF', ('kain_fritsch', 'phys_kf', '/kf_')),
        ('RRTMG-LW', ('rrtmg_lw', '/rad_lw_')), ('RRTMG-SW', ('rrtmg_sw', '/rad_sw_')),
        ('RRTMG-sampling', ('/rad_mcica',)),
        ('Noah-MP', ('noahmp',)), ('surface-layer', ('surface_layer',)),
        ('hydrostatic-prep', ('phys_hydrostatic_real',)),
        ('gravity-wave-drag', ('phys_gwdo',)),
        ('dycore/advection', ('/dynamics/', '/dyn_')),
        ('boundary/nesting', ('boundary_', '/coupling/', 'nest_forcing', 'ring_select')),
        ('layout', ('layout_pin',)),
    ]:
        if any(x in p for x in needles):
            return name
    if '/gpuwrf/' in p:
        return 'glue/other'
    return 'unattributed'


def label(counts):
    known = set(counts) - {'unattributed', 'glue/other'}
    if len(known) == 1:
        return next(iter(known))
    if len(known) > 1:
        return 'mixed:' + '+'.join(sorted(known))
    return 'glue/other' if counts.get('glue/other') else 'unattributed'


def parse(path):
    tables = {k: {} for k in TABLES}
    comps = {}; current = None; section = None
    with path.open() as stream:
        for line in stream:
            line = line.rstrip('\n')
            if line in tables:
                section = line; continue
            if section and line[:1].isdigit():
                key, value = line.split(' ', 1); tables[section][int(key)] = value; continue
            section = None
            m = re.match(r'^(ENTRY )?%?([\w.\-]+) .*\{\s*$', line)
            if m:
                current = m[2]; comps[current] = []
            elif line == '}':
                current = None
            elif current and line.lstrip().startswith(('%', 'ROOT %')) and ' = ' in line:
                comps[current].append(line)

    def sources(line):
        locs = Counter()
        for sid in re.findall(r'stack_frame_id=(\d+)', line):
            f = dict((k,int(v)) for k,v in KV.findall(tables['StackFrames'].get(int(sid), '')))
            loc = dict((k,int(v)) for k,v in KV.findall(tables['FileLocations'].get(f.get('file_location_id'), '')))
            filename = tables['FileNames'].get(loc.get('file_name_id'), 'UNKNOWN').strip('"')
            fn = tables['FunctionNames'].get(loc.get('function_name_id'), 'UNKNOWN').strip('"')
            locs[filename+':'+str(loc.get('line'))+':'+fn] += 1
        return locs

    mappings = defaultdict(list)
    norm = lambda name: re.sub(r'[^A-Za-z0-9_]', '_', name)
    for comp, lines in comps.items():
        for line in lines:
            f = re.search(r'\sfusion\(.*\bcalls=%?([\w.\-]+)', line)
            kernel = None; locs = Counter(); kind = None
            if f:
                kernel = norm(re.search(r'%([\w.\-]+) = ', line)[1]); kind = 'fusion'
                # Parameter/constant source stamps alone are not emitted work.
                for bodyline in comps[f[1]]:
                    if not re.search(r'\s(?:parameter|constant)\(', bodyline):
                        locs.update(sources(bodyline))
                if not locs: locs.update(sources(line))
            elif ' custom-call(' in line:
                n = re.search(r', name = "([^"]+)"', line)
                if n: kernel = norm(n[1]); kind = 'Pallas'; locs.update(sources(line))
            if kernel:
                families = Counter()
                for loc, count in locs.items(): families[family(loc.split(':',1)[0])] += count
                mappings[kernel].append({'family':label(families),'source_families':dict(families),
                                         'sources':dict(locs),'kind':kind,'computation':comp})
    assert mappings and any(x['sources'] for rows in mappings.values() for x in rows)
    return mappings


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    mappings={}; bindings=[]
    index=json.loads((L/'ACTUAL_HLO/INDEX.json').read_text())
    for rec in index['records']:
        path=Path(rec['hlo_file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['hlo_sha256']
        key=(rec['version'],rec['program']);mappings[key]=parse(path);bindings.append(rec)
        print('mapped',*key,len(mappings[key]),flush=True)
    results={}
    for arm, version in [('N32','v032'),('N43','v043'),('R43','v043')]:
        kernels=json.loads((L/'TRACE_SUMMARY'/f'{arm}_kernels.json').read_text())
        totals=defaultdict(lambda:[0,0]); detail=defaultdict(lambda:[0,0]); matches={}
        for k in kernels:
            owner=k['domain'];m=mappings.get((version,owner),{});name=k['name'];via='exact'
            rows=m.get(name,[])
            # Native uniquification adds __N; do not strip arbitrary single
            # underscores/numbers or guess a physics family from a CUDA name.
            if not rows and re.search(r'__\d+$',name):
                rows=m.get(re.sub(r'__\d+$','',name),[]);via='native __N suffix'
            fams=set(x['family'] for x in rows)
            fam=next(iter(fams)) if len(fams)==1 else ('ambiguous-HLO-family' if fams else 'unmatched')
            duration=k['end']-k['start'];totals[fam][0]+=1;totals[fam][1]+=duration
            detail[(owner,fam)][0]+=1;detail[(owner,fam)][1]+=duration
            matches[(owner,name)]={'family':fam,'join':via if rows else 'NONE','candidates':rows}
        assert sum(x[0] for x in totals.values())==len(kernels)
        result={'arm':arm,'version':version,'captured_roots':3,'kernel_count':len(kernels),
                'kernel_sum_ms':sum(x[1] for x in totals.values())/1e6,
                'family_count_ns':dict(sorted(totals.items(),key=lambda x:-x[1][1])),
                'domain_family_count_ns':{d+':'+f:v for (d,f),v in sorted(detail.items())},
                'unmatched_kernel_count':totals.get('unmatched',[0,0])[0],
                'unmatched_kernel_ms':totals.get('unmatched',[0,0])[1]/1e6}
        results[arm]=result
        (a.out/(arm+'_joins.json')).write_text(json.dumps({d+':'+n:v for (d,n),v in matches.items()},indent=2)+'\n')
        print(arm,'kernels',len(kernels),'sum_ms',result['kernel_sum_ms'],
              'unmatched_ms',result['unmatched_kernel_ms'],'top8',list(result['family_count_ns'].items())[:8],flush=True)
    (a.out/'FAMILY_TIMES.json').write_text(json.dumps({'scope':__doc__,'hlo_bindings':bindings,'arms':results},indent=2)+'\n')


if __name__=='__main__':main()
