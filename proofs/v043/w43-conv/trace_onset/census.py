"""Onset/decay symmetry census for trace hydrometeor fields (CPU-only, read-only).
CAVEAT (critic cr-w43-conv, E200): no explicit finite checks -- CPU-zero+NaN diff counts as both0, CPU-zero+Inf diff as
GPU_ONLY, CPU NaN/Inf array as both. Add finite/raw-pair validation before reusing this census.
Per case/domain/field/lead: CPU all-zero? (from CPU-WRF ref wrfout), |GPU-CPU| max (from teacher extract).
Classes: both0 (cpu0 & diff0), GPU_ONLY (cpu0 & diff>0), CPU_ONLY (from integrity degenerate list = GPU constant 0),
both (else). GPU raw wrfouts were already deleted, so GPU_ONLY is inferred exactly: CPU field identically 0 and diff>0."""
import json, glob, os, sys, netCDF4, numpy as np
S='<USER_HOME>/wrf_gpu2_lanes/w43-teacher/score'
FIELDS=('QCLOUD','QRAIN','QICE','QSNOW','QGRAUP','QNICE','QNRAIN','RAINNC','CANLIQ','CANWAT','FWET','QRAINXY','QSNOWXY','CANICE')
out={}
for c in sorted(glob.glob(S+'/*/integ/integrity_all_72h.json')):
    case=c.split('/')[-3]
    integ=json.load(open(c)); cpu_dir=integ['cpu_dir']
    for dom in ('d01','d02','d03'):
        xj=f'{S}/{case}/x/{dom}.json'
        if not os.path.exists(xj): continue
        fsum=json.load(open(xj))['field_summaries']
        deg=integ['domains'][dom].get('degenerate_fields',{})
        files=sorted(glob.glob(f'{cpu_dir}/wrfout_{dom}_*'))
        flds=[f for f in FIELDS if f in fsum]
        cpu={f:[] for f in flds}
        for fn in files:
            ds=netCDF4.Dataset(fn)
            for f in flds:
                a=np.asarray(ds[f][0]); cpu[f].append((os.path.basename(fn)[11:24], bool(np.all(a==0)), float(np.abs(a).max()), int((a!=0).sum())))
            ds.close()
        for f in flds:
            leads={x['lead_h']:x for x in fsum[f]['by_lead']}
            degf={e['file'][11:24] for e in deg.get(f,[])}
            rec={'both0':0,'GPU_ONLY':[],'CPU_ONLY':[],'both':0}
            for lead,(stamp,cpu0,cmax,cn) in enumerate(cpu[f]):
                d=leads.get(lead)
                if d is None: continue
                if stamp in degf: rec['CPU_ONLY'].append((stamp,cmax,cn))
                elif cpu0 and d['max_abs']>0: rec['GPU_ONLY'].append((stamp,d['max_abs']))
                elif cpu0: rec['both0']+=1
                else: rec['both']+=1
            out[f'{case}/{dom}/{f}']=rec
            if rec['GPU_ONLY'] or rec['CPU_ONLY']:
                print(f'{case} {dom} {f}: both={rec["both"]} both0={rec["both0"]} GPU_ONLY={len(rec["GPU_ONLY"])} CPU_ONLY={len(rec["CPU_ONLY"])}', flush=True)
json.dump(out,open(sys.argv[1],'w'),indent=1)
