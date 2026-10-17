"""Original CPU-WRF Swiss crops -> unmodified full FDDAGD oracle, surface modes 1/2."""
from pathlib import Path
import sys,json,hashlib,subprocess,os
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
INT_KEYS = ("if_no_pbl_nudging_uv", "if_no_pbl_nudging_t", "if_no_pbl_nudging_q", "if_zfac_uv", "k_zfac_uv",
            "if_zfac_t", "k_zfac_t", "if_zfac_q", "k_zfac_q", "if_ramping", "interval_m", "end_h", "end_h_sfc",
            "stepfg", "fgdtzero", "specified")
REAL_KEYS = ("guv", "gt", "gq", "dtramp_min", "dt", "dx")
SOURCE=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
def fort(a):
    return np.ascontiguousarray(a.transpose(1,0,2)) if a.ndim==3 else np.ascontiguousarray(a)
def unfort(a,shape):
    nz,ny,nx=shape
    return a.reshape(ny,nz,nx).transpose(1,0,2)
def frame(hour):
    path=SOURCE / f'wrfout_d01_2023-01-15_{hour:02d}:00:00'
    ys,xs=slice(14,19),slice(16,22)
    with Dataset(path) as ds:
        g=lambda n:np.asarray(ds[n][0],np.float32)
        q=g('QVAPOR')[:,ys,xs]
        th=(g('THM')[:,ys,xs]+np.float32(300))/(np.float32(1)+np.float32(np.float32(461.6)/np.float32(287))*q)
        pp=g('P')[:,ys,xs]+g('PB')[:,ys,xs]
        return dict(u=g('U')[:,ys,16:23],v=g('V')[:,14:20,xs],th_phy=th,qv=q,
            p_phy=pp,t_phy=th*(pp/np.float32(1e5))**np.float32(np.float32(287)/np.float32(1004.5)),
            z_at_w=(g('PHB')[:,ys,xs]+g('PH')[:,ys,xs])/np.float32(9.81),pblh=g('PBLH')[ys,xs],
            ht=g('HGT')[ys,xs],mu_2=g('MU')[ys,xs],mub=g('MUB')[ys,xs],
            c1h=g('C1H'),c2h=g('C2H'),u10=g('U10')[ys,xs],v10=g('V10')[ys,xs],
            th2=g('TH2')[ys,xs],q2=g('Q2')[ys,xs])
def run_oracle(exe: Path, case: dict, opts: dict, calls: list, kind: str, work: Path, hfx_path=None) -> list:
    dt = np.float32 if kind == "r4" else np.float64
    f0 = case["frames"][0]
    nz, ny, nx = f0["th_phy"].shape
    inp, out = work / f"in_{kind}.bin", work / f"out_{kind}.bin"
    with open(inp, "wb") as fh:
        w = lambda a, t=dt: fh.write(np.asarray(a, t).tobytes())  # noqa: E731
        w([nx, ny, nz, len(calls), len(case["records"]), len(case["frames"])], np.int32)
        w([int(opts[k]) for k in INT_KEYS], np.int32)
        w([float(opts[k]) for k in REAL_KEYS])
        w([opts["surface_code"],opts["interval_m_sfc"]],np.int32)
        w([opts[k] for k in ("guv_sfc","gt_sfc","gq_sfc","rinblw")])
        w(case["c1h"]); w(case["c2h"]); w(fort(case["ht"]))
        for r in case["records"]:
            for n in ("u_old", "u_new", "v_old", "v_new", "t_old", "t_new", "q_old", "q_new"):
                w(fort(r[n]))
        # native Fortran dimensions (i,j,field,suffix,record)
        sr=np.zeros((nx+1,ny+1,5,2,len(case['records'])),dtype=dt,order='F')
        for ri,rec in enumerate(case['surface']):
            for fi,n in enumerate(('u10','v10','th2','q2','odis')):
                for si,s in enumerate(('old','new')):
                    a=rec[n+'_'+s]
                    sr[:a.shape[1],:a.shape[0],fi,si,ri]=a.T
        fh.write(sr.tobytes(order='F'))
        for f in case["frames"]:
            for n in ("u", "v", "th_phy", "qv", "z_at_w"):
                w(fort(f[n]))
            for n in ("pblh", "mu_2", "mub", "muts"):
                w(fort(f[n]))
            for n in ("t_phy","p_phy","regime","znt"):
                w(fort(f[n]))
        for c in calls:
            w([c["itimestep"], c["secs"], c["rec"], c["frame"]], np.int32)
    subprocess.run([str(exe), str(inp), str(out)] + ([] if hfx_path is None else [str(hfx_path)]), check=True)
    raw = np.fromfile(out, dtype=dt)
    su, sv, sm = (nz, ny, nx + 1), (nz, ny + 1, nx), (nz, ny, nx)
    order = [("u_old", su), ("u_new", su), ("v_old", sv), ("v_new", sv), ("t_old", sm), ("t_new", sm),
             ("q_old", sm), ("q_new", sm)]
    tend = [("run", su), ("rvn", sv), ("rth", sm), ("rqv", sm)]
    blocks = order + [(f"unc_{n}", s) for n, s in tend] + [(f"cpl_{n}", s) for n, s in tend] \
        + [(f"own_{n}", s) for n, s in tend] + [(f"dec_{n}", s) for n, s in tend]
    results, pos = [], 0
    for _ in calls:
        res = {}
        for name, shape in blocks:
            size = int(np.prod(shape))
            res[name] = unfort(raw[pos:pos + size], shape)
            pos += size
        for n in ("sda_hfx","sda_qfx"):
            size=ny*nx
            res[n]=raw[pos:pos+size].reshape(ny,nx);pos+=size
        results.append(res)
    assert pos == raw.size, (pos, raw.size)
    return results

def main():
    build,out=map(Path,sys.argv[1:3]);out.mkdir(parents=True,exist_ok=True)
    work=build/'work';work.mkdir(exist_ok=True)
    f0,f1,f2=[frame(h) for h in (12,13,14)]
    frames=[f1,f2]
    ny,nx=f1['ht'].shape
    yy,xx=np.indices((ny,nx))
    wind_census=os.environ.get('SF_WIND_CENSUS')=='1'
    for f in frames:
        # Independent active-regime perturbations of real columns: stable/free
        # convection, roughness branches, shallow/deep PBL and supersaturation.
        f['regime']=np.where((xx//2 if wind_census else xx)%2,4.,1.).astype(np.float32)
        f['znt']=np.where((yy//2 if wind_census else yy)%2,.05,.4).astype(np.float32)
        f['pblh']=np.where(yy%2,800.,120.).astype(np.float32)
        if wind_census:
            # Construct a first half-layer >10m so WRF's wind-height branch
            # is active even where the original Swiss first layer is shallow.
            floor=np.float32(30)*np.arange(f['z_at_w'].shape[0],dtype=np.float32)[:,None,None]
            f['z_at_w']=f['ht'][None]+np.maximum(f['z_at_w']-f['ht'][None],floor)
        f['muts']=f['mub']+f['mu_2']
    records=[];surfaces=[]
    for a,b in ((f0,f1),(f1,f2)):
        records.append({n+'_'+s:f[k]-np.float32(300) if n=='t' else f[k]
            for n,k in (('u','u'),('v','v'),('t','th_phy'),('q','qv')) for s,f in (('old',a),('new',b))})
        sd={}
        for s,f in (('old',a),('new',b)):
            sd['u10_'+s]=np.pad(f['u10'],((0,0),(0,1)),mode='edge')
            sd['v10_'+s]=np.pad(f['v10'],((0,1),(0,0)),mode='edge')
            sd['th2_'+s]=f['th2']
            sd['q2_'+s]=f['q2']*np.float32(5) # activates the WRF saturation restriction
            sd['odis_'+s]=(xx*20+yy*10+(10 if s=='new' else 0)).astype(np.float32)
        surfaces.append(sd)
    case=dict(frames=frames,records=records,surface=surfaces,ht=f1['ht'],c1h=f1['c1h'],c2h=f1['c2h'])
    base=dict(guv=3e-4,gt=3e-4,gq=3e-4,if_no_pbl_nudging_uv=0,if_no_pbl_nudging_t=0,if_no_pbl_nudging_q=0,
        if_zfac_uv=0,k_zfac_uv=0,if_zfac_t=0,k_zfac_t=0,if_zfac_q=0,k_zfac_q=0,
        if_ramping=0,dtramp_min=0.,interval_m=60,end_h=2,end_h_sfc=1,stepfg=1,fgdtzero=0,specified=1,
        dt=18.,dx=3000.,interval_m_sfc=60,guv_sfc=3e-4,gt_sfc=3e-4,gq_sfc=3e-4,rinblw=100.)
    manifest=dict(pristine=(build/'pristine_sha256.txt').read_text(),cases={},
        source=str(SOURCE),crop=[14,19,16,22],regimes='Swiss real columns + explicit stable/free, roughness, PBL, density, supersaturation perturbations')
    for code in (1,2):
        for variant in (('wind_census',) if wind_census else ('plain','ramp_pos','ramp_neg','hold_zero','zfac')):
            o=dict(base,surface_code=code)
            if variant.startswith('ramp'):
                o.update(if_ramping=1,dtramp_min=30. if variant=='ramp_pos' else -30.)
            if variant=='hold_zero':o.update(stepfg=5,fgdtzero=1)
            if variant=='zfac':o.update(if_zfac_uv=1,k_zfac_uv=1,if_zfac_t=1,k_zfac_t=2,if_zfac_q=1,k_zfac_q=1)
            calls=[dict(itimestep=it,secs=sec,rec=rec,frame=fr) for it,sec,rec,fr in
                ((1,0,1,1),(2,18,0,2),(5,72,0,1),(6,90,0,2),(101,1800,0,2),(201,3600,2,1),
                 (250,4482,0,2),(301,5400,0,1),(410,7362,0,2))]
            for kind in ('r4','r8'):
                dtype=np.float32 if kind=='r4' else np.float64
                results=run_oracle(build/kind/'fdda_oracle',case,o,calls,kind,work)
                arrays={n:case[n].astype(dtype) for n in ('c1h','c2h','ht')}
                for ri,(rec,sfc) in enumerate(zip(records,surfaces)):
                    arrays.update({f'rec{ri+1}_{n}':a.astype(dtype) for n,a in dict(rec,**sfc).items()})
                for fi,f in enumerate(frames):
                    arrays.update({f'frame{fi+1}_{n}':a.astype(dtype) for n,a in f.items() if n not in ('c1h','c2h','ht')})
                for ci,res in enumerate(results):
                    arrays.update({f'call{ci}_{n}':a for n,a in res.items()})
                arrays['calls']=np.array([[c[n] for n in ('itimestep','secs','rec','frame')] for c in calls],np.int32)
                name=f'swiss_sfc{code}_{variant}_{kind}.npz'
                np.savez_compressed(out/name,**arrays)
                manifest['cases'][name]=dict(opts=o,sha256=hashlib.sha256((out/name).read_bytes()).hexdigest(),
                    nonzero=[{n:int(np.count_nonzero(a)) for n,a in r.items() if n.startswith('unc_')} for r in results])
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
