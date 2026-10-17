"""Acquire multi-level/negative-QV pristine NUDOB arms; old operands reused.

Same fixed operand-based 16eps bounds as the staged single-level gate; no bounds
are selected from candidate errors. Real Swiss QV supplies active scaling inputs.
"""
from pathlib import Path
import hashlib,json,subprocess,sys
import netCDF4
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'proofs/s1_obsnudge/savepoints'


def main(build,save):
    source=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfout_d01_2023-01-15_01:00:00')
    with netCDF4.Dataset(source) as ds:real_qv=np.asarray(ds['QVAPOR'][0,:,20,15:27])
    old=json.loads((BASE/'MANIFEST.json').read_text())
    manifest={'oracle':json.loads((build/'build.json').read_text()),'cases':{},
              'driver_sha256':hashlib.sha256((Path(__file__).parent/'spread_driver.F90').read_bytes()).hexdigest(),
              'tolerance':'16eps operand envelope identical to original single-level preregistration'}
    for kind in ('r4','r8'):
        dtype=np.float32 if kind=='r4' else np.float64
        for ivar in (1,2,3,4):
            for mode in ('regime','mm5','no_pbl','gap'):
                base_name=f'{"regime" if mode=="gap" else mode}_{ivar}_{kind}.npz'
                with np.load(BASE/base_name) as s:data={k:s[k].copy() for k in s.files}
                o=old['cases'][base_name]['options'].copy()
                o['j']=5
                nobs=len(data['rio']);nz,nx=data['pbase'].shape;ny=o['jde']-1
                # 2 surface reports + one four-level sounding + 2 independent controls.
                lengths=np.asarray([1,1,4,4,4,4,1,1],dtype)
                levels=np.asarray([1,1,1,2,3,4,1,1],dtype)
                data['rio'][2:6]=dtype(6.25);data['rjo'][2:6]=dtype(5.5)
                data['time_hours'][2:6]=dtype(.015)
                data['rko'][2:6]=np.asarray([5,8,11,17],dtype)
                pind=dtype(.001)*(data['pbase']+data['pp'])
                data['pressure_cb'][2:6]=pind[[4,7,10,16],5]
                factor=dtype(.001 if ivar==4 else 1.)
                data['innovation'][2:6]=np.asarray([2.,-1.,3.,-2.],dtype)*factor
                var=np.zeros((nobs,6),dtype);err=np.zeros((nobs,10),dtype)
                var[:,4]=data['pressure_cb'];var[:,ivar-1]=dtype(1)
                var[2,ivar-1]=dtype(-888888)  # first sounding level missing; other levels still act
                err[:,ivar-1]=data['innovation'];err[:,4]=data['pbl_level']
                err[:,5:8]=data['base_pressure_cb'][:,None];err[:,8]=data['rko']
                err[:,9]=dtype(.004)
                maxgap=dtype(5 if mode=='gap' else 20)
                for scale in ((0,1) if ivar==4 else (0,)):
                    vv,ee=var.copy(),err.copy()
                    if scale:
                        ee[:2,ivar-1]=dtype(-.008)
                        ee[2:6,ivar-1]=np.asarray([-.008,-.006,-.004,-.002],dtype)
                    ee[2,ivar-1]=dtype(-888888)  # errob's error for the missing first level
                    starts=[0,1,2,6,7];width=4
                    p=np.ones((5,width),dtype);e=np.zeros_like(p);v=np.full_like(p,dtype(-888888));q=np.zeros_like(p)
                    for g,start in enumerate(starts):
                        count=int(lengths[start])
                        p[g,:count]=vv[start:start+count,4]
                        e[g,:count]=ee[start:start+count,ivar-1]
                        v[g,:count]=vv[start:start+count,ivar-1]
                        q[g,:count]=ee[start:start+count,9]
                    grouped={k:data[k][starts] for k in ('rio','rjo','rko','time_hours','pressure_cb','base_pressure_cb','pbl_level')}
                    grouped['innovation']=ee[starts,ivar-1]
                    grouped.update(prof_pressure=p,prof_innovation=e,prof_values=v,prof_q_model=q,
                                   nlevels=np.asarray([1,1,4,1,1],np.int32),qv=real_qv.astype(dtype))
                    packet={**data,**grouped}
                    inp,out=build/'spread_in.bin',build/'spread_out.bin'
                    with inp.open('wb') as f:
                        def write(x,dt=dtype):f.write(np.asarray(x,dt).tobytes())
                        write([nx,nz,ny,nobs,len(data['calls']),ivar,o['j'],o['ionf'],o['enabled'],int(o['nudge_pbl']),o['sfc_scheme_horiz'],o['sfc_scheme_vert'],scale],np.int32)
                        write([o[k] for k in ('dx','rinxy','twindo','dtmin','coefficient','sfcfact','sfcfacr','dpsmx')]+[maxgap])
                        for key in ('rio','rjo','rko','time_hours'):write(data[key])
                        for a in (vv,ee,levels,lengths):write(a)
                        for key in ('mu','c1h','c2h','msfx','msfy','pbase','pp','kpbl','vih1','vih2','terrain','z'):
                            write(data[key],np.int32 if key=='kpbl' else dtype)
                        write(real_qv);write(data['initial_aten'])
                        write(np.repeat(data['initial_saved'][:,:,None],6,axis=2))
                        for ints,reals in zip(data['calls'],data['times']):write(ints,np.int32);write(reals)
                    subprocess.run([str(build/kind/'spread'),str(inp),str(out)],check=True,timeout=20)
                    result=np.fromfile(out,dtype).reshape(len(data['calls']),2,nz,nx)
                    assert np.isfinite(result).all()
                    packet.update(oracle_aten=result[:,0],oracle_saved=result[:,1])
                    name=f'profiles_{mode}_{ivar}_scale{scale}_{kind}.npz'
                    np.savez_compressed(save/name,**packet)
                    manifest['cases'][name]={'options':o,'scale':scale,'maxgap':float(maxgap),
                                             'sha256':hashlib.sha256((save/name).read_bytes()).hexdigest(),
                                             'active_values':int(np.count_nonzero(result[:,0]-data['initial_aten']))}
                    inp.unlink();out.unlink()
    (save/'SPREAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
