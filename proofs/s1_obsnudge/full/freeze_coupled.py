"""Decode write-only pristine DM driver probes and seal coupled savepoints."""
import hashlib,json,sys,shutil
from pathlib import Path
import netCDF4
import numpy as np


def read_probe(path):
    with path.open('rb') as f:
        def get(shape,dtype='>f4'):
            return np.fromfile(f,dtype,count=int(np.prod(shape))).reshape(shape,order='F').astype(np.dtype(dtype).newbyteorder('='))
        nx,ny,nz,n,step,nv,ne=get((7,),'>i4')
        a=dict(nx=nx,ny=ny,nz=nz,nstat=n,step=step)
        a['xtime'],a['dt_s'],a['dx']=get((3,))
        for k in ('rio','rjo','rko_before','time_hours'):a[k]=get((n,))
        # These fdob metadata arrays are REAL in original WRF, even though
        # their values represent discrete indices/counts/platform codes.
        for k in ('lev_in_ob','nlevs_ob','plfo'):a[k]=get((n,))
        for k,width in (('varobs_before',nv),('errf_before',ne)):a[k]=get((width,n)).T
        a['u']=get((nx+1,nz,ny)).transpose(1,2,0)
        a['v']=get((nx,nz,ny+1)).transpose(1,2,0)
        for k in ('tb','qv','pbase','pp','z'):a[k]=get((nx,nz,ny)).transpose(1,2,0)
        for k in ('uratx','vratx','tratx','kpbl','regime','pblh','terrain','mut','muu','muv'):
            a[k]=get((nx,ny),'>i4' if k=='kpbl' else '>f4').T
        for k in ('c1h','c2h'):a[k]=get((nz,))
        for k in ('msfux','msfuy','msfvx','msfvy','msftx','msfty'):a[k]=get((nx,ny)).T
        a['weights_before']=get((4,nx,nz,ny)).transpose(0,2,3,1)
        a['tend_before']=np.stack([get((nx,nz,ny)).transpose(1,2,0) for _ in range(4)])
        a['varobs_after']=get((nv,n)).T;a['errf_after']=get((ne,n)).T;a['rko_after']=get((n,))
        a['weights_after']=get((4,nx,nz,ny)).transpose(0,2,3,1)
        a['tend_after']=np.stack([get((nx,nz,ny)).transpose(1,2,0) for _ in range(4)])
        if f.read():raise ValueError('probe trailing bytes')
    return a


def main(case,control,out):
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(case/'OBS_DOMAIN101',out/'OBS_DOMAIN101')
    with netCDF4.Dataset(case/'wrfinput_d01') as ds:
        projection=dict(code=int(ds.MAP_PROJ),lat1=float(ds['XLAT'][0,0,0]),lon1=float(ds['XLONG'][0,0,0]),
                        truelat1=float(ds.TRUELAT1),truelat2=float(ds.TRUELAT2),stdlon=float(ds.STAND_LON),
                        dx=float(ds.DX),dy=float(ds.DY))
    counts={}
    for probe,clean in zip(sorted(case.glob('wrfout*')),sorted(control.glob('wrfout*')),strict=True):
        with netCDF4.Dataset(probe) as p,netCDF4.Dataset(clean) as c:
            assert set(p.variables)==set(c.variables)
            for key in p.variables:
                pa,ca=np.asarray(p[key][:]),np.asarray(c[key][:])
                assert pa.dtype==ca.dtype and pa.shape==ca.shape and pa.tobytes()==ca.tobytes(),key
            counts[probe.name]=len(p.variables)
    entries=[]
    for source in sorted(case.glob('obs_probe_*.bin')):
        data=read_probe(source)
        # ERRF before the very first errob is not a defined input to the science.
        # Do not turn uninitialized storage into a purported truth fixture.
        data.pop('errf_before')
        for key,value in data.items():
            if not np.isfinite(value).all():raise ValueError(f'{source.name}:{key}:nonfinite')
        dest=out/(source.stem+'.npz');np.savez_compressed(dest,**data)
        entries.append(dict(file=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                            raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            count=int(data['nstat']),step=int(data['step']),
                            active_weights=[int(np.count_nonzero(w)) for w in data['weights_after']]))
        print(entries[-1])
    (out/'COUPLED_MANIFEST.json').write_text(json.dumps(dict(
        probe_transparent_variables=counts,arms=entries,
        projection=projection,
        original_provenance=json.loads((control/'PROVENANCE.json').read_text()),
        probe_provenance=json.loads((case.parent/'probe_build/PROVENANCE.json').read_text())),indent=2)+'\n')


if __name__=='__main__':main(*map(lambda p:Path(p).resolve(),sys.argv[1:]))
