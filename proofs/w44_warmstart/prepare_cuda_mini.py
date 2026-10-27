"""Real 24x24x44 three-nest crop; interior BCs from the retained CPU twin."""
from pathlib import Path
import hashlib,json,re,sys,shutil
import numpy as np
from netCDF4 import Dataset
from gpuwrf.cycling.boundary import coupled_field,edge
from gpuwrf.cycling.atmosphere import cycle_atmosphere


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def crop(source,target,xy,*,parent=(1,1),one_time=False):
    x,y=xy;size=24
    cuts={'west_east':(x,size),'west_east_stag':(x,size+1),
          'south_north':(y,size),'south_north_stag':(y,size+1)}
    with Dataset(source)as a,Dataset(target,'w',format=a.file_format)as b:
        a.set_auto_maskandscale(False)
        for n,d in a.dimensions.items():b.createDimension(n,None if d.isunlimited()else cuts.get(n,(0,len(d)))[1])
        attrs={n:a.getncattr(n)for n in a.ncattrs()}
        for n in attrs:
            if any(k in n for k in ('WEST-EAST','SOUTH-NORTH')):
                if n.endswith('GRID_DIMENSION')or 'PATCH_END_STAG'in n:attrs[n]=np.int32(size+1)
                elif 'PATCH_END_UNSTAG'in n:attrs[n]=np.int32(size)
        for n,v in [('I_PARENT_START',parent[0]),('J_PARENT_START',parent[1])]:
            if n in attrs:attrs[n]=np.int32(v)
        if 'XLAT'in a.variables:
            attrs.update(CEN_LAT=a['XLAT'][0,y+size//2,x+size//2],CEN_LON=a['XLONG'][0,y+size//2,x+size//2])
        b.setncatts(attrs)
        for n,v in a.variables.items():
            attrs={k:v.getncattr(k)for k in v.ncattrs()};fill=attrs.pop('_FillValue',None)
            w=b.createVariable(n,v.datatype,v.dimensions,**({'fill_value':fill}if fill is not None else{}));w.setncatts(attrs)
            index=tuple(slice(cuts[d][0],sum(cuts[d]))if d in cuts else slice(0,1)if d=='Time'and one_time else slice(None)for d in v.dimensions)
            w[:]=v[index]


def main():
    assert shutil.disk_usage('<DATA_ROOT>').free>=365*2**30,'disk reserve below 365GiB'
    source,target,cpu=map(Path,sys.argv[1:4]);target.mkdir(exist_ok=False,parents=True)
    positions=[((44,24),(1,1)),((111,48),(9,8)),((84,66),(9,10))]
    for dom,(xy,parent)in enumerate(positions,1):
        for stem in ('wrfinput','wrflowinp'):
            crop(source/f'{stem}_d{dom:02}',target/f'{stem}_d{dom:02}',xy,parent=parent)
    crop(source/'wrfbdy_d01',target/'wrfbdy_d01',(44,24),one_time=True)
    endpoint=target/'CPU_ENDPOINT_d01.nc'
    crop(cpu/'wrfout_d01_2026-05-04_03:00:00',endpoint,(44,24))
    with Dataset(endpoint,'r+')as end:
        # Original WRF history uses QKE; native input's Registry alias is qke_adv.
        q=end['QKE'];v=end.createVariable('qke_adv',q.datatype,q.dimensions)
        v.setncatts({n:q.getncattr(n)for n in q.ncattrs()if n!='_FillValue'});v[:]=q[:]
    fields=[]
    with Dataset(target/'wrfinput_d01')as initial,Dataset(endpoint)as end,Dataset(target/'wrfbdy_d01','r+')as b:
        for name in b.variables:
            if '_B'not in name:continue
            field,suffix=name.rsplit('_B',1)
            if suffix not in ('XS','XE','YS','YE')or field in ('PC','HT_SHAD'):continue
            a,z=coupled_field(initial,field),coupled_field(end,field)
            first,last=edge(a,suffix,5),edge(z,suffix,5)
            b[name][0]=first;b[f'{field}_BT{suffix}'][0]=(last-first)/np.float32(10800)
            fields.append(name)
        for name in b.variables:
            if name=='Times'or'thisbdytime'in name or'nextbdytime'in name:
                stamp='2026-05-04_03:00:00'if'nextbdytime'in name else'2026-05-04_00:00:00'
                b[name][0]=np.frombuffer(stamp.encode(),dtype='S1')
    # Cropping changes every shape/mask digest. Use the actual new real.exe
    # template (SAVE_TOPO=1), while retaining the resolved source (SAVE_TOPO=0).
    # Never rewrite provenance hashes or lie about the source terrain flag.
    raw=target/'cropped_sources';raw.mkdir()
    for dom in (1,2,3):
        name=f'wrfinput_d{dom:02}';p=target/name
        with Dataset(p,'r+')as ds:
            for marker in ('GPUWRF_ATMOSPHERE_CYCLE','GPUWRF_LAND_CYCLE'):
                if marker in ds.ncattrs():ds.delncattr(marker)
        p.rename(raw/name)
    (target/'wrfbdy_d01').rename(raw/'wrfbdy_d01')
    fresh=target/'fresh_real';fresh.mkdir()
    fresh_source=source.parent/'may_pair/20260503_18z_a1'
    for dom,(xy,parent)in enumerate(positions,1):
        crop(fresh_source/f'wrfinput_d{dom:02}',fresh/f'wrfinput_d{dom:02}',xy,parent=parent)
    report=cycle_atmosphere(raw/'wrfinput_d01',fresh/'wrfinput_d01',target/'wrfinput_d01',
        wrfbdy=raw/'wrfbdy_d01',boundary_output=target/'wrfbdy_d01',native=True,
        nests=[(raw/f'wrfinput_d{d:02}',fresh/f'wrfinput_d{d:02}',target/f'wrfinput_d{d:02}')for d in (2,3)])
    (target/'CYCLE_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    # Exact identity is checked after the writer, while the raw views exist.
    for dom in (1,2,3):
        with Dataset(raw/f'wrfinput_d{dom:02}')as a,Dataset(target/f'wrfinput_d{dom:02}')as b:
            for name in a.variables:
                assert np.asarray(a[name][:]).tobytes()==np.asarray(b[name][:]).tobytes(),(dom,name)
    shutil.rmtree(raw)
    shutil.rmtree(fresh)
    text=(source/'namelist.input').read_text()
    for key,value in [('e_we','25,25,25'),('e_sn','25,25,25'),('i_parent_start','1,9,9'),('j_parent_start','1,8,10'),
                      ('interval_seconds','10800')]:
        text,n=re.subn(rf'(?im)^\s*{key}\s*=.*$',f' {key} = {value},',text);assert n==1,key
    (target/'namelist.input').write_text(text)
    manifest=dict(source=str(source),CPU_boundary_endpoint_source=str(cpu/'wrfout_d01_2026-05-04_03:00:00'),
        geometry='3x24x24x44, dx9000/3000/1000 dt54/18/6; all production physics retained',
        crop_offsets=positions,hours=3,root_boundary='WRFDA dry-mass coupled interior t0 and original CPU t3; linear 10800s interval, no synthetic state',
        fields=fields,files=[dict(name=p.name,sha256=sha(p),bytes=p.stat().st_size)for p in sorted(target.iterdir())],
        scope='dispatch/IO/E41 miniature; NOT original full-grid D6 or forecast fidelity qualification')
    (target/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for p in target.iterdir():p.chmod(0o444)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
