"""Short private pristine case for rolling windows and duplicate reports."""
from datetime import timedelta
import importlib.util,shutil,sys
from pathlib import Path
import netCDF4

s=importlib.util.spec_from_file_location('make_case',Path(__file__).with_name('make_cpu_case.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def main(out):
    m.main(out,True)
    base=out.parent
    shutil.copyfile(base/'cpu_probe/wrf.exe',out/'wrf.exe')
    for name in ('qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat'):
        p=base/'cpu_on'/name
        if p.exists() and not (out/name).exists():(out/name).symlink_to(p)
    with netCDF4.Dataset(out/'wrfinput_d01') as d:
        from datetime import datetime
        start=datetime.strptime(d['Times'][0].tobytes().decode(),'%Y-%m-%d_%H:%M:%S')
        lat=float(d['XLAT'][0,20,20]);lon=float(d['XLONG'][0,20,20]);ht=float(d['HGT'][0,20,20])
        ps=float(d['PSFC'][0,20,20]);temp=float(d['T2'][0,20,20])
    miss=(-888888.,-888888.)
    def surface(u=5.,t=None):
        return [miss,miss,(ht,0),(temp+1 if t is None else t,0),(u,0),(-2.,0),(75.,0),(ps,0),miss]
    levels=[[(85000.,0),(1500.,0),(278.,0),(8.,0),(-3.,0),(70.,0)],
            [(75000.,0),(2500.,0),(273.,0),(10.,0),(-1.,0),(60.,0)],
            [(65000.,0),(3500.,0),(268.,0),(12.,0),(1.,0),(50.,0)]]
    events=[(0,'SFC1',False,[surface()]),(0,'SFC2',False,[surface(6.,temp+2)]),
            (0,'TEMP2',True,levels[:2]),(0,'TEMP3',True,levels),
            (18,'SFC18',False,[surface(4.)]),(36,'SFC36',False,[surface(7.)]),
            (54,'OUTSIDE',False,[surface()]),(72,'PILOT3',True,levels),
            (72,'PILOT2',True,levels[:2]),(90,'SFC90',False,[surface(3.)]),
            (126,'TEMP126',True,levels),(600,'FUTURE',False,[surface()])]
    text=''
    for seconds,name,sounding,rows in events:
        record=m.report(start+timedelta(seconds=seconds),lat if name!='OUTSIDE' else 0.,
                        lon if name!='OUTSIDE' else 0.,name,sounding,rows,ht)
        if name.startswith('PILOT'):record=record.replace('FM-35 TEMP      ','FM-32 PILOT     ')
        text+=record
    (out/'OBS_DOMAIN101').write_text(text)
    text=(out/'namelist.input').read_text()
    changes={'run_seconds':216,'end_minute':3,'end_second':36,'history_interval_s':216}
    for k,v in changes.items():text=m.edit(text,k,str(v),'time_control')
    text=m.edit(text,'obs_twindo','.005','fdda')
    (out/'namelist.input').write_text(text)


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
