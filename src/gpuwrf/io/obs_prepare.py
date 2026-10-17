"""Resident observation preparation: literal WRF in4dob QC and read scheduling.

All file I/O, projection, duplicate resolution and report arrival planning happen
at initialization. VAROBS and RKO mutations from errob remain device carry.
No observations are fetched from the host inside a model step.
"""
from dataclasses import dataclass,field
from datetime import timezone
import numpy as np
from gpuwrf.io.obs_domain import ObservationRecord

R=np.float32
MISSING=R(-888888)


def _good(q):return R(q)>=R(0) and R(q)<R(30000)


def _rh2r(rh,t,p_mb):
    # wrf_fddaobs_in.F rh2r(iice=0), including its distinct constants.
    rh1=R(rh)*R(.01)
    esat=R(6.1078)*np.exp(R(17.2693882)*(R(t)-R(273.16))/(R(t)-R(35.86)))
    rsat=R(.62197)*esat/(R(p_mb)-esat)
    return rh1*rsat


def _platform(report):
    name=report.name.ljust(40);platform=report.platform.ljust(16)
    if name[1:9]=='PROFILER':platform=platform[:6]+'PROFILER'+platform[14:]
    if name[1:6]=='ACARS':platform=platform[:6]+'ACARS'+platform[11:]
    if name[:7]=='SATWNDS':platform='SATWNDS    '+platform[11:]
    if name[:8]=='CLASS DA':platform=platform[:6]+'TEMP'+platform[10:]
    code=99
    for token,value in (('METAR',1),('SPECI',2),('SHIP',3),('SYNOP',4),('TEMP',5),('PILOT',6),('SATWI',7),('PROFILER',9),('ACARS',7)):
        if platform[6:6+len(token)]==token:code=value
    if platform[:7]=='SATWNDS':code=7
    if platform[:4]=='SAMS':code=8
    return code,platform


def normalize_report(report:ObservationRecord,projection):
    """VAROBS (levels,6), initial RKO, platform code, projected coordinates."""
    plfo,platform=_platform(report)
    ri,rj=projection.station_coordinates(report.latitude,report.longitude)
    levels=len(report.measurements)
    var=np.full((levels,6),MISSING,np.float32)
    rko=np.full(levels,R(-99) if report.is_sounding else R(1),np.float32)
    for n,measurements in enumerate(report.measurements):
        a=np.asarray([(m.value,m.qc) for m in measurements],np.float32)
        if report.is_sounding:
            a[:,1]=np.where(a[:,1]==R(-777777),R(0),a[:,1])
            a[:,1]=np.where(a[:,0]==MISSING,MISSING,a[:,1])
            p,h,t,u,v,rh=a
            if plfo==5 and (u[1]==R(256) or v[1]==R(256)):u[:]=MISSING;v[:]=MISSING
            if plfo==6:t[:]=MISSING;rh[:]=MISSING
            if _good(t[1]) and (_good(p[1]) or _good(h[1])):var[n,2]=t[0]
            if _good(h[1]):var[n,5]=h[0]
            if _good(p[1]):var[n,4]=p[0]
            if var[n,4]>=R(0):var[n,4]=var[n,4]*R(.001)
            if var[n,4]>R(0) and var[n,4]<=R(8):
                # Original order retains VAROBS(T), assigned before this block.
                u[:]=MISSING;v[:]=MISSING;t[:]=MISSING;rh[:]=MISSING
            if _good(rh[1]) and _good(p[1]) and _good(t[1]):
                var[n,3]=_rh2r(rh[0],t[0],p[0]*R(.01))
        else:
            slp,refp,h,t,u,v,rh,p,precip=a
            rovg=R(R(287)/R(9.81))
            if p[1]<R(0) and _good(slp[1]) and _good(t[1]) and slp[0]>R(90000):
                tbar=t[0]+R(.5)*R(report.elevation)*R(.0065)
                p[0]=slp[0]*np.exp(-R(report.elevation)/(rovg*tbar));p[1]=R(0)
            if p[1]<R(0) and _good(t[1]):
                tbar=t[0]+R(.5)*R(report.elevation)*R(.0065)
                p[0]=R(100000)*np.exp(-R(report.elevation)/(rovg*tbar));p[1]=R(0)
            validp=_good(p[1]) and p[0]>R(70000) and p[0]<R(105000)
            if validp:var[n,4]=p[0]*R(.001)
            if _good(t[1]) and validp:var[n,2]=t[0]
            if plfo==3 and rh[1]>=R(0) and rh[0]<R(70):rh[:]=MISSING
            if _good(rh[1]) and _good(p[1]) and _good(t[1]):
                var[n,3]=_rh2r(rh[0],t[0],p[0]*R(.01))
        if _good(u[1]) and _good(v[1]) and (u[0]!=R(0) or v[0]!=R(0)):
            uv=(u[0],v[0])
            if u[1]==R(129) and v[1]==R(129):uv=projection.rotate_wind(report.longitude,*uv)
            var[n,0],var[n,1]=uv
    if not np.isfinite(var).all():raise ValueError(f'{report.station_id}: non-finite normalized observation')
    if report.is_sounding:
        located=(var[:,4]>R(-800000))|(var[:,5]>R(-800000))
        if not located.all():raise ValueError(f'{report.station_id}: sounding level has neither pressure nor height for errob')
        goodp=var[:,4]>R(-800000)
        if np.any(var[goodp,4]<=0):raise ValueError(f'{report.station_id}: sounding pressure must be positive')
        if goodp.all() and np.any(np.diff(var[:,4])>R(0)):
            raise ValueError(f'{report.station_id}: sounding pressures must descend in file order')
        if not goodp.all() and np.any(np.diff(var[:,5][~goodp])<R(0)):
            raise ValueError(f'{report.station_id}: height-only sounding levels must ascend in file order')
    return var,rko,plfo,platform,R(ri),R(rj)


@dataclass
class _Slot:
    time: np.float32
    rio: np.float32
    rjo: np.float32
    plfo: int
    sounding: bool
    rko: np.ndarray
    variants: list=field(default_factory=list)  # (read-step,VAROBS)
    deactivated: int=2147483647


def prepare_reports(reports,projection,run_start,options,dt_s,nx,ny):
    """Precompute in4dob's one-future-report lookahead and duplicate events.

    Slots remain chronological; replacing a surface/satellite single report
    preserves its RKO, exactly as WRF. Replaced TEMP/PROFILER profiles remain in
    the rolling capacity accounting but cease contributing when replaced.
    """
    if run_start.tzinfo is not None:run_start=run_start.astimezone(timezone.utc).replace(tzinfo=None)
    ionf=int(options['obs_ionf']);window=R(options['obs_twindo']);dt=float(dt_s)
    end=R(options['fdda_end']);dyn=int(options['obs_idynin'])==1
    datend=end-R(options['obs_dtramp']) if dyn and float(options['obs_dtramp'])>0 else end
    def clock(step):
        xtime=R(float(step)*dt/60.)
        return xtime,max(R(xtime/R(60)),R(0))
    def next_read(prev,step):
        if prev is None:return step
        step=max(step,max(0,int(np.ceil((float(prev)-float(window))*3600/dt/ionf)))*ionf)
        while prev>R(clock(step)[1]+window):step+=ionf
        while step>=ionf and step-ionf>=last_read and prev<=R(clock(step-ionf)[1]+window):step-=ionf
        return step
    slots=[];last_time=None;last_read=0
    for report in reports:
        step=next_read(last_time,last_read)
        xtime,xhour=clock(step)
        if xtime>end:break
        seconds=int((report.time-run_start).total_seconds())
        time=R(R(seconds)/R(3600))
        if dyn and R(time*R(60))>datend:break
        tback=R(xhour-window)
        if time<tback:continue
        var,rko,plfo,platform,ri,rj=normalize_report(report,projection)
        # in4dob retains only observations inside these non-staggered bounds.
        if not (R(2)<=ri<=R(nx) and R(2)<=rj<=R(ny)):continue
        if plfo==4 and platform[6:16]=='SYNOP PRET' and report.station_id.ljust(40)[6:15]=='METNET= 3':continue
        duplicate=None
        for index,s in enumerate(slots):
            if s.time<tback or s.deactivated<=step or s.plfo==99:continue
            if s.time==time and s.rio==ri and s.rjo==rj and ((s.plfo<=4 and plfo<=4) or s.plfo==plfo):
                duplicate=index;break
        discard=False;replaced=False
        if duplicate is not None:
            s=slots[duplicate]
            if not report.is_sounding and np.array_equal(s.rko,rko):
                s.variants.append((step,var));replaced=True
            elif report.is_sounding and s.plfo==plfo:
                if plfo in (5,9) or (plfo==6 and len(rko)>=len(s.rko)):s.deactivated=step
                elif plfo==6 and len(rko)<len(s.rko):discard=True
                elif plfo==7 and len(rko)==len(s.rko)==1 and var[0,4]==s.variants[-1][1][0,4]:
                    s.variants.append((step,var));replaced=True
        if discard:continue
        if not replaced:slots.append(_Slot(time,ri,rj,plfo,report.is_sounding,rko,[(step,var)]))
        last_time=time;last_read=step
    if not slots:
        raise ValueError('obs_nudge_opt=1: no observations inside the domain and requested assimilation period')
    # Maximum report-frame size and WRF max_obs accounting at every arrival.
    max_reports=max_levels=required_capacity=0
    times=np.asarray([s.time for s in slots],np.float32)
    arrivals=np.asarray([s.variants[0][0] for s in slots],np.int32)
    for step in sorted(set(arrivals.tolist())):
        tback=R(clock(step)[1]-window)
        active=(arrivals<=step)&(times>=tback)
        max_reports=max(max_reports,int(active.sum()))
        occupied=sum(len(s.rko) for s,a in zip(slots,active) if a)
        max_levels=max(max_levels,occupied)
        # in4dob tests N>NIOBF-1 BEFORE attempting another timestamp/EOF
        # read. A full frame needs a spare slot unless its last report is
        # already the future lookahead that ends this read cycle.
        future=np.any(active & (times>R(clock(step)[1]+window)))
        required_capacity=max(required_capacity,occupied+(0 if future else 1))
    if required_capacity>int(options['max_obs']):
        raise ValueError(f'obs_nudge_opt=1 needs max_obs>={required_capacity} (WRF rolling buffer/EOF lookahead), configured {options["max_obs"]}')
    n=len(slots);width=max(len(s.rko) for s in slots);variants=max(len(s.variants) for s in slots)
    data=np.full((n,variants,width,6),MISSING,np.float32)
    updates=np.full((n,variants),2147483647,np.int32)
    rko=np.ones((n,width),np.float32)
    for i,s in enumerate(slots):
        rko[i,:len(s.rko)]=s.rko
        for j,(step,var) in enumerate(s.variants):data[i,j,:len(s.rko)]=var;updates[i,j]=step
    return dict(times=times,arrivals=arrivals,deactivated=np.asarray([s.deactivated for s in slots],np.int32),
                rio=np.asarray([s.rio for s in slots],np.float32),rjo=np.asarray([s.rjo for s in slots],np.float32),
                rko=rko,variants=data,variant_steps=updates,lengths=np.asarray([len(s.rko) for s in slots],np.int32),
                max_reports=max_reports,max_observation_levels=max_levels)
