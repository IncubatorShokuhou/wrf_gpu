"""Offline selected-variable CSV/NetCDF export of a WRF *.TS surface stream."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .files import SURFACE_FIELDS

UNITS = dict(zip(SURFACE_FIELDS, ('K', 'kg kg-1', 'm s-1', 'm s-1', 'Pa', 'W m-2', 'W m-2',
                                 'W m-2', 'W m-2', 'K', 'K', 'mm', 'mm', 'kg m-2')))


def export_surface(source, target, *, variables=SURFACE_FIELDS, interval_seconds=None, format='csv', station_uid=None):
    """Select original rows at interval alarm crossings; never interpolate.

    With interval=60 and dt=18, keep t=72,126,... (first completed own step
    crossing each positive multiple of 60). The WRF raw stream stays intact.
    One station/domain per file; output retains original forecast hours.
    """
    source, target = Path(source), Path(target)
    variables = tuple(variables)
    if (not variables or len(set(variables)) != len(variables)
            or set(variables)-set(SURFACE_FIELDS)):
        raise ValueError(f"variables must be unique names from {','.join(SURFACE_FIELDS)}")
    if format not in {'csv','netcdf','station-csv'}:
        raise ValueError("format must be csv, netcdf or station-csv")
    if format=='station-csv':
        if not isinstance(station_uid,str) or not station_uid.strip():
            raise ValueError("station-csv requires an explicit station_uid matching the verify station table")
        if set(variables)-{'T2','U10','V10'} or (('U10' in variables)!=('V10' in variables)):
            raise ValueError("station-csv supports T2 and/or the U10,V10 pair; use plain csv for other fields")
    if interval_seconds is not None and (not np.isfinite(interval_seconds) or interval_seconds <= 0):
        raise ValueError("interval_seconds must be finite and positive")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"export target exists: {target}")
    lines = source.read_text(encoding='ascii').splitlines()
    if not lines:
        raise ValueError("empty time series file")
    start_date = lines[0][-19:]
    rows=[]
    for number, line in enumerate(lines[1:],2):
        if len(line) != 226:
            raise ValueError(f"TS line {number}: expected WRF process_time_series=1 record width226")
        try:
            row = [int(line[:2]),float(line[2:15]),int(line[15:20]),int(line[20:25]),int(line[25:30])]
            row += [float(line[31+14*i:44+14*i]) for i in range(14)]
        except ValueError as exc:
            raise ValueError(f"TS line {number}: invalid/overflowed numeric field") from exc
        if not np.isfinite(row).all():
            raise ValueError(f"TS line {number}: NaN/Inf")
        rows.append(row)
    if not rows:
        raise ValueError("time series has no samples")
    data = np.asarray(rows,np.float64)
    if np.any(np.diff(data[:,1]) <= 0) or not np.all(data[:,(0,2,3,4)] == data[0,(0,2,3,4)]):
        raise ValueError("time series hours must increase and station/domain/grid indices must stay fixed")
    if interval_seconds is not None:
        alarm = np.floor((data[:,1]*3600+1.e-6)/float(interval_seconds)).astype(np.int64)
        data = data[alarm > np.r_[0,alarm[:-1]]]
    selected = [SURFACE_FIELDS.index(name)+5 for name in variables]
    target.parent.mkdir(parents=True,exist_ok=True)
    if format == 'csv':
        with target.open('x',newline='',encoding='ascii') as file:
            writer=csv.writer(file)
            writer.writerow(['domain_id','forecast_hour','station_id','i','j',*variables])
            for row in data:
                writer.writerow([int(row[0]),float(row[1]),int(row[2]),int(row[3]),int(row[4]),
                                 *map(float,row[selected])])
    elif format=='station-csv':
        from datetime import datetime,timedelta,timezone
        import pandas as pd
        from gpuwrf.verify.observations import validate_rows
        origin=datetime.strptime(start_date,'%Y-%m-%d_%H:%M:%S').replace(tzinfo=timezone.utc)
        records=[]
        for row in data:
            valid=(origin+timedelta(hours=float(row[1]))).isoformat()
            def add(variable,value,**extra):
                records.append(dict(station_uid=station_uid,valid_time=valid,interval_start=valid,
                                    variable=variable,value=value,**extra))
            if 'T2' in variables:
                add('t2',float(row[5])-273.15)
            if 'U10' in variables:
                u,v=map(float,row[[7,8]])
                speed=float(np.hypot(u,v))
                add('wind',speed,u_ms=u,v_ms=v)
                if speed>0:
                    add('wind_direction',float(np.degrees(np.arctan2(-u,-v))%360))
        frame=validate_rows(pd.DataFrame(records))
        with target.open('x',newline='',encoding='ascii') as file:
            frame.to_csv(file,index=False)
    else:
        from gpuwrf.io.netcdf_lock import Dataset
        with Dataset(target,'w',clobber=False) as ds:
            ds.createDimension('time',None)
            time=ds.createVariable('time','f8',('time',))
            time.units='hours since '+start_date.replace('_',' ')
            time.calendar='standard'
            time[:]=data[:,1]
            ds.station_header=lines[0]
            ds.source_file=source.name
            ds.sampling='original completed WRF steps; no interpolation'
            for index,name in zip(selected,variables):
                variable=ds.createVariable(name,'f8',('time',))
                variable.units=UNITS[name]
                variable[:]=data[:,index]
            for key,index in [('domain_id',0),('station_id',2),('i',3),('j',4)]:
                ds.setncattr(key,int(rows[0][index]))
    return {'source':str(source),'path':str(target),'format':format,
            'rows':len(records) if format=='station-csv' else len(data),
            'samples':len(data),'variables':list(variables)}


def command(args):
    from gpuwrf.cli import _fail
    try:
        receipt=export_surface(args.input,args.output,variables=args.variables.split(','),
                               interval_seconds=args.interval_seconds,format=args.format,station_uid=args.station_uid)
    except (ValueError,OSError) as exc:
        return _fail(str(exc))
    import json
    print(json.dumps(receipt,sort_keys=True))
    return 0
