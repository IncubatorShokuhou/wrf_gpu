"""Localize a literal divergence against print-only original statement values."""
import argparse
import json
import linecache
from pathlib import Path
import re
import sys
import numpy as np
from literal_numpy import LiteralNumpy
from gpuwrf.physics.ntu import scalar_generated as port


def main():
    p=argparse.ArgumentParser();p.add_argument('oracle',type=Path);p.add_argument('routine');p.add_argument('row',type=int);p.add_argument('out',type=Path)
    args=p.parse_args();meta=json.loads((args.oracle/'manifest.json').read_text());data=np.load(args.oracle/'scalars.npz')
    originals={};current=0
    for line in (args.oracle/'trace.txt').read_text().splitlines():
        n,name,value=line.split();n=int(n)
        if n==0:
            current=int(value)
            if current>args.row:break
        elif current==args.row:originals[n,name]=np.float32(value)
    params=meta['signatures'][args.routine];values=data[args.routine+'_input'][args.row-1]
    expected=data[args.routine+'_output'][args.row-1]
    xp=LiteralNumpy();record=[];previous={}
    def observe(frame,event,arg):
        if frame.f_code.co_name!=args.routine:return observe
        if event in ('line','return'):
            prior=previous.get(id(frame))
            if prior:
                source=linecache.getline(frame.f_code.co_filename,prior)
                match=re.search(r"v\['(\w+)'\].*# WRF:(\d+)",source)
                if match:
                    name,n=match[1],int(match[2]);v=np.asarray(frame.f_locals['v'][name])
                    if (n,name) in originals:
                        truth=originals[n,name];candidate=v.item()
                        byte=v.astype(np.float32).tobytes()==truth.tobytes()
                        record.append({'line':n,'name':name,'actual':float(candidate),'expected':float(truth),'BYTE':byte})
            previous[id(frame)]=frame.f_lineno
        return observe
    with np.errstate(all='ignore'):
        sys.settrace(observe)
        try:outputs=getattr(port,args.routine)({k:xp.asarray(values[i]) for i,k in enumerate(params)},xp=xp)
        finally:sys.settrace(None)
    first=next((x for x in record if not x['BYTE']),None)
    result={'routine':args.routine,'row':args.row,'original_observations':len(originals),'compared':len(record),'first_difference':first,'observations':record}
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='observations'}))


if __name__=='__main__':main()
