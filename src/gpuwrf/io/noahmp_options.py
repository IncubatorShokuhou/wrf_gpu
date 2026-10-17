"""Noah-MP option admission and static selection (no JAX/backend imports).

Defaults use the existing program verbatim. Nondefault choices must be listed
here only after their producer and operational consumer have been qualified.
"""
from dataclasses import asdict,dataclass
from pathlib import Path
import math
import re


DEFAULTS = {
    "dveg": 4, "opt_crs": 1, "opt_btr": 1, "opt_run": 3, "opt_sfc": 1,
    "opt_frz": 1, "opt_inf": 1, "opt_rad": 3, "opt_alb": 2, "opt_snf": 1,
    "opt_tbot": 2, "opt_stc": 1, "opt_gla": 1, "opt_rsf": 1,
    "opt_soil": 1, "opt_pedo": 1, "opt_crop": 0, "opt_irr": 0, "opt_irrm": 0,
    "opt_infdv": 0, "opt_tdrn": 0,
}
SUPPORTED = {key: (value,) for key, value in DEFAULTS.items()}
SUPPORTED.update(opt_gla=(1,2),opt_alb=(1, 2), opt_snf=(1, 2, 3, 4, 5), opt_irr=(0,1,2,3), opt_irrm=(0,1,2,3),
                 opt_btr=(1,2,3),opt_rsf=(1,2,3,4),dveg=tuple(range(1,10)),
                 opt_tbot=(1,2),opt_stc=(1,2,3),opt_frz=(1,2),opt_rad=(1,2,3),opt_crs=(1,2),opt_sfc=(1,2),opt_inf=(1,2),opt_run=(1,2,3,4,5,6,7,8),opt_crop=(0,1),opt_soil=(1,2,3),opt_infdv=(0,1,2,3),opt_tdrn=(0,1,2))
BATS_KEYS = ("BATS_COSZ", "BATS_VIS_NEW", "BATS_NIR_NEW", "BATS_VIS_AGE",
             "BATS_NIR_AGE", "BATS_VIS_DIR", "BATS_NIR_DIR")


@dataclass(frozen=True)
class NoahMPOptions:
    """Only nondefault selections ride OperationalNamelist's static aux."""
    opt_alb: int = 2
    opt_snf: int = 1
    bats: tuple[float, ...] | None = None
    opt_irr: int = 0
    opt_irrm: int = 0
    opt_btr: int = 1
    opt_rsf: int = 1
    rsurf_snow: float | None = None
    dveg: int = 4
    opt_tbot: int = 2
    opt_stc: int = 1
    opt_frz: int = 1
    opt_rad: int = 3
    opt_crs: int = 1
    opt_sfc: int = 1
    opt_inf: int = 1
    opt_run: int = 3
    fsatmx: float | None = None
    timean: float | None = None
    opt_crop: int = 0
    opt_soil: int = 1
    opt_infdv: int = 0
    opt_tdrn: int = 0
    drain_layer_opt: int | None = None

    def __post_init__(self):
        if self.opt_crop==2:
            from gpuwrf.io.scheme_catalog import _GECROS_REFUSAL_REASON
            raise ValueError(_GECROS_REFUSAL_REASON)
        for key in ("opt_alb", "opt_snf", "opt_irr", "opt_irrm", "opt_btr", "opt_rsf", "dveg","opt_tbot","opt_stc","opt_frz","opt_rad","opt_crs","opt_sfc","opt_inf","opt_run","opt_crop","opt_soil","opt_infdv","opt_tdrn"):
            if getattr(self, key) not in SUPPORTED[key]:
                raise ValueError(f"Noah-MP {key}: supported values {SUPPORTED[key]}")
        if self.opt_alb == 1 and (self.bats is None or len(self.bats) != 7
                                 or not all(math.isfinite(x) for x in self.bats)
                                 or self.bats[0] <= 0):
            raise ValueError("BATS snow albedo requires seven finite MPTABLE parameters (BATS_COSZ > 0)")
        if self.opt_rsf==4 and (self.rsurf_snow is None or not math.isfinite(self.rsurf_snow) or self.rsurf_snow<=0.):
            raise ValueError("opt_rsf=4 requires finite positive MPTABLE RSURF_SNOW")
        if self.opt_run in (1,2) and (self.fsatmx is None or self.timean is None or not math.isfinite(self.timean) or not 0.<=self.fsatmx<=1.):
            raise ValueError("opt_run=1/2 requires finite actual MPTABLE TIMEAN and FSATMX in [0,1]")
        if self.opt_run==8 and self.opt_infdv==0:raise ValueError('Dynamic VIC opt_run=8 requires opt_infdv=1/2/3; WRF leaves FSUR unassigned for 0')
        if self.opt_tdrn and self.opt_run!=3:raise ValueError('WRF tile drainage only runs with opt_run=3')
        if self.opt_tdrn and self.drain_layer_opt not in range(6):raise ValueError('Tile drainage requires actual MPTABLE DRAIN_LAYER_OPT=0..5')
        if self.opt_run==5:
            changed=[key for key,default in DEFAULTS.items() if key not in ('opt_run','opt_inf') and hasattr(self,key) and getattr(self,key)!=default]
            if changed:raise ValueError('MMF opt_run=5 retains its qualified Noah-MP defaults except opt_inf=1/2; unqualified combination: '+','.join(changed))


def read_bats_parameters(table: Path | None = None) -> tuple[float, ...]:
    """Load the actual WRF table, once during binding, never in a timestep."""
    if table is None:
        from gpuwrf.config.paths import wrf_run_dir
        table = wrf_run_dir() / "MPTABLE.TBL"
    text = "\n".join(line.split("!", 1)[0] for line in Path(table).read_text().splitlines())
    block = re.search(r"&noahmp_global_parameters\b(.*?)/", text, re.I | re.S)
    if block is None:
        raise ValueError("MPTABLE.TBL lacks &noahmp_global_parameters")
    values = []
    for key in BATS_KEYS:
        match = re.search(rf"\b{key}\s*=\s*([-+\d.eEdD]+)", block[1], re.I)
        if match is None:
            raise ValueError(f"MPTABLE.TBL lacks {key}")
        values.append(float(match[1].lower().replace("d", "e")))
    return tuple(values)


def read_global_parameter(key,*,block='noahmp_global_parameters'):
    from gpuwrf.config.paths import wrf_run_dir
    text="\n".join(line.split("!",1)[0] for line in (wrf_run_dir()/"MPTABLE.TBL").read_text().splitlines())
    found=re.search(rf"&{block}\b(.*?)/",text,re.I|re.S)
    match=None if found is None else re.search(rf"\b{key}\s*=\s*([-+\d.eEdD]+)",found[1],re.I)
    if match is None:raise ValueError(f"MPTABLE.TBL lacks {key}")
    return float(match[1].lower().replace("d","e"))


def selected_options(wrf_namelist):
    """Default requests return None to preserve released treedefs and programs."""
    group = wrf_namelist.get("noah_mp") or {}
    def value(key):
        raw = group.get(key, DEFAULTS[key])
        if isinstance(raw, (list, tuple)):
            if len(raw) != 1:
                raise ValueError(f"&noah_mp {key} is a single-entry WRF option")
            raw = raw[0]
        if raw not in SUPPORTED[key]:
            if key=='opt_crop' and raw==2:
                from gpuwrf.io.scheme_catalog import _GECROS_REFUSAL_REASON
                raise ValueError(_GECROS_REFUSAL_REASON)
            raise ValueError(f"&noah_mp {key}={raw!r}: supported values {SUPPORTED[key]}")
        return raw
    alb, snf, irr, irrm = (int(value(key)) for key in ("opt_alb","opt_snf","opt_irr","opt_irrm"))
    btr,rsf=int(value("opt_btr")),int(value("opt_rsf"))
    dveg=int(value("dveg"))
    thermal={key:int(value(key)) for key in ("opt_tbot","opt_stc","opt_frz","opt_rad","opt_crs","opt_sfc","opt_inf","opt_run","opt_crop","opt_soil","opt_infdv","opt_tdrn")}
    if thermal["opt_run"] in (1,2):thermal.update(fsatmx=read_global_parameter("FSATMX"),timean=read_global_parameter("TIMEAN"))
    if thermal['opt_tdrn']:thermal['drain_layer_opt']=int(read_global_parameter('DRAIN_LAYER_OPT',block='noahmp_tiledrain_parameters'))
    if (alb, snf, irr, irrm,btr,rsf,dveg) == (2, 1, 0, 0,1,1,4) and all(thermal[key]==DEFAULTS[key] for key in DEFAULTS if key in thermal):
        return None
    result=NoahMPOptions(alb, snf, read_bats_parameters() if alb == 1 else None, irr, irrm,btr,rsf,
                         read_global_parameter("RSURF_SNOW") if rsf==4 else None,dveg,**thermal)
    # MMF owns RUN5/INF selection in both native and daily pipelines. Its
    # qualified default controls need no duplicate natural-soil auxiliary.
    return None if result.opt_run==5 else result


def irrigation_configuration(namelist):
    """Resolved option tuple for checkpoint identity and E78 guards (backend-dark)."""
    from collections.abc import Mapping
    if isinstance(namelist,Mapping):
        opts=namelist.get("noahmp_options",namelist.get("noah_mp",namelist))
    else:
        opts=getattr(namelist,"noahmp_options",None)
    if opts is None:
        return (0,0)
    get=opts.get if isinstance(opts,Mapping) else lambda key,default:getattr(opts,key,default)
    config=tuple(get(key,0) for key in ("opt_irr","opt_irrm"))
    if any(value not in (0,1,2,3) for value in config):
        raise ValueError("Noah-MP irrigation needs resolved scalar opt_irr/opt_irrm values 0..3")
    return tuple(int(value) for value in config)


def restart_configuration(namelist):
    """Static physics identity, including actual snow-table constants (E78)."""
    from collections.abc import Mapping
    opts=namelist.get("noahmp_options") if isinstance(namelist,Mapping) else getattr(namelist,"noahmp_options",None)
    if opts is None:return None
    if not isinstance(opts,NoahMPOptions):raise ValueError("Noah-MP restart needs resolved NoahMPOptions (E78)")
    config=asdict(opts)
    if config["bats"] is not None:config["bats"]=tuple(config["bats"])
    return config


def validate_option_restart(recorded,namelist=None):
    """Old inactive restarts remain readable; active selections must match."""
    if recorded is not None:
        if not isinstance(recorded,dict):raise ValueError("Noah-MP restart options are malformed (E78)")
        try:
            recorded=asdict(NoahMPOptions(**recorded))
            # JSON arrays become lists while static BATS uses a tuple.
            if recorded["bats"] is not None:recorded["bats"]=tuple(recorded["bats"])
        except (TypeError,ValueError) as error:
            raise ValueError("Noah-MP restart options are malformed (E78)") from error
    if namelist is not None and recorded!=restart_configuration(namelist):
        raise ValueError("Noah-MP restart options differ from requested selection/table (E78)")






def option_refusals(wrf_namelist, *, replay=False):
    """Refuse unported choices, invalid codes and the WRF-inapplicable mosaic."""
    errors = []
    requested = False
    run_choice=(wrf_namelist.get('noah_mp') or {}).get('opt_run',3)
    run_choice=run_choice[0] if isinstance(run_choice,(list,tuple)) and run_choice else run_choice
    for key, raw in (wrf_namelist.get("noah_mp") or {}).items():
        if key not in SUPPORTED:
            continue
        values = raw if isinstance(raw, (list, tuple)) else (raw,)
        if len(values) != 1:
            errors.append(f"&noah_mp {key} is a single-entry WRF option")
        for value in values:
            requested |= value != DEFAULTS[key]
            if value not in SUPPORTED[key]:
                if key=='opt_crop' and value==2:
                    from gpuwrf.io.scheme_catalog import _GECROS_REFUSAL_REASON
                    errors.append(_GECROS_REFUSAL_REASON)
                else:errors.append(f"&noah_mp {key}={value!r}: supported values {SUPPORTED[key]}; "
                                   "other choices are not WRF-defined or ported")
            elif replay and value != DEFAULTS[key] and not (run_choice==5 and key in ('opt_run','opt_inf')):
                errors.append(f"&noah_mp {key} requires native prognostic Noah-MP; "
                              "CPU-history replay does not run this land model")
    physics = wrf_namelist.get("physics") or {}
    group=wrf_namelist.get("noah_mp") or {}
    def scalar(key,default):
        raw=group.get(key,default)
        return raw[0] if isinstance(raw,(tuple,list)) and raw else raw
    if scalar('opt_run',3)==8 and scalar('opt_infdv',0)==0:
        errors.append('Dynamic VIC opt_run=8 requires opt_infdv=1/2/3; WRF leaves FSUR unassigned for 0')
    if scalar('opt_tdrn',0) and scalar('opt_run',3)!=3:
        errors.append('WRF tile drainage only runs with opt_run=3')
    irr=group.get("opt_irr",0)
    irr=irr[0] if isinstance(irr,(list,tuple)) and irr else irr
    if irr:
        lake=physics.get("sf_lake_physics",0)
        lakes=lake if isinstance(lake,(list,tuple)) else (lake,)
        if any(value!=0 for value in lakes):
            errors.append("Noah-MP irrigation plus compact lake tiles is not qualified")
    if scalar('opt_run',3)==5:
        changed=[key for key,default in DEFAULTS.items() if key not in ('opt_run','opt_inf','opt_gla') and scalar(key,default)!=default]
        if changed:errors.append('MMF opt_run=5 retains its qualified Noah-MP defaults except opt_inf=1/2; unqualified combination: '+','.join(changed))
    dveg=group.get("dveg",4)
    dveg=dveg[0] if isinstance(dveg,(list,tuple)) and dveg else dveg
    crop=group.get("opt_crop",0)
    crop=crop[0] if isinstance(crop,(list,tuple)) and crop else crop
    if dveg in (2,5,6) or crop==1:
        crs=group.get("opt_crs",1)
        crs=crs[0] if isinstance(crs,(list,tuple)) and crs else crs
        if crs!=1:errors.append("Dynamic Noah-MP vegetation requires opt_crs=1; WRF CANRES/Jarvis returns PSN=-999.99 for inactive carbon")
        for key in ("sf_lake_physics","sf_urban_physics"):
            raw=physics.get(key,0)
            values=raw if isinstance(raw,(tuple,list)) else (raw,)
            if any(value!=0 for value in values):errors.append(f"Dynamic Noah-MP vegetation plus {key} is not qualified")
    surface = physics.get("sf_surface_physics", 4)
    surfaces = surface if isinstance(surface, (list, tuple)) else (surface,)
    if requested and any(value != 4 for value in surfaces):
        errors.append("Nondefault &noah_mp options require sf_surface_physics=4 on every domain")
    mosaic = physics.get("sf_surface_mosaic", 0)
    mosaic = mosaic[0] if isinstance(mosaic, (list, tuple)) else mosaic
    if mosaic != 0:
        errors.append("&physics sf_surface_mosaic=1 is implemented by WRF only for "
                      "Noah-classic (sf_surface_physics=2); WRF ignores it with Noah-MP "
                      "(module_surface_driver.F:2764,3146). Noah-classic mosaic is REFUSED for v0.4.0: "
                      "per-tile REDPRM/category binding, 24 surface plus 4 soil carries, restart "
                      "and original logarithmic/reciprocal aggregation work are missing; v0.4.x follow-up")
    skin = physics.get("sst_skin", 0)
    skin = skin[0] if isinstance(skin, (list, tuple)) else skin
    if skin not in (0, 1):
        errors.append("&physics sst_skin supports only 0/1")
    elif skin:
        if replay:
            errors.append("sst_skin=1 requires the native full-carry pipeline; CPU-history replay is unsupported")
        for key, wanted in (("sf_surface_physics", 4), ("ra_sw_physics", 4), ("ra_lw_physics", 4), ("sf_lake_physics", 0)):
            raw = physics.get(key, wanted)
            values = raw if isinstance(raw, (list, tuple)) else (raw,)
            if any(value != wanted for value in values):
                errors.append(f"sst_skin=1 requires {key}={wanted} on every domain")
    return errors
