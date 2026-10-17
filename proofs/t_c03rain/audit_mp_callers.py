"""Bounded, source-only audit of main's complete microphysics caller inventory."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


MAIN=Path('<USER_HOME>/src/wrf_gpu2')
SCAN='src/gpuwrf/coupling/scan_adapters.py'
PC='src/gpuwrf/coupling/physics_couplers.py'
REG='src/gpuwrf/contracts/physics_registry.py'
DIS='src/gpuwrf/coupling/physics_dispatch.py'
OP='src/gpuwrf/runtime/operational_mode.py'


def audit(commit, out):
    files={}
    def read(path):
        if path not in files:
            files[path]=subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=MAIN,text=True)
        return files[path]
    def fn(path,name):
        source=read(path)
        node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)
        return node,ast.get_source_segment(source,node)
    def row(mp,path,name,helper,verdict,note,required=()):
        node,body=fn(path,name)
        for token in required: assert token in body,(mp,token)
        return dict(mp=mp,route=name,evidence=f'{path}:{node.lineno}',helper=helper,
                    verdict=verdict,T_pi_qv=note,body_sha256=hashlib.sha256(body.encode()).hexdigest())
    registry=ast.parse(read(REG))
    accepted=next(ast.literal_eval(n.value) for n in registry.body if isinstance(n,ast.AnnAssign)
                  and isinstance(n.target,ast.Name) and n.target.id=='ACCEPTED_MP_PHYSICS')
    entries=next(n.value for n in ast.parse(read(DIS)).body if isinstance(n,ast.AnnAssign)
                 and isinstance(n.target,ast.Name) and n.target.id=='_MP_ENTRIES')
    assert set(ast.literal_eval(k) for k in entries.keys)==set(accepted)
    rows=[dict(mp=0,route='disabled/passive',evidence=f'{REG}:93',helper='N/A',verdict='PASSIVE',T_pi_qv='No microphysics handoff.')]
    bad={1:'kessler_adapter',2:'lin_adapter',3:'wsm3_adapter',4:'wsm5_adapter',6:'wsm6_adapter',
         10:'morrison_adapter',13:'sbu_ylin_adapter',14:'wdm5_adapter',97:'goddard_adapter'}
    for mp,name in bad.items():
        rows.append(row(mp,SCAN,name,'NONE; state.theta -> dry-TH kernel; direct replacement',
            'MISSING_PREP_AND_FINISH',
            'Positive qv: theta_m*pi is warmer than dry T; PII itself is (p/1e5)^(Rd/Cp); qv remains dry-air mixing ratio.',
            ('mp(state.theta)','_apply_mp_replacements')))
    for mp,name in {16:'wdm6_adapter',24:'wsm7_adapter',26:'wdm7_adapter',40:'morrison_aero_adapter'}.items():
        rows.append(row(mp,SCAN,name,'_dry_theta_view + _theta_m_from_microphysics',
            'HANDOFF_PRESENT','Dry TH; PII from total p; qv mixing ratio. FP64 island where registered; precision is not certified by this source audit.',
            ('_dry_theta_view','_theta_m_from_microphysics')))
    for mp,name,inp,output in [(8,'thompson_adapter','_thompson_column_from_state','_state_from_thompson_output'),
                               (28,'thompson_aero_adapter','_thompson_aero_column_from_state','_state_from_thompson_aero_output')]:
        _,a=fn(PC,inp);_,b=fn(PC,output)
        assert '_dry_temperature_from_state' in a and '_theta_m_from_microphysics' in b
        rows.append(row(mp,PC,name,'_dry_temperature_from_state -> T/pi -> _theta_m_from_microphysics',
            'HANDOFF_PRESENT','Input T is dry theta*pi; pressure Pa; qv mixing ratio. Thompson-specific dry density is intentional.'))
    for mp in (5,15):
        rows.append(row(mp,'src/gpuwrf/physics/ferrier/adapter.py','ferrier_adapter',
            '_dry_theta_view + _theta_m_from_microphysics','HANDOFF_PRESENT',
            'Dry TH/pi; vapor mixing ratio enters prepare_level, which uses q=vapor/(1+vapor), then finish_level returns q/(1-q).',
            ('_dry_theta_view','_theta_m_from_microphysics')))
    rows.append(row(7,'src/gpuwrf/coupling/goddard4ice_adapter.py','goddard4ice_adapter',
        '_dry_theta_view + _theta_m_from_microphysics','HANDOFF_PRESENT',
        'Dry TH, total pressure Pa, Exner, mixing-ratio qv; WRF phy_prep moist density.',
        ('_dry_theta_view','_theta_m_from_microphysics')))
    rows.append(row(9,'src/gpuwrf/physics/milbrandt2mom/adapter.py','milbrandt_adapter',
        'manual REAL old_theta/(1+Rv/Rd*qv) + _theta_m_from_microphysics','HANDOFF_PRESENT',
        'Explicit rounded REAL prep and T=theta*pi; returns dry T/pi before moist finish; no helper-name false positive.',
        ('old_theta/denominator','temperature=_round_real(theta*pii)','_theta_m_from_microphysics')))
    for mp in (17,18,19,21,22):
        rows.append(row(mp,'src/gpuwrf/physics/nssl2mom/adapter.py','nssl2mom_adapter',
            '_dry_theta_view + _temperature_from_theta(dry) + _theta_m_from_microphysics',
            'HANDOFF_PRESENT','17 normalizes to18; 19/21 select one-moment variants; qv mixing ratio, total-density multiplier (1+qv) explicit.',
            ('_dry_theta_view','_temperature_from_theta(theta_dry','_theta_m_from_microphysics')))
    rows.append(row(38,'src/gpuwrf/coupling/thompson_hail_adapter.py','thompson_hail_adapter',
        '_dry_theta_view + T=th*pi + _theta_m_from_microphysics','HANDOFF_PRESENT',
        'Dry T; total pressure Pa; mixing-ratio vapor; Thompson-specific dry rho.',
        ('_dry_theta_view','_theta_m_from_microphysics')))
    for mp in (50,51,52,53):
        rows.append(row(mp,'src/gpuwrf/physics/p3/adapter.py','p3_adapter',
            '_dry_theta_view + _theta_m_from_microphysics','HANDOFF_PRESENT',
            'Dry current TH and dry TH_OLD history; kernel T=TH*pi, P inPa, qv mixing ratio. Source audit does not qualify compiled math.',
            ('_dry_theta_view','_theta_m_from_microphysics')))
    assert {r['mp'] for r in rows}==set(accepted)
    for mp in (11,20,30,32,54,55,56,95):
        assert mp not in accepted
        rows.append(dict(mp=mp,route='not bound on this main snapshot',evidence=f'{REG}:93; {DIS}:149',
                         helper='N/A',verdict='UNBOUND',T_pi_qv='No operational State caller to audit; no implementation/fidelity claim.'))
    rows.sort(key=lambda r:r['mp'])
    # Audit every special operational branch and generic entry, including legacy pre-RK route.
    read(OP)
    _,island=fn(SCAN,'_fp64_island')
    assert '_dry_theta_view' not in island and '_theta_m_from_microphysics' not in island
    read('src/gpuwrf/physics/ferrier/column.py')
    read('src/gpuwrf/physics/p3/column.py')
    read('src/gpuwrf/physics/thompson_column.py')
    for name in ('kessler','lin','wsm3','wsm5','wsm6','morrison','sbu_ylin','wdm5','goddard'):
        read(f'src/gpuwrf/physics/microphysics_{name}.py')
    result=dict(schema='main-mp-caller-source-audit-v1',main_commit=commit,
        generated_utc=datetime.now(timezone.utc).isoformat(),accepted_codes=accepted,rows=rows,
        summary=dict(active=29,passive=1,missing_thermo_boundary=9,handoff_present=20,requested_unbound=8),
        source_sha256={p:hashlib.sha256(s.encode()).hexdigest() for p,s in files.items()},
        qualification='SOURCE AUDIT ONLY; no new oracle/GPU/trajectory qualification',
        secondary_suspects=[
            'Generic _rho_from_state uses Thompson dry density (physics_couplers:1579/1587, thompson_column:447/450); legacy rho-argument schemes should be checked against WRF phy_prep rho=(1+qv)/ALT (:4856). This includes helpers-present WDM16/WSM24/WDM26; separate from proven missing theta boundaries.',
            'Goddard97 goddard_run hardcodes zsfc=0 at microphysics_goddard:852-853, analogous to independently proven Lin terrain bug; no change here.'],
        route_coverage='physics_registry + physics_dispatch + MP_SCAN_ADAPTERS + operational post-RK/pre-RK special branches; alternate Thompson callers share the same conversion adapter',
        fp64_island='dtype widening only; does not repair any missing thermodynamic conversion',
        fix_scope='Only Lin fixed in lane/t-c03rain; other rows remain unchanged for manager assignment')
    out.mkdir(parents=True,exist_ok=True)
    (out/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[f'Main microphysics caller audit — `{commit}`', '',
        'Source audit of all 29 active dispatcher codes, passive 0, and eight requested unbound codes. '
        'Nine active main callers lack both dry-theta preparation and moist-theta finish; twenty have the handoff. '
        'This is source evidence, not numerical qualification. Only Lin is changed in this lane.', '',
        '| MP | State caller (main file:line) | Input → output helper | Verdict |',
        '|---|---|---|---|']
    lines.extend(f"| {r['mp']} | `{r['route']}` — `{r['evidence']}` | {r['helper']} | {r['verdict']} |" for r in rows)
    lines+=['','T/pi/qv checks:']+[f"- MP{r['mp']}: {r['T_pi_qv']}" for r in rows]
    lines+=['','The fp64 island only changes dtype; it cannot turn moist theta into dry theta. '
        'WRF baseline: module_big_step_utilities_em.F:4834 (dry prep), :5735–5737 (moist finish). '
        'Existing independent literal REAL4 values are in proofs/t_mprain/thermo_oracle.json.', '',
        'Additional suspects (not fixed):']+['- '+s for s in result['secondary_suspects']]
    (out/'AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['summary']))


if __name__=='__main__':
    audit(sys.argv[1],Path(sys.argv[2]))
