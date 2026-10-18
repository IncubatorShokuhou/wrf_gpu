"""WRF surface_driver's post-LSM horizontal SWDOWN history convention."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.operational_state import HistoryDiagnostics
from gpuwrf.runtime.aot_cheap_key import IMPORT_TIME_ENV_CONSTANTS, module_const_env_hash

FIXTURE = Path(__file__).parent/'fixtures/history_swdown_horizontal_0227.json'
ROOT = Path(__file__).resolve().parents[1]


def inputs(monkeypatch):
    f=json.loads(FIXTURE.read_text())['fields']
    field=lambda key:jnp.asarray([f[key]],jnp.float32)
    rad=SimpleNamespace(swdown=field('SWDNB'),swnorm=field('SWNORM'),glw=field('GLW'),
        coszen=field('COSZEN'),swup=field('SWUPB'),sw_toa_down=field('SWDNT'),
        sw_toa_up=field('SWUPT'),glw_up=field('LWUPB'),lw_toa_down=field('LWDNT'),
        lw_toa_up=field('LWUPT'))
    # MYJSFC/QNSESFC winds were added after this Noah-MP oracle was frozen.
    # Their documented default is None; all original WRF operands stay exact.
    history=HistoryDiagnostics(**{k:field(k.upper()) for k in HistoryDiagnostics._fields
                                 if k.upper() in f})
    state=SimpleNamespace(t_skin=field('TSK'),hfx=field('HFX'))
    namelist=SimpleNamespace(use_noahmp=True,ra_sw_physics=4,ra_lw_physics=4,
        slope_rad=1,grid=None,metrics=None)
    held=(field('SWNORM'),field('GLW'),field('COSZEN'))
    monkeypatch.setattr(op,'_history_pblh',lambda state,grid:field('PBLH'))
    def forbid(*args,**kwargs):
        raise AssertionError('History fix recomputed physics')
    for name in ('surface_layer_diagnostics','overlay_noahmp_land_diagnostics','rrtmg_radiation_diagnostics'):
        monkeypatch.setattr(op,name,forbid)
    return f,state,namelist,rad,history,held


def output(state,namelist,rad,history,held):
    return op.compute_m9_diagnostics(state,namelist,60*3600,
        noahmp_rad=held,radiation_diagnostics=rad,history_diagnostics=history)


def assert_wrf_horizontal(m9,fixture):
    actual=np.asarray(m9.swdown)[0]
    expected=np.asarray(fixture['SWDOWN'],np.float32)
    assert np.max(np.abs(actual-expected)) <= .001, 'WRF horizontal SWDOWN not restored'
    assert np.asarray(m9.swnorm)[0].tobytes()==np.asarray(fixture['SWNORM'],np.float32).tobytes()


def test_real_original_wrf_slopes_choose_horizontal_history(monkeypatch):
    f,state,namelist,rad,history,held=inputs(monkeypatch)
    before=[np.asarray(x).tobytes() for x in (*held,*history,state.t_skin,state.hfx)]
    monkeypatch.setattr(op,'_HISTORY_SWDOWN_HORIZONTAL',True)
    got=output(state,namelist,rad,history,held)
    assert_wrf_horizontal(got,f)
    assert [np.asarray(x).tobytes() for x in (*held,*history,state.t_skin,state.hfx)]==before


def test_off_uses_original_held_soldn_and_other_fields_are_unchanged(monkeypatch):
    f,state,namelist,rad,history,held=inputs(monkeypatch)
    monkeypatch.setattr(op,'_HISTORY_SWDOWN_HORIZONTAL',False)
    off=output(state,namelist,rad,history,held)
    assert np.asarray(off.swdown).tobytes()==np.asarray(held[0],np.float64).tobytes()
    monkeypatch.setattr(op,'_HISTORY_SWDOWN_HORIZONTAL',True)
    on=output(state,namelist,rad,history,held)
    for key in off._fields:
        if key=='swdown':continue
        a,b=np.asarray(getattr(off,key)),np.asarray(getattr(on,key))
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),key


def test_disabled_shortwave_stays_zero(monkeypatch):
    f,state,namelist,rad,history,held=inputs(monkeypatch)
    namelist.ra_sw_physics=0
    monkeypatch.setattr(op,'_HISTORY_SWDOWN_HORIZONTAL',True)
    got=output(state,namelist,rad,history,held)
    assert np.count_nonzero(np.asarray(got.swdown))==0
    assert np.count_nonzero(np.asarray(got.swnorm))==0


def test_actual_selector_deletion_is_a_genuine_oracle_failure(monkeypatch):
    f,state,namelist,rad,history,held=inputs(monkeypatch)
    monkeypatch.setattr(op,'_HISTORY_SWDOWN_HORIZONTAL',False)
    with pytest.raises(AssertionError,match='horizontal SWDOWN'):
        assert_wrf_horizontal(output(state,namelist,rad,history,held),f)


def test_wrong_horizontal_operand_is_a_genuine_oracle_failure(monkeypatch):
    f,state,namelist,rad,history,held=inputs(monkeypatch)
    monkeypatch.setattr(op,'_HISTORY_SWDOWN_HORIZONTAL',True)
    rad.swdown=rad.swnorm
    with pytest.raises(AssertionError,match='horizontal SWDOWN'):
        assert_wrf_horizontal(output(state,namelist,rad,history,held),f)


def test_import_released_default_on_explicit_off_and_registered_resolved_key(monkeypatch):
    # v0.3.3 W2 defaults commit: the key is a released FAST_PATH_DEFAULT (ON when unset); an explicit 0 restores the legacy path.
    env=dict(os.environ,JAX_PLATFORMS='cpu',GPUWRF_FAST_DEFAULTS='1',PYTHONDONTWRITEBYTECODE='1',
             PYTHONPATH=str(ROOT/'src'))
    env.pop('GPUWRF_HISTORY_SWDOWN_HORIZONTAL',None)
    code='from gpuwrf.runtime import operational_mode as op; assert op._HISTORY_SWDOWN_HORIZONTAL is True'
    subprocess.run([sys.executable,'-c',code],env=env,check=True,timeout=60)
    code='from gpuwrf.runtime import operational_mode as op; assert op._HISTORY_SWDOWN_HORIZONTAL is False'
    subprocess.run([sys.executable,'-c',code],env=dict(env,GPUWRF_HISTORY_SWDOWN_HORIZONTAL='0'),check=True,timeout=60)
    assert ('gpuwrf.runtime.operational_mode','_HISTORY_SWDOWN_HORIZONTAL') in IMPORT_TIME_ENV_CONSTANTS
    monkeypatch.setattr(op,'_HISTORY_SWDOWN_HORIZONTAL',False);off=module_const_env_hash()
    monkeypatch.setattr(op,'_HISTORY_SWDOWN_HORIZONTAL',True);on=module_const_env_hash()
    assert off!=on


def _flag_reads(tree):
    return [n for n in ast.walk(tree) if (isinstance(n,ast.Name) and n.id=='_HISTORY_SWDOWN_HORIZONTAL'
            and isinstance(n.ctx,ast.Load)) or (isinstance(n,ast.Attribute) and n.attr=='_HISTORY_SWDOWN_HORIZONTAL')]


def test_off_output_function_ast_is_exact_and_model_producers_unchanged():
    # Durable form of the 2026-10-07 source-isolation control (previously an AST freeze of every
    # operational_mode function against a fixed main commit, which every later unrelated model change
    # broke). The feature property is checked on the CURRENT tree: (1) across ALL of src/gpuwrf the flag
    # is read exactly once, as the sole test of one If in compute_m9_diagnostics, so no model producer
    # depends on it; (2) that selector is AST-identical to the frozen selector on main 1e9f09c07; (3) the
    # value it rebinds (swdown_out) only reaches the M9 history field; (4) removing the selector leaves the
    # OFF output function, whose numerics are pinned by the OFF/ON tests above.
    path=ROOT/'src/gpuwrf/runtime/operational_mode.py'
    reads={}
    for source in sorted((ROOT/'src/gpuwrf').rglob('*.py')):
        found=_flag_reads(ast.parse(source.read_text()))
        if found:reads[source.relative_to(ROOT).as_posix()]=found
    assert list(reads)==['src/gpuwrf/runtime/operational_mode.py'] and len(reads[list(reads)[0]])==1,reads
    baseline=subprocess.check_output(['git','show','1e9f09c07:src/gpuwrf/runtime/operational_mode.py'],cwd=ROOT,text=True)
    selectors={}
    for label,tree in (('frozen',ast.parse(baseline)),('current',ast.parse(path.read_text()))):
        functions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        m9=functions['compute_m9_diagnostics']
        selects=[n for n in m9.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name)
                 and n.test.id=='_HISTORY_SWDOWN_HORIZONTAL']
        assert len(selects)==1 and not selects[0].orelse,label
        selectors[label]=ast.dump(selects[0],include_attributes=False)
        if label=='current':
            assert _flag_reads(m9)==[selects[0].test]
            assert [ast.unparse(x) for x in selects[0].body]==['swdown_out = rad.swdown']
            loads=[n for n in ast.walk(m9) if isinstance(n,ast.Name) and n.id=='swdown_out'
                   and isinstance(n.ctx,ast.Load)]
            sinks=[k for n in ast.walk(m9) if isinstance(n,ast.Call) and ast.unparse(n.func)=='M9Diagnostics'
                   for k in n.keywords if isinstance(k.value,ast.Name) and k.value.id=='swdown_out']
            zeros=[n for n in ast.walk(m9) if isinstance(n,ast.Call) and ast.unparse(n)=='jnp.zeros_like(swdown_out)']
            assert [k.arg for k in sinks]==['swdown'] and len(loads)==len(sinks)+len(zeros)==2
            m9.body.remove(selects[0])
            assert not _flag_reads(m9)
    assert selectors['current']==selectors['frozen']


def test_pristine_surface_restore_source_binding():
    fixture=json.loads(FIXTURE.read_text())
    source=Path('<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/phys/module_surface_driver.F')
    if not source.exists():pytest.skip('original executable source is local evidence')
    data=source.read_bytes()
    assert hashlib.sha256(data).hexdigest()==fixture['source_sha256']
    text=data.decode()
    start=text.index('! topographic slope effects removed from SWDOWN and GSW here for output')
    restore=text[start:start+1100]
    for statement in ('SWSAVE = SWDOWN(i,j)','SWDOWN(i,j) = SWNORM(i,j)',
                      'SWNORM(i,j) = SWSAVE','GSW(i,j) = GSWSAVE(i,j)'):
        assert statement in restore
