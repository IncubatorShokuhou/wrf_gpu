"""MY2's implicit radar requires native carry, including at direct API admission."""
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('radar',[None,0,1])
def test_implicit_and_explicit_radar_refuse_replay_preserve_native(radar):
    from gpuwrf.io.namelist_binding import namelist_honour_report
    physics={'mp_physics':[9]}
    if radar is not None:physics['do_radar_ref']=[radar]
    wrf={'physics':physics}
    errors,_=namelist_honour_report(wrf,('d01',),replay=True)
    assert any('Milbrandt' in e and 'native mode' in e for e in errors),errors
    errors,_=namelist_honour_report(wrf,('d01',),replay=False)
    assert not errors,errors


@pytest.mark.parametrize('radar',[None,0,1])
def test_cli_refuses_milbrandt_replay_preflight(tmp_path,capsys,radar):
    from gpuwrf import cli
    fixture=Path(__file__).resolve().parents[2]/'fixtures/wn3_20260227_cadence.namelist'
    import re
    text=re.sub(r'^\s*mp_physics\s*=.*$', ' mp_physics = 9, 9, 9,',fixture.read_text(),flags=re.M)
    text=re.sub(r'^\s*max_dom\s*=.*$', ' max_dom = 1,',text,flags=re.M)
    if radar is not None:text=text.replace('&physics','&physics\n do_radar_ref = '+str(radar)+',',1)
    (tmp_path/'namelist.input').write_text(text)
    for hour in ('00','01'):(tmp_path/f'wrfout_d01_2026-02-27_{hour}:00:00').touch()
    rc=cli.main(['run','--input-dir',str(tmp_path),'--output-dir',str(tmp_path/'out'),
                 '--max-dom','1','--domain','d01','--hours','1','--dry-run'])
    error=capsys.readouterr().err
    assert rc!=0 and 'Milbrandt' in error and 'native mode' in error,error


@pytest.mark.parametrize('radar',[None,0,1])
def test_direct_daily_api_refuses_before_state_initialization(monkeypatch,tmp_path,radar):
    from gpuwrf.integration import daily_pipeline as daily
    from gpuwrf.io.gen2_accessor import Gen2Run
    text='&domains\n max_dom=1,\n/\n&physics\n mp_physics=9,\n'
    if radar is not None:text+=' do_radar_ref='+str(radar)+',\n'
    (tmp_path/'namelist.input').write_text(text+'/\n')
    # Deliberately no State: refusal must happen before buffer preparation.
    replay=SimpleNamespace(run=Gen2Run(tmp_path))
    monkeypatch.setattr(daily,'build_replay_case',lambda *a,**k:replay)
    with pytest.raises(ValueError,match='Milbrandt.*native mode.*radar'):
        daily._build_real_case(daily.DailyPipelineConfig(run_id=str(tmp_path),domain='d01'))
