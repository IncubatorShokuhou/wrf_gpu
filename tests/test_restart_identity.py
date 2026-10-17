"""Native restart identities must survive the manifest/journal JSON boundary."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
from gpuwrf.io.noahmp_options import NoahMPOptions
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.restart_store import HistoryJournal, RestartStore, nested_identity


@pytest.fixture
def identity_case(tmp_path):
    grid = GridSpec.canary_3km_template()
    namelist = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=18.)
    config = NestedPipelineConfig(tmp_path, tmp_path / 'out', tmp_path / 'proof', hours=1, max_dom=1)
    start = datetime(2024, 6, 1, 12, tzinfo=timezone.utc)
    return config, namelist, start


@pytest.mark.parametrize('options', [
    None,
    NoahMPOptions(opt_irr=2, opt_irrm=3),
    NoahMPOptions(opt_alb=1, bats=(0.4, 0.9, 0.8, 0.2, 0.3, 0.4, 0.5)),
])
def test_native_identity_is_json_canonical(identity_case, options):
    config, namelist, start = identity_case
    namelist = replace(namelist, noahmp_options=options)
    identity = nested_identity(config, {'d01': SimpleNamespace(namelist=namelist)}, start)
    assert identity == json.loads(json.dumps(identity, allow_nan=False))
    assert identity['noahmp_irrigation_config']['d01'] == (
        [0, 0] if options is None else [options.opt_irr, options.opt_irrm])
    if options is not None and options.bats is not None:
        assert identity['noahmp_options_config']['d01']['bats'] == list(options.bats)


def test_native_journal_reopens_with_fresh_identity_and_refuses_changes(identity_case, tmp_path):
    config, namelist, start = identity_case
    bundles = {'d01': SimpleNamespace(namelist=namelist)}
    original = nested_identity(config, bundles, start)
    store = RestartStore(tmp_path / 'store', 10**7, 2, 0)
    store.root.mkdir()
    journal = HistoryJournal(store, original)
    fresh = nested_identity(config, bundles, start)
    HistoryJournal(store, fresh, journal.stream)
    for field in ('source_sha256', 'inputs', 'namelists', 'noahmp_irrigation_config'):
        changed = dict(fresh, **{field: {'changed': True}})
        with pytest.raises(ValueError, match='identity'):
            HistoryJournal(store, changed, journal.stream)
