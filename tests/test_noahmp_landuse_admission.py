"""Refuse dataset mixing at the real Noah-MP input boundary, before allocation."""
from pathlib import Path
import shutil

import numpy as np
import pytest

from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state


SOURCE = Path(__file__).resolve().parents[1] / "examples/switzerland_d01"


@pytest.mark.parametrize("dataset,count", [
    ("USGS", 24), ("USGS", 28), ("NLCD40", 40), ("LCZ", 61), (None, 21),
])
def test_unsupported_dataset_refused_before_any_field_read(tmp_path, monkeypatch, dataset, count):
    # Metadata-only wrfinput: there are deliberately no data variables. A field
    # read is forbidden so the error must be about dataset admission itself.
    with Dataset(tmp_path / "wrfinput_d01", "w") as ds:
        if dataset is not None:
            ds.MMINLU = dataset
        ds.NUM_LAND_CAT = np.int32(count)

    def unexpected_field_read(*args, **kwargs):
        pytest.fail("dataset refusal must precede field reads")

    monkeypatch.setattr(Gen2Run, "load_wrfinput", unexpected_field_read)
    with pytest.raises(ValueError, match="Noah-MP requires MMINLU=.*MODIFIED_IGBP_MODIS_NOAH") as error:
        build_noahmp_land_state(tmp_path, "d01")
    assert f"NUM_LAND_CAT={count}" in str(error.value)
    assert "wrfinput_d01" in str(error.value)


def test_real_input_with_usgs_metadata_is_refused(tmp_path):
    # Reproduce the previous silent admission using real fields. This is an
    # interface mutation of Swiss input, not USGS numerical validation.
    shutil.copyfile(SOURCE / "wrfinput_d01", tmp_path / "wrfinput_d01")
    (tmp_path / "namelist.input").symlink_to(SOURCE / "namelist.input")
    with Dataset(tmp_path / "wrfinput_d01", "r+") as ds:
        ds.MMINLU = "USGS"
        ds.NUM_LAND_CAT = np.int32(24)
    with pytest.raises(ValueError, match="got MMINLU='USGS'"):
        build_noahmp_land_state(tmp_path, "d01")


def test_shipped_modis_input_still_builds_finite_land_state():
    import jax
    land, static, _ = build_noahmp_land_state(SOURCE, "d01")
    assert (static.parameters.isurban, static.parameters.iswater,
            static.parameters.isice, static.parameters.isbarren) == (13, 17, 15, 16)
    for leaf in jax.tree_util.tree_leaves(jax.device_get((land, static))):
        assert np.isfinite(np.asarray(leaf)).all()
