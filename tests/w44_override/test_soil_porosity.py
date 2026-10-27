import numpy as np
import pytest

from gpuwrf.io.wrfinput_consistency import validate_soil_porosity, validate_fields
from test_array_api import view  # shared real-structured array/metadata fixture


def test_explicit_cell_and_layer_caps_are_used_without_table_lookup(monkeypatch):
    def no_table(*args, **kwargs):
        raise AssertionError("explicit porosity consulted SOILPARM")
    monkeypatch.setattr("gpuwrf.io.wrfinput_consistency._porosity", no_table)
    moisture = np.full((4, 2, 3), .39, np.float32)
    land = np.ones((2, 3), np.int32)
    lu = np.full((2, 3), 5)
    assert validate_soil_porosity(moisture, .4, land, lu)
    assert not validate_soil_porosity(moisture, .38, land, lu)
    cap = np.full(moisture.shape, .4, np.float32)
    cap[2, 0, 1] = .38
    assert not validate_soil_porosity(moisture, cap, land, lu)
    cap[2, 0, 1] = .39
    assert validate_soil_porosity(moisture, cap, land, lu)


def test_one_based_per_class_caps_and_no_mutation():
    moisture = np.broadcast_to(np.array([.3, .5], np.float32), (4, 1, 2)).copy()
    cap = np.array([np.nan, .3, .5], np.float32)
    soil = np.array([[1, 2]], np.int32)
    land, lu = np.ones((1, 2)), np.full((1, 2), 5)
    liquid = moisture.copy()
    arrays = (moisture, cap, soil, land, lu, liquid)
    before = [a.tobytes() for a in arrays]
    assert validate_soil_porosity(moisture, cap, land, lu, sh2o=liquid, isltyp=soil)
    assert before == [a.tobytes() for a in arrays]
    soil[0, 1] = 1
    assert not validate_soil_porosity(moisture, cap, land, lu, sh2o=liquid, isltyp=soil)


@pytest.mark.parametrize("index", [0, 3, 1.5, np.nan, np.inf])
def test_invalid_class_indices_refuse(index):
    assert not validate_soil_porosity(np.full((4, 1, 1), .2), [np.nan, .3, .5],
        [[1]], [[5]], isltyp=[[index]])


@pytest.mark.parametrize("liquid,passed", [(.1, True), (.3, True), (.31, False), (-.01, False), (.41, False)])
def test_liquid_water_must_fit_total_and_controlled_porosity(liquid, passed):
    assert validate_soil_porosity(np.full((4, 1, 2), .3), .4, [[1, 1]], [[5, 5]],
        sh2o=np.full((4, 1, 2), liquid)) is passed


@pytest.mark.parametrize("lu,ice,physics,passed", [(15, None, 4, True), (15, 0, 4, True),
    (15, .5, 4, False), (15, 0, 2, False), (5, 0, 4, False)])
def test_noahmp_glacier_exception_and_seaice_control(lu, ice, physics, passed):
    assert validate_soil_porosity(np.ones((4, 1, 1)), .3, [[1]], [[lu]],
        seaice=ice, surface_physics=physics, sh2o=np.zeros((4, 1, 1))) is passed


@pytest.mark.parametrize("which", ["smois", "smcmax", "sh2o", "landmask", "lu_index", "seaice"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_refuse(which, bad):
    args = {"smois": np.full((4, 1, 1), .2), "smcmax": np.full((1, 1), .3),
            "landmask": np.ones((1, 1)), "lu_index": np.full((1, 1), 5.),
            "seaice": np.zeros((1, 1)), "sh2o": np.full((4, 1, 1), .1)}
    args[which].flat[0] = bad
    assert not validate_soil_porosity(**args)


def test_inactive_water_is_not_a_soil_column_and_nonfinite_still_refuses():
    moisture = np.array([[[-1., .3]]])
    assert validate_soil_porosity(moisture, .4, [[0, 1]], [[17, 5]])
    moisture[0, 0, 0] = np.nan
    assert not validate_soil_porosity(moisture, .4, [[0, 1]], [[17, 5]])


@pytest.mark.parametrize("cap", [-.1, 1.1, "bad", np.ones((2, 4))])
def test_invalid_caps_refuse(cap):
    assert not validate_soil_porosity(np.full((4, 1, 2), .2), cap, [[1, 1]], [[5, 5]])


def test_masked_liquid_and_shape_mismatch_refuse():
    moisture = np.full((4, 1, 2), .2)
    liquid = np.ma.array(np.full(moisture.shape, .1), mask=False)
    liquid.mask[0, 0, 0] = True
    assert not validate_soil_porosity(moisture, .3, [[1, 1]], [[5, 5]], sh2o=liquid)
    assert not validate_soil_porosity(moisture, .3, [[1, 1]], [[5, 5]], sh2o=np.full((1, 2), .1))


@pytest.mark.parametrize("with_liquid", [False, True])
def test_existing_table_validator_delegates_to_shared_helper(view, monkeypatch, with_liquid):
    fields, meta, _ = view
    if with_liquid:
        fields["SH2O"] = np.full(fields["SMOIS"].shape, .1, np.float32)
        meta["variable_dimensions"]["SH2O"] = meta["variable_dimensions"]["SMOIS"]
    calls = []
    def observe(*args, **kwargs):
        calls.append((args, kwargs))
        return validate_soil_porosity(*args, **kwargs)
    monkeypatch.setattr("gpuwrf.io.wrfinput_consistency.validate_soil_porosity", observe)
    changed = np.broadcast_to(fields["LANDMASK"] == 1, fields["SMOIS"].shape)
    assert validate_fields(fields, meta, changed_fields={"SMOIS": changed}, checkpoint=with_liquid)["passed"]
    assert len(calls) == 1
    assert (calls[0][1]["sh2o"] is not None) is with_liquid
    np.testing.assert_array_equal(calls[0][0][1], np.full((1, 2, 3), np.float32(.339)))
