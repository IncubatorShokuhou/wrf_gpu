"""Host configuration checks, runnable directly with stdlib unittest (no JAX)."""

from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


# Bypass gpuwrf.__init__, which configures JAX for operational imports.
_SOURCE = Path(__file__).resolve().parents[1] / "src/gpuwrf/tracers/config.py"
_SPEC = importlib.util.spec_from_file_location("_tracer_config_host_tests", _SOURCE)
config = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = config
_SPEC.loader.exec_module(config)


class TracerConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tracer_config_")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "tracers.json"

    def load(self, value, *, shape=(3, 4, 5)):
        self.path.write_text(json.dumps(value), encoding="utf-8")
        return config.load_config(self.path, shape=shape)

    @staticmethod
    def fixture():
        return {
            "version": 1,
            "species": [
                {"name": "SMOKE", "decay_rate_s": 0.001,
                 "deposition_velocity_m_s": 0.01, "wet_scavenging_per_mm": 0.2},
                {"name": "POLLEN"},
            ],
            "emissions": [
                {"species": "POLLEN", "start_s": 0, "end_s": 10,
                 "x": 4, "y": 3, "rate_kg_s": 2.5},
                {"species": "SMOKE", "start_s": 10, "end_s": 20,
                 "x0": 0, "x1": 5, "y0": 0, "y1": 4, "flux_kg_m2_s": 0.03},
            ],
        }

    def test_point_area_species_mapping_defaults_and_source_bytes(self):
        value = self.load(self.fixture())
        self.assertEqual(value.source_sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(value.species[0], config.TracerSpecies("SMOKE", 0.001, 0.01, 0.2))
        self.assertEqual(value.species[1], config.TracerSpecies("POLLEN"))
        self.assertEqual(value.emissions[0], config.Emission(1, 0, 10, 4, 5, 3, 4, 2.5, 0, True))
        self.assertEqual(value.emissions[1], config.Emission(0, 10, 20, 0, 5, 0, 4, 0, 0.03, False))
        self.path.write_bytes(self.path.read_bytes() + b"\n")
        self.assertNotEqual(value.source_sha256, config.load_config(self.path, shape=(3, 4, 5)).source_sha256)

    def test_frozen_configuration(self):
        value = self.load(self.fixture())
        for item, field, replacement in (
            (value, "species", ()), (value.species[0], "name", "DUST"),
            (value.emissions[0], "x0", 0),
        ):
            with self.subTest(field=field), self.assertRaises(FrozenInstanceError):
                setattr(item, field, replacement)
        self.assertIsInstance(value.species, tuple)
        self.assertIsInstance(value.emissions, tuple)

    def test_no_emissions_and_no_arbitrary_species_cap(self):
        value = self.load({"version": 1, "species": [{"name": f"S{i}"} for i in range(128)]})
        self.assertEqual(len(value.species), 128)
        self.assertEqual(value.emissions, ())

    def test_bad_names_and_duplicates(self):
        for name in ("smoke", "A-B", "A B", "", "1SMOKE", "S\n", "é", None, 12,
                     "U", "V", "W", "T", "P", "PB", "RAINNC", "HGT", "XLAT", "T2", "QTEST", "TR17_1"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load({"version": 1, "species": [{"name": name}]})
        with self.assertRaisesRegex(ValueError, "duplicate tracer"):
            self.load({"version": 1, "species": [{"name": "SMOKE"}, {"name": "SMOKE"}]})

    def test_bad_loss_coefficients(self):
        for field in ("decay_rate_s", "deposition_velocity_m_s", "wet_scavenging_per_mm"):
            for invalid in (-1, float("nan"), float("inf"), -float("inf"), True, "0", None,
                            1e300, 3.5e38, 10 ** 400):
                value = {"version": 1, "species": [{"name": "SMOKE", field: invalid}]}
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    self.load(value)

    def test_bad_structure_unknown_keys_and_version(self):
        for invalid in (None, [], {}, {"version": 1}, {"version": 1, "species": []},
                        {"version": 1, "species": {}}, {"version": 1, "species": ["SMOKE"]},
                        {"version": 1, "species": [{}]},
                        {"version": 1, "species": [{"name": "SMOKE", "decay": 1}]}):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.load(invalid)
        for field, invalid in (("version", 2), ("version", "1"), ("version", True),
                               ("version", 1.0), ("emissions", {}), ("extra", 1)):
            value = {"version": 1, "species": [{"name": "SMOKE"}], field: invalid}
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                self.load(value)

    def test_emission_bounds_times_rates_and_unknown_species(self):
        invalid_changes = [
            ("species", "DUST"), ("species", []), ("x", -1), ("x", 5), ("y", 4),
            ("y", -1), ("x", 1.0), ("x", True), ("y", "1"),
            ("start_s", -1), ("start_s", 10), ("end_s", 0), ("end_s", 1e309),
            ("start_s", float("nan")), ("rate_kg_s", -1), ("rate_kg_s", float("inf")),
            ("rate_kg_s", True), ("extra", 1), ("flux_kg_m2_s", 1),
            ("start_s", 1e300), ("end_s", 1e300), ("rate_kg_s", 1e300),
            ("end_s", 3.5e38), ("rate_kg_s", 3.5e38),
            ("end_s", 10 ** 400), ("rate_kg_s", 10 ** 400),
        ]
        for field, invalid in invalid_changes:
            value = self.fixture()
            value["emissions"][0][field] = invalid
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                self.load(value)
        for field in ("species", "start_s", "end_s", "x", "y", "rate_kg_s"):
            value = self.fixture()
            del value["emissions"][0][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.load(value)
        value = self.fixture()
        value["emissions"][0].update(start_s=1e300, end_s=1.1e300)
        with self.assertRaisesRegex(ValueError, "REAL32"):
            self.load(value)

    def test_area_extents_and_flux(self):
        for field, invalid in (("x0", -1), ("x1", 6), ("x0", 5), ("x1", 0),
                               ("y0", -1), ("y1", 5), ("y0", 4), ("y1", 0),
                               ("x1", 3.5), ("y0", True), ("flux_kg_m2_s", -1),
                               ("flux_kg_m2_s", float("nan")), ("flux_kg_m2_s", float("inf")),
                               ("flux_kg_m2_s", 1e300), ("flux_kg_m2_s", 3.5e38),
                               ("flux_kg_m2_s", 10 ** 400),
                               ("rate_kg_s", 1)):
            value = self.fixture()
            value["emissions"][1][field] = invalid
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                self.load(value)
        value = self.fixture()
        value["emissions"] = [None]
        with self.assertRaises(ValueError):
            self.load(value)
        # The format bound is REAL32's finite range, not an arbitrary physical
        # cap. Validation leaves even non-exact decimal inputs unrounded.
        maximum = float.fromhex("0x1.fffffep+127")
        value = self.fixture()
        value["species"][0]["decay_rate_s"] = maximum
        value["species"][0]["deposition_velocity_m_s"] = 0.1
        value["species"][0]["wet_scavenging_per_mm"] = 1e-300
        value["emissions"][0]["rate_kg_s"] = maximum
        value["emissions"][0]["end_s"] = maximum
        value["emissions"][1]["flux_kg_m2_s"] = maximum
        parsed = self.load(value)
        self.assertEqual(parsed.species[0].decay_rate_s, maximum)
        self.assertEqual(parsed.species[0].deposition_velocity_m_s, 0.1)
        self.assertEqual(parsed.species[0].wet_scavenging_per_mm, 1e-300)
        self.assertEqual(parsed.emissions[0].rate_kg_s, maximum)
        self.assertEqual(parsed.emissions[0].end_s, maximum)
        self.assertEqual(parsed.emissions[1].flux_kg_m2_s, maximum)

    def test_reject_duplicate_json_keys_and_invalid_bytes(self):
        for raw in (b'{"version":1,"version":1,"species":[{"name":"SMOKE"}]}',
                    b'{"version":1,"species":[{"name":"SMOKE","name":"DUST"}]}',
                    b"{", b"\xff"):
            self.path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                config.load_config(self.path, shape=(3, 4, 5))

    def test_invalid_shapes_and_overflow(self):
        for shape in ((0, 4, 5), (3, -1, 5), (3, 4, 0), (3, 4), (3, 4, 5, 6),
                      (3, 4, True), (3, 4, 5.0)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.load(self.fixture(), shape=shape)
        value = self.fixture()
        value["species"][0]["decay_rate_s"] = 10 ** 400
        with self.assertRaises(ValueError):
            self.load(value)


if __name__ == "__main__":
    unittest.main()
