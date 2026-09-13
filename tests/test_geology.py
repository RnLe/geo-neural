"""Geological context: the representation choices, and the mistakes they prevent.

Every test here corresponds to a way of destroying geological information that
does not raise on its own. A float class raster, a strike treated as a direction,
an unknown cell merged into a mapped unit and a context raster left out of the
byte total all produce a model that trains, converges and reports a number.
"""
from __future__ import annotations
import pathlib
import tempfile
import unittest

import numpy as np

from geoneural.data import geology


class Strike(unittest.TestCase):
    """A fault striking 30 degrees and one striking 210 degrees are one fault."""

    def test_a_strike_and_its_reverse_are_identical(self):
        angles = np.arange(0.0, 180.0, 7.0)
        forward = geology.strike_features(angles)
        reversed_ = geology.strike_features(angles + 180.0)
        self.assertTrue(np.allclose(forward, reversed_, atol=1e-12))

    def test_a_directional_encoding_would_have_got_this_wrong(self):
        """The control that gives the test above its meaning: the naive sin/cos
        of the angle itself makes a fault and its reverse opposite."""
        angles = np.array([30.0, 210.0])
        naive = np.stack([np.sin(np.deg2rad(angles)), np.cos(np.deg2rad(angles))], axis=-1)
        self.assertFalse(np.allclose(naive[0], naive[1]))

    def test_perpendicular_strikes_stay_distinguishable(self):
        """Axial encoding must not collapse everything: 0 and 90 are different
        faults and must not map to the same feature."""
        features = geology.strike_features(np.array([0.0, 90.0]))
        self.assertGreater(float(np.abs(features[0] - features[1]).max()), 1.0)

    def test_features_lie_on_the_unit_circle(self):
        angles = np.linspace(0.0, 360.0, 37)
        norms = np.linalg.norm(geology.strike_features(angles), axis=-1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-12))


class Classes(unittest.TestCase):
    def test_a_float_class_raster_is_refused(self):
        """A float raster is the signature of interpolated, averaged or
        normalised categories. None of those operations raise on their own and
        all of them destroy the labels."""
        with self.assertRaises(TypeError):
            geology.class_features(np.array([[1.0, 2.0], [3.0, 0.0]]), 4)

    def test_an_out_of_range_code_is_refused(self):
        with self.assertRaises(ValueError):
            geology.class_features(np.array([[0, 9]], dtype=np.int32), 4)

    def test_valid_codes_pass_through_as_integers(self):
        out = geology.class_features(np.array([[0, 1], [2, 3]], dtype=np.int16), 4)
        self.assertTrue(np.issubdtype(out.dtype, np.integer))

    def test_unknown_is_a_class_and_stays_addressable(self):
        """An unmapped cell is not unit zero and not the commonest unit. Keeping
        the mask means a result can be reported with and without it."""
        classes = np.array([[0, 1], [2, 0]], dtype=np.int32)
        mask = geology.unknown_mask(classes)
        self.assertEqual(mask.tolist(), [[True, False], [False, True]])
        self.assertEqual(geology.UNKNOWN_CLASS, 0)


class BoundaryDistance(unittest.TestCase):
    def setUp(self):
        self.classes = np.zeros((32, 32), dtype=np.int32)
        self.classes[8:24, 8:24] = 2

    def test_boundary_cells_are_at_zero(self):
        distance = geology.distance_to_boundary(self.classes, 10.0)
        self.assertEqual(float(distance[8, 8]), 0.0)
        self.assertEqual(float(distance[7, 8]), 0.0)

    def test_distance_grows_away_from_the_boundary(self):
        distance = geology.distance_to_boundary(self.classes, 10.0)
        self.assertGreater(float(distance[16, 16]), float(distance[10, 16]))

    def test_it_is_reported_in_metres_not_cells(self):
        coarse = geology.distance_to_boundary(self.classes, 10.0)
        fine = geology.distance_to_boundary(self.classes, 20.0)
        self.assertAlmostEqual(float(fine.max()) / float(coarse.max()), 2.0)

    def test_a_uniform_map_saturates_at_the_cap_rather_than_at_infinity(self):
        """A saturated value says the boundary is farther than the cap. Infinity
        would be unusable as a network input and zero would be a lie."""
        uniform = np.ones((16, 16), dtype=np.int32)
        distance = geology.distance_to_boundary(uniform, 10.0, max_cells=5)
        self.assertTrue(np.isfinite(distance).all())
        self.assertEqual(float(distance.min()), 50.0)


class Accounting(unittest.TestCase):
    def setUp(self):
        self.classes = np.zeros((64, 64), dtype=np.int32)
        self.classes[16:48, 16:48] = 3

    def test_context_is_charged_as_an_encoded_payload(self):
        result = geology.context_bytes(self.classes)
        self.assertGreater(result["classRasterBytes"], 0)
        self.assertLess(result["classRasterBytes"], self.classes.size * 4,
                        "a class raster of few labels must compress")

    def test_derived_fields_are_counted_separately_from_the_classes(self):
        """A decoder can recompute them, so shipping them is a choice with a byte
        consequence. Both totals are reported so the choice is visible."""
        distance = geology.distance_to_boundary(self.classes, 10.0)
        result = geology.context_bytes(self.classes, {"distanceToBoundary": distance})
        self.assertIn("distanceToBoundary", result["derivedFieldBytes"])
        self.assertGreater(result["totalIfAllShipped"], result["totalIfDerivedRecomputed"])
        self.assertEqual(result["totalIfDerivedRecomputed"], result["classRasterBytes"])

    def test_the_summary_is_json_safe_and_states_what_it_is_not(self):
        import json
        result = geology.summary(self.classes, 10.0, np.full(self.classes.shape, 45.0))
        json.dumps(result)
        for denial in ("not rock", "erodibility", "unknown"):
            with self.subTest(denial):
                self.assertIn(denial, result["qualification"])

    def test_the_summary_reports_how_much_is_unknown(self):
        classes = np.zeros((10, 10), dtype=np.int32)
        classes[:5] = 1
        result = geology.summary(classes, 10.0)
        self.assertAlmostEqual(result["unknownFraction"], 0.5)


if __name__ == "__main__":
    unittest.main()


def _square(west, south, east, north):
    return [[[(west, south), (east, south), (east, north), (west, north), (west, south)]]]


class RasteriseToTheLattice(unittest.TestCase):
    """Burn mapped units onto the atlas lattice without inventing geology.

    The properties that matter are registration, the unknown class, and the fact
    that nothing here reads a colour: class codes come from INSPIRE attributes.
    """

    BBOX = (1000.0, 2000.0, 1040.0, 2040.0)      # 40 m square
    SIDE = 5                                      # 5 nodes, 10 m spacing

    def unit(self, material, geometry, age="carboniferous"):
        return {"material": material, "age": age, "name": "", "localId": "1",
                "geometry": {"type": "MultiPolygon", "coordinates": geometry}}

    def test_pixel_centres_land_on_lattice_nodes(self):
        """A polygon covering exactly the western half must claim exactly the
        nodes in that half; an off-by-half-a-cell transform would not."""
        west, south, east, north = self.BBOX
        middle = west + 20.0
        units = [self.unit("sandstone", _square(west - 5, south - 5, middle + 1, north + 5))]
        out = geology.rasterise(units, self.BBOX, self.SIDE, 10.0)
        classes = out["classes"]
        code = out["legend"]["sandstone"]
        # Nodes at x = 1000, 1010, 1020, 1030, 1040; the first three are covered.
        self.assertTrue((classes[:, :3] == code).all(), f"west half not covered:\n{classes}")
        self.assertTrue((classes[:, 3:] == geology.UNKNOWN_CLASS).all(),
                        f"east half wrongly covered:\n{classes}")

    def test_uncovered_cells_are_unknown_rather_than_a_class(self):
        units = [self.unit("sandstone", _square(1000.0, 2000.0, 1010.0, 2010.0))]
        out = geology.rasterise(units, self.BBOX, self.SIDE, 10.0)
        self.assertGreater(out["unknownCells"], 0)
        self.assertNotIn(geology.UNKNOWN_CLASS, out["legend"].values())
        self.assertAlmostEqual(out["unknownFraction"],
                               out["unknownCells"] / out["classes"].size)

    def test_the_legend_is_deterministic(self):
        units = [self.unit("siltstone", _square(*self.BBOX)),
                 self.unit("sandstone", _square(1000.0, 2000.0, 1010.0, 2010.0))]
        first = geology.rasterise(units, self.BBOX, self.SIDE, 10.0)["legend"]
        second = geology.rasterise(list(reversed(units)), self.BBOX, self.SIDE, 10.0)["legend"]
        self.assertEqual(first, second)
        self.assertEqual(first, {"sandstone": 1, "siltstone": 2})

    def test_age_is_a_separate_field_from_lithology(self):
        units = [self.unit("sandstone", _square(*self.BBOX), age="serpukhovian")]
        material = geology.rasterise(units, self.BBOX, self.SIDE, 10.0, attribute="material")
        age = geology.rasterise(units, self.BBOX, self.SIDE, 10.0, attribute="age")
        self.assertEqual(list(material["legend"]), ["sandstone"])
        self.assertEqual(list(age["legend"]), ["serpukhovian"])

    def test_the_record_states_what_the_raster_is_not(self):
        out = geology.rasterise([self.unit("sandstone", _square(*self.BBOX))],
                                self.BBOX, self.SIDE, 10.0)
        self.assertEqual(out["sourceScale"], "1:100,000")
        self.assertGreaterEqual(out["positionalUncertaintyM"], 10.0)
        self.assertIn("not rock occupancy", out["qualification"])
        # The unit raster omits fault displacement but not the faults themselves:
        # the traces are features in the source and `parse_faults` reads them.
        joined = " ".join(out["omits"]).lower()
        self.assertIn("displacement", joined)
        self.assertIn("depth extent", joined)


class ParseRealGk100(unittest.TestCase):
    """Against the retained GML, because a parser tested only on fixtures is
    tested only against its author's assumptions about the provider's schema."""

    import pathlib as _pathlib
    SOURCE = _pathlib.Path(__file__).resolve().parents[1] / "data" / "sample" / "essen-ruhr" / "geology"

    def setUp(self):
        if not self.SOURCE.exists():
            self.skipTest("GK100 GML not retained in this working copy")

    def test_units_carry_attributes_and_geometry(self):
        units = geology.parse_units(self.SOURCE.glob("*.gml"))
        self.assertGreater(len(units), 100)
        self.assertTrue(all(unit["geometry"]["type"] == "MultiPolygon" for unit in units))
        self.assertGreater(len({unit["material"] for unit in units if unit["material"]}), 3)
        self.assertGreater(len({unit["age"] for unit in units if unit["age"]}), 3)

    def test_the_axis_order_is_checked_rather_than_trusted(self):
        """The GML declares a urn CRS whose formal axis order is northing-first
        while writing easting-first. A transposed file must raise, not silently
        rasterise the map sideways onto the terrain."""
        units = geology.parse_units(self.SOURCE.glob("*.gml"))
        easting, northing = units[0]["geometry"]["coordinates"][0][0][0]
        self.assertLess(easting, 1_000_000.0)
        self.assertGreater(northing, 5_000_000.0)


class AblationControls(unittest.TestCase):
    """The geology ablation arms, and the asymmetry between them that decides
    which one a conclusion may rest on."""

    def field(self, side=64):
        rows = np.arange(side)[:, None]
        columns = np.arange(side)[None, :]
        return ((rows // 8 + columns // 11) % 5 + 1).astype(np.int32)

    def test_misaligning_preserves_everything_except_registration(self):
        original = self.field()
        moved = geology.misalign(original, (21, 16))
        self.assertEqual(sorted(np.bincount(original.reshape(-1)).tolist()),
                         sorted(np.bincount(moved.reshape(-1)).tolist()))
        self.assertFalse(np.array_equal(original, moved))

    def test_misaligned_is_byte_matched_and_shuffled_is_not(self):
        """Permuting cells destroys the structure that made the field compress,
        so `shuffled` costs many times `real` and cannot be compared at rate."""
        original = self.field()
        real = geology.context_bytes(original)["classRasterBytes"]
        moved = geology.context_bytes(geology.misalign(original, (21, 16)))["classRasterBytes"]
        scrambled = geology.context_bytes(geology.shuffle_cells(original))["classRasterBytes"]
        self.assertLess(abs(moved - real) / real, 0.10)
        self.assertGreater(scrambled, 3 * real)

    def test_shuffling_keeps_the_class_frequencies(self):
        original = self.field()
        scrambled = geology.shuffle_cells(original, seed=5)
        self.assertEqual(np.bincount(original.reshape(-1)).tolist(),
                         np.bincount(scrambled.reshape(-1)).tolist())

    def test_generic_context_is_blocky_and_unrelated(self):
        generic = geology.generic_context(64, regions=40, class_count=6, seed=1)
        self.assertEqual(generic.shape, (64, 64))
        self.assertLessEqual(int(generic.max()), 6)
        # Blocky rather than noise: most neighbours agree.
        agree = float((generic[:, :-1] == generic[:, 1:]).mean())
        self.assertGreater(agree, 0.8)

    def test_every_arm_is_present_and_priced(self):
        arms = geology.ablation_contexts(self.field(), 6, regions=40)
        self.assertEqual(set(arms), {"none", "real", "misaligned", "shuffled", "generic"})
        for name, arm in arms.items():
            self.assertGreater(arm["bytes"], 0, name)
            self.assertEqual(arm["classes"].shape, (64, 64))
        self.assertEqual(arms["none"]["distinctClasses"], 1)


class FaultTraces(unittest.TestCase):
    """GK100 fault traces: parsed as map-view polylines, with what they omit named."""

    def gml(self, body: str) -> pathlib.Path:
        self._n = getattr(self, "_n", 0) + 1
        path = pathlib.Path(self.tmp.name) / f"faults-{self._n}.gml"
        path.write_text(
            '<?xml version="1.0"?><wfs:FeatureCollection '
            'xmlns:wfs="http://www.opengis.net/wfs/2.0" '
            'xmlns:gml="http://www.opengis.net/gml/3.2" '
            'xmlns:ge="urn:x-inspire:specification:gmlas:GeologyCore">' + body
            + "</wfs:FeatureCollection>", encoding="utf-8")
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def feature(self, points: str, kind: str = "fault") -> str:
        return ('<wfs:member><ge:GE.GeologicFault gml:id="GE.GeologicFault.1">'
                '<ge:Shape><gml:MultiCurve><gml:curveMember><gml:LineString>'
                f'<gml:posList>{points}</gml:posList>'
                '</gml:LineString></gml:curveMember></gml:MultiCurve></ge:Shape>'
                f'<ge:faulttype_label>{kind}</ge:faulttype_label>'
                '<ge:geologichistory_void>2</ge:geologichistory_void>'
                '</ge:GE.GeologicFault></wfs:member>')

    def test_a_multicurve_trace_is_read_as_a_polyline(self):
        path = self.gml(self.feature("356000 5694000 357000 5695000"))
        faults = geology.parse_faults([path])
        self.assertEqual(len(faults), 1)
        self.assertEqual(faults[0]["faultType"], "fault")
        self.assertTrue(faults[0]["historyVoid"])
        self.assertEqual(faults[0]["geometry"]["coordinates"][0],
                         [[356000.0, 5694000.0], [357000.0, 5695000.0]])

    def test_northing_first_coordinates_are_refused(self):
        """The same axis-order trap `parse_units` asserts against."""
        path = self.gml(self.feature("5694000 356000 5695000 357000"))
        with self.assertRaises(ValueError):
            geology.parse_faults([path])

    def test_distance_is_zero_on_the_trace_and_saturates_away_from_it(self):
        path = self.gml(self.feature("356000 5694000 356000 5695000"))
        faults = geology.parse_faults([path])
        raster = geology.fault_rasters(faults, (356000.0, 5694000.0, 356640.0, 5694640.0),
                                       65, 10.0, max_cells=8)
        self.assertEqual(float(raster["distanceM"].min()), 0.0)
        self.assertEqual(float(raster["distanceM"].max()), 80.0)
        self.assertGreater(raster["traceCells"], 0)

    def test_strike_is_axial_so_a_trace_and_its_reverse_agree(self):
        """A fault striking 30 degrees and one striking 210 are the same fault."""
        bbox = (356000.0, 5694000.0, 356640.0, 5694640.0)
        forward = geology.parse_faults([self.gml(self.feature("356100 5694100 356500 5694500"))])
        reverse = geology.parse_faults([self.gml(self.feature("356500 5694500 356100 5694100"))])
        a = geology.fault_rasters(forward, bbox, 65, 10.0)["strikeSinCos"]
        b = geology.fault_rasters(reverse, bbox, 65, 10.0)["strikeSinCos"]
        self.assertTrue(np.allclose(a, b, atol=1e-9))

    def test_what_the_source_does_not_supply_is_named(self):
        """These channels may condition a decoder and may never support a
        structural claim; the record has to say which."""
        raster = geology.fault_rasters([], (0.0, 0.0, 640.0, 640.0), 65, 10.0)
        joined = " ".join(raster["omits"]).lower()
        for absent in ("dip", "throw", "displacement", "depth extent"):
            self.assertIn(absent, joined)
