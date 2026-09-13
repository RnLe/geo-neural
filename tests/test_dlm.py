"""The landscape-model inspection: parsing the provider's GML, and each test's power."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geoneural.data import dlm

SIDE, SPACING = 201, 10.0
MANIFEST = {"bounds": [0.5, 0.5, 0.5 + (SIDE - 1) * SPACING, 0.5 + (SIDE - 1) * SPACING], "spacing_m": SPACING}
X0 = 1000.5  # the valley floor runs north along this easting


def _valley() -> np.ndarray:
    """A V-shaped valley whose floor descends northward, 1 m per 100 m."""
    rows, cols = np.mgrid[0:SIDE, 0:SIDE].astype(np.float64)
    east = MANIFEST["bounds"][0] + cols * SPACING
    north = MANIFEST["bounds"][3] - rows * SPACING
    return 100.0 + north / 100.0 + 0.05 * np.abs(east - X0)


def _axis(flow: str, reverse: bool = False) -> dict:
    coords = [(X0, 50.5 + k * 100.0) for k in range(19)]
    coords = coords[::-1] if reverse else coords
    return {"type": "AX_Gewaesserachse", "attrs": {"fliessrichtung": flow, "breiteDesGewaessers": "3"},
            "coords": coords}


class Inspection(unittest.TestCase):
    def test_a_correct_atlas_passes_every_test_and_the_mirror_does_not(self):
        grid = _valley()
        # Digitised with the flow (northward down the valley), as the flag says.
        axis = {**_axis("true"), "coords": _axis("true")["coords"][::-1]}
        level = {"type": "AX_Wasserspiegelhoehe", "attrs": {"hoeheDesWasserspiegels": "110.0"},
                 "coords": [(X0, 1000.5)]}
        bridge = {"type": "AX_BauwerkImVerkehrsbereich", "attrs": {"bauwerksfunktion": "1800"},
                  "coords": [(X0 - 5, 900.5), (X0 + 5, 900.5)]}
        report = dlm.inspect(grid, MANIFEST, [axis, level, bridge])
        self.assertEqual(report["orientation"]["flowAlongDigitisation"]["descending"], 1)
        self.assertEqual(report["orientation"]["flowAlongDigitisationMirrored"]["descending"], 0)
        self.assertGreater(report["valleyFloor"]["atlas"]["belowBothBanks"], 0.9)
        self.assertLess(report["crossings"]["absBumpM"]["max"], 0.1)
        self.assertTrue(report["waterLevels"]["allWithinTolerance"])

    def test_a_flag_other_than_true_never_decides_orientation(self):
        grid = _valley()
        report = dlm.inspect(grid, MANIFEST, [_axis("false")])
        self.assertEqual(report["orientation"]["flowAlongDigitisation"],
                         {"descending": 0, "ascending": 0, "notSignificant": 0, "pChance": 1.0})
        self.assertEqual(report["orientation"]["flagFalseAsDigitised"]["ascending"], 1)

    def test_a_water_level_off_by_more_than_the_stated_accuracy_fails(self):
        level = {"type": "AX_Wasserspiegelhoehe", "attrs": {"hoeheDesWasserspiegels": "111.0"},
                 "coords": [(X0, 1000.5)]}
        self.assertFalse(dlm.inspect(_valley(), MANIFEST, [level])["waterLevels"]["allWithinTolerance"])


class Parse(unittest.TestCase):
    def test_features_attributes_and_geometry(self):
        xml = ('<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0" '
               'xmlns:gml="http://www.opengis.net/gml/3.2" xmlns="http://www.adv-online.de/namespaces/adv/gid/7.1">'
               '<wfs:member><AX_Gewaesserachse gml:id="a"><fliessrichtung>true</fliessrichtung>'
               '<position><gml:LineString><gml:posList>1 2 3 4 5 6</gml:posList></gml:LineString></position>'
               '</AX_Gewaesserachse></wfs:member>'
               '<wfs:member><AX_Wasserspiegelhoehe gml:id="b"><hoeheDesWasserspiegels>51.9</hoeheDesWasserspiegels>'
               '<position><gml:Point><gml:pos>7 8</gml:pos></gml:Point></position></AX_Wasserspiegelhoehe>'
               '</wfs:member></wfs:FeatureCollection>')
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "AX_test-000.xml").write_text(xml)
            features = dlm.parse(Path(tmp))
        self.assertEqual([f["type"] for f in features], ["AX_Gewaesserachse", "AX_Wasserspiegelhoehe"])
        self.assertEqual(features[0]["coords"], [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
        self.assertEqual(features[0]["attrs"]["fliessrichtung"], "true")
        self.assertEqual(features[1]["attrs"]["hoeheDesWasserspiegels"], "51.9")


if __name__ == "__main__":
    unittest.main()
