"""Survey epochs come from the provider's tile metadata, and a tile republished
after retrieval is flagged instead of being dated from newer metadata."""
import tempfile
import unittest
import zipfile
from pathlib import Path

from geoneural.data import epochs

CSV = ("Kachelinformationen des DGM1 fuer die Datenabgabe\nLand;Nordrhein-Westfalen\n"
       "Aktualitaet_Kachelinformationen;2026-09-01\nVersion_Standard;3.3\n"
       "Kachelname;Aktualitaet;Erfassungsmethode;Fortfuehrung;Fortfuehrungsmethode;Genauigkeit;"
       "Koordinatenreferenzsystem_Lage;Koordinatenreferenzsystem_Hoehe;Hoehenanomalie\n"
       "dgm1_32_356_5694_1_nw_2025;2025-02-19;5020;2025-02-19;5020;0.2;ETRS89_UTM32;DE_DHHN2016_NH;GCG\n"
       "dgm1_32_357_5694_1_nw_2025;2025-02-21;5020;2025-02-21;5020;0.2;ETRS89_UTM32;DE_DHHN2016_NH;GCG\n"
       "dgm1_32_400_5694_1_nw_2020;2020-01-01;5020;2020-01-01;5020;0.2;ETRS89_UTM32;DE_DHHN2016_NH;GCG\n")
XML = ('<?xml version="1.0"?><opengeodata><datasets><dataset><files>'
       '<file name="dgm1_32_356_5694_1_nw_2025.tif" timestamp="2026-09-08T22:35:03" />'
       '<file name="dgm1_32_357_5694_1_nw_2025.tif" timestamp="2026-09-20T08:00:00" />'
       '<file name="dgm1_32_400_5694_1_nw_2020.tif" timestamp="2021-01-01T00:00:00" />'
       '</files></dataset></datasets></opengeodata>')


class Epochs(unittest.TestCase):
    def test_tiles_under_the_bounds_and_a_late_republication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / "dgm1_meta.zip", "w") as archive:
                archive.writestr("dgm1_nw.csv", CSV)
            (root / "index.xml").write_text(XML)
            report = epochs.epochs(root / "dgm1_meta.zip", root / "index.xml",
                                   [356000.5, 5694000.5, 357900.5, 5694900.5], "2026-09-12T15:38:55+00:00")
        self.assertEqual(report["tiles"], 2)  # the tile at 400 km lies outside
        self.assertEqual(report["republishedAfterRetrieval"], ["dgm1_32_357_5694_1_nw_2025"])
        # Only the tile current at retrieval is dated.
        self.assertEqual(report["surveyEpochs"], {"2025-02-19": 1})
        self.assertEqual(report["source"]["metadataCurrency"], "2026-09-01")

    def test_metadata_without_its_column_header_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dgm1_meta.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("dgm1_nw.csv", "Land;NRW\n")
            with self.assertRaises(ValueError):
                epochs.read_metadata(path)


if __name__ == "__main__":
    unittest.main()
