# Data licenses

The code in this repository is MIT licensed. The data it reads keeps the license of its provider.
Derived files (prepared lattices, pyramids, rasters, the browser bundle) carry the same terms as
their source.

| Source | Provider | Used for | License | Shipped here |
|---|---|---|---|---|
| DGM1 digital terrain model, WCS `nw_dgm` (10 m responses) | Geobasis NRW, Bezirksregierung Koeln | Terrain reference for all regions | [DL-DE-Zero-2.0](https://www.govdata.de/dl-de/zero-2-0) | Essen tiles in `data/sample/` |
| IS GK100 geological map 1:100,000, INSPIRE WFS | Geologischer Dienst NRW | Surface lithology and fault traces | [DL-DE-BY-2.0](https://www.govdata.de/dl-de/by-2-0) | Essen extract in `data/sample/` |
| ATKIS Basis-DLM, WFS | Geobasis NRW | Water levels, flow direction, valleys and crossings (registration checks) | DL-DE-Zero-2.0 | No |
| Height benchmarks (Hoehenfestpunkte) | Geobasis NRW, opengeodata.nrw.de | Registration and datum check | Not recorded | No |

Required attribution for the geology:

> IS GK100 (https://www.wfs.nrw.de/gd/wfs_nw_inspire-gk100), Geologischer Dienst NRW, retrieved
> 2026-09-12; DL-DE-BY-2.0 (https://www.govdata.de/dl-de/by-2-0)

Terrain:

> DGM1, Geobasis NRW (https://www.wcs.nrw.de/geobasis/wcs_nw_dgm), retrieved 2026-09-12;
> DL-DE-Zero-2.0

Every downloaded file has a receipt next to it with its URL, retrieval time and SHA-256. The
height benchmark files are fetched by the user and never redistributed, because their terms were
not recorded at download.
