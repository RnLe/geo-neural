# Data

## Four roles

The code and the viewer keep four kinds of field apart.

| Role | Meaning | Example |
|---|---|---|
| observation | Provider data with its own acquisition history | DGM1 tiles, GK100 polygons |
| reference | The exactly defined field every experiment is scored against | The prepared 10 m lattice |
| reconstruction | A decoded or predicted approximation of the reference | A quantised grid, a neural decoder output |
| scenario | A state generated under stated assumptions | A hillslope after a synthetic forcing |

A reconstruction is not ground truth, the reference is not free of observation error, and a
scenario is not a forecast because its starting surface is real.

## Terrain reference

DGM1 is the bare-earth terrain model of North Rhine-Westphalia (1 m raster, ETRS89 / UTM 32N
horizontally, DHHN2016 heights). The 10 m data used here comes from the provider's WCS with
`SCALEFACTOR=0.1`, so the provider's own coarsening defines it. It is a derived product, not a
lossless copy of the 1 m samples.

`geoneural prepare` turns the downloaded tiles into the reference:

* a node lattice of 1025 x 1025 samples at 10 m. The side is (1025 - 1) x 10 m = 10,240 m, so the
  area is 104.86 km2. Row 0 is north. Bounds are node centres;
* tiles are reprojected bilinearly onto the lattice, two edge samples per tile are trimmed, and
  overlaps are resolved by a fixed tile order. Remaining nodata stops the build: missing terrain is
  never filled with zeros;
* a five-level pyramid (binomial [1 4 6 4 1] filter, then decimation by two) in 341 pages of
  65 x 65 nodes, each page an EAT1 file: a small header and gzip-compressed int32 codes at a 1 cm
  quantum.

The reference array of the Essen region has SHA-256 `ece831c6...2986618`. Rebuilding it from the
shipped tiles (`geoneural reproduce`) must give the same hash before any comparison runs.

## Regions

The six study regions are 10.24 km squares at 10 m (`configs/regions.json` also has two unused
presets). Only Essen is shipped with the repository; the others are fetched with
`geoneural fetch --preset <name>` and built with `geoneural prepare --preset <name>`.

| Preset | South-west node (E, N) | Relief | Character |
|---|---|---|---|
| essen-ruhr | 356000.5, 5694000.5 | 130 m | Ruhr valley, urban and mining terrain |
| muensterland-plain | 394000.5, 5744000.5 | 25 m | Flat lowland |
| lower-rhine | 318000.5, 5715360.5 | 75 m | River plain |
| teutoburg-forest | 466000.5, 5761000.5 | 182 m | Narrow ridge |
| bergisches-land | 379000.5, 5656000.5 | 210 m | Dissected upland |
| rothaar-sauerland | 452000.5, 5671000.5 | 418 m | Low mountain range |

Survey dates differ between regions (2020-12 to 2025-02); within Essen the tiles were surveyed on
2025-02-19 (55 tiles) and 2025-02-21 (66 tiles). The same provider does not mean the same epoch.

## Registration checks

These test that the lattice sits where it should, not how accurate the source is.

* **Lattice and CRS.** Axis order, row direction, extent and corner round trips through pyproj
  (`geoneural verify-lattice`): all structural checks pass.
* **Height benchmarks.** 403 official NRW height benchmarks fall inside Essen. Fitting a shift on
  the 341 benchmarks in the robust core gives (-0.02 m east, +0.41 m north), with a 95 % bootstrap
  interval within about 1.9 m, well inside the 5 m half cell. The residual spread is 0.36 m on the
  lattice as built (no shift), against 32.8 m if the grid were flipped north to south and 27.4 m if
  it were transposed. Median residual (published minus lattice) is +0.28 m. Most benchmarks are bolts on
  walls and bridges, so these residuals are not a ground-accuracy measure.
* **Landscape model.** Two published water levels agree within 0.15 m. Along 52 of 55 digitised
  long water axes with a stated flow direction, the lattice descends along it (chance probability about
  6e-15; the mirrored grid gives 21 against 28). Valley floors lie below both banks at 79 % of
  13,832 samples, against 10 % on the mirrored grid.

## Geology

GK100 is a geological map at 1:100,000. The extract for Essen has 226 mapped units and 34 fault
traces. Units are rasterised onto the lattice by their `material` attribute (nine classes plus an
explicit "not mapped" class, 2.9 % of the nodes), with pixel centres on lattice nodes.

What this layer is not: rasterising a 1:100,000 map at 10 m does not create 10 m geology, a
surface unit says nothing about depth, a fault trace has no dip or throw, and a lithology class
does not determine an erodibility or a diffusivity. Where the physics lab assigns coefficients
to classes, they are labelled as assumptions.

## Water and human terrain

A bare-earth DTM has no bathymetry, culverts or sewers, and the Ruhr terrain carries mining
subsidence, embankments and spoil heaps that no landscape-evolution model explains. The drainage
diagnostic is a routing comparison between two surfaces, not a flood or ground-stability model.
