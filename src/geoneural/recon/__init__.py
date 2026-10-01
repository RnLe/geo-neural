"""Reconstruction: predicting terrain heights that were not observed.

A separate track from compression, with its own split and input contract. Three observation tasks share one
neural residual model and one set of non-neural comparators:

* coarse-to-fine: a declared averaging operator turns the reference into a coarse grid (40 m from 10 m, or 10 m
  from 1 m); every method receives that grid and the operator.
* missing blocks: square holes of 8, 32 and 128 cells, filled from the surrounding reference only.
* sparse noisy samples: 1 % or 5 % of nodes with correlated noise of declared size and length scale.

Regions are the statistical unit. Models are trained on some regions, selected on another by a declared rule and
scored on one that took no part in either; test regions are a parameter so a later confirmation cohort is run
with recipes frozen on the development regions. Geology (GK100 material classes) is mapped context and may be
an input; it is reported with and without.
"""
