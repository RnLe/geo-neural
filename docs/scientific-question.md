# The question

At the same complete storage cost, can a learned predictor, geological context or a landscape-process prior
reconstruct terrain more accurately, and preserve its drainage better, than strong conventional codecs and an
equally capable model without those additions?

"Complete storage cost" means every byte a decoder needs for this field: the coded stream, its headers, any model
weights or context maps that are not shared, and the share of any model that is. Generic decoder software is not
counted, for conventional and learned codecs alike.

## Four tasks that must not be mixed

| Task | What the method sees | What a result can claim |
|---|---|---|
| Compression | The encoder sees the whole reference; the decoder sees only the product | The known terrain can be stored in fewer bytes or more accurately |
| Reconstruction | Only declared coarse, partial or noisy observations and permitted context | Unobserved terrain can be estimated, with measured error and uncertainty |
| Forward dynamics | Initial surface, coefficients, forcing and boundaries | Simulator states can be predicted at stated cost and accuracy |
| Inverse | Observations constrain parameters through a forward model | Some parameter combinations can be inferred under stated assumptions |

## Three meanings of "beat"

1. **Compression:** fewer complete bytes at the same maximum error (and no worse drainage), against the smallest
   conventional product that keeps the same bound.
2. **Scientific utility:** better drainage or derivative fidelity at the same complete bytes, under a height-error
   ceiling.
3. **Systems:** lower encode or decode cost at a declared storage trade-off.

## Hypotheses

| ID | Hypothesis | Falsified when |
|---|---|---|
| H1 | A learned predictor reduces complete bytes at a fixed maximum error | No robust gain over the best conventional product once the model is counted |
| H2 | Real geology adds coding information | The gain disappears against shifted, shuffled or wrong-region maps, or once the map is charged |
| H3 | Terrain-aware bit allocation preserves drainage more efficiently | No stream improvement at equal bytes beyond the reference noise floor |
| H4 | A process-derived prior adds benefit beyond real data | The gain is matched by extra real-data training or by a non-physical procedural prior |
| H5 | A structured learned closure improves accuracy under conservation and stability | Only one-step error improves, or rollouts fail |
| H6 | An inverse estimator recovers identifiable combinations with calibrated uncertainty | It reports precise values along an unidentifiable direction, or coverage is off |

H1 to H4 form the compression study, H5 and H6 its physical companion, and the reconstruction track asks a
separate question with its own split. A negative answer is an acceptable outcome for each.

The comparison rules, thresholds and the confirmation procedure are in [protocol-v2.md](protocol-v2.md).
