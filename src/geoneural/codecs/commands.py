"""CLI commands of the multilevel coder and its campaigns."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from geoneural.common import HOME


def summarise_h1(rows: list[dict]) -> dict:
    """Per bound: learned (corpus and standalone, mean over seeds) against the cubic control, SZ3 and the smallest
    conventional product that kept the bound on every node."""
    from geoneural.evaluation.statistics import paired_log_ratio
    out = {}
    for b in sorted({r["boundM"] for r in rows}):
        at = [r for r in rows if r["boundM"] == b]

        def per_region(pred):
            vals: dict[str, list[float]] = {}
            for r in at:
                if pred(r):
                    vals.setdefault(r["region"], []).append(r["bytes"])
            return {k: float(np.mean(v)) for k, v in vals.items()}

        conv = {}
        for r in at:
            if r["family"] == "conventional" and r["boundViolations"] == 0:
                conv[r["region"]] = min(conv.get(r["region"], np.inf), r["bytes"])
        best_name = {}
        for r in at:
            if r["family"] == "conventional" and r["boundViolations"] == 0 and r["bytes"] == conv.get(r["region"]):
                best_name[r["region"]] = r["coder"]
        learned_c = per_region(lambda r: r["coder"] == "learned" and r["product"] == "corpus")
        learned_s = per_region(lambda r: r["coder"] == "learned" and r["product"] == "standalone")
        ctx = per_region(lambda r: r["coder"] == "cubic-ctx")
        sz3 = per_region(lambda r: r["coder"] == "sz3" and r["boundViolations"] == 0)
        out[str(b)] = {"bestConventionalCoder": best_name,
                       "learnedCorpusVsBestConventional": paired_log_ratio(learned_c, conv),
                       "learnedStandaloneVsBestConventional": paired_log_ratio(learned_s, conv),
                       "learnedCorpusVsCubicCtx": paired_log_ratio(learned_c, ctx),
                       "cubicCtxVsBestConventional": paired_log_ratio(ctx, conv),
                       "learnedCorpusVsSz3": paired_log_ratio(learned_c, sz3)}
    return out


def cmd_codec_h1(a):
    from geoneural.codecs import campaign, foreign
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    rows = campaign.h1(out, tuple(a.regions), tuple(a.bounds), tuple(a.seeds))
    rec = protocol.record(
        "compression", "h1-learned-multilevel", "exploratory",
        results={"rows": rows, "summary": summarise_h1(rows)},
        recipe={"predictor": {"widths": [32, 32], "steps": 5000, "rounds": 2, "loss": "Laplace code length"},
                "lattice_m": 0.001, "bounds": list(a.bounds), "codecVersions": foreign.versions(),
                "product": "standalone 1025 x 1025 raster; learned also as corpus with the model counted once"},
        split={"scheme": "leave one region out", "regions": list(a.regions)}, seeds=list(a.seeds),
        timing={"qualified": False, "note": "shared machine; encode/decode seconds are indicative only"})
    print(protocol.write(rec, out / "h1.json"))


def summarise_h2(rows: list[dict]) -> dict:
    """Per bound: total bytes of each context arm against the matched no-context arm, regions as units."""
    from geoneural.evaluation.statistics import paired_log_ratio
    out = {}
    for b in sorted({r["boundM"] for r in rows}):
        at = [r for r in rows if r["boundM"] == b]

        def per_region(arm, free=False):
            vals: dict[str, list[float]] = {}
            for r in at:
                if r["arm"] == arm:
                    vals.setdefault(r["region"], []).append(r["bytes"] - (r["contextBytes"] if free else 0))
            return {k: float(np.mean(v)) for k, v in vals.items()}

        none = per_region("none")
        out[str(b)] = {arm: paired_log_ratio(per_region(arm), none) for arm in
                       ("constant", "real", "shifted", "shuffled", "wrong-region")}
        out[str(b)]["realWithMapAtDecoder"] = paired_log_ratio(per_region("real", free=True), none)
    return out


def cmd_codec_h2(a):
    from geoneural.codecs import campaign
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    rows = campaign.h2(out, tuple(a.regions), tuple(a.bounds), tuple(a.seeds))
    rec = protocol.record(
        "compression", "h2-geology", "exploratory", results={"rows": rows, "summary": summarise_h2(rows)},
        recipe={"start": "H1 leave-one-region-out model of the same fold and seed",
                "extraTraining": "one closed-loop round, 3000 steps, identical for every arm",
                "context": "GK100 material classes, 40 m raster, nearest node, 4-dimensional embedding",
                "charged": sorted(campaign.CHARGED)},
        split={"scheme": "leave one region out", "regions": list(a.regions)}, seeds=list(a.seeds))
    print(protocol.write(rec, out / "h2.json"))


def cmd_codec_h3(a):
    from geoneural.codecs import campaign
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    rows = campaign.h3(out, tuple(a.regions))
    rec = protocol.record(
        "compression", "h3-allocation", "exploratory",
        results={"rows": rows, "atEqualBytes": campaign.at_equal_bytes(rows, campaign.h3_metric)},
        recipe={"rules": campaign.H3_RULES, "bounds": list(campaign.H3_BOUNDS),
                "learnedModel": "H1 leave-one-region-out model, seed 0"},
        split={"scheme": "leave one region out", "regions": list(a.regions)}, seeds=[0])
    print(protocol.write(rec, out / "h3.json"))


def cmd_codec_paged(a):
    from geoneural.codecs import campaign
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    rows = campaign.paged_campaign(out, tuple(a.regions))
    rec = protocol.record(
        "compression", "paged-product", "exploratory", results={"rows": rows},
        recipe={"page": campaign.PAGE + 1, "directoryBytesPerPage": 8, "learnedModel": "H1 model, seed 0"},
        split={"scheme": "leave one region out", "regions": list(a.regions)}, seeds=[0])
    print(protocol.write(rec, out / "paged.json"))


def cmd_codec_sz3best(a):
    from geoneural.codecs import campaign, sz3tuned
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    rows = campaign.sz3_best_rows(tuple(a.regions))
    rec = protocol.record("compression", "sz3-best", "exploratory", results={"rows": rows},
                          recipe={"configs": sz3tuned.CONFIGS, "sz3": sz3tuned.version(),
                                  "rule": "smallest stream holding the bound, chosen per field and bound"},
                          split={"regions": list(a.regions)})
    print(protocol.write(rec, out / "sz3-best.json"))


def summarise_arms(rows: list[dict], reference: str, arms) -> dict:
    from geoneural.evaluation.statistics import paired_log_ratio
    out = {}
    for b in sorted({r["boundM"] for r in rows}):
        def per_region(arm):
            vals: dict[str, list[float]] = {}
            for r in rows:
                if r["boundM"] == b and r["arm"] == arm:
                    vals.setdefault(r["region"], []).append(r["bytes"])
            return {k: float(np.mean(v)) for k, v in vals.items()}
        ref = per_region(reference)
        out[str(b)] = {arm: paired_log_ratio(per_region(arm), ref) for arm in arms if arm != reference}
    return out


def cmd_codec_h4(a):
    from geoneural.codecs import campaign
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    rows = campaign.h4(out, tuple(a.regions), seeds=tuple(a.seeds))
    rec = protocol.record(
        "compression", "h4-process-prior", "exploratory",
        results={"rows": rows, "vsRealExtra": summarise_arms(rows, "real-extra", campaign.H4_ARMS),
                 "vsProceduralReal": summarise_arms(rows, "procedural-real", campaign.H4_ARMS)},
        recipe={"pretraining": "two closed-loop rounds on 48 synthetic 513 x 513 fields per kind",
                "fineTuning": "two closed-loop rounds on the five training regions",
                "synthetic": "see .data/synthetic/*/manifest.json"},
        split={"scheme": "leave one region out", "regions": list(a.regions)}, seeds=list(a.seeds))
    print(protocol.write(rec, out / "h4.json"))


def cmd_encode(a):
    from geoneural.codecs import package
    from geoneural.codecs.predictor import Predictor
    from geoneural.common import read_json
    z = np.load(a.atlas / "reference.npy")
    model = Predictor.from_bytes(a.model.read_bytes()) if a.model else None
    blob, recon, info = package.encode(z, a.bound, a.coder, model, embed_model=not a.corpus,
                                       spatial=package.spatial_from_atlas(read_json(a.atlas / "atlas.json")))
    a.out.write_bytes(blob)
    print(f"{a.out}: {len(blob)} bytes, max error {info['maxErrorM']:.6f} m, {info['breakdown']}")


def cmd_decode(a):
    from geoneural.codecs import package
    from geoneural.codecs.predictor import Predictor
    model = Predictor.from_bytes(a.model.read_bytes()) if a.model else None
    field = package.decode(a.product.read_bytes(), model)
    np.save(a.out, field.astype(np.float32) if a.float32 else field)
    print(a.out)


def cmd_train_predictor(a):
    from geoneural.codecs import campaign
    from geoneural.codecs.predictor import fit
    model = fit([campaign.load(r)[0] for r in a.regions], tuple(a.bounds), rounds=a.rounds,
                widths=tuple(a.widths), steps=a.steps, seed=a.seed)
    a.out.write_bytes(model.to_bytes())
    print(f"{a.out}: {model.nbytes()} bytes, sha256 {model.sha256()[:16]}, {model.meta}")


def register(add):
    from geoneural.codecs import campaign
    s = add("codec-h1", cmd_codec_h1, "Learned multilevel coder against conventional codecs, leave one region out")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--bounds", nargs="+", type=float, default=list(campaign.BOUNDS))
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    s.add_argument("--out", type=Path)
    s = add("codec-h2", cmd_codec_h2, "Geology context inside the learned coder, with matched controls")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--bounds", nargs="+", type=float, default=list(campaign.BOUNDS))
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    s.add_argument("--out", type=Path)
    s = add("codec-h3", cmd_codec_h3, "Bound allocation near the decoded drainage network and on gentle slopes")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--out", type=Path)
    s = add("codec-paged", cmd_codec_paged, "The random-access product: 257 x 257 pages coded alone")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--out", type=Path)
    s = add("codec-sz3best", cmd_codec_sz3best, "SZ3 at its best configuration per field (needs GEONEURAL_SZ3)")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--out", type=Path)
    s = add("codec-h4", cmd_codec_h4, "Process-prior pretraining of the shared predictor, with matched controls")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--seeds", nargs="+", type=int, default=[0])
    s.add_argument("--out", type=Path)
    s = add("encode", cmd_encode, "Encode an atlas reference into a .gnc product")
    s.add_argument("--atlas", type=Path, required=True)
    s.add_argument("--bound", type=float, required=True, help="Maximum absolute error in metres")
    s.add_argument("--coder", choices=["cubic-order0", "cubic-ctx", "learned"], default="cubic-ctx")
    s.add_argument("--model", type=Path, help="Shared predictor (.gnm) for the learned coder")
    s.add_argument("--corpus", action="store_true", help="Reference the shared model instead of embedding it")
    s.add_argument("--out", type=Path, required=True)
    s = add("decode", cmd_decode, "Decode a .gnc product using only the file (and a shared model if it names one)")
    s.add_argument("product", type=Path)
    s.add_argument("--model", type=Path)
    s.add_argument("--float32", action="store_true")
    s.add_argument("--out", type=Path, required=True)
    s = add("train-predictor", cmd_train_predictor, "Train the shared predictor of the multilevel coder")
    s.add_argument("--regions", nargs="+", required=True)
    s.add_argument("--bounds", nargs="+", type=float, default=list(campaign.BOUNDS))
    s.add_argument("--widths", nargs=2, type=int, default=[32, 32])
    s.add_argument("--steps", type=int, default=5000)
    s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, required=True)
