"""Complete deployment-byte accounting for a published package.

Deployment size includes every level payload, codebook, embedding, shared weight,
conditioning field, correction, index, framing and required metadata. Research
references and optimizer state are excluded from deployment size but reported
separately as storage cost; they are not eliminated, only owned elsewhere.

Accounting is deliberately strict: a file this module cannot classify raises
rather than being dropped, because an unnoticed exclusion makes a rate result
look better than it is.
"""
from __future__ import annotations
from pathlib import Path

from geoneural.codecs.contracts import DeploymentBytes

#: Suffix/name rules, most specific first. Each maps to a DeploymentBytes field
#: or to the research-only set, which never enters a deployment total.
RESEARCH_ONLY = "research_only"

_RULES: tuple[tuple[str, str], ...] = (
    ("reference.npy", RESEARCH_ONLY),
    ("reference.tif", RESEARCH_ONLY),
    ("nodata-report.json", RESEARCH_ONLY),
    # A decoder must know which samples are missing, so a shipped validity mask is deployment data. (An
    # all-valid field can say so with a flag instead; the reference atlases refuse missing values altogether.)
    ("nodata-mask.npy", "metadata_and_index"),
    ("optimizer.safetensors", RESEARCH_ONLY),
    ("weights.safetensors", "shared_weights"),
    ("factors.safetensors", "shared_weights"),
    ("codes.safetensors", "regional_codes"),
    ("codes.npy", "regional_codes"),
    ("categories.npy", "conditioning_and_history"),
    ("context.json", "conditioning_and_history"),
    ("history.json", "conditioning_and_history"),
    ("features.json", "critical_features"),
    (".eat.gz", "field_payload"),
    (".res.gz", "residuals"),
    (".json", "metadata_and_index"),
)


def classify(path: Path) -> str:
    """Name the accounting bucket for one published file."""
    name = path.name
    for pattern, bucket in _RULES:
        if name == pattern or (pattern.startswith(".") and name.endswith(pattern)):
            return bucket
    raise ValueError(
        f"Unclassified package file {name!r}. Add an explicit rule; deployment bytes "
        "must never omit a shipped file."
    )


def package_bytes(directory: Path, allow_unclassified: bool = False) -> dict:
    """Account every byte under a published package directory.

    Returns the deployment total, the separately-owned research bytes, and the
    per-file breakdown that lets a reviewer re-derive both.
    """
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Not a package directory: {root}")
    buckets = {
        "shared_weights": 0,
        "regional_codes": 0,
        "conditioning_and_history": 0,
        "residuals": 0,
        "critical_features": 0,
        "metadata_and_index": 0,
        "field_payload": 0,
    }
    research = 0
    files: list[dict] = []
    unclassified: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        size = path.stat().st_size
        try:
            bucket = classify(path)
        except ValueError:
            if not allow_unclassified:
                unclassified.append(str(path.relative_to(root)))
                continue
            bucket = "metadata_and_index"
        if bucket == RESEARCH_ONLY:
            research += size
        else:
            buckets[bucket] += size
        files.append({"path": str(path.relative_to(root)), "bytes": size, "bucket": bucket})
    if unclassified:
        raise ValueError(
            "Unclassified package files: " + ", ".join(unclassified[:16]) +
            ". Deployment bytes must account for every shipped file."
        )
    deployment = DeploymentBytes(**buckets)
    return {
        "schema": "geoneural-package-bytes-v1",
        "directory": str(root),
        "deployment": dict(buckets),
        "deployment_total_bytes": deployment.total(),
        "research_only_bytes": research,
        "files": files,
        "qualification": (
            "Deployment bytes are the shipped package only. Research references, source "
            "acquisitions and optimizer state are reported separately and are not eliminated "
            "by runtime compression."
        ),
    }
