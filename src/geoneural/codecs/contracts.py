"""Interface contracts for future streaming, conditioning and physics adapters."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Protocol
import numpy as np


class ErrorKind(str,Enum):
    SAMPLE_BOUND='sample-bound'
    CONTINUOUS_BOUND='continuous-bound'
    ESTIMATED='estimated'
    UNKNOWN='unknown'


@dataclass(frozen=True)
class RegionRequest:
    source_id: str
    chart_id: str
    level: int
    index_x: int
    index_y: int
    sample_side: int
    maximum_output_bytes: int


@dataclass(frozen=True)
class Uncertainty:
    """One uncertainty channel, kept distinct from the others.

    Observation uncertainty, inference uncertainty and codec error against the
    chosen reference are three separate quantities. A small codec error does not
    reduce either of the others. `value_metres` is None when the quantity is not
    calibrated; it is never silently set to zero, and `basis` must say what
    established the value or why it is absent.
    """
    kind: str
    basis: str
    value_metres: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("observation", "inference", "codec"):
            raise ValueError(f"Unknown uncertainty channel {self.kind!r}")
        if not self.basis:
            raise ValueError("An uncertainty needs a stated basis, including when it is unquantified")
        if self.value_metres is not None and not self.value_metres >= 0.0:
            raise ValueError("Uncertainty in metres must be non-negative and finite")


def unquantified(kind: str, basis: str) -> Uncertainty:
    """Declare a channel that exists but has no calibrated value yet."""
    return Uncertainty(kind=kind, basis=basis, value_metres=None)


@dataclass(frozen=True)
class FieldPatch:
    values: np.ndarray
    horizontal_crs: str
    vertical_crs: str
    spacing_metres: float
    source_id: str
    error_kind: ErrorKind
    error_metres: float | None
    observation_uncertainty: Uncertainty
    inference_uncertainty: Uncertainty


@dataclass(frozen=True)
class DeploymentBytes:
    """Every shipped byte, split by owner.

    `total` sums `vars(self)`, so a new component is counted automatically. Keep
    every field a non-negative int and do not add `slots=True`: both would defeat
    that automatic counting and reintroduce silent exclusions.

    `field_payload` holds explicitly stored field samples, such as conventional
    compressed pages. It defaults to zero so positional construction with the
    first six fields keeps working.
    """
    shared_weights: int
    regional_codes: int
    conditioning_and_history: int
    residuals: int
    critical_features: int
    metadata_and_index: int
    field_payload: int = 0

    def total(self) -> int:
        items=tuple(vars(self).values())
        if any(v<0 for v in items): raise ValueError('Negative deployment accounting')
        return sum(items)


class AtlasDecoder(Protocol):
    def decode(self, request: RegionRequest) -> FieldPatch: ...
    def deployment_bytes(self) -> DeploymentBytes: ...
    def release(self) -> None: ...


class ConditioningProvider(Protocol):
    def sample(self, projected_xy: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """Return values/classes plus an explicit availability mask."""
        ...
    def content_identity(self) -> str: ...


class HistoryPrior(Protocol):
    def present_day_features(self, request: RegionRequest) -> FieldPatch: ...
    def posterior_provenance(self) -> dict: ...
