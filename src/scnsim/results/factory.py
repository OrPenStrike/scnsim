from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, fields
from types import MappingProxyType
from typing import TypeVar

from ..errors import HBCaseFailure
from .base import (
    AnalysisResult, MatrixView, ParameterPointIdentity,
    Result, ResultIdentity, _VERIFIED_TOKEN, _freeze, _is_verified_result_identity,
    _is_verified_identity, _sha256, _valid_source_index,
)
from .derived import ReportResult
from .hb import HBCaseOutcome, HBBatchResult
from .matrix import (
    DiagonalRootResult, DirectQuantityResult, DirectSolveResult,
    OperatorElementRootResult, OperatorPointResult, OperatorResult,
    ReconciliationEvidence,
)
from .optimization import OptimizationBest, OptimizationResult
from .sweep import ParameterSweepResult

T = TypeVar("T")

def _verified_result(cls: type[T], /, **values: object) -> T:
    """Private verified-decoder hook; never call this on unverified evidence.

    The caller supplies exactly the public dataclass fields for ``cls``.  The
    hook validates receipt identity fields and recursively detaches mappings,
    sequences, and NumPy/Pint arrays before the value becomes user-visible.
    """

    if not isinstance(cls, type) or not issubclass(cls, (Result, MatrixView, ResultIdentity, ParameterPointIdentity, ReconciliationEvidence, OptimizationBest, OperatorPointResult)):
        raise TypeError("_verified_result only constructs SCNSim result values")
    if cls is HBCaseOutcome:
        expected = {
            "id", "failure", "effective_sources", "operating_point_closure", "bias_state", "pump_state", "s", "y", "z", "traces", "states", "state_node_map",
        }
        if set(values) != expected:
            raise TypeError("verified HBCaseOutcome fields mismatch")
        failure = values["failure"]
        success = failure is None
        if not isinstance(values["id"], str) or not values["id"]:
            raise ValueError("HB case id must be nonempty")
        if failure is not None and not isinstance(failure, HBCaseFailure):
            raise TypeError("HB failure must be HBCaseFailure")
        required = ("operating_point_closure", "bias_state", "pump_state", "s", "y", "z", "traces", "states", "state_node_map")
        if success != all(values[name] is not None for name in required):
            raise ValueError("HB success must provide every success-only surface")
        if not success and any(values[name] is not None for name in required):
            raise ValueError("HB failure must not retain success-only surfaces")
        if not isinstance(values["effective_sources"], (tuple, list)):
            raise TypeError("HB outcome effective_sources must be an ordered sequence")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_id", values["id"])
        object.__setattr__(instance, "_failure", failure)
        for name in ("effective_sources", "operating_point_closure", "bias_state", "pump_state", "s", "y", "z", "traces", "states", "state_node_map"):
            object.__setattr__(instance, f"_{name}", _freeze(values[name]))
        return instance
    if cls is HBBatchResult:
        if set(values) != {"identity", "discretization", "cases", "topology_evidence", "_presentation"}:
            raise TypeError("verified HBBatchResult fields mismatch")
        identity, cases = values["identity"], values["cases"]
        if not _is_verified_result_identity(identity) or not isinstance(cases, Mapping) or not cases or not isinstance(values["topology_evidence"], Mapping):
            raise TypeError("verified HBBatchResult requires identity and nonempty cases")
        materialized = dict(cases)
        if any(
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(outcome, HBCaseOutcome)
            or outcome.id != identifier
            for identifier, outcome in materialized.items()
        ):
            raise TypeError("verified HBBatchResult cases are malformed")
        instance = object.__new__(cls)
        object.__setattr__(instance, "identity", identity)
        object.__setattr__(instance, "discretization", _freeze(values["discretization"]))
        object.__setattr__(instance, "cases", MappingProxyType(materialized))
        object.__setattr__(instance, "topology_evidence", _freeze(values["topology_evidence"]))
        object.__setattr__(instance, "_presentation", _freeze(values["_presentation"]))
        object.__setattr__(instance, "_verified_result_token", _VERIFIED_TOKEN)
        return instance
    expected = {item.name: item for item in fields(cls) if item.init}
    missing = set(expected) - set(values)
    extra = set(values) - set(expected)
    required_missing = {
        name for name in missing
        if expected[name].default is MISSING and expected[name].default_factory is MISSING
    }
    if required_missing or extra:
        raise TypeError(f"verified {cls.__name__} fields mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for name in missing:
        descriptor = expected[name]
        values[name] = descriptor.default_factory() if descriptor.default_factory is not MISSING else descriptor.default
    if cls is ReportResult:
        html = values.get("html")
        inputs = values.get("inputs")
        presentation_sha256 = values.get("presentation_sha256")
        if not isinstance(html, str) or not html:
            raise TypeError("verified ReportResult requires nonempty HTML")
        if (
            not isinstance(inputs, (tuple, list))
            or not inputs
            or not all(_is_verified_analysis_result(item) for item in inputs)
        ):
            raise TypeError("verified ReportResult requires nonempty verified inputs")
        _sha256(presentation_sha256, name="presentation_sha256")
        if presentation_sha256 not in html:
            raise ValueError("ReportResult HTML must contain its presentation identity")
    if cls is ResultIdentity:
        for name, value in values.items():
            _sha256(value, name=name)
    if cls is ParameterPointIdentity:
        batch = values.get("batch")
        source_index = values.get("source_index")
        if not _is_verified_identity(batch) or not _valid_source_index(source_index):
            raise TypeError("parameter-point identity fields are invalid")
        _sha256(values.get("parameters_sha256"), name="parameters_sha256")
    if issubclass(cls, AnalysisResult) and not _is_verified_result_identity(values.get("identity")):
        raise TypeError("analysis results require a verified batch or point identity")
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, _freeze(value))
    if cls is ResultIdentity:
        object.__setattr__(instance, "_verified_identity_token", _VERIFIED_TOKEN)
    if cls is ParameterPointIdentity:
        object.__setattr__(instance, "_verified_point_identity_token", _VERIFIED_TOKEN)
    if issubclass(cls, AnalysisResult):
        object.__setattr__(instance, "_verified_result_token", _VERIFIED_TOKEN)
    return instance

def _is_verified_analysis_result(value: object) -> bool:
    return (
        type(value) in (
            DirectSolveResult,
            DiagonalRootResult,
            OperatorElementRootResult,
            DirectQuantityResult,
            OperatorResult,
            OptimizationResult,
            HBBatchResult,
            ParameterSweepResult,
        )
        and getattr(value, "_verified_result_token", None) is _VERIFIED_TOKEN
        and _is_verified_result_identity(getattr(value, "identity", None))
    )
