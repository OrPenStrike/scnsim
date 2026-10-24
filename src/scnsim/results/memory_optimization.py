"""Detached Results for Optimization executions with no Workspace storage.

The live execution owner supplies already-computed typed trial records. This
module binds those records to immutable public values without receipt identities,
fixed readers, numeric codecs, or persistence claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
import math
from typing import TYPE_CHECKING, Literal
from uuid import UUID

import numpy as np

from ..authoring import ParameterRef, ParameterSet
from ..construction import unavailable
from ..diagnostics.memory_operation import InMemoryBenchmarkResult
from ..errors import SCNSimStateError
from ..numerics.models import EvaluationResult
from ..visualization.presentation import Theme
from .base import LineDiscretization, Result, _sha256
from .factory import _MEMORY_RESULT_TOKEN, _verified_result

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


@dataclass(frozen=True, slots=True)
class InMemoryOptimizationTrial(Result):
    """One caller-owned optimizer occurrence, including typed failures."""

    stage_id: str
    ordinal: int
    evaluation_ordinal: int
    kind: str
    parameters: ParameterSet
    status: str
    cost: float
    objective_components: tuple[Mapping[str, object], ...]
    failure: Mapping[str, object] | None
    gradient: np.ndarray | None
    model_identity: Mapping[str, object]
    bodies: Mapping[int, EvaluationResult]
    observable_metadata: tuple[Mapping[str, object], ...]
    certificates: tuple[object, ...]
    discretization: tuple[LineDiscretization, ...] | None = None
    optimizer_unavailability: Mapping[str, object] | None = None

    def __init__(self) -> None:
        unavailable("InMemoryOptimizationTrial construction")

    @property
    def point(self) -> ParameterSet:
        return self.parameters


@dataclass(frozen=True, slots=True)
class InMemoryOptimizationStage(Result):
    """One materialized stage whose history shares the result's trial values."""

    stage_id: str
    strategy: str
    model_identity: Mapping[str, object]
    objective_ids: tuple[str, ...]
    variable_refs: tuple[ParameterRef, ...]
    status: str
    termination_reason: str | None
    evaluation_count: int
    iteration_count: int
    best_ordinal: int | None
    accepted_ordinal: int | None
    observed_best: InMemoryOptimizationTrial | None
    accepted_iterate: InMemoryOptimizationTrial | None
    trials: tuple[InMemoryOptimizationTrial, ...]

    def __init__(self) -> None:
        unavailable("InMemoryOptimizationStage construction")

    @property
    def history(self) -> tuple[InMemoryOptimizationTrial, ...]:
        return self.trials

    def __repr__(self) -> str:
        return (
            "InMemoryOptimizationStage("
            f"id={self.stage_id!r}, strategy={self.strategy!r}, "
            f"status={self.status!r}, evaluations={self.evaluation_count}, "
            f"iterations={self.iteration_count})"
        )


@dataclass(frozen=True, slots=True)
class InMemoryOptimizationResult(Result):
    """Materialized Optimization output with live-execution provenance only.

    ``plan_sha256``, ``request_sha256`` and ``execution_id`` identify the
    in-memory call, not a persisted attempt or a reopenable Workspace result.
    """

    plan_sha256: str
    request_sha256: str
    execution_id: str
    manifest: Mapping[str, object]
    outcome: str
    summary: Mapping[str, object]
    transitions: tuple[Mapping[str, object], ...]
    baseline: InMemoryOptimizationTrial | None
    best: InMemoryOptimizationTrial | None
    stages: tuple[InMemoryOptimizationStage, ...]
    trials: tuple[InMemoryOptimizationTrial, ...]
    comparison: Mapping[str, object]
    comparison_model_identity: Mapping[str, object] | None
    comparison_stage_id: str | None
    _presentation: Mapping[str, object] = field(repr=False, compare=False)
    _benchmark: InMemoryBenchmarkResult | None = field(default=None, repr=False, compare=False)
    _memory_report_token: object = field(default=None, init=False, repr=False, compare=False)

    def __init__(self) -> None:
        unavailable("InMemoryOptimizationResult construction")

    @property
    def ledger(self) -> tuple[InMemoryOptimizationTrial, ...]:
        """The ordered materialized trial sequence (not a durable ledger)."""

        return self.trials

    @property
    def history(self) -> tuple[InMemoryOptimizationTrial, ...]:
        return self.trials

    @property
    def candidate_count(self) -> int:
        return len(self.trials)

    @property
    def persistence(self) -> Literal["memory"]:
        return "memory"

    @property
    def persistence_kind(self) -> Literal["memory"]:
        return "memory"

    @property
    def benchmark(self) -> InMemoryBenchmarkResult:
        benchmark = object.__getattribute__(self, "_benchmark")
        if benchmark is None:
            raise RuntimeError("in-memory Optimization result finalization is incomplete")
        return benchmark

    def __repr__(self) -> str:
        best = "unavailable" if self.best is None else f"{self.best.cost:.12g}"
        return (
            "InMemoryOptimizationResult("
            f"execution_id={self.execution_id!r}, stages={len(self.stages)}, "
            f"candidates={len(self.trials)}, best_cost={best})"
        )

    def stage(self, stage_id: str | None = None) -> InMemoryOptimizationStage:
        if stage_id is None:
            if len(self.stages) == 1:
                return self.stages[0]
            if not self.stages:
                raise SCNSimStateError(
                    "in-memory Optimization has no stage projection",
                    stage="result_access",
                    evidence={"stage_count": 0},
                )
            raise SCNSimStateError(
                "stage_id is required when an in-memory Optimization has multiple stages",
                stage="result_access",
                evidence={"stage_ids": tuple(stage.stage_id for stage in self.stages)},
            )
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(stage_id)

    def plot(
        self,
        *,
        kind: Literal["history", "objective", "residual", "parameter", "table", "comparison"] = "history",
        objective: str | None = None,
        parameter: ParameterRef | None = None,
        stage_id: str | None = None,
        theme: Theme = Theme.AUTO,
    ) -> Figure:
        from ..visualization.plots.optimization import optimization_plot

        return optimization_plot(
            self, kind=kind, objective=objective, parameter=parameter,
            stage_id=stage_id, theme=theme,
        )

    def show(self, **presentation: object) -> None:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(**presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        row: int,
        col: int,
        kind: Literal["history", "objective", "residual", "parameter", "table", "comparison"],
        objective: str | None = None,
        parameter: ParameterRef | None = None,
        stage_id: str | None = None,
    ) -> Figure:
        from ..visualization.plots.optimization import optimization_add_to

        return optimization_add_to(
            self, fig, row=row, col=col, kind=kind, objective=objective,
            parameter=parameter, stage_id=stage_id,
        )


def _parameter_set(
    record: object,
    parameter_lookup: Mapping[tuple[str, str], ParameterRef],
) -> ParameterSet:
    """Convert one detached point through the canonical authoring boundary."""

    # The record was recursively detached at the memory execution boundary,
    # so RLGC array fields are tuples here. The established ParameterSet
    # decoder normalizes those JSON-sequence shapes before its exact roundtrip
    # comparison; numerical result arrays still bypass the persistent codec.
    from .decode import VerifiedResultDecoder

    decoder = object.__new__(VerifiedResultDecoder)
    object.__setattr__(decoder, "_parameter_lookup", parameter_lookup)
    return decoder._decode_parameter_set(record)


def _selected_trial(
    reference: object,
    trials: Mapping[tuple[str, int], InMemoryOptimizationTrial],
    *,
    name: str,
) -> InMemoryOptimizationTrial | None:
    if reference is None:
        return None
    if not isinstance(reference, Mapping):
        raise TypeError(f"in-memory Optimization {name} reference is malformed")
    stage_id = reference["stage_id"]
    ordinal = reference["ordinal"]
    if not isinstance(stage_id, str) or isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise TypeError(f"in-memory Optimization {name} reference is malformed")
    try:
        return trials[(stage_id, ordinal)]
    except KeyError as error:
        raise ValueError(f"in-memory Optimization {name} reference has no trial") from error


def _candidate_cost(cost: object, status: object) -> float:
    """Keep the one allowed non-finite cost attached to its failed trial."""

    if type(cost) is not float or status not in {"success", "failure"}:
        raise TypeError("in-memory Optimization candidate cost or status is malformed")
    if status == "failure":
        if cost != math.inf:
            raise ValueError("failed in-memory Optimization trial must retain +infinity cost")
        return cost
    if not math.isfinite(cost):
        raise ValueError("successful in-memory Optimization trial cost must be finite")
    return cost


def _make_trial(
    row: object,
    *,
    parameter_lookup: Mapping[tuple[str, str], ParameterRef],
) -> InMemoryOptimizationTrial:
    if not isinstance(row, Mapping):
        raise TypeError("in-memory Optimization trial is malformed")
    stage_id = row["stage_id"]
    ordinal = row["ordinal"]
    evaluation_ordinal = row["evaluation_ordinal"]
    status = row["status"]
    if (
        not isinstance(stage_id, str) or not stage_id
        or isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0
        or isinstance(evaluation_ordinal, bool) or not isinstance(evaluation_ordinal, int) or evaluation_ordinal < 0
        or status not in {"success", "failure"}
    ):
        raise TypeError("in-memory Optimization trial identity or status is malformed")
    failure = row["failure"]
    if (status == "failure") != isinstance(failure, Mapping):
        raise TypeError("in-memory Optimization failure data disagrees with trial status")
    if status == "success" and failure is not None:
        raise TypeError("successful in-memory Optimization trial cannot carry failure data")
    cost = _candidate_cost(row["cost"], status)
    gradient = row["gradient"]
    if gradient is not None and not isinstance(gradient, np.ndarray):
        raise TypeError("in-memory Optimization gradient must be a detached ndarray or None")
    if status == "failure" and gradient is not None:
        raise TypeError("failed in-memory Optimization trial cannot carry a gradient")
    raw_components = row["objective_components"]
    if not isinstance(raw_components, Sequence) or isinstance(raw_components, (str, bytes)):
        raise TypeError("in-memory Optimization objective components are malformed")
    components: list[Mapping[str, object]] = []
    for component in raw_components:
        if not isinstance(component, Mapping) or component.get("status") not in {
            "success", "failure", "unevaluated"
        }:
            raise TypeError("in-memory Optimization objective component is malformed")
        components.append(component)

    raw_bodies = row["bodies"]
    if not isinstance(raw_bodies, Mapping):
        raise TypeError("in-memory Optimization observable bodies are malformed")
    bodies: dict[int, EvaluationResult] = {}
    for index, body in raw_bodies.items():
        if isinstance(index, bool) or not isinstance(index, int) or not isinstance(body, EvaluationResult):
            raise TypeError("in-memory Optimization observable body is malformed")
        bodies[index] = body
    if status == "failure" and bodies:
        raise TypeError("failed in-memory Optimization trial cannot carry success bodies")

    raw_metadata = row["observable_metadata"]
    if not isinstance(raw_metadata, Sequence) or isinstance(raw_metadata, (str, bytes)):
        raise TypeError("in-memory Optimization observable metadata is malformed")
    if not all(isinstance(item, Mapping) for item in raw_metadata):
        raise TypeError("in-memory Optimization observable metadata rows are malformed")
    certificates = row["certificates"]
    if not isinstance(certificates, tuple):
        raise TypeError("in-memory Optimization certificates must be an ordered detached tuple")
    certificate_fields = {
        "site", "role", "stage", "derivative_order", "dependency", "basis_kind",
        "basis_indices", "work_index", "frequency_source_index", "status", "eta",
        "residual_numerator", "residual_denominator", "performed", "present",
    }
    for certificate in certificates:
        if not is_dataclass(certificate) or not certificate_fields.issubset(
            {item.name for item in fields(certificate)}
        ):
            raise TypeError("in-memory Optimization certificate lacks its role, basis, or payload")

    model_identity = row["model_identity"]
    if not isinstance(model_identity, Mapping):
        raise TypeError("in-memory Optimization model identity is malformed")
    optimizer_unavailability = row.get("optimizer_unavailability")
    if optimizer_unavailability is not None:
        if not isinstance(optimizer_unavailability, Mapping):
            raise TypeError("in-memory optimizer unavailability is malformed")
        kind = optimizer_unavailability.get("kind")
        detail = optimizer_unavailability.get("detail")
        role = optimizer_unavailability.get("role")
        evidence = optimizer_unavailability.get("evidence")
        if (
            not isinstance(kind, str) or not kind
            or not isinstance(detail, str) or not detail
            or role != "candidate_value_gradient"
            or not isinstance(evidence, Mapping)
        ):
            raise TypeError("in-memory optimizer unavailability is malformed")
    if not isinstance(row["kind"], str) or not row["kind"]:
        raise TypeError("in-memory Optimization trial kind is malformed")
    # Detach public approval without changing the execution-owned trial record.
    point = _parameter_set(
        dict(row["parameters"], allow_extrapolation=[]), parameter_lookup
    )

    discretization = None
    if status == "success":
        from .memory_direct import _memory_discretization

        discretization = _memory_discretization(row["discretization"])

    return _verified_result(
        InMemoryOptimizationTrial,
        stage_id=stage_id,
        ordinal=ordinal,
        evaluation_ordinal=evaluation_ordinal,
        kind=row["kind"],
        parameters=point,
        status=status,
        cost=cost,
        objective_components=tuple(components),
        failure=failure,
        gradient=gradient,
        model_identity=model_identity,
        bodies=bodies,
        observable_metadata=tuple(raw_metadata),
        certificates=certificates,
        discretization=discretization,
        optimizer_unavailability=optimizer_unavailability,
    )


def build_memory_optimization_result(
    *,
    payload: Mapping[str, object],
    request: Mapping[str, object],
    plan_sha256: str,
    request_sha256: str,
    execution_id: str,
    parameter_lookup: Mapping[tuple[str, str], ParameterRef],
    benchmark: InMemoryBenchmarkResult | None = None,
) -> InMemoryOptimizationResult:
    """Build the live typed Optimization Result directly from its RAM payload."""

    if not isinstance(payload, Mapping) or payload.get("persistence") != "memory":
        raise TypeError("in-memory Optimization payload has no memory persistence authority")
    if request.get("operation") != "optimize_direct":
        raise TypeError("in-memory Optimization requires its optimize_direct request")
    spec = request.get("spec")
    if not isinstance(spec, Mapping):
        raise TypeError("in-memory Optimization request has no Spec mapping")
    _sha256(plan_sha256, name="plan_sha256")
    _sha256(request_sha256, name="request_sha256")
    UUID(execution_id)
    if payload["execution_id"] != execution_id:
        raise ValueError("in-memory Optimization payload belongs to another execution")
    if not isinstance(parameter_lookup, Mapping):
        raise TypeError("in-memory Optimization requires the captured parameter lookup")
    from types import MappingProxyType

    parameter_lookup = MappingProxyType(dict(parameter_lookup))
    raw_trials = payload["trials"]
    raw_stages = payload["stages"]
    if not isinstance(raw_trials, Sequence) or isinstance(raw_trials, (str, bytes)):
        raise TypeError("in-memory Optimization trial sequence is malformed")
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, (str, bytes)):
        raise TypeError("in-memory Optimization stage sequence is malformed")

    trials = tuple(_make_trial(row, parameter_lookup=parameter_lookup) for row in raw_trials)
    by_stage_ordinal: dict[tuple[str, int], InMemoryOptimizationTrial] = {}
    for trial in trials:
        key = (trial.stage_id, trial.ordinal)
        if key in by_stage_ordinal:
            raise ValueError("in-memory Optimization trial occurrence is duplicated")
        by_stage_ordinal[key] = trial

    stages: list[InMemoryOptimizationStage] = []
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, Mapping):
            raise TypeError("in-memory Optimization stage is malformed")
        stage_id = raw_stage["stage_id"]
        if not isinstance(stage_id, str) or not stage_id:
            raise TypeError("in-memory Optimization stage id is malformed")
        variable_refs = tuple(
            parameter_lookup[(reference["definitions_id"], reference["parameter_id"])]
            for reference in raw_stage["variable_refs"]
        )
        raw_stage_trials = raw_stage["trials"]
        if not isinstance(raw_stage_trials, Sequence) or isinstance(raw_stage_trials, (str, bytes)):
            raise TypeError("in-memory Optimization stage trial sequence is malformed")
        stage_trials = tuple(
            by_stage_ordinal[(stage_id, row["ordinal"]) ]
            for row in raw_stage_trials
        )
        best_ordinal = raw_stage["best_ordinal"]
        accepted_ordinal = raw_stage["accepted_ordinal"]
        observed_best = (
            None if best_ordinal is None
            else by_stage_ordinal[(stage_id, best_ordinal)]
        )
        accepted_iterate = (
            None if accepted_ordinal is None
            else by_stage_ordinal[(stage_id, accepted_ordinal)]
        )
        stages.append(_verified_result(
            InMemoryOptimizationStage,
            stage_id=stage_id,
            strategy=raw_stage["strategy"],
            model_identity=raw_stage["model_identity"],
            objective_ids=tuple(raw_stage["objective_ids"]),
            variable_refs=variable_refs,
            status=raw_stage["status"],
            termination_reason=raw_stage["termination_reason"],
            evaluation_count=raw_stage["eval_count"],
            iteration_count=raw_stage["iter_count"],
            best_ordinal=best_ordinal,
            accepted_ordinal=accepted_ordinal,
            observed_best=observed_best,
            accepted_iterate=accepted_iterate,
            trials=stage_trials,
        ))
    stage_values = tuple(stages)
    if len({stage.stage_id for stage in stage_values}) != len(stage_values):
        raise ValueError("in-memory Optimization stage identity is duplicated")

    comparison = payload["comparison"]
    if not isinstance(comparison, Mapping):
        raise TypeError("in-memory Optimization comparison projection is malformed")
    baseline = _selected_trial(comparison["baseline_ref"], by_stage_ordinal, name="baseline")
    best = _selected_trial(comparison["winner_ref"], by_stage_ordinal, name="winner")
    comparison_model_identity = comparison["comparison_model_identity"]
    comparison_stage_id = comparison["comparison_stage_id"]
    if comparison_model_identity is not None and not isinstance(comparison_model_identity, Mapping):
        raise TypeError("in-memory Optimization comparison model identity is malformed")
    if comparison_stage_id is not None and not isinstance(comparison_stage_id, str):
        raise TypeError("in-memory Optimization comparison stage id is malformed")

    if benchmark is not None and not isinstance(benchmark, InMemoryBenchmarkResult):
        raise TypeError("in-memory Optimization benchmark has the wrong type")
    presentation = {
        "initial_parameters": None if baseline is None else baseline.parameters,
        "initial_candidate": baseline,
        "best_candidate": best,
        "objectives": tuple(spec["objectives"]),
        "variables": tuple(spec["variables"]),
        "best_evaluation_ordinal": None if best is None else best.evaluation_ordinal,
    }
    def execution_ref(trial: InMemoryOptimizationTrial | None) -> Mapping[str, object] | None:
        if trial is None:
            return None
        return {
            "execution_id": execution_id,
            "stage_id": trial.stage_id,
            "ordinal": trial.ordinal,
        }

    comparison_projection = {
        "comparison_model_identity": comparison_model_identity,
        "comparison_stage_id": comparison_stage_id,
        "declarations": comparison["declarations"],
        "baseline_ref": execution_ref(baseline),
        "winner_ref": execution_ref(best),
    }
    result = _verified_result(
        InMemoryOptimizationResult,
        plan_sha256=plan_sha256,
        request_sha256=request_sha256,
        execution_id=execution_id,
        manifest=payload["manifest"],
        outcome=payload["outcome"],
        summary=payload["summary"],
        transitions=tuple(payload["transitions"]),
        baseline=baseline,
        best=best,
        stages=stage_values,
        trials=trials,
        comparison=comparison_projection,
        comparison_model_identity=comparison_model_identity,
        comparison_stage_id=comparison_stage_id,
        _presentation=presentation,
        _benchmark=benchmark,
    )
    object.__setattr__(result, "_memory_report_token", _MEMORY_RESULT_TOKEN)
    return result


def attach_memory_optimization_benchmark(
    result: InMemoryOptimizationResult,
    benchmark: InMemoryBenchmarkResult,
) -> InMemoryOptimizationResult:
    """Attach the recorder's post-resource snapshot before exposing the Result."""

    if type(result) is not InMemoryOptimizationResult:
        raise TypeError("benchmark can be attached only to an in-memory Optimization result")
    if not isinstance(benchmark, InMemoryBenchmarkResult):
        raise TypeError("in-memory Optimization result requires its in-memory benchmark")
    if object.__getattribute__(result, "_benchmark") is not None:
        raise RuntimeError("in-memory Optimization benchmark was already finalized")
    object.__setattr__(result, "_benchmark", benchmark)
    return result


__all__ = [
    "InMemoryOptimizationResult",
    "InMemoryOptimizationStage",
    "InMemoryOptimizationTrial",
    "attach_memory_optimization_benchmark",
    "build_memory_optimization_result",
]
