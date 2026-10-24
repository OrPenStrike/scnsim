"""Decode model-bound Optimization projections from a fixed result reader."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from ..canonical import float64_from_hex
from ..errors import EvidenceIntegrityError
from .factory import _verified_result
from .lazy_optimization import LazyOptimizationStageHistory, _public_optimization_candidate
from .optimization import OptimizationBest, OptimizationResult, OptimizationStage


_FAILED_CANDIDATE_INFINITY = "7ff0000000000000"


def _candidate_cost(record: Mapping[str, object], status: str) -> float:
    """Decode finite costs and the one recorded failed-occurrence sentinel."""

    encoded = record.get("cost_f64")
    if encoded == _FAILED_CANDIDATE_INFINITY:
        if status != "failure":
            raise EvidenceIntegrityError(
                "Optimization infinite cost is not bound to a failed candidate",
                stage="result_decode",
            )
        return math.inf
    if not isinstance(encoded, str):
        raise EvidenceIntegrityError(
            "Optimization candidate cost is malformed", stage="result_decode"
        )
    return float64_from_hex(encoded)


def _candidate(decoder, record, model_identity, decode_discretization):
    if not isinstance(record, Mapping):
        raise EvidenceIntegrityError(
            "Optimization stage candidate projection is malformed",
            stage="result_decode",
        )
    parameters = record.get("parameters")
    ordinal = record.get("evaluation_ordinal")
    status = record.get("status")
    if (
        not isinstance(parameters, Mapping)
        or not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or not isinstance(status, str)
    ):
        raise EvidenceIntegrityError(
            "Optimization stage candidate lacks a complete saved occurrence",
            stage="result_decode",
        )
    parameters = _public_optimization_candidate(record)["parameters"]
    return _verified_result(
        OptimizationBest,
        parameters=decoder._decode_parameter_set(parameters),
        cost=_candidate_cost(record, status),
        discretization=decode_discretization(record.get("discretization")),
        evaluation_ordinal=ordinal,
        status=status,
        model_identity=model_identity,
        locator=record.get("locator"),
    )


def decode_staged_optimization(
    decoder, identity, reader, *, decode_discretization, adapt_candidate
):
    """Build compact stages and the exact comparison-model winner only."""

    stage_rows = reader.project("stages")
    if not isinstance(stage_rows, Sequence) or isinstance(stage_rows, (str, bytes)):
        raise EvidenceIntegrityError(
            "Optimization stage projection is malformed", stage="result_decode"
        )
    stages = []
    for row in stage_rows:
        if not isinstance(row, Mapping):
            raise EvidenceIntegrityError(
                "Optimization stage row is malformed", stage="result_decode"
            )
        stage_id = row.get("stage_id")
        model_identity = row.get("model_identity")
        objective_ids = row.get("objective_ids")
        variable_refs = row.get("variable_refs")
        if (
            not isinstance(stage_id, str)
            or not isinstance(model_identity, Mapping)
            or not isinstance(objective_ids, Sequence)
            or isinstance(objective_ids, (str, bytes))
            or not isinstance(variable_refs, Sequence)
            or isinstance(variable_refs, (str, bytes))
        ):
            raise EvidenceIntegrityError(
                "Optimization stage identity or declarations are malformed",
                stage="result_decode",
            )
        observed = row.get("observed_best")
        accepted = row.get("accepted_iterate")
        stages.append(_verified_result(
            OptimizationStage,
            stage_id=stage_id,
            strategy=row["strategy"],
            model_identity=model_identity,
            objective_ids=tuple(objective_ids),
            variable_refs=tuple(decoder._decode_parameter_ref(item) for item in variable_refs),
            observed_best=None if observed is None else _candidate(
                decoder, observed, model_identity, decode_discretization
            ),
            accepted_iterate=None if accepted is None else _candidate(
                decoder, accepted, model_identity, decode_discretization
            ),
            termination_reason=row.get("termination_reason"),
            status=row["status"],
            evaluation_count=row["evaluation_count"],
            iteration_count=row["iteration_count"],
            history=LazyOptimizationStageHistory(reader, stage_id, model_identity),
        ))

    comparison = reader.project("comparison")
    if not isinstance(comparison, Mapping):
        raise EvidenceIntegrityError(
            "Optimization comparison projection is malformed", stage="result_decode"
        )
    comparison_identity = comparison.get("comparison_model_identity")
    comparison_stage_id = comparison.get("comparison_stage_id")
    declarations = comparison.get("declarations")
    if not isinstance(declarations, Mapping):
        raise EvidenceIntegrityError(
            "Optimization comparison declarations are malformed", stage="result_decode"
        )
    objectives = declarations.get("objectives")
    variables = declarations.get("variables")
    if (
        not isinstance(objectives, Sequence)
        or isinstance(objectives, (str, bytes))
        or not isinstance(variables, Sequence)
        or isinstance(variables, (str, bytes))
    ):
        raise EvidenceIntegrityError(
            "Optimization comparison declarations are malformed", stage="result_decode"
        )

    baseline_record = comparison.get("baseline")
    winner_record = comparison.get("winner")
    if baseline_record is not None and not isinstance(baseline_record, Mapping):
        raise EvidenceIntegrityError(
            "Optimization comparison baseline is malformed", stage="result_decode"
        )
    if winner_record is not None and not isinstance(winner_record, Mapping):
        raise EvidenceIntegrityError(
            "Optimization comparison winner is malformed", stage="result_decode"
        )
    if comparison_identity is not None and not isinstance(comparison_identity, Mapping):
        raise EvidenceIntegrityError(
            "Optimization comparison model identity is malformed", stage="result_decode"
        )
    if winner_record is not None and comparison_identity is None:
        raise EvidenceIntegrityError(
            "Optimization comparison winner has no model identity", stage="result_decode"
        )

    # These are the only two full occurrence bodies selected for comparison.
    baseline = None if baseline_record is None else adapt_candidate(baseline_record)
    winner = None if winner_record is None else adapt_candidate(winner_record)
    top_best = (
        None if winner is None else _candidate(
            decoder, winner, comparison_identity, decode_discretization
        )
    )
    baseline_parameters = (
        None if baseline is None
        else decoder._decode_parameter_set(baseline["parameters"])
    )
    presentation = {
        "initial_parameters": baseline_parameters,
        "initial_candidate": baseline,
        "best_candidate": winner,
        "objectives": tuple(objectives),
        "variables": tuple(variables),
        "best_evaluation_ordinal": None if top_best is None else top_best.evaluation_ordinal,
        "comparison_model_identity": comparison_identity,
        "comparison_stage_id": comparison_stage_id,
    }
    return _verified_result(
        OptimizationResult,
        identity=identity,
        discretization=None if top_best is None else top_best.discretization,
        _fixed_reader=reader,
        best=top_best,
        stages=tuple(stages),
        comparison_model_identity=comparison_identity,
        comparison_stage_id=comparison_stage_id,
        ledger=(),
        candidate_discretization=(),
        _presentation=presentation,
    )
