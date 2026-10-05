"""Optimization result envelope verification."""

from __future__ import annotations

from collections.abc import Mapping

from .common import (
    _SHA256,
    _finite_f64,
    _integrity,
    _required_extrapolation_rows,
    _verify_discretization,
    _verify_extrapolation_evidence,
    _verify_parameter_set_document,
)

def _verify_optimization_result(result: Mapping[str, object], request: Mapping[str, object], plan: Mapping[str, object], common: set[str]) -> None:
    expected = common | {"baseline", "best", "completed_generations", "unused_evaluations", "ledger_artifacts"}
    baseline = result.get("baseline")
    best = result.get("best")
    if set(result) != expected or not isinstance(baseline, dict) or not isinstance(best, dict):
        raise _integrity("Optimization Result envelope is open or incomplete.")
    best_fields = {"evaluation_ordinal", "cost_f64", "parameters"}
    if "discretization" in best:
        best_fields.add("discretization")
        _verify_discretization(best["discretization"], plan)
    if (
        set(best) != best_fields
        or not isinstance(best.get("evaluation_ordinal"), int)
        or isinstance(best.get("evaluation_ordinal"), bool)
        or best["evaluation_ordinal"] < 0
        or not _finite_f64(best.get("cost_f64"))
    ):
        raise _integrity("Optimization winner envelope is open.")
    _verify_parameter_set_document(best["parameters"], require_empty_authorization=True)
    expected_baseline = {
        "evaluation_ordinal", "origin", "generation", "population_column",
        "optimizer_coordinates_f64", "parameters", "cache_hit",
        "extrapolation_evidence", "outcome",
    }
    if "discretization" in baseline:
        expected_baseline.add("discretization")
        _verify_discretization(baseline["discretization"], plan)
    baseline_outcome = baseline.get("outcome")
    if (
        set(baseline) != expected_baseline
        or baseline.get("evaluation_ordinal") != 0
        or baseline.get("origin") != "baseline"
        or baseline.get("generation") != 0
        or baseline.get("population_column") is not None
        or baseline.get("cache_hit") is not False
        or not isinstance(baseline.get("optimizer_coordinates_f64"), list)
        or not baseline["optimizer_coordinates_f64"]
        or any(not _finite_f64(value) for value in baseline["optimizer_coordinates_f64"])
        or not isinstance(baseline_outcome, dict)
        or set(baseline_outcome) != {"status", "cost_f64", "objective_components"}
        or baseline_outcome.get("status") != "success"
        or not _finite_f64(baseline_outcome.get("cost_f64"))
        or not isinstance(baseline_outcome.get("objective_components"), list)
    ):
        raise _integrity("Optimization baseline envelope is open or malformed.")
    _verify_parameter_set_document(baseline["parameters"], require_empty_authorization=True)
    _verify_extrapolation_evidence(
        baseline.get("extrapolation_evidence"),
        allowed_sources={"none", "optimization_spec"},
        required_rows=_required_extrapolation_rows(
            plan,
            baseline["parameters"],
            authorization_source="optimization_spec",
            optimization_authorizations=request.get("spec", {}).get("allow_extrapolation", [])
            if isinstance(request.get("spec"), dict) else [],
        ),
    )
    generations = result.get("completed_generations")
    unused = result.get("unused_evaluations")
    ledgers = result.get("ledger_artifacts")
    if (
        not isinstance(generations, int)
        or isinstance(generations, bool)
        or generations < 1
        or not isinstance(unused, int)
        or isinstance(unused, bool)
        or unused < 0
        or not isinstance(ledgers, list)
        or len(ledgers) != generations
    ):
        raise _integrity("Optimization Result has no generation ledger catalog.")
    for generation, ledger in enumerate(ledgers, 1):
        text = str(generation).zfill(6)
        if (
            not isinstance(ledger, dict)
            or set(ledger) != {"id", "path", "sha256", "media_type", "byte_length"}
            or ledger.get("id") != f"generation_{text}"
            or ledger.get("path") != f"artifacts/generations/{text}.json"
            or ledger.get("media_type") != "application/json"
            or not isinstance(ledger.get("byte_length"), int)
            or isinstance(ledger.get("byte_length"), bool)
            or ledger["byte_length"] < 1
            or _SHA256.fullmatch(str(ledger.get("sha256", ""))) is None
        ):
            raise _integrity("Optimization ledger catalog entry is open or malformed.")
