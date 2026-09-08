"""Resolve a captured parameter point without a live Plan or diagram state.

The immutable snapshot owns field bindings and their original ParameterRefs.
This module applies those bindings and shared physical domains only; it neither
captures authoring state nor changes compiler topology or numerical requests.
"""

from __future__ import annotations

from ._authoring_snapshot import AuthoringSnapshot, ResolvedPlanPoint
from ._parameters import ParameterSet
from ._physical_values import (
    ParameterSpec,
    RLGCParameterSpec,
    _validate_captured_field,
)
from .errors import SCNSimValidationError
from .units import require_quantity


def resolve_parameter_point(
    snapshot: AuthoringSnapshot,
    supplied: ParameterSet | None = None,
) -> ResolvedPlanPoint:
    """Resolve the exact captured bindings, retaining units and failure semantics."""
    if not isinstance(snapshot, AuthoringSnapshot):
        raise SCNSimValidationError("invalid AuthoringSnapshot", stage="authoring")
    refs, fields = snapshot._resolution_refs, snapshot._resolution_fields
    if not refs and snapshot.semantic_record["parameter_closure"]["definitions"]:
        raise SCNSimValidationError(
            "snapshot lacks captured parameter bindings", stage="authoring"
        )
    supplied = ParameterSet() if supplied is None else supplied
    if not isinstance(supplied, ParameterSet) or set(supplied.values) - set(
        refs.values()
    ):
        raise SCNSimValidationError(
            "invalid ParameterSet for Plan", stage="authoring"
        )
    if any(
        refs[ref._key()]._definition_record() != ref._definition_record()
        for ref in supplied.values
    ):
        raise SCNSimValidationError(
            "ParameterSet definition conflicts with captured Plan",
            stage="authoring",
        )
    if any(
        ref._key() not in refs
        or refs[ref._key()]._definition_record() != ref._definition_record()
        or isinstance(ref.spec, RLGCParameterSpec)
        for ref in supplied.allow_extrapolation
    ):
        raise SCNSimValidationError(
            "invalid extrapolation authorization", stage="authoring"
        )
    effective_values = {
        ref: supplied.values.get(ref, ref.baseline) for ref in refs.values()
    }
    effective_source_units = {
        ref: supplied._source_units.get(ref, ref._source_unit)
        for ref in refs.values()
        if isinstance(ref.spec, ParameterSpec)
        and supplied._source_units.get(ref, ref._source_unit) is not None
    }
    effective = ParameterSet._from_normalized(
        effective_values,
        source_units=effective_source_units,
        allow_extrapolation=supplied.allow_extrapolation,
    )
    resolved = {}
    for (path, field), (
        base, binding, ref, unit, positive, nonnegative
    ) in fields.items():
        if ref is None:
            value = base
        elif binding["kind"] == "affine":
            slope, intercept, support = binding["_affine_data"]
            candidate = effective.values[ref]
            low, high = support
            if (
                not low.to(ref.spec.si_unit).magnitude
                <= candidate.to(ref.spec.si_unit).magnitude
                <= high.to(ref.spec.si_unit).magnitude
                and ref not in supplied.allow_extrapolation
            ):
                raise SCNSimValidationError(
                    "AffineMap value lies outside support", stage="authoring"
                )
            value = require_quantity(slope * candidate + intercept, unit, name=field)
        else:
            value = effective.values[ref]
        resolved[(path, field)] = _validate_captured_field(
            value,
            unit,
            field,
            positive,
            nonnegative,
        )
    return ResolvedPlanPoint.create(
        snapshot=snapshot,
        effective_parameters=effective,
        parameter_record=effective._record(),
        resolved_fields=resolved,
    )
