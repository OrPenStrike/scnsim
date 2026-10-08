"""Canonical prepared-request and parameter-source envelopes.

This encoder consumes finalized declarations and preserves the operation-level
algorithm identity, including JAX quantity dependencies of CMA; native Julia
identifiers remain independent. It does not select Views or resolve parameters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from ..authoring.identity import (
    _iter_mappings,
    _mapping,
    _parameter_ref_key,
    canonical_parameter_set,
)
from ..canonical import _nfc, _sha256, _validation, canonical_value


_DIRECT_ALGORITHMS = {
    "solve_direct": "scnsim.direct_response.v1",
    "solve_hb": "scnsim.hb_response.josephsoncircuits.v1",
    "optimize_direct": "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v9",
}


_JAX_ALGORITHMS = {
    "solve_direct": "scnsim.jax.sparse_superlu.direct_response.v1",
    "diagonal_root": "scnsim.jax.sparse_superlu.diagonal_root.newton32.v1",
    "response_element": "scnsim.jax.sparse_superlu.response_element.v1",
    "operator_element_root": "scnsim.jax.sparse_superlu.operator_element_root.newton32.v1",
    "hybridized_pole": "scnsim.jax.sparse_superlu.hybridized_pole.newton32.v1",
    "transfer_zero": "scnsim.jax.sparse_superlu.transfer_zero.newton32.v1",
    "residue_normalized_coupling": "scnsim.jax.sparse_superlu.residue_normalized_coupling.v1",
    "operator": "scnsim.jax.sparse_superlu.direct_operator.v1",
    "optimize_direct": "scnsim.jax.sparse_superlu.direct_cmaes.cmaes_0_13_1.v2",
}


_EVALUATION_ALGORITHMS = {
    "diagonal_root": "scnsim.diagonal_root.newton32.v2",
    "operator_element_root": "scnsim.operator_element_root.newton32.v1",
    "hybridized_pole": "scnsim.hybridized_pole.newton32.v1",
    "transfer_zero": "scnsim.transfer_zero.newton32.v4",
    "residue_normalized_coupling": "scnsim.residue_normalized_coupling.v2",
    "response_element": "scnsim.response_element.v1",
    "operator": "scnsim.direct_operator.v1",
}


def canonical_parameter_source(source: Mapping[str, object]) -> dict[str, object]:
    """Close a point or lazily enumerable ordered parameter-space descriptor."""

    document = dict(source)
    kind = document.get("kind")
    if kind == "point":
        if set(document) != {"kind", "parameters"}:
            raise _validation("point parameter source fields are invalid")
        document["parameters"] = canonical_parameter_set(
            _mapping(document["parameters"], "parameters")
        )
    elif kind == "grid":
        if set(document) != {"kind", "base_parameters", "axes", "shape"}:
            raise _validation("grid parameter source fields are invalid")
        document["base_parameters"] = canonical_parameter_set(
            _mapping(document["base_parameters"], "base_parameters")
        )
        axes = [dict(item) for item in _iter_mappings(document["axes"], "axes")]
        shape = document["shape"]
        if (
            not axes
            or not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes))
            or len(shape) != len(axes)
        ):
            raise _validation("grid shape must match its nonempty ordered axes")
        keys: list[tuple[str, str]] = []
        for index, axis in enumerate(axes):
            if set(axis) != {"parameter", "values"}:
                raise _validation("grid axis fields are invalid")
            key = _parameter_ref_key(axis["parameter"])
            values = axis["values"]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
                raise _validation("grid axes must be nonempty arrays")
            if isinstance(shape[index], bool) or not isinstance(shape[index], int) or shape[index] != len(values):
                raise _validation("grid shape disagrees with an axis length")
            keys.append(key)
        if len(set(keys)) != len(keys):
            raise _validation("grid axis parameters must be unique")
        document["axes"] = axes  # Author order is semantic.
        document["shape"] = list(shape)
    elif kind == "points":
        if set(document) != {"kind", "baseline_parameters", "points"}:
            raise _validation("listed parameter source fields are invalid")
        document["baseline_parameters"] = canonical_parameter_set(
            _mapping(document["baseline_parameters"], "baseline_parameters")
        )
        points = document["points"]
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or not points:
            raise _validation("listed parameter source must contain points")
        document["points"] = [
            canonical_parameter_set(_mapping(item, "listed point")) for item in points
        ]
    else:
        raise _validation("unknown parameter source kind", kind=kind)
    return canonical_value(document)  # type: ignore[return-value]


def canonical_request_document(
    *,
    plan_sha256: str,
    operation: str,
    view: Mapping[str, object],
    spec: Mapping[str, object],
    parameter_source: Mapping[str, object],
    runtime_semantic: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact closed declarative request envelope.

    The operation remains deliberately coarse: every scalar Direct evaluation
    shares ``evaluate_direct`` while its closed Spec discriminator selects the
    Human-defined algorithm identity.  This keeps the request envelope stable
    without a second operation family.
    """

    selected_operation = _nfc(operation, field="operation")
    runtime = dict(runtime_semantic)
    spec_type = spec.get("type")
    expected_algorithm = (
        _EVALUATION_ALGORITHMS.get(str(spec_type))
        if selected_operation == "evaluate_direct"
        else _DIRECT_ALGORITHMS.get(selected_operation)
    )
    if runtime.get("backend") == "jax":
        expected_algorithm = _JAX_ALGORITHMS.get(
            str(spec_type) if selected_operation == "evaluate_direct" else selected_operation
        )
    expected_spec = {
        "solve_direct": "direct_solve",
        "solve_hb": "hb_solve",
        "optimize_direct": "optimization",
    }.get(selected_operation)
    if expected_algorithm is None:
        raise _validation("operation or Spec is outside the runtime", operation=selected_operation, spec_type=spec_type)
    if expected_spec is not None and spec_type != expected_spec:
        raise _validation("operation requires a different Spec", operation=selected_operation, spec_type=spec_type)
    if runtime.get("algorithm_id") != expected_algorithm:
        raise _validation("request algorithm does not match operation and Spec", operation=selected_operation, spec_type=spec_type)
    return canonical_value({
        "schema": "scnsim.request",
        "schema_version": 2,
        "plan_sha256": _sha256(plan_sha256, field="plan_sha256"),
        "operation": selected_operation,
        "view": dict(view),
        "spec": dict(spec),
        "parameter_source": canonical_parameter_source(parameter_source),
        "runtime_semantic": runtime,
    })  # type: ignore[return-value]
