"""Authoring capture and frozen Run lookup handoff, independent of Workspace.

This record contains no backend, operation, callback, connection or Run handle.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from ..authoring import ParameterRef
from ..authoring.identity import canonical_plan_snapshot
from ..authoring.resolution import resolve_parameter_point
from ..canonical import canonical_json_bytes, sha256_hex
from ..errors import CompilerInvariantError, SCNSimValidationError
from ..workspace import _plan_coordinates

def _parameter_key(parameter: ParameterRef) -> tuple[str, str]:
    """Return one independent definitions-collection/local identity."""

    if not isinstance(parameter, ParameterRef):
        raise TypeError("parameter must be ParameterRef")
    definitions_id = getattr(parameter, "definitions_id", None)
    identifier = getattr(parameter, "id", None)
    if not isinstance(definitions_id, str) or not definitions_id or not isinstance(identifier, str) or not identifier:
        raise TypeError("ParameterRef has no canonical SCNSim parameter identity")
    return definitions_id, identifier

def _plan_has_affine_binding(value: object) -> bool:
    """Recognize affine expansion without interpreting its unimplemented support."""

    if isinstance(value, Mapping):
        return value.get("kind") == "affine" or any(
            _plan_has_affine_binding(item) for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_plan_has_affine_binding(item) for item in value)
    return False

def compatible_parameter(parameter_lookup, parameter: ParameterRef) -> ParameterRef:
        current = parameter_lookup.get(_parameter_key(parameter))
        if current is None or current._definition_record() != parameter._definition_record():
            raise SCNSimValidationError(
                "parameter is not a compatible consumed definition in this Plan",
                stage="preflight",
            )
        return current

def _freeze_document(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_document(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_document(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class CapturedRunBinding:
    snapshot: object
    baseline_point: object
    plan_document: Mapping[str, object]
    plan_bytes: bytes
    plan_sha256: str
    parameter_lookup: Mapping[tuple[str, str], ParameterRef]
    public_coordinates: frozenset[str]
    coordinate_order: tuple[str, ...]
    coordinate_lookup: Mapping[str, str | None]
    affine_plan: bool


def capture_run_binding(plan):
    snapshot = plan._capture_authoring_snapshot()
    baseline_point = resolve_parameter_point(snapshot)
    document = canonical_plan_snapshot(snapshot)
    encoded = canonical_json_bytes(document)
    parameter_lookup = MappingProxyType({
        _parameter_key(p): p for p in baseline_point.effective_parameters.values
    })
    # Validate the canonical record before its lists become immutable tuples.
    coordinate_order, public_coordinates = _plan_coordinates(document)
    if not public_coordinates:
        raise CompilerInvariantError("Plan public coordinate table is malformed", stage="plan_seal")
    lookup: dict[str, str | None] = {}
    for node in document["connectivity"]["node_coordinates"]:
        compiler_id = str(node["compiler_node_id"])
        lookup[compiler_id] = compiler_id
        for alias in node["public_aliases"]:
            identifier = alias.get("id")
            if isinstance(identifier, str):
                if identifier not in lookup:
                    lookup[identifier] = compiler_id
                elif lookup[identifier] != compiler_id:
                    lookup[identifier] = None
            if alias.get("kind") == "exposed_coordinate":
                scope = alias.get("scope")
                if isinstance(scope, list) and all(isinstance(item, str) for item in scope) and isinstance(identifier, str):
                    scoped = canonical_json_bytes({"scope": scope, "id": identifier}).decode("utf-8")
                    previous = lookup.setdefault(scoped, compiler_id)
                    if previous != compiler_id:
                        raise CompilerInvariantError(
                            "typed public coordinate resolves to multiple physical nodes",
                            stage="plan_seal",
                        )
    coordinate_lookup = MappingProxyType(lookup)
    return CapturedRunBinding(snapshot, baseline_point, _freeze_document(document), encoded,
                              sha256_hex(encoded), parameter_lookup,
                              frozenset(public_coordinates), tuple(coordinate_order), coordinate_lookup,
                              _plan_has_affine_binding(document))
