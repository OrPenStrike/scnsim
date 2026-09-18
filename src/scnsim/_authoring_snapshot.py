"""Immutable normalized authoring handoff; no canonical hashing or compiler logic."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from types import MappingProxyType
import numpy as np
from ._parameters import ParameterSet
from .units import Quantity, registry


def freeze(value: object) -> object:
    # This is an in-process immutable model, not a JSON encoder: point-field
    # keys intentionally remain typed ``(occurrence_path, field_id)`` tuples.
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, Quantity):
        magnitude = np.array(value.magnitude, copy=True)
        magnitude.setflags(write=False)
        return registry.Quantity(magnitude, value.units)
    return value


@dataclass(frozen=True, slots=True)
class AuthoringSnapshot:
    semantic_record: Mapping[str, object]
    source_provenance: Mapping[str, object]
    _resolution_refs: Mapping[tuple[str, str], object] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    _resolution_fields: Mapping[tuple[tuple[str, ...], str], object] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    @classmethod
    def create(
        cls,
        *,
        semantic_record: Mapping[str, object],
        source_provenance: Mapping[str, object],
        resolution_refs: Mapping[tuple[str, str], object] | None = None,
        resolution_fields: Mapping[tuple[tuple[str, ...], str], object] | None = None,
    ) -> "AuthoringSnapshot":
        return cls(
            freeze(dict(semantic_record)),
            freeze(dict(source_provenance)),
            MappingProxyType(dict(resolution_refs or {})),
            MappingProxyType(dict(resolution_fields or {})),
        )


@dataclass(frozen=True, slots=True)
class ResolvedPlanPoint:
    snapshot: AuthoringSnapshot
    effective_parameters: ParameterSet
    parameter_record: Mapping[str, object]
    resolved_fields: Mapping[tuple[tuple[str, ...], str], object]

    @classmethod
    def create(
        cls,
        *,
        snapshot: AuthoringSnapshot,
        effective_parameters: ParameterSet,
        parameter_record: Mapping[str, object],
        resolved_fields: Mapping[tuple[tuple[str, ...], str], object],
    ) -> "ResolvedPlanPoint":
        return cls(
            snapshot,
            effective_parameters,
            freeze(dict(parameter_record)),
            freeze(dict(resolved_fields)),
        )
