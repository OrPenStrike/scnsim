"""Independent definitions and immutable parameter points for structured authoring."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from html import escape
import numpy as np
from ._physical_values import (
    ParameterSpec,
    RLGC,
    RLGCParameterSpec,
    identifier,
    quantity_record,
)
from .units import Quantity, registry, require_quantity


def _value(value: object, spec: object, name: str) -> object:
    if isinstance(spec, ParameterSpec):
        resolved = require_quantity(value, spec.si_unit, name=name)
        magnitude = np.asarray(resolved.to(spec.si_unit).magnitude)
        if magnitude.ndim != 0 or not np.isfinite(magnitude).all():
            raise ValueError(f"{name} must be a finite scalar Quantity")
        return registry.Quantity(float(magnitude), spec.si_unit)
    if not isinstance(spec, RLGCParameterSpec) or not isinstance(value, RLGC):
        raise TypeError(f"{name} must have its declared physical parameter type")
    if (
        value.conductors != spec.conductors
        or value.reference_conductor != spec.reference_conductor
    ):
        raise ValueError(f"{name} RLGC basis conflicts with its definition")
    return value


def _freeze_value(value: object) -> object:
    """Detach scalar data retained by a definition, point, or grid axis.

    Pint quantities are mutable (including through ``.ito()``), so a mapping
    proxy alone is not an immutable parameter boundary.  RLGC owns immutable
    numeric tuples and has copy-on-read public properties already.
    """
    if isinstance(value, Quantity):
        magnitude = np.array(value.magnitude, copy=True)
        magnitude.setflags(write=False)
        return registry.Quantity(magnitude, value.units)
    return value


def _public_value(value: object) -> object:
    """Return an independently mutable view of a stored physical value."""
    if isinstance(value, Quantity):
        return registry.Quantity(np.array(value.magnitude, copy=True), value.units)
    return value


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ParameterRef:
    definitions_id: str
    id: str
    _baseline: object
    spec: ParameterSpec | RLGCParameterSpec
    _source_unit: str | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "ParameterRef values are created by ParameterDefinitions.parameter()"
        )

    @classmethod
    def _create(
        cls,
        definitions_id: str,
        id: str,
        baseline: object,
        spec: ParameterSpec | RLGCParameterSpec,
        source_unit: str | None = None,
    ) -> "ParameterRef":
        value = object.__new__(cls)
        object.__setattr__(value, "definitions_id", definitions_id)
        object.__setattr__(value, "id", id)
        object.__setattr__(value, "_baseline", _freeze_value(baseline))
        object.__setattr__(value, "spec", spec)
        object.__setattr__(value, "_source_unit", source_unit)
        return value

    @property
    def baseline(self) -> object:
        return _public_value(self._baseline)

    def _key(self) -> tuple[str, str]:
        return self.definitions_id, self.id

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ParameterRef) and self._key() == other._key()

    def _key_record(self) -> dict[str, str]:
        return {"definitions_id": self.definitions_id, "parameter_id": self.id}

    def _definition_record(self) -> dict[str, object]:
        b = (
            self._baseline._record()
            if isinstance(self._baseline, RLGC)
            else quantity_record(self._baseline, self.spec.si_unit)
        )  # type: ignore
        return {**self._key_record(), "spec": self.spec._record(), "baseline": b}

    def show(self) -> object:
        from .results import HtmlPresentation

        return HtmlPresentation(
            f"<pre>ParameterRef(definitions_id={escape(self.definitions_id)!r}, parameter_id={escape(self.id)!r}, baseline={escape(str(self.baseline))})</pre>"
        )


class ParameterDefinitions:
    __slots__ = ("id", "_refs", "_frozen")

    def __init__(self, *, id: str) -> None:
        object.__setattr__(self, "id", identifier(id, field="definitions id"))
        object.__setattr__(self, "_refs", {})
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("ParameterDefinitions identity is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ParameterDefinitions identity is immutable")

    def parameter(
        self, *, id: str, baseline: object, spec: ParameterSpec | RLGCParameterSpec
    ) -> ParameterRef:
        local = identifier(id, field="parameter id")
        if not isinstance(spec, (ParameterSpec, RLGCParameterSpec)):
            raise TypeError("spec must be ParameterSpec or RLGCParameterSpec")
        source_unit = (
            str(baseline.units)
            if isinstance(spec, ParameterSpec) and isinstance(baseline, Quantity)
            else None
        )
        candidate = ParameterRef._create(
            self.id, local, _value(baseline, spec, local), spec, source_unit
        )
        old = self._refs.get(local)
        if old is not None:
            if old._definition_record() != candidate._definition_record():
                raise ValueError("conflicting definition for parameter identity")
            return old
        self._refs[local] = candidate
        return candidate

    def show(self) -> object:
        from .results import HtmlPresentation

        rows = "\n".join(
            f"{escape(ref.id)}: {escape(str(ref.baseline))}"
            for ref in self._refs.values()
        )
        return HtmlPresentation(
            f"<pre>ParameterDefinitions({escape(self.id)})\n{rows}</pre>"
        )


class ParameterSet:
    __slots__ = ("_values", "_source_units", "_allow_extrapolation", "_frozen")

    def __init__(
        self,
        values: Mapping[ParameterRef, object] | None = None,
        *,
        allow_extrapolation: Sequence[ParameterRef] = (),
    ) -> None:
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise TypeError("values must map ParameterRef values")
        out = {}
        source_units = {}
        for ref, value in values.items():
            if not isinstance(ref, ParameterRef):
                raise TypeError("ParameterSet keys must be ParameterRef")
            out[ref] = _freeze_value(_value(value, ref.spec, ref.id))
            if isinstance(ref.spec, ParameterSpec) and isinstance(value, Quantity):
                source_units[ref] = str(value.units)
        if any(not isinstance(ref, ParameterRef) for ref in allow_extrapolation):
            raise TypeError("allow_extrapolation must contain ParameterRef")
        self._initialize(out, source_units, allow_extrapolation)

    def _initialize(
        self,
        values: Mapping[ParameterRef, object],
        source_units: Mapping[ParameterRef, str],
        allow_extrapolation: Sequence[ParameterRef],
    ) -> None:
        """Freeze already validated canonical values with detached unit evidence."""
        if not set(source_units) <= set(values):
            raise ValueError("source units must belong to ParameterSet values")
        if any(not isinstance(unit, str) for unit in source_units.values()):
            raise TypeError("source unit evidence must be strings")
        object.__setattr__(
            self,
            "_values",
            MappingProxyType(
                dict(
                    sorted(
                        (
                            (ref, _freeze_value(value))
                            for ref, value in values.items()
                        ),
                        key=lambda item: item[0]._key(),
                    )
                )
            ),
        )
        object.__setattr__(
            self,
            "_source_units",
            MappingProxyType(
                dict(sorted(source_units.items(), key=lambda x: x[0]._key()))
            ),
        )
        object.__setattr__(
            self,
            "_allow_extrapolation",
            tuple(sorted(set(allow_extrapolation), key=ParameterRef._key)),
        )
        object.__setattr__(self, "_frozen", True)

    @classmethod
    def _from_normalized(
        cls,
        values: Mapping[ParameterRef, object],
        *,
        source_units: Mapping[ParameterRef, str],
        allow_extrapolation: Sequence[ParameterRef],
    ) -> "ParameterSet":
        """Copy canonical values without erasing caller-authored unit spellings."""
        result = object.__new__(cls)
        result._initialize(values, source_units, allow_extrapolation)
        return result

    @classmethod
    def _copy(cls, point: "ParameterSet") -> "ParameterSet":
        if not isinstance(point, cls):
            raise TypeError("point must be ParameterSet")
        return cls._from_normalized(
            point._values,
            source_units=point._source_units,
            allow_extrapolation=point.allow_extrapolation,
        )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("ParameterSet is immutable")
        object.__setattr__(self, name, value)

    @property
    def values(self) -> Mapping[ParameterRef, object]:
        return MappingProxyType(
            {ref: _public_value(value) for ref, value in self._values.items()}
        )

    @property
    def allow_extrapolation(self) -> tuple[ParameterRef, ...]:
        return self._allow_extrapolation

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ParameterSet is immutable")

    def _record(self) -> dict[str, object]:
        return {
            "type": "parameter_set_v2",
            "bindings": [
                {
                    "parameter": r._key_record(),
                    "value": v._record()
                    if isinstance(v, RLGC)
                    else quantity_record(v, r.spec.si_unit),
                }
                for r, v in self._values.items()
            ],
            "allow_extrapolation": [r._key_record() for r in self._allow_extrapolation],
        }

    _canonical_record = _record


class ParameterSpace:
    """Frozen author-ordered grid/list declaration; runtime owns execution."""

    __slots__ = (
        "_axes",
        "fixed",
        "_points",
        "_axis_source_units",
        "kind",
        "_frozen",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use ParameterSpace.grid() or .points()")

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("ParameterSpace is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def grid(
        cls,
        *,
        axes: Mapping[ParameterRef, object],
        fixed: ParameterSet = ParameterSet(),
    ) -> "ParameterSpace":
        if not isinstance(axes, Mapping) or not axes:
            raise ValueError("axes must be a nonempty mapping")
        if not isinstance(fixed, ParameterSet):
            raise TypeError("fixed must be ParameterSet")
        copied = []
        axis_source_units = []
        for ref, raw in axes.items():
            if not isinstance(ref, ParameterRef):
                raise TypeError("axis keys must be ParameterRef")
            if isinstance(raw, Quantity):
                magnitude = raw.magnitude
                if getattr(magnitude, "ndim", 0) != 1:
                    raise ValueError("Quantity axis must be one-dimensional")
                values = tuple(
                    _freeze_value(_value(raw[index], ref.spec, ref.id))
                    for index in range(len(magnitude))
                )
                unit_evidence = tuple(str(raw.units) for _ in values)
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                values = tuple(
                    _freeze_value(_value(item, ref.spec, ref.id)) for item in raw
                )
                unit_evidence = tuple(
                    str(item.units)
                    if isinstance(ref.spec, ParameterSpec)
                    and isinstance(item, Quantity)
                    else None
                    for item in raw
                )
            else:
                raise TypeError(
                    "axis values must be a one-dimensional Quantity or sequence"
                )
            if not values:
                raise ValueError("axes must be nonempty")
            copied.append((ref, values))
            axis_source_units.append(unit_evidence)
        if {ref for ref, _ in copied} & set(fixed.values):
            raise ValueError("fixed and axes must be disjoint")
        result = object.__new__(cls)
        object.__setattr__(result, "_axes", tuple(copied))
        object.__setattr__(
            result,
            "fixed",
            ParameterSet._copy(fixed),
        )
        object.__setattr__(result, "_axis_source_units", tuple(axis_source_units))
        object.__setattr__(result, "kind", "grid")
        object.__setattr__(result, "_frozen", True)
        return result

    @property
    def axes(self) -> tuple[tuple[ParameterRef, tuple[object, ...]], ...]:
        """Return detached axis quantities without exposing grid storage."""
        return tuple(
            (ref, tuple(_public_value(value) for value in values))
            for ref, values in self._axes
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ParameterSpace is immutable")

    @classmethod
    def points(cls, points: Sequence[ParameterSet]) -> "ParameterSpace":
        if (
            not isinstance(points, Sequence)
            or isinstance(points, (str, bytes))
            or not points
        ):
            raise ValueError("points must be nonempty ParameterSet sequence")
        if any(not isinstance(x, ParameterSet) for x in points):
            raise TypeError("points must contain ParameterSet")
        result = object.__new__(cls)
        object.__setattr__(
            result,
            "_points",
            tuple(ParameterSet._copy(x) for x in points),
        )
        object.__setattr__(result, "_axis_source_units", ())
        object.__setattr__(result, "kind", "points")
        object.__setattr__(result, "_frozen", True)
        return result
