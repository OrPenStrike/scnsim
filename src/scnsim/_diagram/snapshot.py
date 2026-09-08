"""Immutable, non-sealing Plan capture shared by diagram projections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .. import units
from .._canonical import (
    canonical_json_bytes,
    canonical_plan_document,
    complex_quantity_envelope,
    quantity_envelope,
    sha256_hex,
)
from ..authoring import CircuitPlan, ParameterSet


def _freeze(value: object) -> object:
    """Recursively make captured JSON-shaped evidence read-only."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _source_identity(
    *, scope: str, component_path: Sequence[str] = (), parameter_id: str, field: str
) -> str:
    return canonical_json_bytes(
        {
            "scope": scope,
            "component_path": list(component_path),
            "parameter_id": parameter_id,
            "field": field,
        }
    ).decode("utf-8")


def _source_unit_record(identity: str, value: object, si_unit: str) -> dict[str, object]:
    magnitude = np.asarray(getattr(value, "magnitude"))
    probe = value if magnitude.ndim == 0 else units.registry.Quantity(float(magnitude.flat[0]), value.units)
    source_magnitude = getattr(probe, "magnitude", None)
    encoded = (
        complex_quantity_envelope(probe, si_unit=si_unit, registry=units.registry)
        if isinstance(source_magnitude, complex) or getattr(getattr(source_magnitude, "dtype", None), "kind", None) == "c"
        else quantity_envelope(probe, si_unit=si_unit, registry=units.registry)
    )
    return {
        "identity": identity,
        "source_unit": str(value.units),
        "canonical_si_unit": encoded["si_unit"],
        "canonical_dimensionality": encoded["dimensionality"],
    }


def _capture_provenance(plan: CircuitPlan) -> Mapping[str, object]:
    """Capture source spelling and authoring ground calls outside Plan identity."""

    source_units: list[dict[str, object]] = []
    ground_call_groups: list[dict[str, object]] = []

    def add_component(component: object, path: tuple[str, ...]) -> None:
        for parameter in component._parameters.values():
            source_units.append(
                _source_unit_record(
                    _source_identity(
                        scope="plan_parameter",
                        component_path=path,
                        parameter_id=parameter.id,
                        field="baseline",
                    ),
                    parameter.baseline,
                    parameter.unit,
                )
            )
        if component._rlgc_source is not None:
            units_by_field = {
                "resistance_per_length": "ohm / meter",
                "inductance_per_length": "henry / meter",
                "conductance_per_length": "siemens / meter",
                "capacitance_per_length": "farad / meter",
                "extraction_frequency": "hertz",
            }
            for field, value in component._rlgc_source.items():
                source_units.append(
                    _source_unit_record(
                        _source_identity(
                            scope="plan_rlgc", component_path=path,
                            parameter_id="rlgc", field=field,
                        ),
                        value, units_by_field[field],
                    )
                )
        realization = component._realization
        for parameter_id, source in component._affine_sources.items():
            binding = realization["bindings"][parameter_id]
            support = source["support"]
            values = (
                ("slope", source["slope"], binding["slope"]["si_unit"]),
                ("intercept", source["intercept"], binding["intercept"]["si_unit"]),
                ("support_lower", support[0], binding["support"][0]["si_unit"]),
                ("support_upper", support[1], binding["support"][1]["si_unit"]),
            )
            for field, value, si_unit in values:
                source_units.append(
                    _source_unit_record(
                        _source_identity(
                            scope="plan_affine", component_path=path,
                            parameter_id=parameter_id, field=field,
                        ),
                        value, si_unit,
                    )
                )
        for group in component._ground_groups:
            ground_call_groups.append(
                {
                    "component_path": list(path),
                    "endpoints": [
                        {
                            "component_path": [*path, *endpoint["component_path"]],
                            "pin_id": endpoint["pin_id"],
                        }
                        for endpoint in group
                    ],
                }
            )
        for child in realization.get("children", ()):
            add_component(child, (*path, child.id))

    for group in plan._ground_groups:
        ground_call_groups.append(
            {
                "component_path": [],
                "endpoints": [pin._endpoint() for pin in group],
            }
        )
    for component in plan.components:
        add_component(component, (component.id,))
    for port in plan.ports:
        source_units.append(
            _source_unit_record(
                _source_identity(
                    scope="plan_port", parameter_id=port.id,
                    field="reference_impedance",
                ),
                port.reference_impedance, "ohm",
            )
        )
    source_units.sort(key=lambda item: str(item["identity"]))
    return MappingProxyType(
        {
            "source_units": tuple(_freeze(item) for item in source_units),
            "ground_call_groups": tuple(_freeze(item) for item in ground_call_groups),
        }
    )


@dataclass(frozen=True, slots=True)
class _CapturedPlan:
    """One immutable baseline declaration for a diagram pipeline."""

    document: Mapping[str, object]
    canonical_bytes: bytes
    plan_sha256: str
    provenance: Mapping[str, object]
    baseline_parameters: Mapping[str, object]


def capture_plan(plan: CircuitPlan) -> _CapturedPlan:
    """Capture a complete Plan identity without sealing or changing ``plan``."""

    if not isinstance(plan, CircuitPlan):
        raise TypeError("capture_plan() requires CircuitPlan")
    document = canonical_plan_document(plan._captured_snapshot())
    canonical_bytes = canonical_json_bytes(document)
    baseline_parameters = ParameterSet(
        {
            parameter: parameter.baseline
            for component in plan.components
            for parameter in component._parameters.values()
        }
    )._canonical_record()
    return _CapturedPlan(
        document=_freeze(document),  # type: ignore[arg-type]
        canonical_bytes=canonical_bytes,
        plan_sha256=sha256_hex(canonical_bytes),
        provenance=_capture_provenance(plan),
        baseline_parameters=_freeze(baseline_parameters),  # type: ignore[arg-type]
    )
