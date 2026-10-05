"""Pure selectors over already-materialized numerical Results.

This module owns exact channel, frequency, and optimization-history lookup.
It has no plotting dependency so specifications and presentation can share
the same selection behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from pint import Quantity

Channel = str | tuple[str, tuple[int, ...]]


def _channel_index(
    channels: Sequence[tuple[str, tuple[int, ...]]],
    selector: Channel | None,
    *,
    role: str,
    default: bool,
) -> int:
    if not channels:
        raise ValueError(f"matrix has no {role} channels")
    if selector is None:
        if default:
            return 0
        raise ValueError(f"{role}_channel is required")
    if isinstance(selector, str):
        matches = [index for index, (coordinate, _) in enumerate(channels) if coordinate == selector]
        if not matches:
            raise ValueError(f"unknown {role} channel coordinate: {selector}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous {role} channel coordinate; use (coordinate, mode)")
        return matches[0]
    if (
        not isinstance(selector, tuple)
        or len(selector) != 2
        or not isinstance(selector[0], str)
        or not isinstance(selector[1], tuple)
        or any(not isinstance(value, int) or isinstance(value, bool) for value in selector[1])
    ):
        raise TypeError(f"{role}_channel must be a coordinate string or (coordinate, mode) tuple")
    try:
        return channels.index(selector)
    except ValueError as error:
        raise ValueError(f"unknown {role} channel: {selector!r}") from error


def _frequency_index(view: Any, frequency: Quantity | None) -> int:
    if not isinstance(frequency, Quantity) or np.asarray(frequency.magnitude).ndim != 0:
        raise TypeError("frequency must be one scalar Quantity")
    try:
        wanted = float(frequency.to(view.frequencies.units).magnitude)
    except Exception as error:
        raise ValueError("frequency is incompatible with the stored grid") from error
    values = np.asarray(view.frequencies.magnitude)
    matches = np.flatnonzero(values == wanted)
    if matches.size != 1:
        raise KeyError("frequency was not materialized exactly once")
    return int(matches[0])


def _optimization_series(
    result: Any,
    *,
    kind: Literal["history", "objective", "residual", "parameter"],
    objective: str | None,
    parameter: Any | None,
) -> tuple[list[int], list[float | None], list[str], str, str]:
    from ..canonical import float64_from_hex
    from ..authoring import ParameterRef

    if kind in {"objective", "residual"}:
        if not isinstance(objective, str) or not objective:
            raise ValueError(f"kind={kind!r} requires objective=<objective ID>")
        if parameter is not None:
            raise ValueError("parameter is valid only for kind='parameter'")
    elif kind == "parameter":
        if not isinstance(parameter, ParameterRef):
            raise TypeError("kind='parameter' requires parameter=ParameterRef")
        if objective is not None:
            raise ValueError("objective is valid only for objective or residual history")
        retained = next(
            (
                item
                for item in result.best.parameters.values
                if item.definitions_id == parameter.definitions_id
                and item.id == parameter.id
            ),
            None,
        )
        if retained is None:
            raise ValueError(
                "selected parameter history is absent from the Optimization Result: "
                f"{parameter.definitions_id}.{parameter.id}"
            )
        if retained._definition_record() != parameter._definition_record():
            raise ValueError(
                "selected parameter definition disagrees with the Optimization Result"
            )
        parameter = retained
    elif objective is not None or parameter is not None:
        raise ValueError("objective and parameter are invalid for total-cost history")

    ordinals: list[int] = []
    values: list[float | None] = []
    statuses: list[str] = []
    found = False
    unit = "dimensionless"
    for ledger in result.ledger:
        candidates = ledger.get("candidates") if isinstance(ledger, Mapping) else None
        if not isinstance(candidates, Sequence):
            raise ValueError("Optimization ledger candidates are malformed")
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not isinstance(candidate.get("outcome"), Mapping):
                raise ValueError("Optimization candidate evidence is malformed")
            outcome = candidate["outcome"]
            ordinal = candidate.get("evaluation_ordinal")
            if not isinstance(ordinal, int):
                raise ValueError("Optimization candidate ordinal is malformed")
            value: float | None = None
            status = str(outcome.get("status", ""))
            if kind == "history":
                encoded = outcome.get("cost_f64")
                value = float64_from_hex(encoded) if isinstance(encoded, str) else None
            elif kind in {"objective", "residual"}:
                components = outcome.get("objective_components")
                if not isinstance(components, Sequence):
                    raise ValueError("Optimization objective evidence is malformed")
                component = next(
                    (
                        item for item in components
                        if isinstance(item, Mapping) and item.get("objective_id") == objective
                    ),
                    None,
                )
                if component is not None:
                    found = True
                    status = str(component.get("status", status))
                    field = "weighted_cost_f64" if kind == "objective" else "normalized_residual_f64"
                    encoded = component.get(field)
                    value = float64_from_hex(encoded) if isinstance(encoded, str) else None
            else:
                bindings = candidate.get("parameters", {}).get("bindings") if isinstance(candidate.get("parameters"), Mapping) else None
                if not isinstance(bindings, Sequence):
                    raise ValueError("Optimization candidate parameters are malformed")
                key = {"definitions_id": parameter.definitions_id, "parameter_id": parameter.id}
                binding = next(
                    (item for item in bindings if isinstance(item, Mapping) and item.get("parameter") == key),
                    None,
                )
                if binding is not None and isinstance(binding.get("value"), Mapping):
                    found = True
                    envelope = binding["value"]
                    encoded = envelope.get("si_value_f64")
                    stored_unit = envelope.get("si_unit")
                    if isinstance(encoded, str) and isinstance(stored_unit, str):
                        quantity = Quantity(float64_from_hex(encoded), stored_unit).to(
                            parameter.spec.si_unit
                        )
                        value = float(quantity.magnitude)
                        unit = str(quantity.units)
            ordinals.append(ordinal)
            values.append(value)
            statuses.append(status)
    if kind in {"objective", "residual", "parameter"} and not found:
        selected = objective if parameter is None else f"{parameter.definitions_id}.{parameter.id}"
        raise ValueError(f"selected {kind} history is absent from the completed ledger: {selected}")
    if kind == "history":
        label = "total cost"
    elif kind == "objective":
        label = f"{objective} weighted cost"
    elif kind == "residual":
        label = f"{objective} normalized residual"
    else:
        label = f"{parameter.definitions_id}.{parameter.id}"
    return ordinals, values, statuses, label, unit


__all__ = ["Channel", "_channel_index", "_frequency_index", "_optimization_series"]
