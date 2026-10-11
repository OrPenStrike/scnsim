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
) -> tuple[list[int], list[float | None], list[str], str, str, list[str]]:
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

    selector = (
        objective if kind in {"objective", "residual"}
        else None if kind != "parameter"
        else {"definitions_id": parameter.definitions_id, "parameter_id": parameter.id}
    )
    rows = result._fixed_reader.project(kind, selector)
    ordinals: list[int] = []
    values: list[float | None] = []
    statuses: list[str] = []
    details: list[str] = []
    found = False
    unit = "dimensionless"
    for row in rows:
        ordinal = row["evaluation_ordinal"]
        value: float | None = None
        status = str(row["status"])
        details.append("; ".join(
            f"{item.get('objective_id')}: {item.get('status')} "
            f"({len(item.get('terms', ())) } terms)"
            for item in row.get("components", ())
        ))
        if kind == "history":
            encoded = row.get("cost_f64")
            value = float64_from_hex(encoded) if isinstance(encoded, str) else None
        elif kind in {"objective", "residual"}:
            components = row["components"]
            component = next(
                (item for item in components if item["objective_id"] == objective),
                None,
            )
            if component is not None:
                found = True
                status = str(component["status"])
                field = "weighted_cost_f64" if kind == "objective" else "normalized_residual_f64"
                encoded = component.get(field)
                value = float64_from_hex(encoded) if isinstance(encoded, str) else None
        else:
            bindings = row["parameters"]["bindings"]
            key = {"definitions_id": parameter.definitions_id, "parameter_id": parameter.id}
            binding = next(
                (item for item in bindings if item["parameter"] == key),
                None,
            )
            if binding is not None:
                found = True
                envelope = binding["value"]
                quantity = Quantity(
                    float64_from_hex(envelope["si_value_f64"]), envelope["si_unit"]
                ).to(parameter.spec.si_unit)
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
    return ordinals, values, statuses, label, unit, details


__all__ = ["Channel", "_channel_index", "_frequency_index", "_optimization_series"]
