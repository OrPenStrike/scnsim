"""Convert sealed compiler grid records into immutable result values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, cast

from pint import Quantity

from .. import units
from ..canonical import quantity_from_envelope
from ..errors import EvidenceIntegrityError
from .base import LineDiscretization


def _decode_discretization(value: object) -> tuple[LineDiscretization, ...] | None:
    if value is None:
        return None  # Historical results predate line-grid evidence.
    # Receipt JSON decodes as a list; ExplanationResult freezes the same
    # compiler evidence into a tuple before exposing it to callers.
    if not isinstance(value, (list, tuple)):
        raise EvidenceIntegrityError("line discretization is malformed", stage="result_decode")
    rows = []

    def decoded_quantity(item: object) -> Quantity:
        return cast(Quantity, quantity_from_envelope(
            cast(Mapping[str, object], item), registry=units.registry,
        ))

    for record in value:
        if not isinstance(record, Mapping):
            raise EvidenceIntegrityError("line discretization entry is malformed", stage="result_decode")
        rows.append(LineDiscretization(
            component_path=tuple(cast(Sequence[str], record["component_path"])),
            kind=cast(Literal["fixed_count", "electrical_resolution"], record["kind"]),
            length=decoded_quantity(record["length"]),
            n_sections=cast(int, record["n_sections"]),
            dx=decoded_quantity(record["dx"]),
            modal_velocities=tuple(decoded_quantity(item) for item in cast(Sequence[object], record.get("modal_velocities", ()))),
            hmax=None if "hmax" not in record else decoded_quantity(record["hmax"]),
            policy=cast(Mapping[str, object] | None, record.get("policy")),
        ))
    return tuple(rows)


__all__ = ["_decode_discretization"]
