"""Public data-ingestion declarations for circuit-model authoring.

This module converts supported external solver artifacts into SCNSim authoring
values.  It does not run AEDT, import PyAEDT, or make an external artifact a
second circuit authority.
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import math
from os import PathLike
from pathlib import Path
import re

import numpy as np

from .authoring import RLGC, _identifier
from .errors import SCNSimValidationError
from .units import registry


_DECIMAL = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_PRIMARY_TITLES = (
    ("Capacitance Matrix", "C", "pF/meter"),
    ("Conductance Matrix", "G", "mho/meter"),
    ("Inductance Matrix", "L", "nH/meter"),
    ("Resistance Matrix", "R", "ohm/meter"),
)
_HEADER_UNITS = "C Units:pF/meter, G Units:mho/meter, R Units:ohm/meter, L Units:nH/meter"
_SUPPLEMENTAL_TITLES = {
    "Capacitance Matrix Coupling Coefficient",
    "Spice Capacitance Matrix",
}


def load_q2d_rlgc(
    path: str | PathLike[str],
    *,
    reference_conductor: str,
    conductor_map: Mapping[str, str] | None = None,
) -> RLGC:
    """Load frozen N-conductor RLGC matrices from an AEDT Q2D raw CSV.

    The reader requires exactly one primary capacitance, conductance,
    inductance, and resistance block with identical native order.
    ``conductor_map`` must be a complete bijective rename and cannot reorder or
    discard mutual terms.  Positive head-to-tail current is the caller's
    asserted extractor +z convention; this strict file profile does not infer
    direction. Source labels, units, extraction frequency, conductor order,
    caller assertion, and content hash remain provenance; this helper does not
    run AEDT or prove solver convergence.
    """

    try:
        raw = Path(path).read_bytes()
    except (OSError, TypeError) as exc:
        raise SCNSimValidationError("Q2D CSV cannot be read", stage="q2d_load") from exc
    content_sha256 = sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SCNSimValidationError("Q2D CSV must be strict UTF-8", stage="q2d_load") from exc
    if text.startswith("\ufeff"):
        raise SCNSimValidationError("Q2D CSV must not include a UTF-8 byte-order mark", stage="q2d_load")
    if "\r" in text.replace("\r\n", ""):
        raise SCNSimValidationError("Q2D CSV must use LF or CRLF records", stage="q2d_load")
    records = text.splitlines()
    headers, frequency = _q2d_headers(records)
    native_labels, matrices = _q2d_primary_blocks(records[5:])
    public_labels, map_records = _conductor_map(native_labels, conductor_map)
    reference = _identifier(reference_conductor, field="reference_conductor")
    if reference in public_labels:
        raise SCNSimValidationError("reference_conductor cannot occur in the Q2D conductor map", stage="q2d_load")
    values = {
        "C": np.asarray(matrices["C"], dtype=np.float64) * registry.picofarad / registry.meter,
        "G": np.asarray(matrices["G"], dtype=np.float64) * registry.mho / registry.meter,
        "L": np.asarray(matrices["L"], dtype=np.float64) * registry.nanohenry / registry.meter,
        "R": np.asarray(matrices["R"], dtype=np.float64) * registry.ohm / registry.meter,
    }
    source = {
        "source_kind": "aedt_q2d_csv",
        "profile_id": "aedt_q2d_xy_cross_section_positive_z",
        "profile_boundary": "caller_asserted",
        "content_sha256": content_sha256,
        "problem_type": "CG, RL",
        "native_conductor_order": list(native_labels),
        "conductor_map": map_records,
        "header_records": headers,
        "source_unit_records": [
            {"matrix": "C", "source_unit": "pF/meter"},
            {"matrix": "G", "source_unit": "mho/meter"},
            {"matrix": "L", "source_unit": "nH/meter"},
            {"matrix": "R", "source_unit": "ohm/meter"},
            {"matrix": "frequency", "source_unit": "GHz"},
        ],
    }
    return RLGC._from_source(
        conductors=public_labels,
        reference_conductor=reference,
        resistance_per_length=values["R"],
        inductance_per_length=values["L"],
        conductance_per_length=values["G"],
        capacitance_per_length=values["C"],
        extraction_frequency=frequency * registry.gigahertz,
        source=source,
    )


def _q2d_headers(records: list[str]) -> tuple[list[str], float]:
    if len(records) < 5:
        raise SCNSimValidationError("Q2D CSV is missing its five required header records", stage="q2d_load")
    headers = records[:5]
    if (
        not re.fullmatch(r".+:LastAdaptive", headers[0])
        or headers[1] != "Problem Type:  CG, RL"
        or headers[2] != _HEADER_UNITS
        or headers[3] != "Reduce Matrix:  Original"
        or not headers[4].startswith("Frequency:  ")
        or not headers[4].endswith("GHz")
    ):
        raise SCNSimValidationError("Q2D CSV headers do not match AEDT 2024.2 CG, RL", stage="q2d_load")
    value = headers[4][len("Frequency:  "):-len("GHz")]
    frequency = _finite_decimal(value, field="Frequency")
    if frequency <= 0:
        raise SCNSimValidationError("Q2D extraction frequency must be positive", stage="q2d_load")
    return headers, frequency


def _q2d_primary_blocks(records: list[str]) -> tuple[tuple[str, ...], dict[str, tuple[tuple[float, ...], ...]]]:
    found: dict[str, tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]] = {}
    expected = 0
    index = 0
    titles = {title: (code, unit) for title, code, unit in _PRIMARY_TITLES}
    while index < len(records):
        if records[index] == "":
            index += 1
            continue
        title = records[index]
        primary = titles.get(title)
        if primary is None:
            if title not in _SUPPLEMENTAL_TITLES:
                raise SCNSimValidationError("Q2D CSV contains an unknown supplemental block", stage="q2d_load")
            index = _skip_supplemental_block(records, index)
            continue
        code, _ = primary
        if expected >= len(_PRIMARY_TITLES) or code != _PRIMARY_TITLES[expected][1] or code in found:
            raise SCNSimValidationError("Q2D primary C/G/L/R blocks must occur exactly once in order", stage="q2d_load")
        labels, matrix, index = _q2d_matrix_block(records, index + 1, code)
        found[code] = (labels, matrix)
        expected += 1
    if expected != len(_PRIMARY_TITLES):
        raise SCNSimValidationError("Q2D CSV is missing one or more primary C/G/L/R matrix blocks", stage="q2d_load")
    labels = found["C"][0]
    if any(found[code][0] != labels for _, code, _ in _PRIMARY_TITLES):
        raise SCNSimValidationError("Q2D primary matrix blocks must share one native conductor order", stage="q2d_load")
    return labels, {code: found[code][1] for _, code, _ in _PRIMARY_TITLES}


def _skip_supplemental_block(records: list[str], index: int) -> int:
    while index < len(records) and records[index] != "":
        index += 1
    return index


def _q2d_matrix_block(
    records: list[str], index: int, code: str
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...], int]:
    while index < len(records) and records[index] == "":
        index += 1
    if index >= len(records) or records[index] == "":
        raise SCNSimValidationError(f"Q2D {code} primary block lacks its CSV header", stage="q2d_load")
    header = _csv_fields(records[index], field=f"{code} matrix header")
    if not header or header[0] != "" or len(header) < 2:
        raise SCNSimValidationError(f"Q2D {code} matrix header is invalid", stage="q2d_load")
    labels = tuple(_identifier(label, field=f"{code} native conductor") for label in header[1:])
    if len(set(labels)) != len(labels):
        raise SCNSimValidationError(f"Q2D {code} matrix native labels must be unique", stage="q2d_load")
    rows: list[tuple[float, ...]] = []
    index += 1
    for row_index, label in enumerate(labels):
        if index >= len(records) or records[index] == "":
            raise SCNSimValidationError(f"Q2D {code} matrix has too few rows", stage="q2d_load")
        row = _csv_fields(records[index], field=f"{code} matrix row")
        if len(row) != len(labels) + 1 or row[0] != label:
            raise SCNSimValidationError(f"Q2D {code} matrix row order or width is invalid", stage="q2d_load")
        rows.append(tuple(_finite_decimal(value, field=f"{code}[{row_index}]") for value in row[1:]))
        index += 1
    if index < len(records) and records[index] != "" and records[index] not in {title for title, _, _ in _PRIMARY_TITLES}:
        raise SCNSimValidationError(f"Q2D {code} matrix has extra rows or records", stage="q2d_load")
    return labels, tuple(rows), index


def _csv_fields(record: str, *, field: str) -> list[str]:
    try:
        rows = list(csv.reader([record], strict=True))
    except csv.Error as exc:
        raise SCNSimValidationError(f"{field} is not valid CSV", stage="q2d_load") from exc
    if len(rows) != 1:
        raise SCNSimValidationError(f"{field} is not one CSV record", stage="q2d_load")
    return [item.strip() for item in rows[0]]


def _finite_decimal(value: str, *, field: str) -> float:
    if not _DECIMAL.fullmatch(value):
        raise SCNSimValidationError(f"{field} must be a finite decimal", stage="q2d_load")
    try:
        result = float(Decimal(value))
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise SCNSimValidationError(f"{field} must be a finite decimal", stage="q2d_load") from exc
    if not math.isfinite(result):
        raise SCNSimValidationError(f"{field} must be finite", stage="q2d_load")
    return result


def _conductor_map(
    native_labels: tuple[str, ...], conductor_map: Mapping[str, str] | None
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if conductor_map is None:
        mapping = {label: label for label in native_labels}
    elif isinstance(conductor_map, Mapping):
        mapping = dict(conductor_map)
    else:
        raise TypeError("conductor_map must be a complete mapping or None")
    if set(mapping) != set(native_labels):
        raise SCNSimValidationError("conductor_map must name every Q2D native conductor exactly once", stage="q2d_load")
    public = tuple(_identifier(mapping[label], field="public conductor") for label in native_labels)
    if len(set(public)) != len(public):
        raise SCNSimValidationError("conductor_map public labels must be bijective", stage="q2d_load")
    return public, [
        {"native_label": native, "public_label": public_label}
        for native, public_label in zip(native_labels, public)
    ]


__all__ = ["load_q2d_rlgc"]
