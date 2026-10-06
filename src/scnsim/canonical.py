"""Canonical bytes, hashes, binary64, quantities, and safe relative paths.

These shared representation primitives own no Plan, Run, workspace lifetime,
or presentation state. Domain document builders consume this one encoder."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import re
import struct
import unicodedata
from .errors import EvidenceIntegrityError, SCNSimValidationError


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


_IDENTIFIER = re.compile(r"^[^/\\\x00-\x1f\x7f]+$")


_UNITS: dict[str, str] = {
    "farad": "capacitance",
    "henry": "inductance",
    "ohm": "resistance",
    "siemens": "conductance",
    "hertz": "inverse_time",
    "radian / second": "inverse_time",
    "ampere": "current",
    "volt": "voltage",
    "meter": "length",
    "meter / second": "velocity",
    "weber": "magnetic_flux",
    "ohm / meter": "resistance_per_length",
    "henry / meter": "inductance_per_length",
    "siemens / meter": "conductance_per_length",
    "farad / meter": "capacitance_per_length",
    "siemens / second": "conductance_per_time",
    "dimensionless": "dimensionless",
}


def _validation(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="canonical_identity", evidence=evidence)


def _integrity(message: str, **evidence: object) -> EvidenceIntegrityError:
    return EvidenceIntegrityError(message, stage="canonical_identity", evidence=evidence)


def _nfc(value: str, *, field: str = "string") -> str:
    if not isinstance(value, str):
        raise _validation("canonical strings must be str", field=field)
    return unicodedata.normalize("NFC", value)


def _identifier(value: str, *, field: str = "identifier") -> str:
    normalized = _nfc(value, field=field)
    if not normalized or not _IDENTIFIER.fullmatch(normalized):
        raise _validation("invalid canonical identifier", field=field, value=value)
    return normalized


def _sha256(value: str, *, field: str = "sha256") -> str:
    normalized = _nfc(value, field=field)
    if not _SHA256.fullmatch(normalized):
        raise _validation("expected lowercase SHA-256", field=field, value=value)
    return normalized


def canonical_value(value: object) -> object:
    """Return a closed JSON value with NFC strings and no JSON floats.

    Physical floats must be represented by :func:`float64_hex`; accepting a
    JSON number here would make Python and Julia decimal printers part of the
    identity protocol.
    """

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _nfc(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise _validation("canonical JSON forbids floating JSON numbers")
    if isinstance(value, (bytes, bytearray, memoryview, Path, os.PathLike)):
        raise _validation("canonical JSON forbids binary values and filesystem paths")
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            normalized_key = _nfc(key, field="object key")
            if normalized_key in normalized:
                raise _validation("NFC-normalized object keys collide", key=normalized_key)
            normalized[normalized_key] = canonical_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    # NumPy integer scalars are accepted without making NumPy a required import.
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        scalar = value.item()  # type: ignore[union-attr]
        if isinstance(scalar, int):
            return scalar
        raise _validation("canonical JSON requires encoded finite Float64 values")
    raise _validation("canonical JSON received an unsupported value", type=type(value).__name__)


def canonical_json_bytes(value: object) -> bytes:
    """Encode one closed canonical UTF-8 JSON document."""

    return json.dumps(
        canonical_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_hex(value: object | bytes) -> str:
    """Hash canonical JSON or already-canonical raw bytes."""

    payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
    return sha256(payload).hexdigest()


def float64_hex(value: object) -> str:
    """Encode one finite real IEEE-754 binary64 scalar as big-endian hex."""

    if isinstance(value, bool):
        raise _validation("boolean is not a Float64")
    try:
        scalar = float(value)  # NumPy scalar support without coupling the API to NumPy.
    except (TypeError, ValueError) as error:
        raise _validation("expected a real Float64 scalar", value_type=type(value).__name__) from error
    if not math.isfinite(scalar):
        raise _validation("Float64 identity values must be finite")
    return struct.pack(">d", scalar).hex()


def float64_from_hex(value: str) -> float:
    """Decode an exact finite Float64 identity token."""

    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value):
        raise _integrity("invalid Float64 hex token", value=value)
    scalar = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(scalar):
        raise _integrity("nonfinite Float64 token", value=value)
    return scalar


def _quantity_type(value: object) -> bool:
    return value.__class__.__name__ == "Quantity" and hasattr(value, "to") and hasattr(value, "magnitude")


def quantity_envelope(
    value: object,
    *,
    si_unit: str,
    dimensionality: str | None = None,
    registry: object | None = None,
) -> dict[str, str]:
    """Encode a scalar Pint quantity in one of the closed V1 unit families."""

    unit = _nfc(si_unit, field="si_unit")
    expected_dimension = _UNITS.get(unit)
    if expected_dimension is None:
        raise _validation("unsupported canonical SI unit", si_unit=si_unit)
    dimension = expected_dimension if dimensionality is None else _nfc(dimensionality, field="dimensionality")
    if dimension != expected_dimension:
        raise _validation("SI unit and dimensionality do not match", si_unit=unit, dimensionality=dimension)
    if not _quantity_type(value):
        raise _validation("physical values must be Pint Quantity instances", value_type=type(value).__name__)
    value_registry = getattr(value, "_REGISTRY", None)
    if registry is not None and value_registry is not registry:
        raise _validation("quantity belongs to a foreign Pint registry")
    try:
        converted = value.to(unit)  # type: ignore[union-attr]
    except Exception as error:
        raise _validation("quantity has incompatible dimensionality", si_unit=unit) from error
    magnitude = _coherent_magnitude(value, converted, unit)
    if isinstance(magnitude, complex) or getattr(magnitude, "ndim", 0) != 0:
        raise _validation("quantity envelope requires one real scalar")
    return {
        "type": "quantity_f64",
        "si_value_f64": float64_hex(magnitude),
        "si_unit": unit,
        "dimensionality": dimension,
    }


def complex_quantity_envelope(
    value: object,
    *,
    si_unit: str,
    dimensionality: str | None = None,
    registry: object | None = None,
) -> dict[str, str]:
    """Encode one finite complex Pint quantity with explicit real/imag bits."""

    unit = _nfc(si_unit, field="si_unit")
    expected_dimension = _UNITS.get(unit)
    if expected_dimension is None:
        raise _validation("unsupported canonical SI unit", si_unit=si_unit)
    dimension = expected_dimension if dimensionality is None else _nfc(dimensionality, field="dimensionality")
    if dimension != expected_dimension or not _quantity_type(value):
        raise _validation("invalid complex quantity envelope")
    if registry is not None and getattr(value, "_REGISTRY", None) is not registry:
        raise _validation("quantity belongs to a foreign Pint registry")
    try:
        converted = value.to(unit)  # type: ignore[union-attr]
    except Exception as error:
        raise _validation("quantity has incompatible dimensionality", si_unit=unit) from error
    magnitude = _coherent_magnitude(value, converted, unit)
    if getattr(magnitude, "ndim", 0) != 0:
        raise _validation("complex quantity envelope requires one scalar")
    scalar = complex(magnitude)
    return {
        "type": "complex_quantity_f64",
        "real_si_f64": float64_hex(scalar.real),
        "imag_si_f64": float64_hex(scalar.imag),
        "si_unit": unit,
        "dimensionality": dimension,
    }


def _coherent_magnitude(value: object, converted: object, si_unit: str) -> object:
    """Convert multiplicative Pint units through decimal scale spelling.

    Pint correctly checks dimensions, but converting two common metric prefixes
    through binary floats can leave adjacent representable values.  Semantic
    identity needs the coherent SI *value*, so normal scalar source spelling is
    multiplied by the registry's multiplicative source/target factors before
    its one final binary64 rounding.
    """

    magnitude = getattr(converted, "magnitude")
    if getattr(magnitude, "ndim", 0) != 0:
        return magnitude
    try:
        source_factor = value._REGISTRY.get_base_units(value._units)[0]  # type: ignore[union-attr]
        target_factor = 1 if si_unit == "dimensionless" else value._REGISTRY.get_base_units(si_unit)[0]  # type: ignore[union-attr]
        factor = Decimal(str(source_factor)) / Decimal(str(target_factor))
        source = value.magnitude  # type: ignore[union-attr]
        scalar = complex(source)
        if (
            isinstance(source, complex)
            or getattr(getattr(source, "dtype", None), "kind", None) == "c"
            or scalar.imag != 0.0
        ):
            return complex(
                float(Decimal(str(scalar.real)) * factor),
                float(Decimal(str(scalar.imag)) * factor),
            )
        return float(Decimal(str(source)) * factor)
    except (AttributeError, InvalidOperation, ValueError, TypeError, ZeroDivisionError) as error:
        raise _validation(
            "quantity cannot be converted to coherent SI without a binary64 fallback",
            source_magnitude=str(getattr(value, "magnitude", "<unavailable>")),
            source_unit=str(getattr(value, "units", "<unavailable>")),
            target_unit=si_unit,
        ) from error


def quantity_from_envelope(value: Mapping[str, object], *, registry: object) -> object:
    """Reconstruct a scalar Pint quantity after closed-envelope validation."""

    required = {"type", "si_value_f64", "si_unit", "dimensionality"}
    if set(value) != required or value.get("type") != "quantity_f64":
        raise _integrity("invalid quantity envelope")
    unit = value["si_unit"]
    dimension = value["dimensionality"]
    if not isinstance(unit, str) or _UNITS.get(unit) != dimension:
        raise _integrity("quantity unit/dimensionality mismatch")
    magnitude = float64_from_hex(value["si_value_f64"] if isinstance(value["si_value_f64"], str) else "")
    try:
        return registry.Quantity(magnitude, unit)  # type: ignore[union-attr]
    except Exception as error:
        raise _integrity("unable to reconstruct Pint quantity", si_unit=unit) from error


def complex_quantity_from_envelope(value: Mapping[str, object], *, registry: object) -> object:
    """Reconstruct a complex scalar Pint quantity after closed validation."""

    required = {"type", "real_si_f64", "imag_si_f64", "si_unit", "dimensionality"}
    if set(value) != required or value.get("type") != "complex_quantity_f64":
        raise _integrity("invalid complex quantity envelope")
    unit = value.get("si_unit")
    dimension = value.get("dimensionality")
    if not isinstance(unit, str) or _UNITS.get(unit) != dimension:
        raise _integrity("complex quantity unit/dimensionality mismatch")
    real = float64_from_hex(value["real_si_f64"] if isinstance(value["real_si_f64"], str) else "")
    imaginary = float64_from_hex(value["imag_si_f64"] if isinstance(value["imag_si_f64"], str) else "")
    try:
        return registry.Quantity(complex(real, imaginary), unit)  # type: ignore[union-attr]
    except Exception as error:
        raise _integrity("unable to reconstruct complex Pint quantity", si_unit=unit) from error


def relative_path(value: str) -> str:
    """Validate the single portable relative-path spelling accepted by V1."""

    normalized = _nfc(value, field="relative_path")
    if (
        not normalized
        or normalized.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", normalized)
        or "\\" in normalized
        or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
    ):
        raise _validation("invalid relative artifact path", path=value)
    parts = normalized.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise _validation("relative artifact path escapes its root", path=value)
    return normalized


def safe_join(root: Path, value: str) -> Path:
    """Join a validated relative path without accepting a symlink escape."""

    relative = relative_path(value)
    root_resolved = root.resolve(strict=True)
    target = root_resolved.joinpath(*relative.split("/"))
    try:
        target.resolve(strict=False).relative_to(root_resolved)
    except ValueError as error:
        raise _integrity("artifact path escapes workspace root", path=relative) from error
    return target


def _nonempty(value: object, field: str) -> str:
    normalized = _nfc(value, field=field) if isinstance(value, str) else ""
    if not normalized:
        raise _validation("expected nonempty string", field=field)
    return normalized
