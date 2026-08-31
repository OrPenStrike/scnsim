"""Closed canonical bytes and evidence helpers for the executable dev3 slice.

This module is deliberately not a generic schema engine.  The shipped JSON
Schema is the field authority; these helpers own the bytes that Python writes
and the narrow primitive Direct envelopes that Python can read in dev3.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import base64
import csv
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from io import StringIO
from itertools import product
from pathlib import Path
import json
import math
import os
import re
import struct
import subprocess
import unicodedata

from .errors import EvidenceIntegrityError, SCNSimValidationError
from .errors import ScaffoldUnavailableError


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_IDENTIFIER = re.compile(r"^[^/\\\x00-\x1f\x7f]+$")
_CHUNK = re.compile(r"^[0-9]+(?:\.[0-9]+){0,4}$")

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
    "weber": "magnetic_flux",
    "ohm / meter": "resistance_per_length",
    "henry / meter": "inductance_per_length",
    "siemens / meter": "conductance_per_length",
    "farad / meter": "capacitance_per_length",
    "siemens / second": "conductance_per_time",
    "dimensionless": "dimensionless",
}

_DEV3_ALGORITHMS = {
    "solve_direct": "scnsim.direct_response.v1",
    "evaluate_direct": "scnsim.diagonal_root.newton32.v1",
    "optimize_direct": "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v2",
}
_DEV3_RESULTS = {"direct_response", "diagonal_root", "optimization"}


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
    if getattr(magnitude, "ndim", 0) != 0 or isinstance(magnitude, complex):
        return magnitude
    try:
        source_factor = value._REGISTRY.get_base_units(value._units)[0]  # type: ignore[union-attr]
        target_factor = 1 if si_unit == "dimensionless" else value._REGISTRY.get_base_units(si_unit)[0]  # type: ignore[union-attr]
        return float(Decimal(str(value.magnitude)) * Decimal(str(source_factor)) / Decimal(str(target_factor)))  # type: ignore[union-attr]
    except (AttributeError, InvalidOperation, ValueError, TypeError, ZeroDivisionError):
        return magnitude


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


def _endpoint(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != {"component_path", "pin_id"}:
        raise _validation("endpoint has unsupported fields")
    raw_path = value.get("component_path")
    if not isinstance(raw_path, Sequence) or isinstance(raw_path, (str, bytes)) or not raw_path:
        raise _validation("endpoint component_path must be nonempty segments")
    path = [_identifier(segment, field="component_path") for segment in raw_path]
    return {"component_path": path, "pin_id": _identifier(value.get("pin_id"), field="pin_id")}


def canonical_endpoints(endpoints: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Canonical endpoint-set order shared by Plan nodes and internal IDs."""

    encoded = [_endpoint(endpoint) for endpoint in endpoints]
    if not encoded:
        raise _validation("a node needs at least one endpoint")
    encoded.sort(key=lambda item: (tuple(item["component_path"]), item["pin_id"]))
    if len({(tuple(item["component_path"]), item["pin_id"]) for item in encoded}) != len(encoded):
        raise _validation("node endpoints must be unique")
    return encoded


def internal_node_id(endpoints: Iterable[Mapping[str, object]]) -> str:
    """Return the opaque endpoint-derived anonymous-node identity."""

    payload = {
        "schema": "scnsim.internal_node",
        "schema_version": 1,
        "endpoints": canonical_endpoints(endpoints),
    }
    return f"internal-{sha256_hex(payload)}"


def _component_key(component: Mapping[str, object]) -> tuple[str, ...]:
    path = component.get("component_path")
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
        raise _validation("component snapshot requires component_path")
    return tuple(_identifier(item, field="component_path") for item in path)


def _sort_id_maps(values: Iterable[Mapping[str, object]], *, key: str = "id") -> list[dict[str, object]]:
    copied = [dict(value) for value in values]
    copied.sort(key=lambda value: _identifier(value.get(key), field=key))
    if len({_identifier(value.get(key), field=key) for value in copied}) != len(copied):
        raise _validation("canonical identifiers must be unique", field=key)
    return copied


def canonical_plan_document(snapshot: Mapping[str, object]) -> dict[str, object]:
    """Close the primitive/full-V1 Plan ordering before computing its hash.

    The dev3 compiler accepts primitive realizations only; this encoder retains
    full schema ordering so a later slice does not alter Plan identity bytes.
    """

    document = dict(snapshot)
    document["schema"] = "scnsim.plan"
    document["schema_version"] = 1
    document["ground_id"] = "ground"
    required = {
        "schema", "schema_version", "plan_id", "ground_id", "catalog_sources",
        "components", "nodes", "grounded_endpoints", "ports", "couplings",
    }
    if set(document) != required:
        raise _validation("plan snapshot fields do not match identity-v1", fields=sorted(document))
    document["plan_id"] = _identifier(document["plan_id"], field="plan_id")
    catalogs = [dict(item) for item in _iter_mappings(document["catalog_sources"], "catalog_sources")]
    catalogs.sort(key=lambda item: _nfc(_as_str(item.get("catalog_id"), "catalog_id")))
    if len({_nfc(_as_str(item.get("catalog_id"), "catalog_id")) for item in catalogs}) != len(catalogs):
        raise _validation("catalog_id values must be unique")
    document["catalog_sources"] = catalogs
    components = [_canonical_component(item) for item in _iter_mappings(document["components"], "components")]
    components.sort(key=_component_key)
    if len({_component_key(item) for item in components}) != len(components):
        raise _validation("component paths must be unique")
    document["components"] = components
    nodes = [_canonical_node(item) for item in _iter_mappings(document["nodes"], "nodes")]
    nodes.sort(key=lambda item: _nfc(_as_str(item.get("node_id"), "node_id")))
    if len({_nfc(_as_str(item.get("node_id"), "node_id")) for item in nodes}) != len(nodes):
        raise _validation("node IDs must be unique")
    document["nodes"] = nodes
    document["grounded_endpoints"] = canonical_endpoints(_iter_mappings(document["grounded_endpoints"], "grounded_endpoints")) if document["grounded_endpoints"] else []
    ports = [dict(item) for item in _iter_mappings(document["ports"], "ports")]
    # Port declaration order is physical and intentionally retained.
    if len({_identifier(item.get("port_id"), field="port_id") for item in ports}) != len(ports):
        raise _validation("port IDs must be unique")
    document["ports"] = ports
    document["couplings"] = _sort_id_maps(_iter_mappings(document["couplings"], "couplings"))
    return canonical_value(document)  # type: ignore[return-value]


def _canonical_component(component: Mapping[str, object]) -> dict[str, object]:
    result = dict(component)
    path = _component_key(result)
    result["component_path"] = list(path)
    result["parameter_bindings"] = _sort_id_maps(_iter_mappings(result.get("parameter_bindings"), "parameter_bindings"))
    result["inductive_branches"] = _sort_id_maps(_iter_mappings(result.get("inductive_branches"), "inductive_branches"))
    realization = result.get("realization")
    if isinstance(realization, Mapping) and realization.get("kind") == "composite":
        nested = dict(realization)
        nested["children"] = [_canonical_component(item) for item in _iter_mappings(nested.get("children"), "children")]
        nested["children"].sort(key=_component_key)
        if len({_component_key(item) for item in nested["children"]}) != len(nested["children"]):
            raise _validation("nested component paths must be unique")
        nested["private_nodes"] = [
            {**dict(node), "endpoints": canonical_endpoints(_iter_mappings(dict(node).get("endpoints"), "private node endpoints"))}
            for node in _iter_mappings(nested.get("private_nodes"), "private_nodes")
        ]
        nested["private_nodes"].sort(key=lambda node: _identifier(node.get("id"), field="private_node.id"))
        if len({_identifier(node.get("id"), field="private_node.id") for node in nested["private_nodes"]}) != len(nested["private_nodes"]):
            raise _validation("private node IDs must be unique")
        nested["grounded_endpoints"] = canonical_endpoints(_iter_mappings(nested.get("grounded_endpoints"), "grounded_endpoints")) if nested.get("grounded_endpoints") else []
        nested["couplings"] = _sort_id_maps(_iter_mappings(nested.get("couplings"), "couplings"))
        nested["public_pin_map"] = _sort_id_maps(
            _iter_mappings(nested.get("public_pin_map"), "public_pin_map"), key="public_id"
        )
        nested["public_coordinate_map"] = _sort_id_maps(
            _iter_mappings(nested.get("public_coordinate_map"), "public_coordinate_map"), key="public_id"
        )
        nested["public_inductive_branch_map"] = _sort_id_maps(
            _iter_mappings(nested.get("public_inductive_branch_map"), "public_inductive_branch_map"), key="public_id"
        )
        parameter_maps = [
            dict(item) for item in _iter_mappings(nested.get("public_parameter_maps"), "public_parameter_maps")
        ]
        for parameter_map in parameter_maps:
            consumers = [
                dict(item) for item in _iter_mappings(parameter_map.get("consumers"), "parameter consumers")
            ]
            consumers.sort(key=lambda item: _parameter_ref_key(item.get("target")))
            if len({_parameter_ref_key(item.get("target")) for item in consumers}) != len(consumers):
                raise _validation("parameter consumer targets must be unique")
            parameter_map["consumers"] = consumers
        parameter_maps.sort(key=lambda item: _parameter_ref_key(item.get("parameter")))
        if len({_parameter_ref_key(item.get("parameter")) for item in parameter_maps}) != len(parameter_maps):
            raise _validation("public parameter maps must be unique")
        nested["public_parameter_maps"] = parameter_maps
        result["realization"] = nested
    return result


def _canonical_node(node: Mapping[str, object]) -> dict[str, object]:
    result = dict(node)
    result["node_id"] = _identifier(result.get("node_id"), field="node_id")
    result["endpoints"] = canonical_endpoints(_iter_mappings(result.get("endpoints"), "node endpoints"))
    return result


def _as_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise _validation("expected string", field=field)
    return value


def _iter_mappings(value: object, field: str) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _validation("expected array", field=field)
    output: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise _validation("array item must be an object", field=field)
        output.append(item)
    return output


def canonical_parameter_set(parameters: Mapping[str, object]) -> dict[str, object]:
    """Sort fully resolved ParameterSet bindings by `(component_path, parameter_id)`."""

    document = dict(parameters)
    if set(document) != {"type", "bindings", "allow_extrapolation"} or document.get("type") != "parameter_set":
        raise _validation("invalid parameter-set envelope")
    bindings = _iter_mappings(document["bindings"], "parameter bindings")
    bindings.sort(key=lambda item: _parameter_ref_key(_mapping(item, "parameter binding").get("parameter")))
    if len({_parameter_ref_key(item.get("parameter")) for item in bindings}) != len(bindings):
        raise _validation("parameter bindings must be unique")
    document["bindings"] = [dict(item) for item in bindings]
    authorizations = _iter_mappings(document["allow_extrapolation"], "allow_extrapolation")
    authorizations.sort(key=lambda item: _parameter_ref_key(item))
    if len({_parameter_ref_key(item) for item in authorizations}) != len(authorizations):
        raise _validation("allow_extrapolation values must be unique")
    document["allow_extrapolation"] = [dict(item) for item in authorizations]
    return canonical_value(document)  # type: ignore[return-value]


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _validation("expected object", field=field)
    return value


def _parameter_ref_key(value: object) -> tuple[tuple[str, ...], str]:
    ref = _mapping(value, "parameter_ref")
    if set(ref) != {"component_path", "parameter_id"}:
        raise _validation("parameter ref fields are invalid")
    path = ref["component_path"]
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)) or not path:
        raise _validation("parameter ref component path is invalid")
    return tuple(_identifier(item, field="component_path") for item in path), _identifier(ref["parameter_id"], field="parameter_id")


def canonical_request_document(
    *,
    plan_sha256: str,
    operation: str,
    ref_lineage: Mapping[str, object],
    spec: Mapping[str, object],
    parameters: Mapping[str, object],
    runtime_semantic: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact closed dev3 request envelope."""

    selected_operation = _nfc(operation, field="operation")
    expected_algorithm = _DEV3_ALGORITHMS.get(selected_operation)
    if expected_algorithm is None:
        raise _validation("operation is outside the dev3 runtime", operation=selected_operation)
    runtime = dict(runtime_semantic)
    if runtime.get("algorithm_id") != expected_algorithm:
        raise _validation("request algorithm does not match operation", operation=selected_operation)
    spec_type = spec.get("type")
    if selected_operation == "solve_direct" and spec_type != "direct_solve":
        raise _validation("solve_direct requires direct_solve spec")
    if selected_operation == "evaluate_direct" and spec_type != "diagonal_root":
        raise _validation("dev3 evaluate_direct requires diagonal_root spec")
    if selected_operation == "optimize_direct" and spec_type != "optimization":
        raise _validation("optimize_direct requires optimization spec")
    return canonical_value({
        "schema": "scnsim.request",
        "schema_version": 1,
        "plan_sha256": _sha256(plan_sha256, field="plan_sha256"),
        "operation": selected_operation,
        "ref_lineage": dict(ref_lineage),
        "spec": dict(spec),
        "parameters": canonical_parameter_set(parameters),
        "runtime_semantic": runtime,
    })  # type: ignore[return-value]


def attempt_ordinal_text(ordinal: int) -> str:
    """Format a positive attempt ordinal with its contractually minimum width."""

    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise _validation("attempt ordinal must be a positive integer")
    return str(ordinal).zfill(6)


def attempt_paths(request_sha256: str, ordinal: int, staging_uuid: str) -> tuple[str, str, str]:
    """Return `(ordinal_text, final_directory, staging_directory)` for one attempt."""

    request = _sha256(request_sha256, field="request_sha256")
    nonce = _nfc(staging_uuid, field="staging_uuid")
    if not _UUID4.fullmatch(nonce):
        raise _validation("staging nonce must be lowercase UUIDv4")
    text = attempt_ordinal_text(ordinal)
    prefix = f"requests/{request}/attempts"
    return text, f"{prefix}/{text}", f"{prefix}/.staging-{text}-{nonce}"


def canonical_attempt_document(
    *,
    request_sha256: str,
    ordinal: int,
    staging_uuid: str,
    started_at_utc: str,
    julia_executable_sha256: str,
    os_name: str,
    architecture: str,
    cpu: str,
    attempt_state: str = "allocated",
    julia_threads: int | None = None,
    blas_threads: int | None = None,
    blas_vendor: str | None = None,
    resume_ledger_sha256: str | None = None,
) -> dict[str, object]:
    """Build an allocated or launched attempt envelope with exact paths."""

    state = _nfc(attempt_state, field="attempt_state")
    if state not in {"allocated", "launched"}:
        raise _validation("invalid attempt state")
    if state == "launched" and (julia_threads is None or blas_threads is None or blas_vendor is None):
        raise _validation("launched attempt requires Julia and BLAS evidence")
    if state == "allocated" and any(value is not None for value in (julia_threads, blas_threads, blas_vendor)):
        raise _validation("allocated attempt cannot include launch-only evidence")
    text, directory, staging = attempt_paths(request_sha256, ordinal, staging_uuid)
    document: dict[str, object] = {
        "schema": "scnsim.attempt",
        "schema_version": 1,
        "request_sha256": _sha256(request_sha256, field="request_sha256"),
        "ordinal": ordinal,
        "ordinal_text": text,
        "directory": directory,
        "staging_directory": staging,
        "attempt_state": state,
        "started_at_utc": _utc(started_at_utc),
        "julia_executable_sha256": _sha256(julia_executable_sha256, field="julia_executable_sha256"),
        "os": _nonempty(os_name, "os"),
        "architecture": _nonempty(architecture, "architecture"),
        "cpu": _nonempty(cpu, "cpu"),
    }
    if state == "launched":
        if not isinstance(julia_threads, int) or julia_threads < 1 or not isinstance(blas_threads, int) or blas_threads < 1:
            raise _validation("thread counts must be positive")
        document.update({"julia_threads": julia_threads, "blas_threads": blas_threads, "blas_vendor": _nonempty(blas_vendor, "blas_vendor")})
    if resume_ledger_sha256 is not None:
        document["resume_ledger_sha256"] = _sha256(resume_ledger_sha256, field="resume_ledger_sha256")
    return canonical_value(document)  # type: ignore[return-value]


def _nonempty(value: object, field: str) -> str:
    normalized = _nfc(value, field=field) if isinstance(value, str) else ""
    if not normalized:
        raise _validation("expected nonempty string", field=field)
    return normalized


def _utc(value: str) -> str:
    normalized = _nonempty(value, "utc_timestamp")
    if not normalized.endswith("Z"):
        raise _validation("timestamps must use UTC Z spelling")
    return normalized


def canonical_result_document(document: Mapping[str, object]) -> dict[str, object]:
    """Close the three receipt-backed result discriminators materialized in dev3."""

    result = dict(document)
    result["schema"] = "scnsim.result"
    result["schema_version"] = 1
    kind = result.get("result_kind")
    if kind not in _DEV3_RESULTS:
        raise _validation("result discriminator is outside dev3", result_kind=kind)
    return canonical_value(result)  # type: ignore[return-value]


def canonical_receipt_document(document: Mapping[str, object]) -> dict[str, object]:
    """Close a receipt-last terminal envelope without giving it a self hash."""

    receipt = dict(document)
    receipt["schema"] = "scnsim.receipt"
    receipt["schema_version"] = 1
    outcome = receipt.get("outcome")
    if outcome not in {"success", "failure", "interrupted"}:
        raise _validation("invalid receipt outcome")
    if outcome == "success" and not isinstance(receipt.get("result_sha256"), str):
        raise _validation("success receipt requires result SHA-256")
    return canonical_value(receipt)  # type: ignore[return-value]


def plan_workspace_document(*, workspace_instance_id: str, plan_sha256: str) -> dict[str, object]:
    """Build the immutable leaf workspace binding document."""

    return _workspace_document({
        "kind": "plan_workspace",
        "workspace_instance_id": _uuid(workspace_instance_id),
        "plan_sha256": _sha256(plan_sha256, field="plan_sha256"),
    })


def replaceable_workspace_document(
    *, workspace_instance_id: str, leaf_instance_id: str, plan_sha256: str
) -> dict[str, object]:
    leaf = _uuid(leaf_instance_id)
    return _workspace_document({
        "kind": "replaceable_workspace",
        "workspace_instance_id": _uuid(workspace_instance_id),
        "active_leaf": {
            "directory": f"leaves/{leaf}",
            "workspace_instance_id": leaf,
            "plan_sha256": _sha256(plan_sha256, field="plan_sha256"),
        },
    })


def versioned_workspace_document(
    *, workspace_instance_id: str, iterations: Iterable[Mapping[str, object]]
) -> dict[str, object]:
    index = [dict(item) for item in iterations]
    index.sort(key=lambda item: item.get("ordinal", 0))
    expected = 1
    seen_hashes: set[str] = set()
    for item in index:
        ordinal = item.get("ordinal")
        if ordinal != expected:
            raise _validation("versioned workspace iterations must be contiguous")
        plan = _sha256(item.get("plan_sha256"), field="plan_sha256")
        if plan in seen_hashes:
            raise _validation("versioned workspace cannot repeat a Plan")
        seen_hashes.add(plan)
        directory = f"iteration{str(ordinal).zfill(2)}"
        if item.get("directory") != directory:
            raise _validation("versioned workspace directory does not match ordinal")
        item["workspace_instance_id"] = _uuid(item.get("workspace_instance_id"))
        expected += 1
    return _workspace_document({
        "kind": "versioned_workspace",
        "workspace_instance_id": _uuid(workspace_instance_id),
        "next_iteration": expected,
        "iterations": index,
    })


def _workspace_document(fields: Mapping[str, object]) -> dict[str, object]:
    return canonical_value({"schema": "scnsim.workspace", "schema_version": 1, **fields})  # type: ignore[return-value]


def _uuid(value: object) -> str:
    if not isinstance(value, str) or not _UUID4.fullmatch(value):
        raise _validation("workspace identity must be lowercase UUIDv4")
    return value


def zarr_group_metadata_bytes() -> bytes:
    """The one exact V2 root-group metadata byte sequence accepted by V1."""

    return b'{"zarr_format":2}'


def zarr_array_metadata_bytes(*, shape: Sequence[int], chunks: Sequence[int]) -> bytes:
    """Return exact compact V2 Float64 C-order dataset metadata bytes."""

    if not shape or len(shape) != len(chunks) or any(not isinstance(item, int) or item < 1 for item in (*shape, *chunks)):
        raise _validation("invalid Zarr shape/chunks")
    return canonical_json_bytes({
        "chunks": list(chunks), "compressor": None, "dimension_separator": ".", "dtype": "<f8",
        "fill_value": None, "filters": None, "order": "C", "shape": list(shape), "zarr_format": 2,
    })


def zarr_artifact_manifest(
    *, artifact_directory: Path, artifact_id: str, artifact_path: str
) -> dict[str, object]:
    """Validate one Julia-written V2 artifact tree and return its canonical manifest.

    This deliberately accepts only the root group plus `values` or paired
    `real`/`imag` Float64 arrays; callers compare the returned metadata with
    their typed result catalog.
    """

    if artifact_directory.is_symlink():
        raise _integrity("Zarr artifact directory is symlinked", path=str(artifact_directory))
    root = artifact_directory.resolve(strict=True)
    if not root.is_dir():
        raise _integrity("Zarr artifact directory is missing", path=str(artifact_directory))
    entries: list[tuple[str, Path]] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise _integrity("Zarr artifact contains a symlink", path=str(candidate))
        if candidate.is_file():
            relative = candidate.relative_to(root).as_posix()
            entries.append((relative_path(relative), candidate))
        elif candidate.is_dir():
            continue
        else:
            raise _integrity("Zarr artifact contains a non-regular filesystem entry", path=str(candidate))
    entries.sort(key=lambda item: item[0])
    files = {path: candidate for path, candidate in entries}
    if files.get(".zgroup") is None or files[".zgroup"].read_bytes() != zarr_group_metadata_bytes():
        raise _integrity("Zarr root metadata bytes differ from V1 contract")
    datasets = _zarr_datasets(files)
    allowed = {".zgroup"}
    for dataset in datasets:
        allowed.add(f"{dataset}/.zarray")
        allowed.update(path for path in files if path.startswith(f"{dataset}/") and not path.endswith("/.zarray"))
    if set(files) != allowed:
        raise _integrity("Zarr artifact contains unsupported metadata or paths", paths=sorted(set(files) - allowed))
    manifest_files = [
        {"path": path, "mode": "regular", "byte_length": file.stat().st_size, "sha256": sha256(file.read_bytes()).hexdigest()}
        for path, file in entries
    ]
    return canonical_value({
        "schema": "scnsim.artifact_manifest", "schema_version": 1,
        "artifact_id": _identifier(artifact_id, field="artifact_id"),
        "artifact_path": relative_path(artifact_path), "zarr_format": 2,
        "group_metadata_path": ".zgroup", "datasets": [
            {"path": dataset, "metadata_path": f"{dataset}/.zarray", "chunk_paths": sorted(
                path for path in files if path.startswith(f"{dataset}/") and path != f"{dataset}/.zarray"
            )}
            for dataset in datasets
        ],
        "files": manifest_files,
    })  # type: ignore[return-value]


def _zarr_datasets(files: Mapping[str, Path]) -> list[str]:
    present = [name for name in ("values", "real", "imag") if f"{name}/.zarray" in files]
    if present not in (["values"], ["real", "imag"]):
        raise _integrity("Zarr artifact must contain values or paired real/imag datasets")
    for dataset in present:
        metadata_path = f"{dataset}/.zarray"
        try:
            metadata = json.loads(files[metadata_path].read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _integrity("Zarr array metadata is invalid JSON", path=metadata_path) from error
        shape = metadata.get("shape") if isinstance(metadata, Mapping) else None
        chunks = metadata.get("chunks") if isinstance(metadata, Mapping) else None
        if not isinstance(shape, list) or not isinstance(chunks, list):
            raise _integrity("Zarr array metadata lacks shape/chunks", path=metadata_path)
        if files[metadata_path].read_bytes() != zarr_array_metadata_bytes(shape=shape, chunks=chunks):
            raise _integrity("Zarr array metadata bytes differ from V1 contract", path=metadata_path)
        counts = [math.ceil(size / chunk) for size, chunk in zip(shape, chunks)]
        expected = {
            f"{dataset}/" + ".".join(str(index) for index in indices)
            for indices in product(*(range(count) for count in counts))
        }
        actual = {
            path for path in files
            if path.startswith(f"{dataset}/") and path != metadata_path
        }
        if actual != expected:
            raise _integrity(
                "Zarr chunk grid is incomplete or has extra chunks",
                missing=sorted(expected - actual),
                extra=sorted(actual - expected),
            )
        for path in actual:
            chunk = path.removeprefix(f"{dataset}/")
            if not _CHUNK.fullmatch(chunk):
                raise _integrity("Zarr chunk name is invalid", path=path)
            indices = [int(item) for item in chunk.split(".")]
            elements = math.prod(
                min(chunk_size, size - index * chunk_size)
                for size, chunk_size, index in zip(shape, chunks, indices)
            )
            if files[path].stat().st_size != 8 * elements:
                raise _integrity("Zarr chunk byte length disagrees with its grid position", path=path)
    return present


def catalog_source_record(obj_or_class: object) -> dict[str, object]:
    """Return the reserved portable provenance record for built-in components.

    The private implementation class is only used to recognize the singleton;
    it is intentionally absent from the returned semantic identity.
    """

    subject = obj_or_class if isinstance(obj_or_class, type) else type(obj_or_class)
    if getattr(subject, "__module__", None) != "scnsim.authoring" or getattr(subject, "__name__", None) != "_BuiltinComponents":
        raise ScaffoldUnavailableError(
            "custom catalog provenance is not executable until the Composite slice",
            stage="catalog_identity",
            evidence={"module": getattr(subject, "__module__", None)},
        )
    try:
        package = distribution("scnsim")
    except PackageNotFoundError as error:
        raise _validation("installed SCNSim distribution metadata is unavailable") from error
    direct_url = package.read_text("direct_url.json")
    if direct_url:
        try:
            direct = json.loads(direct_url)
        except json.JSONDecodeError as error:
            raise _validation("SCNSim direct_url metadata is invalid") from error
        directory_info = direct.get("dir_info") if isinstance(direct, Mapping) else None
        if isinstance(directory_info, Mapping) and bool(directory_info.get("editable", False)):
            return _editable_builtin_catalog(package.version)
    return _wheel_builtin_catalog(package)


def _wheel_builtin_catalog(package: object) -> dict[str, object]:
    record = package.read_text("RECORD")  # type: ignore[union-attr]
    if record is None:
        raise _validation("installed wheel lacks RECORD provenance")
    rows: list[dict[str, object]] = []
    record_self_rows = 0
    for row in csv.reader(StringIO(record)):
        if len(row) != 3:
            raise _validation("wheel RECORD row has invalid field count")
        path, encoded_hash, size_text = row
        normalized_path = relative_path(path)
        parts = normalized_path.split("/")
        if (
            "__pycache__" in parts
            or normalized_path.endswith(".pyc")
            or (
                len(parts) >= 2
                and parts[-2].endswith(".dist-info")
                and parts[-1]
                in {"INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json", "uv_build.json"}
            )
        ):
            continue
        is_record_self = normalized_path.endswith(".dist-info/RECORD")
        if is_record_self:
            if encoded_hash or size_text:
                raise _validation("wheel RECORD self row must have empty hash and size")
            record_self_rows += 1
        elif not encoded_hash or not size_text:
            raise _validation(
                "wheel RECORD rows must bind every included file by hash and size",
                path=normalized_path,
            )
        if encoded_hash:
            algorithm, separator, digest = encoded_hash.partition("=")
            if algorithm != "sha256" or not separator or not digest:
                raise _validation("wheel RECORD uses a non-SHA-256 hash", path=normalized_path)
            try:
                expected = base64.urlsafe_b64decode(digest + "=" * (-len(digest) % 4))
            except Exception as error:
                raise _validation("wheel RECORD hash is malformed", path=normalized_path) from error
            installed = package.locate_file(path)  # type: ignore[union-attr]
            if installed.is_symlink() or not installed.is_file() or sha256(installed.read_bytes()).digest() != expected:
                raise _validation("wheel RECORD content hash does not match", path=normalized_path)
        if size_text:
            installed = package.locate_file(path)  # type: ignore[union-attr]
            if installed.is_symlink() or not size_text.isdecimal() or installed.stat().st_size != int(size_text):
                raise _validation("wheel RECORD size does not match", path=normalized_path)
        rows.append({"path": normalized_path, "hash": encoded_hash, "size": size_text})
    rows.sort(key=lambda row: row["path"])
    if len({row["path"] for row in rows}) != len(rows):
        raise _validation("wheel RECORD contains duplicate paths")
    if record_self_rows != 1:
        raise _validation("wheel RECORD must contain exactly one empty self row")
    return {
        "catalog_id": "scnsim.components",
        "catalog_kind": "builtin",
        "module": "scnsim",
        "public_symbol": "components",
        "source_kind": "wheel_record",
        "source_sha256": sha256_hex({"schema": "scnsim.wheel_record", "schema_version": 2, "rows": rows}),
        "distribution": _normalize_distribution_name(package.metadata["Name"]),  # type: ignore[union-attr]
        "version": package.version,  # type: ignore[union-attr]
    }


def _editable_builtin_catalog(version: str) -> dict[str, object]:
    package_root = Path(import_module("scnsim").__file__).resolve().parent
    git_root = _git_output(package_root, "rev-parse", "--show-toplevel")
    commit = _git_output(package_root, "rev-parse", "HEAD")
    source_rows = _source_tree_manifest(package_root)
    status = _git_output(package_root, "status", "--porcelain=v1", "--no-renames", "--untracked-files=all")
    overlay: list[dict[str, object]] = []
    for line in status.splitlines():
        if len(line) < 4:
            raise _validation("Git status output is malformed")
        path = line[3:]
        candidate = Path(git_root, path)
        try:
            relative = candidate.resolve(strict=False).relative_to(package_root).as_posix()
        except ValueError:
            continue
        entry: dict[str, object] = {"status": line[:2], "path": relative_path(relative)}
        if candidate.is_file() and not candidate.is_symlink():
            entry["sha256"] = sha256(candidate.read_bytes()).hexdigest()
        overlay.append(entry)
    overlay.sort(key=lambda entry: (entry["path"], entry["status"]))
    return {
        "catalog_id": "scnsim.components",
        "catalog_kind": "builtin",
        "module": "scnsim",
        "public_symbol": "components",
        "source_kind": "editable_git",
        "source_sha256": sha256_hex({"schema": "scnsim.package_source", "schema_version": 1, "files": source_rows}),
        "git_commit": _sha256_or_git(commit),
        "dirty_overlay_sha256": sha256_hex({"schema": "scnsim.git_overlay", "schema_version": 1, "entries": overlay}),
        # editable Git schema intentionally has no distribution/version fields.
    }


def _git_output(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(cwd), *args], text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise _validation("editable SCNSim catalog requires a readable Git repository", stderr=completed.stderr.strip())
    return completed.stdout.rstrip("\n")


def _source_tree_manifest(package_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in package_root.rglob("*"):
        if path.is_symlink():
            raise _validation("editable package source contains a symlink")
        if path.is_file() and "__pycache__" not in path.parts:
            rows.append({
                "path": relative_path(path.relative_to(package_root).as_posix()),
                "mode": "regular",
                "sha256": sha256(path.read_bytes()).hexdigest(),
            })
    rows.sort(key=lambda row: row["path"])
    return rows


def _normalize_distribution_name(name: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", _nonempty(name, "distribution")).lower()
    return normalized


def _sha256_or_git(value: str) -> str:
    # The full V1 schema allows Git SHA-1 or SHA-256 object IDs.  Do not force
    # the current checkout's Git object format into the evidence protocol.
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise _validation("Git commit is not a lowercase object ID")
    return value
