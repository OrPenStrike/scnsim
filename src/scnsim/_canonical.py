"""Canonical bytes and evidence for structured Plans and runtime envelopes.

This module is deliberately not a generic schema engine.  The shipped JSON
Schema is the field authority; these helpers own the bytes that Python writes
for captured Plans, parameter points, requests, and their execution evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import ast
import base64
import csv
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution, distributions
from io import StringIO
from itertools import product
from pathlib import Path
import inspect
import json
import math
import os
import re
import struct
import tokenize
import subprocess
import unicodedata
from urllib.parse import unquote, urlparse

from .errors import EvidenceIntegrityError, SCNSimValidationError


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

_DIRECT_ALGORITHMS = {
    "solve_direct": "scnsim.direct_response.v1",
    "solve_hb": "scnsim.hb_response.josephsoncircuits.v1",
    "optimize_direct": "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v4",
}
_EVALUATION_ALGORITHMS = {
    "diagonal_root": "scnsim.diagonal_root.newton32.v1",
    "hybridized_pole": "scnsim.hybridized_pole.newton32.v1",
    "transfer_zero": "scnsim.transfer_zero.newton32.v1",
    "residue_normalized_coupling": "scnsim.residue_normalized_coupling.v1",
    "response_element": "scnsim.response_element.v1",
    "operator": "scnsim.direct_operator.v1",
}
_DIRECT_RESULTS = frozenset({
    "direct_response", "diagonal_root", "hybridized_pole", "transfer_zero",
    "residue_normalized_coupling", "response_element", "operator", "optimization", "hb_batch",
})


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


def _sort_id_maps(values: Iterable[Mapping[str, object]], *, key: str = "id") -> list[dict[str, object]]:
    copied = [dict(value) for value in values]
    copied.sort(key=lambda value: _identifier(value.get(key), field=key))
    if len({_identifier(value.get(key), field=key) for value in copied}) != len(copied):
        raise _validation("canonical identifiers must be unique", field=key)
    return copied


def canonical_plan_document(snapshot: Mapping[str, object]) -> dict[str, object]:
    """Close the one normalized structured-authoring record as a V2 Plan.

    ``AuthoringSnapshot.source_provenance`` is intentionally not accepted
    here.  Recursive traversal and compiler lookup are projections of these
    normalized tables, never additional records hashed beside them.
    """

    document = dict(snapshot)
    required = {
        "schema", "schema_version", "plan_id", "scope_hierarchy",
        "occurrences", "physical_leaves", "connectivity", "parameter_closure",
    }
    if set(document) != required or document.get("schema") != "scnsim.authoring_snapshot" or document.get("schema_version") != 2:
        raise _validation("authoring snapshot does not match the structured V2 handoff", fields=sorted(document))
    document["schema"] = "scnsim.plan"
    document["plan_id"] = _identifier(document["plan_id"], field="plan_id")

    def path_of(value: Mapping[str, object], *, field: str = "path") -> tuple[str, ...]:
        path = value.get(field)
        if not isinstance(path, Sequence) or isinstance(path, (str, bytes)) or not path:
            raise _validation("structured record requires a nonempty occurrence path", field=field)
        return tuple(_identifier(item, field=field) for item in path)

    occurrences = [dict(item) for item in _iter_mappings(document["occurrences"], "occurrences")]
    occurrences.sort(key=path_of)
    if len({path_of(item) for item in occurrences}) != len(occurrences):
        raise _validation("occurrence paths must be unique")
    document["occurrences"] = occurrences

    leaves = [dict(item) for item in _iter_mappings(document["physical_leaves"], "physical_leaves")]
    leaves.sort(key=path_of)
    if len({path_of(item) for item in leaves}) != len(leaves):
        raise _validation("physical-leaf paths must be unique")
    occurrence_paths = {path_of(item) for item in occurrences}
    if any(path_of(item) not in occurrence_paths for item in leaves):
        raise _validation("physical leaf has no owning occurrence")
    document["physical_leaves"] = leaves

    connectivity = dict(_mapping(document["connectivity"], "connectivity"))
    expected_connectivity = {
        "physical_endpoints", "endpoint_nets", "node_coordinates", "canonical_ground", "ports", "couplings",
    }
    if set(connectivity) != expected_connectivity or connectivity.get("canonical_ground") != "ground":
        raise _validation("structured connectivity fields are invalid")
    endpoints = [dict(item) for item in _iter_mappings(connectivity["physical_endpoints"], "physical_endpoints")]
    endpoints.sort(key=lambda item: (path_of(item), _identifier(item.get("pin"), field="pin")))
    endpoint_keys = [(path_of(item), _identifier(item.get("pin"), field="pin")) for item in endpoints]
    if len(set(endpoint_keys)) != len(endpoint_keys):
        raise _validation("physical endpoints must be unique")
    connectivity["physical_endpoints"] = endpoints
    endpoint_nets = [dict(item) for item in _iter_mappings(connectivity["endpoint_nets"], "endpoint_nets")]
    endpoint_nets.sort(key=lambda item: canonical_json_bytes(item))
    if len({canonical_json_bytes(item.get("endpoint")) for item in endpoint_nets}) != len(endpoint_nets):
        raise _validation("structural endpoints must map to one final net")
    connectivity["endpoint_nets"] = endpoint_nets
    node_coordinates = [
        dict(item) for item in _iter_mappings(connectivity["node_coordinates"], "node_coordinates")
    ]
    final_nets: list[str] = []
    compiler_nodes: list[str] = []
    for node in node_coordinates:
        if set(node) != {"final_net", "compiler_node_id", "visibility", "public_aliases"}:
            raise _validation("node-coordinate fields are invalid")
        final_nets.append(_identifier(node["final_net"], field="final_net"))
        compiler_nodes.append(_identifier(node["compiler_node_id"], field="compiler_node_id"))
        if node["visibility"] not in {"public", "internal"}:
            raise _validation("node-coordinate visibility is invalid")
        aliases = node["public_aliases"]
        if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
            raise _validation("node-coordinate aliases must be an ordered array")
    if len(set(final_nets)) != len(final_nets) or len(set(compiler_nodes)) != len(compiler_nodes):
        raise _validation("node-coordinate identities must be unique")
    if any(item.get("net") not in set(final_nets) | {"ground"} for item in endpoints):
        raise _validation("physical endpoint targets an unknown final net")
    connectivity["node_coordinates"] = node_coordinates  # Compiler order is explicit.
    ports = [dict(item) for item in _iter_mappings(connectivity["ports"], "ports")]
    port_ids = [_identifier(item.get("id"), field="port.id") for item in ports]
    if len(set(port_ids)) != len(port_ids):
        raise _validation("Port IDs must be unique")
    connectivity["ports"] = ports  # Declaration order is semantic.
    connectivity["couplings"] = _sort_id_maps(
        _iter_mappings(connectivity["couplings"], "couplings")
    )
    document["connectivity"] = connectivity

    closure = dict(_mapping(document["parameter_closure"], "parameter_closure"))
    if set(closure) != {"definitions", "field_bindings"}:
        raise _validation("parameter closure fields are invalid")
    definitions = [dict(item) for item in _iter_mappings(closure["definitions"], "definitions")]
    definitions.sort(key=lambda item: _parameter_ref_key(item))
    definition_keys = [_parameter_ref_key(item) for item in definitions]
    if len(set(definition_keys)) != len(definition_keys):
        raise _validation("consumed parameter definitions must be unique")
    bindings = [dict(item) for item in _iter_mappings(closure["field_bindings"], "field_bindings")]
    bindings.sort(key=lambda item: (path_of(item), _identifier(item.get("field"), field="field")))
    binding_keys = [(path_of(item), _identifier(item.get("field"), field="field")) for item in bindings]
    if len(set(binding_keys)) != len(binding_keys):
        raise _validation("physical fields cannot have competing parameter bindings")
    for binding in bindings:
        if _parameter_ref_key(binding.get("parameter")) not in set(definition_keys):
            raise _validation("physical field binding names an unconsumed parameter")
    closure["definitions"] = definitions
    closure["field_bindings"] = bindings
    document["parameter_closure"] = closure
    return canonical_value(document)  # type: ignore[return-value]


def canonical_plan_snapshot(snapshot: object) -> dict[str, object]:
    """Canonicalize an ``AuthoringSnapshot`` without importing model types."""

    semantic_record = getattr(snapshot, "semantic_record", None)
    if not isinstance(semantic_record, Mapping):
        raise TypeError("snapshot must expose an immutable semantic_record mapping")
    return canonical_plan_document(semantic_record)


def canonical_parameters_sha256(parameter_record: Mapping[str, object]) -> str:
    """Identify one complete effective point, independently of source spelling."""

    return sha256_hex({
        "schema": "scnsim.parameter_point_identity",
        "schema_version": 2,
        "parameters": canonical_parameter_set(parameter_record),
    })


def canonical_resolved_plan_point(
    point: object, *, plan_sha256: str
) -> dict[str, object]:
    """Encode the immutable point handed to the standalone compiler audit."""

    snapshot = getattr(point, "snapshot", None)
    parameter_record = getattr(point, "parameter_record", None)
    resolved_fields = getattr(point, "resolved_fields", None)
    if snapshot is None or not isinstance(parameter_record, Mapping) or not isinstance(resolved_fields, Mapping):
        raise TypeError("point must be a ResolvedPlanPoint")
    plan = canonical_plan_snapshot(snapshot)
    expected_plan_sha = sha256_hex(plan)
    if _sha256(plan_sha256, field="plan_sha256") != expected_plan_sha:
        raise _validation("resolved point is paired with a different Plan")
    units_by_field: dict[tuple[tuple[str, ...], str], str] = {}
    for leaf in _iter_mappings(plan["physical_leaves"], "physical_leaves"):
        path = _component_path(leaf.get("path"))
        for field in _iter_mappings(leaf.get("fields"), "physical leaf fields"):
            identifier = _identifier(field.get("id"), field="field")
            unit = _as_str(field.get("unit"), "field.unit")
            units_by_field[(path, identifier)] = unit
    if set(resolved_fields) != set(units_by_field):
        raise _validation("resolved physical fields do not exactly cover the Plan")

    from .units import registry

    rows: list[dict[str, object]] = []
    for key in sorted(units_by_field):
        value = resolved_fields[key]
        unit = units_by_field[key]
        record = getattr(value, "_record", None)
        if unit == "rlgc":
            if not callable(record):
                raise _validation("resolved RLGC field has no structured record")
            encoded = record()
            if not isinstance(encoded, Mapping) or encoded.get("type") != "rlgc":
                raise _validation("resolved RLGC field is malformed")
        else:
            encoded = quantity_envelope(value, si_unit=unit, registry=registry)
        rows.append({"path": list(key[0]), "field": key[1], "value": encoded})
    parameters = canonical_parameter_set(parameter_record)
    return canonical_value({
        "schema": "scnsim.resolved_plan_point",
        "schema_version": 2,
        "plan_sha256": expected_plan_sha,
        "parameters": parameters,
        "parameters_sha256": canonical_parameters_sha256(parameters),
        "resolved_fields": rows,
    })  # type: ignore[return-value]


def canonical_diagram_digests(
    plan: object, *, representation: str
) -> dict[str, str]:
    """Return the representation-tagged diagram facts owned by canonicalization."""

    if representation not in {"authoring", "compiled"}:
        raise ValueError("representation must be 'authoring' or 'compiled'")
    if isinstance(plan, Mapping):
        raw = dict(plan)
        if raw.get("schema") == "scnsim.plan":
            raw["schema"] = "scnsim.authoring_snapshot"
        document = canonical_plan_document(raw)
    else:
        document = canonical_plan_snapshot(plan)
    plan_sha256 = sha256_hex(document)
    common = {
        "schema_version": 2,
        "representation": representation,
        "plan_sha256": plan_sha256,
    }
    connectivity = sha256_hex({
        **common,
        "schema": "scnsim.diagram_connectivity_identity",
        "connectivity": document["connectivity"],
    })
    semantic = sha256_hex({
        **common,
        "schema": "scnsim.diagram_semantic_identity",
        "scope_hierarchy": document["scope_hierarchy"],
        "occurrences": document["occurrences"],
        "physical_leaves": document["physical_leaves"],
        "parameter_closure": document["parameter_closure"],
    })
    return {
        "plan_sha256": plan_sha256,
        "connectivity_sha256": connectivity,
        "semantic_sha256": semantic,
    }


def canonical_expanded_graph_sha256(
    *,
    plan_sha256: str,
    node_order: Sequence[str],
    resolved_bindings: Sequence[Mapping[str, object]],
    expanded_branch_rows: Sequence[Mapping[str, object]],
) -> str:
    """Bind the exact compiler expansion fields promised to diagram evidence."""

    return sha256_hex({
        "schema": "scnsim.expanded_graph_identity",
        "schema_version": 2,
        "plan_sha256": _sha256(plan_sha256, field="plan_sha256"),
        "node_order": list(node_order),
        "resolved_bindings": list(resolved_bindings),
        "expanded_branch_rows": list(expanded_branch_rows),
    })


def _component_path(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise _validation("component path must be nonempty segments")
    return tuple(_identifier(segment, field="component_path") for segment in value)


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
    """Close one V2 ParameterSet by logical definition/local identity."""

    document = dict(parameters)
    if set(document) != {"type", "bindings", "allow_extrapolation"} or document.get("type") != "parameter_set_v2":
        raise _validation("invalid parameter-set envelope")
    bindings = _iter_mappings(document["bindings"], "parameter bindings")
    if any(set(item) != {"parameter", "value"} for item in bindings):
        raise _validation("parameter binding fields are invalid")
    for item in bindings:
        reference = _mapping(item["parameter"], "parameter_ref")
        if set(reference) != {"definitions_id", "parameter_id"}:
            raise _validation("parameter ref fields are invalid")
    bindings.sort(key=lambda item: _parameter_ref_key(_mapping(item, "parameter binding").get("parameter")))
    if len({_parameter_ref_key(item.get("parameter")) for item in bindings}) != len(bindings):
        raise _validation("parameter bindings must be unique")
    document["bindings"] = [dict(item) for item in bindings]
    authorizations = _iter_mappings(document["allow_extrapolation"], "allow_extrapolation")
    if any(set(item) != {"definitions_id", "parameter_id"} for item in authorizations):
        raise _validation("parameter authorization ref fields are invalid")
    authorizations.sort(key=lambda item: _parameter_ref_key(item))
    if len({_parameter_ref_key(item) for item in authorizations}) != len(authorizations):
        raise _validation("allow_extrapolation values must be unique")
    document["allow_extrapolation"] = [dict(item) for item in authorizations]
    return canonical_value(document)  # type: ignore[return-value]


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _validation("expected object", field=field)
    return value


def _parameter_ref_key(value: object) -> tuple[str, str]:
    ref = _mapping(value, "parameter_ref")
    if not {"definitions_id", "parameter_id"}.issubset(ref):
        raise _validation("parameter ref fields are invalid")
    return (
        _identifier(ref["definitions_id"], field="definitions_id"),
        _identifier(ref["parameter_id"], field="parameter_id"),
    )


def canonical_parameter_source(source: Mapping[str, object]) -> dict[str, object]:
    """Close a point or lazily enumerable ordered parameter-space descriptor."""

    document = dict(source)
    kind = document.get("kind")
    if kind == "point":
        if set(document) != {"kind", "parameters"}:
            raise _validation("point parameter source fields are invalid")
        document["parameters"] = canonical_parameter_set(
            _mapping(document["parameters"], "parameters")
        )
    elif kind == "grid":
        if set(document) != {"kind", "base_parameters", "axes", "shape"}:
            raise _validation("grid parameter source fields are invalid")
        document["base_parameters"] = canonical_parameter_set(
            _mapping(document["base_parameters"], "base_parameters")
        )
        axes = [dict(item) for item in _iter_mappings(document["axes"], "axes")]
        shape = document["shape"]
        if (
            not axes
            or not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes))
            or len(shape) != len(axes)
        ):
            raise _validation("grid shape must match its nonempty ordered axes")
        keys: list[tuple[str, str]] = []
        for index, axis in enumerate(axes):
            if set(axis) != {"parameter", "values"}:
                raise _validation("grid axis fields are invalid")
            key = _parameter_ref_key(axis["parameter"])
            values = axis["values"]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
                raise _validation("grid axes must be nonempty arrays")
            if isinstance(shape[index], bool) or not isinstance(shape[index], int) or shape[index] != len(values):
                raise _validation("grid shape disagrees with an axis length")
            keys.append(key)
        if len(set(keys)) != len(keys):
            raise _validation("grid axis parameters must be unique")
        document["axes"] = axes  # Author order is semantic.
        document["shape"] = list(shape)
    elif kind == "points":
        if set(document) != {"kind", "baseline_parameters", "points"}:
            raise _validation("listed parameter source fields are invalid")
        document["baseline_parameters"] = canonical_parameter_set(
            _mapping(document["baseline_parameters"], "baseline_parameters")
        )
        points = document["points"]
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or not points:
            raise _validation("listed parameter source must contain points")
        document["points"] = [
            canonical_parameter_set(_mapping(item, "listed point")) for item in points
        ]
    else:
        raise _validation("unknown parameter source kind", kind=kind)
    return canonical_value(document)  # type: ignore[return-value]


def canonical_request_document(
    *,
    plan_sha256: str,
    operation: str,
    view: Mapping[str, object],
    spec: Mapping[str, object],
    parameter_source: Mapping[str, object],
    runtime_semantic: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact closed declarative request envelope.

    The operation remains deliberately coarse: every scalar Direct evaluation
    shares ``evaluate_direct`` while its closed Spec discriminator selects the
    Human-defined algorithm identity.  This keeps the request envelope stable
    without a second operation family.
    """

    selected_operation = _nfc(operation, field="operation")
    runtime = dict(runtime_semantic)
    spec_type = spec.get("type")
    expected_algorithm = (
        _EVALUATION_ALGORITHMS.get(str(spec_type))
        if selected_operation == "evaluate_direct"
        else _DIRECT_ALGORITHMS.get(selected_operation)
    )
    expected_spec = {
        "solve_direct": "direct_solve",
        "solve_hb": "hb_solve",
        "optimize_direct": "optimization",
    }.get(selected_operation)
    if expected_algorithm is None:
        raise _validation("operation or Spec is outside the runtime", operation=selected_operation, spec_type=spec_type)
    if expected_spec is not None and spec_type != expected_spec:
        raise _validation("operation requires a different Spec", operation=selected_operation, spec_type=spec_type)
    if runtime.get("algorithm_id") != expected_algorithm:
        raise _validation("request algorithm does not match operation and Spec", operation=selected_operation, spec_type=spec_type)
    return canonical_value({
        "schema": "scnsim.request",
        "schema_version": 2,
        "plan_sha256": _sha256(plan_sha256, field="plan_sha256"),
        "operation": selected_operation,
        "view": dict(view),
        "spec": dict(spec),
        "parameter_source": canonical_parameter_source(parameter_source),
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
    """Close receipt-backed result discriminators materialized by the runtime."""

    result = dict(document)
    result["schema"] = "scnsim.result"
    result["schema_version"] = 2
    kind = result.get("result_kind")
    if kind not in _DIRECT_RESULTS | {"parameter_sweep"}:
        raise _validation("result discriminator is outside the runtime", result_kind=kind)
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

    if (
        not shape
        or len(shape) != len(chunks)
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < 0
            or (item == 0 and index != 0)
            for index, item in enumerate(shape)
        )
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in chunks)
    ):
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


def catalog_source_record(
    obj_or_class: object, *, factory: object | None = None
) -> dict[str, object]:
    """Capture one catalog-wide portable source identity.

    ``factory`` remains an authoring call-site compatibility input; it never
    enters the returned record because the component snapshot owns the invoked
    factory name.
    """

    subject = obj_or_class if isinstance(obj_or_class, type) else type(obj_or_class)
    if getattr(subject, "__module__", None) == "scnsim.authoring" and getattr(subject, "__name__", None) == "_BuiltinComponents":
        return _builtin_catalog_source()
    records = [_custom_catalog_source(candidate, factory=factory) for candidate in _catalog_lineage(subject)]
    record = records[-1]
    record["source_sha256"] = sha256_hex({
        "schema": "scnsim.catalog_lineage_source",
        "schema_version": 1,
        "classes": records,
    })
    return record


def _custom_catalog_source(subject: type[object], *, factory: object | None) -> dict[str, object]:
    module_name = _catalog_module_name(subject)
    qualified_class = _catalog_qualified_class(subject)
    identity = {
        "catalog_id": f"{module_name}:{qualified_class}",
        "catalog_kind": "custom",
        "module": module_name,
        "qualified_class": qualified_class,
    }
    module = import_module(module_name)
    source_path = _module_source_path(module)
    if source_path is not None:
        package = _distribution_owning(source_path, module_name)
        if package is not None:
            if _editable_distribution(package):
                package_root = _package_root(module_name, source_path)
                if package_root is None:
                    raise _validation("editable custom catalog must be inside a Python package")
                return _editable_catalog(identity, package_root)
            return _wheel_catalog(identity, package)
        return {
            **identity,
            "source_kind": "module_source",
            "source_sha256": sha256(_normalized_source_bytes(source_path)).hexdigest(),
        }
    return {
        **identity,
        "source_kind": "notebook_source",
        "source_sha256": sha256(_notebook_source_bytes(subject, factory)).hexdigest(),
    }


def _catalog_lineage(subject: type[object]) -> list[type[object]]:
    lineage = [
        candidate for candidate in reversed(subject.__mro__)
        if candidate is not object and not (
            candidate.__module__ == "scnsim.authoring" and candidate.__name__ == "Library"
        )
    ]
    if not lineage or lineage[-1] is not subject:
        raise _validation("catalog class does not have a closed Library lineage")
    return lineage


def _builtin_catalog_source() -> dict[str, object]:
    """Return the reserved provenance record for the public singleton."""

    try:
        package = distribution("scnsim")
    except PackageNotFoundError as error:
        raise _validation("installed SCNSim distribution metadata is unavailable") from error
    identity = {
        "catalog_id": "scnsim.components",
        "catalog_kind": "builtin",
        "module": "scnsim",
        "public_symbol": "components",
    }
    if _editable_distribution(package):
        package_root = Path(import_module("scnsim").__file__).resolve().parent
        return _editable_catalog(identity, package_root)
    return _wheel_catalog(identity, package)


def _wheel_catalog(identity: Mapping[str, object], package: object) -> dict[str, object]:
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
        **identity,
        "source_kind": "wheel_record",
        "source_sha256": sha256_hex({"schema": "scnsim.wheel_record", "schema_version": 2, "rows": rows}),
        "distribution": _normalize_distribution_name(package.metadata["Name"]),  # type: ignore[union-attr]
        "version": package.version,  # type: ignore[union-attr]
    }


def _editable_catalog(identity: Mapping[str, object], package_root: Path) -> dict[str, object]:
    git_root = _git_output(package_root, "rev-parse", "--show-toplevel")
    commit = _git_output(package_root, "rev-parse", "HEAD")
    source_rows = _source_tree_manifest(package_root, Path(git_root))
    status = _git_output_bytes(package_root, "status", "--porcelain=v1", "-z", "--no-renames", "--untracked-files=all")
    overlay: list[dict[str, object]] = []
    for record in status.split(b"\0"):
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise _validation("Git status output is malformed")
        raw_path = record[3:]
        try:
            changed_path = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _validation("Git status path is not UTF-8") from error
        candidate = Path(git_root, changed_path)
        try:
            relative = candidate.absolute().relative_to(package_root).as_posix()
        except ValueError:
            continue
        if _excluded_source_path(relative):
            continue
        entry: dict[str, object] = {"status": record[:2].decode("ascii"), "path": relative_path(relative)}
        if candidate.is_file() and not candidate.is_symlink():
            entry["sha256"] = sha256(candidate.read_bytes()).hexdigest()
        overlay.append(entry)
    overlay.sort(key=lambda entry: (entry["path"], entry["status"]))
    return {
        **identity,
        "source_kind": "editable_git",
        "source_sha256": sha256_hex({"schema": "scnsim.package_source", "schema_version": 1, "files": source_rows}),
        "git_commit": _sha256_or_git(commit),
        "dirty_overlay_sha256": sha256_hex({"schema": "scnsim.git_overlay", "schema_version": 1, "entries": overlay}),
    }


def _git_output(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(cwd), *args], text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise _validation("editable SCNSim catalog requires a readable Git repository", stderr=completed.stderr.strip())
    return completed.stdout.rstrip("\n")


def _git_output_bytes(cwd: Path, *args: str) -> bytes:
    completed = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, check=False)
    if completed.returncode != 0:
        raise _validation("editable catalog requires a readable Git repository", stderr=completed.stderr.decode(errors="replace").strip())
    return completed.stdout


def _source_tree_manifest(package_root: Path, git_root: Path) -> list[dict[str, object]]:
    modes = _git_file_modes(git_root)
    rows: list[dict[str, object]] = []
    for path in package_root.rglob("*"):
        if path.is_symlink():
            raise _validation("editable package source contains a symlink")
        relative = path.relative_to(package_root).as_posix()
        if path.is_file() and not _excluded_source_path(relative):
            git_relative = path.relative_to(git_root).as_posix()
            rows.append({
                "path": relative_path(relative),
                "mode": modes.get(git_relative, _filesystem_git_mode(path)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            })
    rows.sort(key=lambda row: row["path"])
    return rows


def _git_file_modes(git_root: Path) -> dict[str, str]:
    output = _git_output_bytes(git_root, "ls-files", "-s", "-z")
    modes: dict[str, str] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            prefix, raw_path = record.split(b"\t", 1)
            mode, _object_id, stage = prefix.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise _validation("Git index output is malformed") from error
        if stage != "0" or mode not in {"100644", "100755"}:
            raise _validation("catalog source has an unsupported Git file mode", mode=mode)
        modes[path] = mode
    return modes


def _filesystem_git_mode(path: Path) -> str:
    return "100755" if path.stat().st_mode & 0o111 else "100644"


def _excluded_source_path(value: str) -> bool:
    parts = value.split("/")
    return "__pycache__" in parts or any(part in {".git", ".hg", ".svn", ".pytest_cache", ".mypy_cache", "build", "dist"} or part.endswith(".egg-info") for part in parts) or value.endswith((".pyc", ".pyo"))


def _catalog_module_name(subject: type[object]) -> str:
    module = getattr(subject, "__module__", None)
    if not isinstance(module, str) or not module or module == "__main__":
        return "__main__" if module == "__main__" else _raise_catalog_identity("catalog class has no portable module name")
    return _nfc(module, field="catalog_module")


def _catalog_qualified_class(subject: type[object]) -> str:
    qualified = getattr(subject, "__qualname__", None)
    if not isinstance(qualified, str) or not qualified or "<locals>" in qualified:
        raise _validation("catalog class has no portable qualified name")
    return _nfc(qualified, field="qualified_class")


def _raise_catalog_identity(message: str) -> object:
    raise _validation(message)


def _module_source_path(module: object) -> Path | None:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or raw_path.startswith("<"):
        return None
    path = Path(raw_path)
    if path.suffix != ".py" or path.is_symlink() or not path.is_file():
        raise _validation("custom catalog module must have a readable regular Python source file")
    return path.resolve()


def _distribution_owning(source_path: Path, module_name: str) -> object | None:
    matches: list[object] = []
    for candidate in distributions():
        files = candidate.files
        if files is not None:
            for file in files:
                located = Path(candidate.locate_file(file))
                if located.exists() and located.resolve() == source_path:
                    matches.append(candidate)
                    break
            else:
                root = _editable_distribution_root(candidate, module_name)
                if root is not None:
                    try:
                        source_path.relative_to(root)
                    except ValueError:
                        continue
                    matches.append(candidate)
        else:
            root = _editable_distribution_root(candidate, module_name)
            if root is not None:
                try:
                    source_path.relative_to(root)
                except ValueError:
                    continue
                matches.append(candidate)
    if len(matches) > 1:
        raise _validation("custom catalog source belongs to multiple installed distributions")
    return matches[0] if matches else None


def _editable_distribution(package: object) -> bool:
    direct_url = package.read_text("direct_url.json")  # type: ignore[union-attr]
    if not direct_url:
        return False
    try:
        direct = json.loads(direct_url)
    except json.JSONDecodeError as error:
        raise _validation("catalog direct_url metadata is invalid") from error
    info = direct.get("dir_info") if isinstance(direct, Mapping) else None
    return isinstance(info, Mapping) and bool(info.get("editable", False))


def _editable_distribution_root(package: object, module_name: str) -> Path | None:
    if not _editable_distribution(package):
        return None
    top_level = package.read_text("top_level.txt")  # type: ignore[union-attr]
    if top_level is None or module_name.split(".", 1)[0] not in {line.strip() for line in top_level.splitlines()}:
        return None
    direct_url = package.read_text("direct_url.json")  # type: ignore[union-attr]
    direct = json.loads(direct_url)
    url = direct.get("url") if isinstance(direct, Mapping) else None
    parsed = urlparse(url) if isinstance(url, str) else None
    if parsed is None or parsed.scheme != "file":
        return None
    try:
        return Path(unquote(parsed.path)).resolve(strict=True)
    except OSError as error:
        raise _validation("editable catalog source root is unreadable") from error


def _package_root(module_name: str, source_path: Path) -> Path | None:
    parts = module_name.split(".")
    if module_name == "__main__" or not parts:
        return None
    directory = source_path.parent
    for _ in range(len(parts) - (1 if source_path.name == "__init__.py" else 2)):
        directory = directory.parent
    return directory if (directory / "__init__.py").is_file() else None


def _normalized_source_bytes(path: Path) -> bytes:
    try:
        with tokenize.open(path) as source:
            text = source.read()
    except (OSError, SyntaxError, UnicodeError) as error:
        raise _validation("custom catalog source cannot be decoded", path=str(path)) from error
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).encode("utf-8")


def _notebook_source_bytes(subject: type[object], _factory: object | None = None) -> bytes:
    """Close the complete custom Library declaration visible in a notebook."""

    classes: list[dict[str, str]] = []
    factories: list[dict[str, str]] = []
    try:
        for candidate in _catalog_lineage(subject):
            class_source, factory_sources = _notebook_declaration_sources(candidate)
            classes.append({
                "qualified_class": _catalog_qualified_class(candidate),
                "source": _normalized_notebook_source(class_source),
            })
            for name, descriptor in candidate.__dict__.items():
                if name.startswith("_"):
                    continue
                function = descriptor.__func__ if isinstance(descriptor, (classmethod, staticmethod)) else descriptor
                if inspect.isfunction(function):
                    factories.append({
                        "qualified_class": _catalog_qualified_class(candidate),
                        "name": _identifier(name, field="factory"),
                        "source": _normalized_notebook_source(factory_sources[name]),
                    })
    except (OSError, TypeError) as error:
        raise _validation("notebook catalog source is unavailable") from error
    return canonical_json_bytes({
        "schema": "scnsim.notebook_catalog_source",
        "schema_version": 1,
        "classes": classes,
        "factories": factories,
    })


def _notebook_declaration_sources(candidate: type[object]) -> tuple[str, dict[str, str]]:
    """Return one class declaration and its factories from source or IPython history.

    Notebook cells deliberately have no importable module file.  When Python
    cannot recover their source through ``inspect``, the current IPython
    kernel's raw input history is the only accepted alternate evidence.  The
    history match is bound to every unwrapped factory's execution location;
    a namesake in another cell is never provenance for the live catalog.
    """

    functions = _catalog_factory_functions(candidate)
    try:
        return (
            inspect.getsource(candidate),
            {name: inspect.getsource(function) for name, function in functions.items()},
        )
    except (OSError, TypeError):
        return _ipython_notebook_declaration_sources(candidate, functions)


def _catalog_factory_functions(candidate: type[object]) -> dict[str, object]:
    """Return the unwrapped public factory functions declared by one catalog."""

    functions: dict[str, object] = {}
    for name, descriptor in candidate.__dict__.items():
        if name.startswith("_"):
            continue
        function = (
            descriptor.__func__
            if isinstance(descriptor, (classmethod, staticmethod))
            else descriptor
        )
        if inspect.isfunction(function):
            functions[name] = inspect.unwrap(function)
    return functions


def _ipython_notebook_declaration_sources(
    candidate: type[object], functions: Mapping[str, object]
) -> tuple[str, dict[str, str]]:
    """Recover one exact notebook declaration from live IPython cell history.

    A Library with no local factory code has no executable location by which a
    history declaration could be proven current, so it remains fail-closed.
    """

    if not functions:
        raise OSError("notebook catalog has no local factory source location")
    locations = {
        name: _ipython_function_location(function)
        for name, function in functions.items()
    }
    execution_counts = {
        execution_count
        for execution_count, _, _ in locations.values()
        if execution_count is not None
    }
    if len(execution_counts) > 1 or (
        execution_counts
        and any(
            execution_count is None
            for execution_count, _, _ in locations.values()
        )
    ):
        raise OSError("notebook catalog factories do not share one declaration cell")
    execution_count = next(iter(execution_counts), None)
    matches: list[tuple[str, ast.ClassDef, dict[str, str]]] = []
    for source in _ipython_history_sources(execution_count):
        try:
            parsed = ast.parse(source)
        except SyntaxError:
            continue
        class_node = _ipython_class_node(parsed, candidate.__qualname__)
        if class_node is None:
            continue
        members = {
            node.name: node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        factory_sources: dict[str, str] = {}
        for name, (_, line_number, _) in locations.items():
            node = members.get(name)
            if node is None or not _node_covers_line(node, line_number):
                break
            factory_sources[name] = _ast_source_segment(source, node)
        else:
            if _history_source_matches_functions(source, functions, locations):
                matches.append((source, class_node, factory_sources))
    if len(matches) != 1:
        identities = {
            _notebook_declaration_identity(source, class_node, factory_sources)
            for source, class_node, factory_sources in matches
        }
        if not matches or len(identities) != 1:
            raise OSError("notebook execution source is missing or ambiguous")
    source, class_node, factory_sources = matches[0]
    return _ast_source_segment(source, class_node), factory_sources


def _ipython_function_location(function: object) -> tuple[int | None, int, str]:
    """Return the trusted IPython execution and line for one live function."""

    code = getattr(function, "__code__", None)
    filename = getattr(code, "co_filename", "")
    match = re.fullmatch(r"<ipython-input-([0-9]+)-[0-9a-f]+>", filename)
    line_number = getattr(code, "co_firstlineno", None)
    temporary_kernel_file = re.fullmatch(
        r"(?:.*/)?ipykernel_[^/]+/[0-9]+\.py", filename
    )
    if (
        not isinstance(line_number, int)
        or line_number < 1
        or (match is None and temporary_kernel_file is None)
    ):
        raise OSError("notebook factory has no IPython execution location")
    return (
        int(match.group(1)) if match is not None else None,
        line_number,
        filename,
    )


def _ipython_history_sources(execution_count: int | None) -> list[str]:
    """Return raw current-session IPython cells for one execution or all cells."""

    try:
        from IPython import get_ipython
    except ImportError as error:
        raise OSError("IPython history is unavailable") from error
    shell = get_ipython()
    history = getattr(shell, "history_manager", None)
    if history is None:
        raise OSError("IPython history is unavailable")
    matches = [
        source
        for _, line, source in history.get_range(raw=True)
        if (execution_count is None or line == execution_count)
        and isinstance(source, str)
    ]
    if not matches:
        raise OSError("notebook execution source is missing or ambiguous")
    return matches


def _history_source_matches_functions(
    source: str,
    functions: Mapping[str, object],
    locations: Mapping[str, tuple[int | None, int, str]],
) -> bool:
    """Bind raw history to a current code object's trusted execution identity."""

    for name, function in functions.items():
        execution_count, _, filename = locations[name]
        if execution_count is not None:
            continue
        if not _ipykernel_filename_matches_source(filename, source):
            return False
        code = getattr(function, "__code__", None)
        if code is None or getattr(code, "co_filename", None) != filename:
            return False
    return True


def _ipykernel_filename_matches_source(filename: str, source: str) -> bool:
    """Verify IPykernel's source-derived code-object filename without I/O."""

    try:
        from ipykernel.compiler import get_tmp_hash_seed, murmur2_x86
    except ImportError:
        return False
    expected = f"{murmur2_x86(source, get_tmp_hash_seed())}.py"
    return Path(filename).name == expected


def _notebook_declaration_identity(
    source: str, class_node: ast.ClassDef, factory_sources: Mapping[str, str]
) -> bytes:
    """Return the exact normalized declaration identity used to collapse reruns."""

    return canonical_json_bytes({
        "class": _normalized_notebook_source(_ast_source_segment(source, class_node)),
        "factories": [
            {"name": name, "source": _normalized_notebook_source(factory_sources[name])}
            for name in sorted(factory_sources)
        ],
    })


def _ipython_class_node(tree: ast.AST, qualified_name: str) -> ast.ClassDef | None:
    """Find one non-local class declaration by its exact qualified path."""

    parts = qualified_name.split(".")
    if not parts or any(not part or part == "<locals>" for part in parts):
        return None
    nodes: list[ast.AST] = [tree]
    for part in parts:
        matches = [
            child
            for node in nodes
            for child in getattr(node, "body", ())
            if isinstance(child, ast.ClassDef) and child.name == part
        ]
        if len(matches) != 1:
            return None
        nodes = matches
    return nodes[0] if isinstance(nodes[0], ast.ClassDef) else None


def _node_covers_line(node: ast.AST, line_number: int) -> bool:
    """Accept a function's definition or decorator line, but no other source."""

    decorator_lines = [decorator.lineno for decorator in getattr(node, "decorator_list", ())]
    first_line = min([getattr(node, "lineno", 0), *decorator_lines])
    last_line = getattr(node, "end_lineno", 0)
    return first_line <= line_number <= last_line


def _ast_source_segment(source: str, node: ast.AST) -> str:
    """Extract a declaration including decorators from the authoritative cell."""

    lines = source.splitlines(keepends=True)
    decorator_lines = [decorator.lineno for decorator in getattr(node, "decorator_list", ())]
    first_line = min([getattr(node, "lineno", 0), *decorator_lines])
    last_line = getattr(node, "end_lineno", 0)
    if first_line < 1 or last_line < first_line or last_line > len(lines):
        raise OSError("notebook declaration has invalid source coordinates")
    return "".join(lines[first_line - 1:last_line])


def _normalized_notebook_source(source: str) -> str:
    return unicodedata.normalize("NFC", inspect.cleandoc(source).replace("\r\n", "\n").replace("\r", "\n"))


def _normalize_distribution_name(name: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", _nonempty(name, "distribution")).lower()
    return normalized


def _sha256_or_git(value: str) -> str:
    # The full V1 schema allows Git SHA-1 or SHA-256 object IDs.  Do not force
    # the current checkout's Git object format into the evidence protocol.
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise _validation("Git commit is not a lowercase object ID")
    return value
