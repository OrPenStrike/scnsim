"""Shared schema leaves and value verification for stored workspace evidence."""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Mapping
from datetime import datetime, timezone

from ...errors import EvidenceIntegrityError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_IDENTIFIER = re.compile(r"^[^/\\\x00-\x1f\x7f]+$")
_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)

def _integrity(message: str, **evidence: object) -> EvidenceIntegrityError:
    return EvidenceIntegrityError(message, stage="workspace", evidence=evidence)

def _valid_utc_timestamp(value: object) -> bool:
    """Recognize an ISO-8601 date-time with the schema's exact UTC spelling."""

    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo == timezone.utc

def _valid_sha(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _integrity("Expected a lowercase SHA-256 digest.", value=value)
    return value

def _valid_uuid(value: object) -> str:
    if not isinstance(value, str) or _UUID4.fullmatch(value) is None:
        raise _integrity("Expected a lowercase canonical UUIDv4.", value=value)
    return value

def _identifiers(value: object, *, field: str, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value) or any(
        not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in value
    ):
        raise _integrity(f"{field} is not an ordered identifier array.")
    if len(set(value)) != len(value):
        raise _integrity(f"{field} repeats an identifier.")
    return list(value)

def _parameter_key_integrity(value: object) -> tuple[str, str]:
    if not isinstance(value, dict) or set(value) != {"definitions_id", "parameter_id"}:
        raise _integrity("ParameterRef is malformed.")
    definitions = value.get("definitions_id"); identifier = value.get("parameter_id")
    if not isinstance(definitions, str) or _IDENTIFIER.fullmatch(definitions) is None or not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
        raise _integrity("ParameterRef identity is malformed.")
    return definitions, identifier

def _verify_branch_refs(value: object, *, field: str, nonempty: bool) -> None:
    if not isinstance(value, list) or (nonempty and not value):
        raise _integrity(f"{field} is malformed.")
    keys: list[tuple[tuple[str, ...], str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"component_path", "branch_id"}:
            raise _integrity(f"{field} has an open branch identity.")
        path, branch = item.get("component_path"), item.get("branch_id")
        if not isinstance(path, list) or not path or any(not isinstance(segment, str) or _IDENTIFIER.fullmatch(segment) is None for segment in path) or not isinstance(branch, str) or _IDENTIFIER.fullmatch(branch) is None:
            raise _integrity(f"{field} has a malformed branch identity.")
        keys.append((tuple(path), branch))
    if keys != sorted(set(keys)):
        raise _integrity(f"{field} is not sorted and unique.")

def _verify_bounds(value: object) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise _integrity("Optimization bounds are malformed.")
    _verify_quantity_compatible(value[0], value[1])
    if _f64_value(value[0]["si_value_f64"]) >= _f64_value(value[1]["si_value_f64"]):
        raise _integrity("Optimization bounds are not ordered.")

def _verify_quantity_compatible(left: object, right: object) -> None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise _integrity("Quantity pair is malformed.")
    unit, dimensionality = left.get("si_unit"), left.get("dimensionality")
    _verify_quantity_role(left, complex_value=False, unit=unit, dimensionality=dimensionality)
    _verify_quantity_role(right, complex_value=False, unit=unit, dimensionality=dimensionality)

def _verify_quantity_role(value: object, *, complex_value: bool, unit: str, dimensionality: str) -> None:
    if not isinstance(value, dict):
        raise _integrity("Typed quantity Result field is not an object.")
    magnitude_fields = {"real_si_f64", "imag_si_f64"} if complex_value else {"si_value_f64"}
    expected_type = "complex_quantity_f64" if complex_value else "quantity_f64"
    if (
        set(value) != {"type", "si_unit", "dimensionality"} | magnitude_fields
        or value.get("type") != expected_type
        or value.get("si_unit") != unit
        or value.get("dimensionality") != dimensionality
        or any(not _finite_f64(value[field]) for field in magnitude_fields)
    ):
        raise _integrity("Typed quantity Result field has the wrong physical role.")

def _complex_quantity_value(value: object, *, unit: str, dimensionality: str) -> complex:
    _verify_quantity_role(
        value, complex_value=True, unit=unit, dimensionality=dimensionality
    )
    assert isinstance(value, Mapping)
    return complex(
        _f64_value(value["real_si_f64"]),
        _f64_value(value["imag_si_f64"]),
    )

def _verify_quantity_any(value: object) -> None:
    if not isinstance(value, dict) or value.get("type") != "quantity_f64":
        raise _integrity("Typed quantity is malformed.")
    unit, dimensionality = value.get("si_unit"), value.get("dimensionality")
    if not isinstance(unit, str) or not isinstance(dimensionality, str):
        raise _integrity("Typed quantity has no physical role.")
    if (unit, dimensionality) not in _canonical_quantity_roles():
        raise _integrity("Typed quantity uses a closed-vocabulary-invalid physical role.")
    _verify_quantity_role(value, complex_value=False, unit=unit, dimensionality=dimensionality)

def _finite_f64(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{16}", value) is not None
        and math.isfinite(struct.unpack(">d", bytes.fromhex(value))[0])
    )

def _canonical_quantity_roles() -> frozenset[tuple[str, str]]:
    """Reuse the identity schema's closed SI-unit/dimensionality vocabulary."""

    from ...canonical import _UNITS

    return frozenset(_UNITS.items())

def _verify_parameter_set_document(value: object, *, require_empty_authorization: bool = False) -> None:
    if not isinstance(value, dict) or set(value) != {"type", "bindings", "allow_extrapolation"} or value.get("type") != "parameter_set_v2":
        raise _integrity("ParameterSet envelope is open or malformed.")
    bindings = value.get("bindings")
    authorizations = value.get("allow_extrapolation")
    if not isinstance(bindings, list) or not isinstance(authorizations, list):
        raise _integrity("ParameterSet arrays are malformed.")
    keys: list[tuple[str, str]] = []
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {"parameter", "value"}:
            raise _integrity("ParameterSet binding is open or malformed.")
        reference = binding.get("parameter")
        keys.append(_parameter_key_integrity(reference))
        _verify_parameter_value(binding.get("value"))
    if keys != sorted(set(keys)):
        raise _integrity("ParameterSet bindings are not sorted and unique.")
    authorization_keys: list[tuple[str, str]] = []
    for reference in authorizations:
        authorization_keys.append(_parameter_key_integrity(reference))
    if authorization_keys != sorted(set(authorization_keys)) or any(key not in keys for key in authorization_keys):
        raise _integrity("ParameterSet authorizations are not sorted active references.")
    if require_empty_authorization and authorization_keys:
        raise _integrity("Optimization candidate ParameterSet inherited extrapolation authorization.")

def _verify_parameter_value(value: object) -> None:
    if isinstance(value, dict) and value.get("type") == "quantity_f64":
        _verify_quantity_any(value)
        return
    matrix_fields = {
        "resistance_per_length": ("ohm / meter", "resistance_per_length"),
        "inductance_per_length": ("henry / meter", "inductance_per_length"),
        "conductance_per_length": ("siemens / meter", "conductance_per_length"),
        "capacitance_per_length": ("farad / meter", "capacitance_per_length"),
    }
    required = {
        "type", "conductors", "reference_conductor", "orientation", "source",
        *matrix_fields, "extraction_frequency",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("type") != "rlgc":
        raise _integrity("ParameterSet value has an unsupported physical role.")
    conductors = value.get("conductors")
    if (
        not isinstance(conductors, list)
        or not conductors
        or any(not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in conductors)
        or len(set(conductors)) != len(conductors)
        or not isinstance(value.get("reference_conductor"), str)
        or _IDENTIFIER.fullmatch(value["reference_conductor"]) is None
        or value["reference_conductor"] in conductors
        or value.get("orientation") != "extractor_positive_z_is_head_to_tail"
        or not isinstance(value.get("source"), dict)
    ):
        raise _integrity("RLGC parameter basis or provenance is malformed.")
    size = len(conductors)
    for field, (unit, dimensionality) in matrix_fields.items():
        matrix = value[field]
        if (
            not isinstance(matrix, dict)
            or set(matrix) != {"type", "shape", "values_f64", "si_unit", "dimensionality"}
            or matrix.get("type") != "quantity_matrix_f64"
            or matrix.get("shape") != [size, size]
            or matrix.get("si_unit") != unit
            or matrix.get("dimensionality") != dimensionality
            or not isinstance(matrix.get("values_f64"), list)
            or len(matrix["values_f64"]) != size * size
            or any(not _finite_f64(item) for item in matrix["values_f64"])
        ):
            raise _integrity("RLGC parameter matrix is malformed.")
    frequency = value.get("extraction_frequency")
    if frequency is not None:
        _verify_quantity_role(
            frequency, complex_value=False, unit="hertz", dimensionality="inverse_time"
        )

def _verify_extrapolation_evidence(
    value: object,
    *,
    allowed_sources: set[str],
    required_rows: list[dict[str, object]] | None = None,
) -> None:
    """Validate one evidence row per explicitly out-of-support fan-out edge."""

    if not isinstance(value, list):
        raise _integrity("Extrapolation evidence is not an array.")
    keys: list[tuple[tuple[str, str], tuple[tuple[str, ...], str]]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"parameter", "consumer_target", "support", "input_value", "side", "distance", "authorization_source"}:
            raise _integrity("Extrapolation evidence row is malformed.")
        parameter = _parameter_key_integrity(row.get("parameter"))
        target_record = row.get("consumer_target")
        if not isinstance(target_record, Mapping) or set(target_record) != {"path", "field"}:
            raise _integrity("Extrapolation consumer target is malformed.")
        path, field = target_record.get("path"), target_record.get("field")
        if not isinstance(path, list) or not path or any(not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in path) or not isinstance(field, str) or _IDENTIFIER.fullmatch(field) is None:
            raise _integrity("Extrapolation consumer target is malformed.")
        target = (tuple(path), field)
        support = row.get("support")
        if not isinstance(support, list) or len(support) != 2:
            raise _integrity("Extrapolation support interval is malformed.")
        _verify_quantity_compatible(support[0], support[1])
        _verify_quantity_compatible(support[0], row.get("input_value"))
        _verify_quantity_compatible(support[0], row.get("distance"))
        lower, upper = _f64_value(support[0]["si_value_f64"]), _f64_value(support[1]["si_value_f64"])
        input_value = _f64_value(row["input_value"]["si_value_f64"])
        distance = _f64_value(row["distance"]["si_value_f64"])
        side = row.get("side")
        expected = lower - input_value if side == "lower" else input_value - upper if side == "upper" else None
        if lower >= upper or expected is None or expected <= 0.0 or distance <= 0.0 or struct.pack(">d", expected).hex() != row["distance"].get("si_value_f64"):
            raise _integrity("Extrapolation evidence does not reproduce its canonical distance.")
        if row.get("authorization_source") not in allowed_sources:
            raise _integrity("Extrapolation evidence has an unauthorized source.")
        keys.append((parameter, target))
    if keys != sorted(set(keys)):
        raise _integrity("Extrapolation evidence is not sorted and unique per fan-out edge.")
    if required_rows is not None and value != required_rows:
        raise _integrity("Extrapolation evidence omits or alters a required affine fan-out edge.")

def _required_extrapolation_rows(
    plan: Mapping[str, object],
    parameters: object,
    *,
    authorization_source: str,
    optimization_authorizations: object | None = None,
    require_authorized: bool = True,
) -> list[dict[str, object]]:
    """Derive affine support crossings from normalized physical-field bindings."""

    if authorization_source not in {"parameter_set", "optimization_spec"}:
        raise _integrity("Extrapolation evidence authority is unknown.")
    _verify_parameter_set_document(parameters)
    assert isinstance(parameters, Mapping)
    values = {
        _parameter_key_integrity(binding["parameter"]): binding["value"]
        for binding in parameters["bindings"]
    }
    raw_authorizations = (
        parameters["allow_extrapolation"]
        if authorization_source == "parameter_set"
        else optimization_authorizations
    )
    if not isinstance(raw_authorizations, list):
        raise _integrity("Extrapolation authorization collection is malformed.")
    authorized = {_parameter_key_integrity(item) for item in raw_authorizations}
    leaves = plan.get("physical_leaves")
    if not isinstance(leaves, list):
        raise _integrity("Plan physical-field inventory is malformed.")
    rows: list[dict[str, object]] = []
    from ...canonical import float64_hex

    for leaf in leaves:
        if not isinstance(leaf, Mapping) or not isinstance(leaf.get("path"), list) or not isinstance(leaf.get("fields"), list):
            raise _integrity("Plan physical leaf is malformed.")
        path = list(leaf["path"])
        for field in leaf["fields"]:
            binding = field.get("binding") if isinstance(field, Mapping) else None
            if not isinstance(binding, Mapping) or binding.get("kind") != "affine":
                continue
            if set(binding) != {"kind", "input", "slope", "intercept", "support"}:
                raise _integrity("Affine physical-field binding is malformed.")
            parameter = _parameter_key_integrity(binding["input"])
            input_value = values.get(parameter)
            if not isinstance(input_value, Mapping) or input_value.get("type") != "quantity_f64":
                raise _integrity("Affine input has no scalar resolved parameter value.")
            support = binding["support"]
            if not isinstance(support, list) or len(support) != 2:
                raise _integrity("Affine support interval is malformed.")
            _verify_quantity_compatible(support[0], support[1])
            _verify_quantity_compatible(support[0], input_value)
            lower = _f64_value(support[0]["si_value_f64"])
            upper = _f64_value(support[1]["si_value_f64"])
            selected = _f64_value(input_value["si_value_f64"])
            if lower >= upper:
                raise _integrity("Affine support interval is not ordered.")
            if lower <= selected <= upper:
                continue
            authority = authorization_source if parameter in authorized else "none"
            if require_authorized and authority == "none":
                raise _integrity("Successful request has unauthorized affine extrapolation.")
            side, distance = (
                ("lower", lower - selected)
                if selected < lower
                else ("upper", selected - upper)
            )
            distance_record = dict(input_value)
            distance_record["si_value_f64"] = float64_hex(distance)
            rows.append({
                "parameter": dict(binding["input"]),
                "consumer_target": {"path": path, "field": field["id"]},
                "support": [dict(support[0]), dict(support[1])],
                "input_value": dict(input_value),
                "side": side,
                "distance": distance_record,
                "authorization_source": authority,
            })
    rows.sort(key=lambda row: (
        _parameter_key_integrity(row["parameter"]),
        (tuple(row["consumer_target"]["path"]), row["consumer_target"]["field"]),
    ))
    return rows

def _verify_discretization(value: object, plan: Mapping[str, object]) -> None:
    leaves = [leaf for leaf in plan["physical_leaves"] if leaf["model"] == "transmission_line"]
    if value is None:
        if any("discretization" in leaf["model_metadata"] for leaf in leaves):
            raise _integrity("Electrical-resolution Result has no realized grid.")
        return  # Pre-policy fixed-count evidence remains readable.
    if not isinstance(value, list) or len(value) != len(leaves):
        raise _integrity("Result line-grid inventory disagrees with the Plan.")
    for row, leaf in zip(value, leaves):
        if not isinstance(row, dict) or row.get("component_path") != leaf["path"]:
            raise _integrity("Result line-grid path or order is malformed.")
        policy = leaf["model_metadata"].get("discretization")
        expected = {"component_path", "kind", "length", "n_sections", "dx"}
        expected |= {"policy", "modal_velocities", "hmax"} if policy is not None else set()
        if set(row) != expected or row.get("kind") != ("electrical_resolution" if policy is not None else "fixed_count"):
            raise _integrity("Result line-grid fields disagree with its declaration.")
        n = row.get("n_sections")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise _integrity("Result line-grid section count is invalid.")
        _verify_quantity_role(row.get("length"), complex_value=False, unit="meter", dimensionality="length")
        _verify_quantity_role(row.get("dx"), complex_value=False, unit="meter", dimensionality="length")
        length = _f64_value(row["length"]["si_value_f64"])
        dx = _f64_value(row["dx"]["si_value_f64"])
        if length <= 0 or dx <= 0 or dx != length / n:
            raise _integrity("Result line-grid length and step disagree.")
        if policy is None:
            if n != leaf["model_metadata"].get("n_sections"):
                raise _integrity("Fixed-count Result grid disagrees with the Plan.")
            continue
        if row["policy"] != policy:
            raise _integrity("Electrical-resolution Result policy disagrees with the Plan.")
        _verify_quantity_role(row["hmax"], complex_value=False, unit="meter", dimensionality="length")
        hmax = _f64_value(row["hmax"]["si_value_f64"])
        velocities = row["modal_velocities"]
        if not isinstance(velocities, list) or len(velocities) != len(leaf["pin_order"]) // 2 or hmax <= 0:
            raise _integrity("Electrical-resolution modal inventory is malformed.")
        for velocity in velocities:
            _verify_quantity_role(velocity, complex_value=False, unit="meter / second", dimensionality="velocity")
            if _f64_value(velocity["si_value_f64"]) <= 0:
                raise _integrity("Electrical-resolution modal velocity is invalid.")

def _f64_value(value: object) -> float:
    if not _finite_f64(value):
        raise _integrity("Expected one finite Float64 bit string.")
    return struct.unpack(">d", bytes.fromhex(str(value)))[0]
