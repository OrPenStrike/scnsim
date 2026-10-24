"""Evidence-v3 columnar records in immutable OperationStore byte objects.

Only storage encoding lives here. The surrounding record binds ordered ports,
coordinates, units and execution identities; each column also binds its exact
record path, dtype, physical shape and contiguous byte range. No solver runs
while encoding or reading a record.
"""
from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping

import numpy as np

from ..canonical import canonical_json_bytes
from ..errors import EvidenceIntegrityError, UnsupportedEvidenceVersionError
from ..numeric_encoding import _ARRAY_DTYPES, record_data, decode_record

MAGIC = b"SCNSIMB3"
_COLUMN = "$scnsim_column_v3"
_HEADER_KEYS = {"schema", "schema_version", "record", "columns", "context", "chunk_ordinal"}
_COLUMN_KEYS = {"ordinal", "path", "dtype", "byte_order", "shape", "order",
                "axes", "unit", "complex_representation", "offset", "byte_length", "representation"}


def _error(message, **evidence):
    return EvidenceIntegrityError(message, stage="evidence_v3", evidence=evidence)


def _array(value):
    if isinstance(value, np.ndarray):
        dtype = _ARRAY_DTYPES[value.dtype.name]
        source = np.asarray(value, dtype=dtype)
        return source.dtype.name, list(source.shape), source.tobytes(order="C")
    if isinstance(value, Mapping) and {"dtype", "shape", "data_hex"} <= set(value):
        if set(value) != {"dtype", "shape", "data_hex"}:
            raise _error("Numeric array record has unknown fields.")
        name = value["dtype"]
        if not isinstance(name, str) or name not in _ARRAY_DTYPES:
            raise _error("Numeric column dtype is unsupported.", dtype=name)
        shape = value["shape"]
        if not isinstance(shape, (tuple, list)) or any(type(n) is not int or n < 0 for n in shape):
            raise _error("Numeric column shape is malformed.")
        try:
            payload = bytes.fromhex(value["data_hex"])
        except (TypeError, ValueError) as error:
            raise _error("Numeric column bytes are malformed.") from error
        if len(payload) != math.prod(shape) * np.dtype(_ARRAY_DTYPES[name]).itemsize:
            raise _error("Numeric column byte length differs from its shape.")
        return name, list(shape), payload
    return None


def _unit(path):
    if path and path[-1] == "frequencies_hz":
        return "hertz"
    if len(path) >= 2 and path[-2] == "families":
        return {"S": "dimensionless", "Y": "siemens", "Z": "ohm"}.get(path[-1])
    return None


def _context(record):
    if not isinstance(record, dict):
        return {}
    return {key: record[key] for key in (
        "workspace_instance_id", "plan_sha256", "request_sha256", "task_id",
        "attempt_id", "operation_id", "array_context") if key in record}


def record_bytes(value):
    """Encode array columns literally, retaining the scalar identity codec."""
    columns, payloads = [], []
    offset = 0

    def visit(item, path):
        nonlocal offset
        array = _array(item)
        representation = "array_record"
        if (array is None and path and path[-1] == "frequencies_hz"
                and isinstance(item, (list, tuple))
                and all(isinstance(value, (float, np.floating)) for value in item)):
            values = np.asarray(item, dtype="<f8")
            array = ("float64", [len(values)], values.tobytes(order="C"))
            representation = "scalar_sequence"
        if array is not None:
            name, shape, payload = array
            ordinal = len(columns)
            columns.append({"ordinal": ordinal, "path": path, "dtype": name,
                            "byte_order": "little", "shape": shape, "order": "C",
                            "axes": list(range(len(shape))), "unit": _unit(path),
                            "complex_representation": "interleaved_real_imag" if name.startswith("complex") else "not_complex",
                            "offset": offset, "byte_length": len(payload),
                            "representation": representation})
            offset += len(payload)
            payloads.append(payload)
            return {_COLUMN: ordinal}
        if isinstance(item, Mapping):
            if _COLUMN in item:
                raise _error("Record contains a reserved numeric column marker.")
            return {key: visit(item[key], [*path, key]) for key in sorted(item)}
        if isinstance(item, (list, tuple)):
            return [visit(child, [*path, index]) for index, child in enumerate(item)]
        return record_data(item)

    record = visit(value, [])
    if not columns:
        return canonical_json_bytes(record)
    header = canonical_json_bytes({"schema": "scnsim.numeric_block", "schema_version": 3,
                                   "record": record, "columns": columns,
                                   "context": _context(record),
                                   "chunk_ordinal": record.get("ordinal") if isinstance(record, dict) else None})
    return b"".join((MAGIC, struct.pack("<Q", len(header)), header, *payloads))


def _current_manifest(value):
    if (isinstance(value, dict) and value.get("schema") in {
            "scnsim.direct_manifest", "scnsim.sweep_manifest",
            "scnsim.sensitivity_manifest", "scnsim.optimization_manifest"}
            and value.get("evidence_version") != 3):
        raise UnsupportedEvidenceVersionError(
            "Manifest evidence was written by a pre-1.0 development build that 1.0 does not support; use a new Workspace and recompute.",
            stage="evidence_v3", evidence={"supported_evidence_version": 3})
    return value


def record_document(payload):
    """Verify framing and reconstruct the existing exact typed record leaves."""
    if not payload.startswith(MAGIC):
        try:
            value = json.loads(payload)
        except (ValueError, UnicodeError) as error:
            raise _error("Evidence record framing is malformed.") from error
        def has_legacy_array(item):
            if isinstance(item, dict):
                return ({"dtype", "shape", "data_hex"} <= set(item)
                        or any(has_legacy_array(child) for child in item.values()))
            if isinstance(item, list):
                return any(has_legacy_array(child) for child in item)
            return False
        if has_legacy_array(value):
            raise UnsupportedEvidenceVersionError(
                "JSON numerical array evidence was written by a pre-1.0 development build that 1.0 does not support; use a new Workspace and recompute.",
                stage="evidence_v3", evidence={"supported_evidence_version": 3})
        return _current_manifest(decode_record(value))
    if len(payload) < 16:
        raise _error("Numeric block header is truncated.")
    header_length = struct.unpack("<Q", payload[8:16])[0]
    boundary = 16 + header_length
    if boundary > len(payload):
        raise _error("Numeric block header is truncated.")
    try:
        header = json.loads(payload[16:boundary])
    except (ValueError, UnicodeError) as error:
        raise _error("Numeric block header is malformed.") from error
    if (not isinstance(header, dict) or set(header) != _HEADER_KEYS
            or header["schema"] != "scnsim.numeric_block" or header["schema_version"] != 3
            or canonical_json_bytes(header) != payload[16:boundary]
            or not isinstance(header["columns"], list) or not header["columns"]
            or (header["chunk_ordinal"] is not None and
                (type(header["chunk_ordinal"]) is not int or header["chunk_ordinal"] < 0))
            or header["context"] != _context(header["record"])
            or header["chunk_ordinal"] != (header["record"].get("ordinal")
                                           if isinstance(header["record"], dict) else None)):
        raise _error("Numeric block header is not a closed canonical v3 header.")
    columns, offset = header["columns"], 0
    arrays = []
    for ordinal, column in enumerate(columns):
        if not isinstance(column, dict) or set(column) != _COLUMN_KEYS:
            raise _error("Numeric column header is open or malformed.")
        name, shape = column["dtype"], column["shape"]
        if (not isinstance(column["path"], list)
                or any(type(key) not in (str, int) for key in column["path"])):
            raise _error("Numeric column path is malformed.")
        if (not isinstance(name, str) or name not in _ARRAY_DTYPES
                or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)):
            raise _error("Numeric column dtype or shape is malformed.")
        length = math.prod(shape) * np.dtype(_ARRAY_DTYPES[name]).itemsize
        if (type(column["ordinal"]) is not int or column["ordinal"] != ordinal
                or type(column["offset"]) is not int or column["offset"] != offset
                or type(column["byte_length"]) is not int or column["byte_length"] != length
                or column["representation"] not in {"array_record", "scalar_sequence"}
                or (column["representation"] == "scalar_sequence" and
                    (name != "float64" or len(shape) != 1 or not column["path"]
                     or column["path"][-1] != "frequencies_hz"))
                or column["byte_order"] != "little" or column["order"] != "C"
                or column["axes"] != list(range(len(shape))) or column["unit"] != _unit(column["path"])
                or column["complex_representation"] != ("interleaved_real_imag" if name.startswith("complex") else "not_complex")
                or not isinstance(column["path"], list)
                or any(type(key) not in (str, int) for key in column["path"])):
            raise _error("Numeric column layout is malformed.", ordinal=ordinal)
        end = boundary + offset + length
        if end > len(payload):
            raise _error("Numeric column payload is truncated.", ordinal=ordinal)
        arrays.append({"dtype": name, "shape": shape,
                       "data_hex": payload[boundary + offset:end].hex()})
        offset += length
    if boundary + offset != len(payload):
        raise _error("Numeric block has trailing bytes.")
    seen = []

    def hydrate(item, path):
        if isinstance(item, dict):
            if _COLUMN in item:
                ordinal = item[_COLUMN]
                if (set(item) != {_COLUMN} or type(ordinal) is not int
                        or ordinal != len(seen) or ordinal >= len(columns)
                        or columns[ordinal]["path"] != path):
                    raise _error("Numeric column occurrence differs from its header.")
                seen.append(ordinal)
                if columns[ordinal]["representation"] == "scalar_sequence":
                    array = arrays[ordinal]
                    return np.frombuffer(bytes.fromhex(array["data_hex"]), dtype="<f8").tolist()
                return arrays[ordinal]
            return {key: hydrate(item[key], [*path, key]) for key in sorted(item)}
        if isinstance(item, list):
            return [hydrate(child, [*path, index]) for index, child in enumerate(item)]
        return item

    result = hydrate(header["record"], [])
    if len(seen) != len(columns):
        raise _error("Numeric block contains an unreferenced column.")
    return _current_manifest(decode_record(result))
