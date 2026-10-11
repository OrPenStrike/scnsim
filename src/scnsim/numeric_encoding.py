"""Single exact numeric-record codec shared by computation and evidence readers.

Owns binary64 scalar tags and actual-dtype row-major array representation only;
no request, Workspace, runtime discovery or numerical execution responsibility.
"""
from __future__ import annotations
import json
import struct
from collections.abc import Mapping
import numpy as np
from .canonical import canonical_json_bytes, float64_from_hex

def record_data(value):
    """Encode measurement scalars explicitly; never round physical boundaries."""
    if isinstance(value, (float, np.floating)):
        return {"f64": struct.pack(">d", float(value)).hex()}
    if isinstance(value, Mapping):
        return {key: record_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [record_data(item) for item in value]
    return value


def record_bytes(value: dict) -> bytes:
    return canonical_json_bytes(record_data(value))


def decode_record(value):
    if isinstance(value, dict):
        if set(value) == {"f64"}:
            return struct.unpack(">d", bytes.fromhex(value["f64"]))[0]
        return {key: decode_record(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_record(item) for item in value]
    return value


def record_document(value: bytes) -> dict:
    return decode_record(json.loads(value))


def array_record(value: object) -> dict[str, object]:
    """Exact row-major numeric payload, preserving selected numerical precision."""
    source = np.asarray(value)
    complex_ = np.iscomplexobj(source)
    dtype = "<c8" if source.dtype == np.dtype("complex64") else "<f4" if source.dtype == np.dtype("float32") else "<c16" if complex_ else "<f8"
    array = np.asarray(source, dtype=dtype)
    return {
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "data_hex": array.tobytes(order="C").hex(),
    }


def array_from_record(record: dict[str, object]) -> np.ndarray:
    """Reconstruct immutable typed arrays; native decoding/shape errors propagate."""
    dtype = {"float32": "<f4", "float64": "<f8", "complex64": "<c8", "complex128": "<c16"}[record["dtype"]]
    return np.frombuffer(bytes.fromhex(record["data_hex"]), dtype=dtype).reshape(record["shape"])



def bits(value: float) -> str:
    """Numeric evidence may include infinity; unlike parameter identity encoding."""
    return struct.pack(">d", float(value)).hex()

def unbits(value: str) -> float:
    return struct.unpack(">d", bytes.fromhex(value))[0]

def complex_record(value: complex) -> dict:
    return {"real_f64": bits(value.real), "imag_f64": bits(value.imag)}

def complex_value(record: dict) -> complex:
    return complex(float64_from_hex(record["real_f64"]), float64_from_hex(record["imag_f64"]))
