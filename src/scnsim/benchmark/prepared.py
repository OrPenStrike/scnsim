"""Canonical experimental task declaration; no runtime discovery or launch.

The shared descriptor binds immutable Plan/request bytes and benchmark policy.
Arm/sample identity and actual environment evidence are added by the task owner.
Configured julia_threads/julia_blas_threads remain integer or null in the
canonical policy; the task owner binds their effective per-quota native counts.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from hashlib import sha256

import numpy as np

from ..canonical import canonical_json_bytes, float64_hex, float64_from_hex
from ..execution.prepared import PreparedAnalysis
from ..authoring.identity import canonical_parameter_set
from .models import BenchmarkSpec, MeshGroup, MeshSpec


def mesh_from_record(record: dict) -> MeshSpec:
    sections = lambda items: tuple((tuple(path), count) for path, count in items)
    return MeshSpec(
        kind=record["kind"], sections=sections(record["sections"]),
        parameter_key=tuple(record["parameter_key"]) if record["parameter_key"] is not None else None,
        groups=tuple(MeshGroup(float64_from_hex(g["lower_f64"]), float64_from_hex(g["upper_f64"]), g["lower_inclusive"], g["upper_inclusive"], sections(g["sections"]))
                     for g in record["groups"]),
        derivation_bytes=canonical_json_bytes(record["derivation"]),
    )


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
    """Exact row-major little-endian binary64 wire payload, without pickle."""
    source = np.asarray(value)
    complex_ = np.iscomplexobj(source)
    array = np.asarray(source, dtype="<c16" if complex_ else "<f8")
    return {
        "dtype": "complex128" if complex_ else "float64",
        "shape": list(array.shape),
        "data_hex": array.tobytes(order="C").hex(),
    }


def array_from_record(record: dict[str, object]) -> np.ndarray:
    """Reconstruct immutable typed arrays; native decoding/shape errors propagate."""
    dtype = {"float64": "<f8", "complex128": "<c16"}[record["dtype"]]
    return np.frombuffer(bytes.fromhex(record["data_hex"]), dtype=dtype).reshape(record["shape"])


@dataclass(frozen=True, slots=True)
class PreparedBenchmark:
    plan_bytes: bytes
    analysis: PreparedAnalysis
    declaration_bytes: bytes

    @classmethod
    def create(
        cls,
        *,
        plan_bytes: bytes,
        analysis: PreparedAnalysis,
        benchmark: BenchmarkSpec,
    ) -> PreparedBenchmark:
        policy = asdict(benchmark)
        policy["cohort"] = [canonical_parameter_set(point) for point in json.loads(policy.pop("cohort_bytes"))]
        mesh = policy["mesh"]
        mesh["derivation"] = json.loads(mesh.pop("derivation_bytes"))
        for group in mesh["groups"]:
            group["lower_f64"] = float64_hex(group.pop("lower"))
            group["upper_f64"] = float64_hex(group.pop("upper"))
        original = analysis.request()
        semantic_analysis = {
            key: original[key]
            for key in ("operation", "view", "spec", "parameter_source")
        }
        if semantic_analysis["operation"] == "optimize_direct":
            optimizer = semantic_analysis["spec"]["optimizer"]
            optimizer["box_transform_id"] = (
                "scnsim-python-cmaes-0.13.1-native-domains.v1"
                if optimizer["box_transform_id"].endswith("native-domains.v1")
                else "scnsim-python-cmaes-0.13.1-linquad-unit-box.v1"
            )
        declaration = {
            "schema": "scnsim.benchmark_request",
            "schema_version": 1,
            "plan_sha256": sha256(plan_bytes).hexdigest(),
            "analysis": semantic_analysis,
            "source_analysis_sha256": analysis.request_sha256,
            "benchmark": policy,
        }
        return cls(plan_bytes, analysis, record_bytes(declaration))

    @property
    def request_sha256(self) -> str:
        return sha256(self.declaration_bytes).hexdigest()

    def declaration(self) -> dict[str, object]:
        return json.loads(self.declaration_bytes)

    def wire(self) -> dict[str, object]:
        """Canonical process handoff of real immutable bytes, with no live Run."""
        return {"plan_hex": self.plan_bytes.hex(), "declaration_hex": self.declaration_bytes.hex(),
                "analysis_request_hex": self.analysis.request_bytes.hex(),
                "source_unit_hex": [value.hex() for value in self.analysis.source_unit_bytes]}

    @classmethod
    def from_wire(cls, record: dict) -> PreparedBenchmark:
        analysis = PreparedAnalysis(bytes.fromhex(record["analysis_request_hex"]),
                                    tuple(bytes.fromhex(value) for value in record["source_unit_hex"]))
        return cls(bytes.fromhex(record["plan_hex"]), analysis, bytes.fromhex(record["declaration_hex"]))
