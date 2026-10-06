"""Scientific result artifact catalog and quantity evidence verification."""

from __future__ import annotations

import struct
from collections.abc import Mapping

from .common import (
    _SHA256,
    _complex_quantity_value,
    _f64_value,
    _identifiers,
    _integrity,
    _valid_sha,
    _verify_quantity_role,
)

def _verify_root_like_result(result: Mapping[str, object], fields: set[str]) -> None:
    common = {"schema", "schema_version", "result_kind", "request_sha256", "attempt_sha256"}
    if set(result) != common | {"scalar_catalog", "array_catalog"}:
        raise _integrity("Root Result envelope is malformed.")
    scalars = result.get("scalar_catalog")
    if not isinstance(scalars, dict) or set(scalars) != fields:
        raise _integrity("Root scalar catalog is incomplete.")
    _verify_quantity_role(scalars["root"], complex_value=True, unit="radian / second", dimensionality="inverse_time")
    _verify_quantity_role(scalars["frequency"], complex_value=False, unit="hertz", dimensionality="inverse_time")
    _verify_quantity_role(scalars["linewidth"], complex_value=False, unit="hertz", dimensionality="inverse_time")
    _verify_quantity_role(scalars["slope"], complex_value=True, unit="siemens", dimensionality="conductance")
    _valid_sha(scalars["evidence_sha256"])

def _verify_null_vector_artifact(value: object, expected_coordinates: list[str]) -> None:
    if not isinstance(value, dict):
        raise _integrity("Hybridized-pole null-vector artifact is malformed.")
    common = {
        "id", "path", "sha256", "media_type", "file_manifest", "dtype", "shape",
        "chunks", "complex_storage", "group_metadata", "datasets", "axes", "unit",
        "dimensionality", "chunk_policy", "coordinate_ids",
    }
    if (
        set(value) != common
        or value.get("id") != "null_vector"
        or value.get("path") != "artifacts/null_vector.zarr"
        or value.get("file_manifest") != "artifacts/null_vector.manifest.json"
        or _SHA256.fullmatch(str(value.get("sha256", ""))) is None
        or value.get("media_type") != "application/vnd+zarr-v2"
        or value.get("dtype") != "complex128"
        or value.get("complex_storage") != "paired_float64_real_imag"
        or value.get("group_metadata") != {"zarr_format": 2}
        or value.get("unit") != "dimensionless"
        or value.get("dimensionality") != "dimensionless"
        or value.get("chunk_policy") != "single_complete_array_v1"
    ):
        raise _integrity("Hybridized-pole null-vector artifact is malformed.")
    coordinates = value.get("coordinate_ids"); shape = value.get("shape"); chunks = value.get("chunks")
    if (
        not isinstance(coordinates, list)
        or len(coordinates) < 2
        or any(not isinstance(item, str) or not item for item in coordinates)
        or len(set(coordinates)) != len(coordinates)
        or coordinates != expected_coordinates
        or shape != [len(coordinates)]
        or chunks != [len(coordinates)]
        or value.get("axes") != [{"id": "retained_coordinate", "kind": "coordinate", "values": coordinates}]
    ):
        raise _integrity("Hybridized-pole null-vector ordering is malformed.")
    _verify_zarr_datasets(value.get("datasets"), shape=shape, chunks=chunks, names=["real", "imag"])

def _expected_probe_load_state(lineage: object) -> list[dict[str, str]]:
    if not isinstance(lineage, Mapping) or not isinstance(lineage.get("original"), Mapping):
        raise _integrity("View lineage has no original Port order.")
    ports = _identifiers(lineage["original"].get("port_order"), field="Original Port order", nonempty=False)
    ptc = lineage.get("ptc")
    selected = set() if ptc is None else set(_identifiers(ptc.get("selected_ports"), field="PTC selected Ports"))
    return [
        {"port_id": port, "state": "compensated" if port in selected else "raw"}
        for port in ports
    ]

def _verify_operator_artifact(
    value: object,
    frequency_count: int,
    expected_coordinates: list[str],
    expected_probes: list[dict[str, str]],
) -> None:
    if not isinstance(value, dict):
        raise _integrity("Operator artifact is malformed.")
    common = {
        "id", "path", "sha256", "media_type", "file_manifest", "dtype", "shape",
        "chunks", "complex_storage", "group_metadata", "datasets", "axes", "unit",
        "dimensionality", "chunk_policy", "coordinate_ids", "probe_load_state",
    }
    if (
        set(value) != common
        or value.get("id") != "operator"
        or value.get("path") != "artifacts/operator.zarr"
        or value.get("file_manifest") != "artifacts/operator.manifest.json"
        or _SHA256.fullmatch(str(value.get("sha256", ""))) is None
        or value.get("media_type") != "application/vnd+zarr-v2"
        or value.get("dtype") != "complex128"
        or value.get("complex_storage") != "paired_float64_real_imag"
        or value.get("group_metadata") != {"zarr_format": 2}
        or value.get("unit") != "siemens / second"
        or value.get("dimensionality") != "conductance_per_time"
        or value.get("chunk_policy") != "frequency_slab_full_matrix_v1"
    ):
        raise _integrity("Operator artifact is malformed.")
    coordinates = value.get("coordinate_ids"); shape = value.get("shape"); chunks = value.get("chunks")
    if (
        not isinstance(coordinates, list)
        or not coordinates
        or any(not isinstance(item, str) or not item for item in coordinates)
        or len(set(coordinates)) != len(coordinates)
        or coordinates != expected_coordinates
        or shape != [frequency_count, len(coordinates), len(coordinates)]
        or chunks != [min(frequency_count, 1024), len(coordinates), len(coordinates)]
        or value.get("axes") != [
            {"id": "frequency", "kind": "frequency", "artifact_id": "frequencies"},
            {"id": "row_coordinate", "kind": "row_coordinate", "values": coordinates},
            {"id": "column_coordinate", "kind": "column_coordinate", "values": coordinates},
        ]
    ):
        raise _integrity("Operator artifact axes are malformed.")
    probes = value.get("probe_load_state")
    if (
        not isinstance(probes, list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"port_id", "state"}
            or not isinstance(item.get("port_id"), str)
            or not item["port_id"]
            or item.get("state") not in {"raw", "compensated"}
            for item in probes
        )
        or probes != expected_probes
    ):
        raise _integrity("Operator artifact probe-load state is malformed.")
    _verify_zarr_datasets(value.get("datasets"), shape=shape, chunks=chunks, names=["real", "imag"])

def _verify_zarr_datasets(
    value: object,
    *,
    shape: object,
    chunks: object,
    names: list[str],
) -> None:
    """Close the shared no-codec Zarr V2 metadata contract."""

    if not isinstance(value, list) or [item.get("path") if isinstance(item, dict) else None for item in value] != names:
        raise _integrity("Zarr artifact datasets are malformed.")
    expected = {
        "zarr_format": 2, "shape": shape, "chunks": chunks, "dtype": "<f8",
        "compressor": None, "fill_value": None, "order": "C", "filters": None,
        "dimension_separator": ".",
    }
    for dataset in value:
        if not isinstance(dataset, dict) or set(dataset) != {"path", "metadata"} or dataset.get("metadata") != expected:
            raise _integrity("Zarr artifact dataset metadata is malformed.")

def _verify_direct_artifact(value: object, role: str) -> int:
    if not isinstance(value, dict):
        raise _integrity("Direct artifact catalog entry is not an object.", artifact_id=role)
    common = {
        "id", "path", "sha256", "media_type", "file_manifest", "dtype", "shape",
        "chunks", "complex_storage", "group_metadata", "datasets", "axes", "unit",
        "dimensionality", "chunk_policy",
    }
    matrix = role != "frequencies"
    expected = common | ({"coordinate_ids", "probe_load_state"} if matrix else set())
    role_units = {
        "frequencies": ("hertz", "inverse_time"),
        "s": ("dimensionless", "dimensionless"),
        "y": ("siemens", "conductance"),
        "z": ("ohm", "resistance"),
    }
    if (
        set(value) != expected
        or value.get("id") != role
        or value.get("path") != f"artifacts/{role}.zarr"
        or value.get("file_manifest") != f"artifacts/{role}.manifest.json"
        or value.get("media_type") != "application/vnd+zarr-v2"
        or value.get("group_metadata") != {"zarr_format": 2}
        or (value.get("unit"), value.get("dimensionality")) != role_units[role]
    ):
        raise _integrity("Direct artifact has the wrong catalog role.", artifact_id=role)
    shape = value.get("shape")
    chunks = value.get("chunks")
    if matrix:
        valid_shape = (
            isinstance(shape, list) and len(shape) == 3
            and all(isinstance(item, int) and not isinstance(item, bool) and item >= 1 for item in shape)
            and shape[1] == shape[2]
        )
        valid_chunks = bool(valid_shape and isinstance(chunks, list) and chunks == [min(shape[0], 1024), shape[1], shape[2]])
        valid_storage = value.get("dtype") == "complex128" and value.get("complex_storage") == "paired_float64_real_imag" and value.get("chunk_policy") == "frequency_slab_full_matrix_v1"
        coordinates = value.get("coordinate_ids")
        probes = value.get("probe_load_state")
        if not isinstance(coordinates, list) or len(coordinates) != shape[1] or any(not isinstance(item, str) or not item for item in coordinates) or len(set(coordinates)) != len(coordinates):
            raise _integrity("Direct matrix coordinate catalog is invalid.", artifact_id=role)
        if not isinstance(probes, list) or any(not isinstance(item, dict) or set(item) != {"port_id", "state"} or item.get("state") not in {"raw", "compensated"} for item in probes):
            raise _integrity("Direct matrix probe-load catalog is invalid.", artifact_id=role)
        valid_axes = value.get("axes") == [
            {"id": "frequency", "kind": "frequency", "artifact_id": "frequencies"},
            {"id": "output_coordinate", "kind": "coordinate_output", "values": coordinates},
            {"id": "input_coordinate", "kind": "coordinate_input", "values": coordinates},
        ]
        dataset_names = ["real", "imag"]
    else:
        valid_shape = isinstance(shape, list) and len(shape) == 1 and isinstance(shape[0], int) and not isinstance(shape[0], bool) and shape[0] >= 1
        valid_chunks = bool(valid_shape and isinstance(chunks, list) and chunks == [min(shape[0], 1024)])
        valid_storage = value.get("dtype") == "float64" and value.get("complex_storage") == "real" and value.get("chunk_policy") == "frequency_capped_1024_v1"
        valid_axes = value.get("axes") == [{"id": "frequency", "kind": "frequency", "artifact_id": "frequencies"}]
        dataset_names = ["values"]
    if not (valid_shape and valid_chunks and valid_storage and valid_axes):
        raise _integrity("Direct artifact shape, storage, or axes are invalid.", artifact_id=role)
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or [item.get("path") if isinstance(item, dict) else None for item in datasets] != dataset_names:
        raise _integrity("Direct artifact datasets are invalid.", artifact_id=role)
    metadata_expected = {
        "zarr_format", "shape", "chunks", "dtype", "compressor", "fill_value",
        "order", "filters", "dimension_separator",
    }
    for dataset in datasets:
        if set(dataset) != {"path", "metadata"} or not isinstance(dataset.get("metadata"), dict):
            raise _integrity("Direct dataset envelope is open.", artifact_id=role)
        metadata = dataset["metadata"]
        if set(metadata) != metadata_expected or metadata != {
            "zarr_format": 2, "shape": shape, "chunks": chunks, "dtype": "<f8",
            "compressor": None, "fill_value": None, "order": "C", "filters": None,
            "dimension_separator": ".",
        }:
            raise _integrity("Direct dataset metadata disagrees with its artifact.", artifact_id=role)
    return shape[0]

def _verify_residue_coupling_evidence(
    evidence: object,
    spec: Mapping[str, object],
    *,
    expected_projection: str | None = None,
    projected_value: object = None,
) -> None:
    fields = {"branch_a_root", "branch_b_root", "evaluation_omega", "coupling"}
    if not isinstance(evidence, Mapping) or set(evidence) != fields:
        raise _integrity("Residue coupling evidence is open or malformed.")
    values = {
        name: _complex_quantity_value(
            evidence[name], unit="radian / second", dimensionality="inverse_time"
        )
        for name in fields
    }
    frequency = spec.get("frequency")
    if frequency != "complex_root_midpoint":
        _verify_quantity_role(
            frequency, complex_value=False, unit="hertz", dimensionality="inverse_time"
        )
    if expected_projection is not None:
        if expected_projection not in {"real", "imag", "magnitude"}:
            raise _integrity("Residue coupling projection is invalid.")
        _verify_quantity_role(
            projected_value,
            complex_value=False,
            unit="radian / second",
            dimensionality="inverse_time",
        )
        assert isinstance(projected_value, Mapping)
        if expected_projection == "magnitude":
            if _f64_value(projected_value["si_value_f64"]) < 0.0:
                raise _integrity("Residue coupling magnitude is negative.")
        else:
            projected = getattr(values["coupling"], expected_projection)
            if struct.pack(">d", projected) != bytes.fromhex(str(projected_value["si_value_f64"])):
                raise _integrity("Residue coupling projection disagrees with its full complex evidence.")
