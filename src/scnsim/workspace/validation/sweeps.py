"""Parameter-sweep checkpoints, chunks, and embedded point verification."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import product

from ...canonical import canonical_json_bytes as _canonical_bytes, sha256_hex as _sha256
from ..artifacts import _verify_manifest_tree
from ..primitives import _inside
from ..records import PointCheckpoint
from ..storage import _decode_bytes, _load_canonical, _path_entry_exists
from .common import (
    _integrity,
    _parameter_key_integrity,
    _valid_sha,
    _valid_utc_timestamp,
    _verify_parameter_set_document,
)
from .requests import _verify_failure_document, _verify_v1_lineage
from .results import _verify_result_document

def _merge_parameter_records(
    base: Mapping[str, object], overlay: Mapping[str, object]
) -> dict[str, object]:
    by_key = {
        _parameter_key_integrity(row["parameter"]): dict(row)
        for row in base["bindings"]
    }
    order = [_parameter_key_integrity(row["parameter"]) for row in base["bindings"]]
    for row in overlay["bindings"]:
        key = _parameter_key_integrity(row["parameter"])
        if key not in by_key:
            raise _integrity("Listed point references a parameter outside its baseline.")
        by_key[key] = dict(row)
    authorizations = {
        _parameter_key_integrity(item): dict(item)
        for item in (*base["allow_extrapolation"], *overlay["allow_extrapolation"])
    }
    return {
        "type": "parameter_set_v2",
        "bindings": [by_key[key] for key in order],
        "allow_extrapolation": [authorizations[key] for key in sorted(authorizations)],
    }

def _parameter_source_points(source: Mapping[str, object]) -> Iterator[tuple[object, dict[str, object]]]:
    if source["kind"] == "points":
        for ordinal, overlay in enumerate(source["points"]):
            yield ordinal, _merge_parameter_records(source["baseline_parameters"], overlay)
        return
    axes = source["axes"]
    shape = source["shape"]
    for indices in product(*(range(size) for size in shape)):
        overlay = {
            "type": "parameter_set_v2",
            "bindings": [
                {"parameter": axis["parameter"], "value": axis["values"][index]}
                for axis, index in zip(axes, indices)
            ],
            "allow_extrapolation": [],
        }
        yield list(indices), _merge_parameter_records(source["base_parameters"], overlay)

def _verify_point_checkpoint_record(record: Mapping[str, object], point_root: Path,
        request: Mapping[str, object], plan: Mapping[str, object], ordinal: int,
        producer_attempt_sha256: str) -> None:
    from ...authoring.identity import canonical_parameters_sha256

    source = request.get("parameter_source")
    if not isinstance(source, Mapping) or source.get("kind") not in {"grid", "points"}:
        raise _integrity("Point checkpoint request is not a parameter sweep.")
    expected = list(_parameter_source_points(source))
    if ordinal < 0 or ordinal >= len(expected) or not isinstance(record, Mapping) or set(record) != {
        "schema", "schema_version", "request_sha256", "metadata", "files"
    } or record.get("schema") != "scnsim.point_checkpoint_record" or record.get("schema_version") != 1 or record.get("request_sha256") != _sha256(_canonical_bytes(request)):
        raise _integrity("Point checkpoint record envelope is invalid.")
    metadata, files = record["metadata"], record["files"]
    source_index, parameters = expected[ordinal]
    common = {"ordinal", "source_index", "parameters", "parameters_sha256", "status", "producer_attempt_sha256"}
    if (not isinstance(metadata, Mapping) or metadata.get("ordinal") != ordinal or
        metadata.get("source_index") != source_index or metadata.get("parameters") != parameters or
        metadata.get("parameters_sha256") != canonical_parameters_sha256(parameters) or
        metadata.get("producer_attempt_sha256") != producer_attempt_sha256 or
        metadata.get("status") not in {"success", "failure"}):
        raise _integrity("Point checkpoint changed its exact request point.")
    _verify_parameter_set_document(parameters)
    if point_root.is_symlink() or not point_root.is_dir() or not isinstance(files, list):
        raise _integrity("Point checkpoint payload directory is missing or unsafe.")
    actual = sorted(path.relative_to(point_root).as_posix() for path in point_root.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in point_root.rglob("*")) or [row.get("path") if isinstance(row, Mapping) else None for row in files] != actual:
        raise _integrity("Point checkpoint inventory differs from its payload tree.")
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "byte_length"} or not isinstance(row["byte_length"], int) or row["byte_length"] < 1:
            raise _integrity("Point checkpoint file row is malformed.")
        path = _inside(point_root, row["path"])
        if path.stat().st_size != row["byte_length"] or _sha256(path.read_bytes()) != _valid_sha(row["sha256"]):
            raise _integrity("Point checkpoint file bytes differ from record.")
    if metadata["status"] == "failure":
        if set(metadata) not in {frozenset(common | {"failure"}), frozenset(common | {"failure", "ref_lineage"})} or files:
            raise _integrity("Failed point checkpoint contains success payload.")
        _verify_failure_document(metadata["failure"], request["operation"])
        if "ref_lineage" in metadata:
            _verify_v1_lineage(metadata["ref_lineage"], plan)
        return
    if set(metadata) != common | {"ref_lineage", "payload_path"} or metadata["payload_path"] != f"artifacts/parameter_points/points/{ordinal:06d}/payload.json":
        raise _integrity("Successful point checkpoint metadata is invalid.")
    payload = _load_canonical(point_root / "payload.json")
    if payload.get("schema") != "scnsim.parameter_point_payload" or payload.get("schema_version") != 2:
        raise _integrity("Point checkpoint payload envelope is invalid.")
    prefix = f"artifacts/parameter_points/points/{ordinal:06d}/"
    def localize(value: object) -> object:
        if isinstance(value, Mapping):
            return {key: (item[len(prefix):] if key in {"path", "file_manifest"} and isinstance(item, str) and item.startswith(prefix) else localize(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [localize(item) for item in value]
        return value
    point_result = localize(payload)
    assert isinstance(point_result, dict)
    point_result.update({"schema": "scnsim.result", "request_sha256": record["request_sha256"],
        "attempt_sha256": producer_attempt_sha256, "parameters": parameters,
        "parameters_sha256": metadata["parameters_sha256"], "ref_lineage": metadata["ref_lineage"]})
    point_request = dict(request)
    point_request["parameter_source"] = {"kind": "point", "parameters": parameters}
    _verify_result_document(point_result, point_request, record["request_sha256"],
        producer_attempt_sha256, plan)
    for manifest_path in point_root.rglob("*.manifest.json"):
        manifest = _load_canonical(manifest_path)
        if manifest.get("schema") == "scnsim.artifact_manifest":
            artifact_relative = manifest.get("artifact_path")
            if not isinstance(artifact_relative, str) or not artifact_relative.startswith(prefix):
                raise _integrity("Point checkpoint artifact manifest path is invalid.")
            _verify_manifest_tree(_inside(point_root, artifact_relative[len(prefix):]), manifest)

def _verify_point_checkpoints(request_directory: Path, request: Mapping[str, object],
        plan: Mapping[str, object]) -> tuple[PointCheckpoint, ...]:
    root = request_directory / "point-checkpoints"
    anchor_path = request_directory / "point-checkpoint-anchor.json"
    if _path_entry_exists(anchor_path):
        anchor = _load_canonical(anchor_path)
        if set(anchor) != {"schema", "schema_version", "request_sha256"} or anchor.get("schema") != "scnsim.point_checkpoint_anchor" or anchor.get("schema_version") != 1 or anchor.get("request_sha256") != _sha256(_canonical_bytes(request)):
            raise _integrity("Point checkpoint request anchor is invalid.")
        if not _path_entry_exists(root):
            raise _integrity("Published point checkpoint index is missing.")
    if not _path_entry_exists(root):
        return ()
    if root.is_symlink() or not root.is_dir():
        raise _integrity("Point checkpoint directory is unsafe.")
    index = _load_canonical(root / "index.json")
    request_sha = _sha256(_canonical_bytes(request))
    entries = index.get("entries")
    if set(index) != {"schema", "schema_version", "request_sha256", "entries"} or index.get("schema") != "scnsim.point_checkpoint_index" or index.get("schema_version") != 1 or index.get("request_sha256") != request_sha or not isinstance(entries, list):
        raise _integrity("Point checkpoint index is invalid.")
    names = {path.name for path in root.iterdir()}
    if names != {"index.json", *(f"{i:06d}" for i in range(len(entries)))}:
        raise _integrity("Point checkpoint index does not cover its directories.")
    verified = []
    for ordinal, entry in enumerate(entries):
        directory = root / f"{ordinal:06d}"
        if not isinstance(entry, Mapping) or set(entry) != {"ordinal", "seal_sha256"} or entry["ordinal"] != ordinal or directory.is_symlink() or not directory.is_dir() or {p.name for p in directory.iterdir()} != {"record.json", "source-attempt.json", "seal.json", "point"}:
            raise _integrity("Point checkpoint index entry is invalid.")
        if any((directory / name).is_symlink() or not (directory / name).is_file() for name in ("record.json", "source-attempt.json", "seal.json")):
            raise _integrity("Point checkpoint files are unsafe.")
        seal_bytes = (directory / "seal.json").read_bytes()
        if _sha256(seal_bytes) != _valid_sha(entry["seal_sha256"]):
            raise _integrity("Point checkpoint seal differs from index.")
        seal = _decode_bytes(seal_bytes, "point checkpoint seal")
        record_bytes = (directory / "record.json").read_bytes()
        source_bytes = (directory / "source-attempt.json").read_bytes()
        if set(seal) != {"schema", "schema_version", "request_sha256", "ordinal", "record_sha256", "source_attempt_sha256", "published_at_utc"} or seal.get("schema") != "scnsim.point_checkpoint_seal" or seal.get("schema_version") != 1 or seal.get("request_sha256") != request_sha or seal.get("ordinal") != ordinal or seal.get("record_sha256") != _sha256(record_bytes) or seal.get("source_attempt_sha256") != _sha256(source_bytes) or not _valid_utc_timestamp(seal.get("published_at_utc")):
            raise _integrity("Point checkpoint seal is invalid.")
        source_attempt = _decode_bytes(source_bytes, "point source attempt")
        if source_attempt.get("request_sha256") != request_sha or source_attempt.get("attempt_state") != "launched":
            raise _integrity("Point checkpoint source attempt is invalid.")
        record = _decode_bytes(record_bytes, "point checkpoint record")
        _verify_point_checkpoint_record(record, directory / "point", request, plan,
            ordinal, seal["source_attempt_sha256"])
        verified.append(PointCheckpoint(record, entry["seal_sha256"], directory))
    return tuple(verified)

def _verify_parameter_sweep_artifacts(
    directory: Path,
    result: Mapping[str, object],
    receipt: Mapping[str, object],
) -> None:
    from ...authoring.identity import canonical_parameters_sha256

    link = result["manifest"]
    if receipt.get("artifacts") != [link]:
        raise _integrity("Parameter-sweep receipt does not bind its sole manifest.")
    manifest_path = _inside(directory, link["path"])
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size != link["byte_length"]
        or _sha256(manifest_path.read_bytes()) != link["sha256"]
    ):
        raise _integrity("Parameter-sweep manifest link does not bind its file.")
    manifest = _load_canonical(manifest_path)
    if (
        set(manifest) != {"schema", "schema_version", "request_sha256", "attempt_sha256", "point_count", "files"}
        or manifest.get("schema") != "scnsim.parameter_points_manifest"
        or manifest.get("schema_version") != 2
        or manifest.get("request_sha256") != result["request_sha256"]
        or manifest.get("attempt_sha256") != result["attempt_sha256"]
        or manifest.get("point_count") != result["point_count"]
        or not isinstance(manifest.get("files"), list)
    ):
        raise _integrity("Parameter-sweep manifest is malformed.")
    root = _inside(directory, "artifacts/parameter_points")
    if root.is_symlink() or not root.is_dir():
        raise _integrity("Parameter-sweep artifact root is missing or unsafe.")
    actual_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if any(path.is_symlink() for path in root.rglob("*")):
        raise _integrity("Parameter-sweep artifact tree contains a symlink.")
    rows = manifest["files"]
    if [row.get("path") if isinstance(row, Mapping) else None for row in rows] != actual_files:
        raise _integrity("Parameter-sweep manifest does not exactly cover its file tree.")
    manifest_by_path: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256", "byte_length"}
            or not isinstance(row.get("byte_length"), int)
            or isinstance(row.get("byte_length"), bool)
            or row["byte_length"] < 1
        ):
            raise _integrity("Parameter-sweep file manifest row is malformed.")
        path = _inside(root, row["path"])
        if path.stat().st_size != row["byte_length"] or _sha256(path.read_bytes()) != _valid_sha(row["sha256"]):
            raise _integrity("Parameter-sweep file manifest hash is incorrect.")
        manifest_by_path[row["path"]] = row

    request = _load_canonical(directory.parent.parent / "request.json")
    plan = _load_canonical(directory.parent.parent.parent.parent / "plan.json")
    source = request.get("parameter_source")
    if not isinstance(source, Mapping):
        raise _integrity("Parameter-sweep request source is unavailable.")
    expected_points = list(_parameter_source_points(source))
    checkpoints = _verify_point_checkpoints(directory.parent.parent, request, plan)
    if len(checkpoints) != len(expected_points):
        raise _integrity("Final sweep does not bind every published point checkpoint.")
    point_ordinal = 0
    for chunk_link in result["chunks"]:
        relative = str(chunk_link["path"])[len("artifacts/parameter_points/"):]
        row = manifest_by_path.get(relative)
        if row is None or row["sha256"] != chunk_link["sha256"]:
            raise _integrity("Parameter-sweep chunk is absent from its manifest.")
        chunk = _load_canonical(_inside(directory, chunk_link["path"]))
        if (
            set(chunk) != {"schema", "schema_version", "request_sha256", "attempt_sha256", "chunk_ordinal", "first_point", "points"}
            or chunk.get("schema") != "scnsim.parameter_point_chunk"
            or chunk.get("schema_version") != 2
            or chunk.get("request_sha256") != result["request_sha256"]
            or chunk.get("attempt_sha256") != result["attempt_sha256"]
            or chunk.get("chunk_ordinal") != chunk_link["chunk_ordinal"]
            or chunk.get("first_point") != chunk_link["first_point"]
            or not isinstance(chunk.get("points"), list)
            or len(chunk["points"]) != chunk_link["point_count"]
        ):
            raise _integrity("Parameter-sweep chunk envelope is malformed.")
        for point in chunk["points"]:
            expected_source_index, expected_parameters = expected_points[point_ordinal]
            common = {"ordinal", "source_index", "parameters", "parameters_sha256", "status", "producer_attempt_sha256", "checkpoint_seal_sha256"}
            if (
                not isinstance(point, Mapping)
                or point.get("ordinal") != point_ordinal
                or point.get("source_index") != expected_source_index
                or point.get("parameters") != expected_parameters
                or point.get("parameters_sha256") != canonical_parameters_sha256(expected_parameters)
                or point.get("status") not in {"success", "failure"}
            ):
                raise _integrity("Parameter-sweep point identity is malformed.")
            checkpoint = checkpoints[point_ordinal]
            if (point.get("checkpoint_seal_sha256") != checkpoint.seal_sha256 or
                {key: value for key, value in point.items() if key != "checkpoint_seal_sha256"} != checkpoint.record["metadata"]):
                raise _integrity("Final sweep point differs from its original checkpoint record.")
            point_prefix_relative = f"points/{point_ordinal:06d}/"
            actual_point_files = {path[len(point_prefix_relative):]: {**row, "path": path[len(point_prefix_relative):]} for path, row in manifest_by_path.items()
                if path.startswith(point_prefix_relative)}
            checkpoint_files = {row["path"]: row for row in checkpoint.record["files"]}
            if actual_point_files != checkpoint_files:
                raise _integrity("Final sweep point bytes differ from published checkpoint.")
            _verify_parameter_set_document(point["parameters"])
            if point["status"] == "failure":
                failure_fields = set(point)
                if failure_fields != common | {"failure"} and failure_fields != common | {"failure", "ref_lineage"}:
                    raise _integrity("Failed parameter point leaks success evidence.")
                _verify_failure_document(point["failure"], request["operation"])
                if "ref_lineage" in point:
                    _verify_v1_lineage(point["ref_lineage"], plan)
            else:
                if set(point) != common | {"ref_lineage", "payload_path"}:
                    raise _integrity("Successful parameter point is incomplete.")
                payload_path = f"artifacts/parameter_points/points/{point_ordinal:06d}/payload.json"
                if point.get("payload_path") != payload_path:
                    raise _integrity("Successful parameter point payload path is malformed.")
                payload = _load_canonical(_inside(directory, payload_path))
                if payload.get("schema") != "scnsim.parameter_point_payload" or payload.get("schema_version") != 2:
                    raise _integrity("Parameter point payload envelope is malformed.")
                point_prefix = f"artifacts/parameter_points/points/{point_ordinal:06d}/"

                def verify_nested_artifacts(value: object) -> None:
                    if isinstance(value, Mapping):
                        if "file_manifest" in value:
                            artifact_id = value.get("id")
                            artifact_path = value.get("path")
                            file_manifest = value.get("file_manifest")
                            digest = value.get("sha256")
                            if (
                                not isinstance(artifact_id, str)
                                or not artifact_id
                                or not isinstance(artifact_path, str)
                                or not artifact_path.startswith(point_prefix + "artifacts/")
                                or not isinstance(file_manifest, str)
                                or not file_manifest.startswith(point_prefix + "artifacts/")
                            ):
                                raise _integrity("Parameter point artifact path is malformed.")
                            artifact_relative = artifact_path[len("artifacts/parameter_points/"):]
                            manifest_relative = file_manifest[len("artifacts/parameter_points/"):]
                            row = manifest_by_path.get(manifest_relative)
                            manifest_file = _inside(directory, file_manifest)
                            artifact_root = _inside(directory, artifact_path)
                            if (
                                row is None
                                or row.get("sha256") != digest
                                or manifest_file.is_symlink()
                                or not manifest_file.is_file()
                                or artifact_root.is_symlink()
                                or not artifact_root.is_dir()
                            ):
                                raise _integrity("Parameter point artifact is not bound by its batch manifest.")
                            artifact_manifest = _load_canonical(manifest_file)
                            if (
                                artifact_manifest.get("schema") != "scnsim.artifact_manifest"
                                or artifact_manifest.get("artifact_id") != artifact_id
                                or artifact_manifest.get("artifact_path") != artifact_path
                            ):
                                raise _integrity("Parameter point artifact manifest identity is malformed.")
                            _verify_manifest_tree(artifact_root, artifact_manifest)
                            if artifact_relative not in manifest_by_path and not any(
                                name.startswith(artifact_relative.rstrip("/") + "/")
                                for name in manifest_by_path
                            ):
                                raise _integrity("Parameter point artifact tree is absent from the batch manifest.")
                        for nested in value.values():
                            verify_nested_artifacts(nested)
                    elif isinstance(value, list):
                        for nested in value:
                            verify_nested_artifacts(nested)

                def localize_artifact_paths(value: object) -> object:
                    if isinstance(value, Mapping):
                        return {
                            key: (
                                item[len(point_prefix):]
                                if key in {"path", "file_manifest"}
                                and isinstance(item, str)
                                and item.startswith(point_prefix)
                                else localize_artifact_paths(item)
                            )
                            for key, item in value.items()
                        }
                    if isinstance(value, list):
                        return [localize_artifact_paths(item) for item in value]
                    return value

                verify_nested_artifacts(payload)
                point_request = dict(request)
                point_request["parameter_source"] = {
                    "kind": "point",
                    "parameters": point["parameters"],
                }
                localized = localize_artifact_paths(payload)
                if not isinstance(localized, dict):
                    raise _integrity("Parameter point payload is malformed.")
                point_result = localized
                point_result["schema"] = "scnsim.result"
                point_result["request_sha256"] = result["request_sha256"]
                point_result["attempt_sha256"] = point["producer_attempt_sha256"]
                point_result["parameters"] = point["parameters"]
                point_result["parameters_sha256"] = point["parameters_sha256"]
                point_result["ref_lineage"] = point["ref_lineage"]
                _verify_result_document(
                    point_result,
                    point_request,
                    result["request_sha256"],
                    point["producer_attempt_sha256"],
                    plan,
                )
            point_ordinal += 1
    if point_ordinal != result["point_count"]:
        raise _integrity("Parameter-sweep chunks do not cover every point.")
