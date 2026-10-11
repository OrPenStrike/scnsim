"""Execution-owned logs, cleanup, and staged terminal evidence validation.

These helpers inspect the current staging interval and delegate persisted-data
integrity to workspace owners. They do not promote attempts or decode Results."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
from ..canonical import canonical_json_bytes
from ..workspace.manifests import zarr_artifact_manifest
from ..workspace.artifacts import (
    _direct_request_frequencies,
    _read_zarr,
    _validate_direct_values,
    _verify_zarr_catalog_metadata,
)
from ..workspace import (
    BaselineCheckpoint,
    _inside,
    _verify_artifact_inventory,
    _verify_generation_artifacts,
    _verify_result_document,
)
from ..errors import BackendProtocolError, EvidenceIntegrityError


def _require_staging_directory(staging: Path) -> None:
    if staging.parent.is_symlink() or staging.is_symlink() or not staging.is_dir():
        raise EvidenceIntegrityError(
            "attempt staging is not a regular directory",
            stage="workspace",
            evidence={"path": str(staging)},
        )


def _remove_untrusted(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _write_logs(staging: Path, stdout: Sequence[str], stderr: Sequence[str]) -> None:
    _require_staging_directory(staging)
    directory = staging / "logs"
    if directory.exists() or directory.is_symlink():
        _remove_untrusted(directory)
    if not stdout and not stderr:
        return
    directory.mkdir()
    if stdout:
        path = directory / "stdout.log"
        if path.is_symlink():
            path.unlink()
        path.write_text("".join(stdout), encoding="utf-8")
    if stderr:
        path = directory / "stderr.log"
        if path.is_symlink():
            path.unlink()
        path.write_text("".join(stderr), encoding="utf-8")


def _discard_untrusted_outputs(staging: Path, *, keep_ledgers: bool = False) -> None:
    _require_staging_directory(staging)
    outcome = staging / "outcome.json"
    if outcome.exists() or outcome.is_symlink():
        logs = staging / "logs"
        if logs.is_symlink() or logs.exists() and not logs.is_dir():
            _remove_untrusted(logs)
        logs.mkdir(exist_ok=True)
        destination = logs / "untrusted-outcome.json"
        if destination.exists() or destination.is_symlink():
            _remove_untrusted(destination)
        if outcome.is_symlink():
            outcome.unlink()
        elif outcome.is_file():
            shutil.move(outcome, destination)
        else:
            _remove_untrusted(outcome)
    result = staging / "result.json"
    if result.exists() or result.is_symlink():
        _remove_untrusted(result)
    artifacts = staging / "artifacts"
    if artifacts.exists() or artifacts.is_symlink():
        if artifacts.is_symlink():
            artifacts.unlink()
        else:
            generations = artifacts / "generations"
            if keep_ledgers and not generations.is_symlink() and generations.is_dir():
                for child in artifacts.iterdir():
                    if child != generations:
                        _remove_untrusted(child)
                if not any(generations.iterdir()):
                    generations.rmdir()
                    artifacts.rmdir()
            else:
                _remove_untrusted(artifacts)
    allowed = {"attempt.json", "logs"}
    if keep_ledgers and (staging / "artifacts").is_dir():
        allowed.add("artifacts")
    for child in staging.iterdir():
        if child.name not in allowed:
            _remove_untrusted(child)


def _validate_terminal_staging_layout(staging: Path, *, success: bool) -> None:
    _require_staging_directory(staging)
    allowed = {"attempt.json", "logs", "outcome.json", "artifacts"}
    if success:
        allowed.add("result.json")
    unexpected = sorted(
        child.name for child in staging.iterdir() if child.name not in allowed
    )
    if unexpected:
        raise BackendProtocolError(
            "terminal staging contains unsupported entries",
            stage="outcome",
            evidence={"entries": unexpected},
        )


def _decode_direct_point_arrays(
    directory: Path,
    payload: Mapping[str, object],
    request: Mapping[str, object],
    *,
    artifact_prefix: str | None = None,
) -> dict[str, np.ndarray]:
    """Materialize Direct point values through the ordinary point decode path."""

    arrays = payload["array_catalog"]
    source_arrays = arrays
    if artifact_prefix is not None:
        def localize(value: object) -> object:
            if isinstance(value, Mapping):
                return {
                    key: (
                        item[len(artifact_prefix):]
                        if key in {"path", "file_manifest"}
                        and isinstance(item, str)
                        and item.startswith(artifact_prefix)
                        else localize(item)
                    )
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [localize(item) for item in value]
            return value

        arrays = localize(arrays)

    def read_array(role: str, *, complex_values: bool) -> np.ndarray:
        artifact = cast(Mapping[str, object], arrays[role])
        if artifact_prefix is None:
            return _read_zarr(directory, artifact, complex_values=complex_values)
        source_artifact = cast(Mapping[str, object], source_arrays[role])
        return _read_zarr(
            directory,
            artifact,
            complex_values=complex_values,
            manifest_artifact_path=cast(str, source_artifact["path"]),
        )

    direct_values = {
        "frequencies": read_array("frequencies", complex_values=False),
        "s": read_array("s", complex_values=True),
        "y": read_array("y", complex_values=True),
        "z": read_array("z", complex_values=True),
    }
    _validate_direct_values(
        direct_values["frequencies"],
        direct_values["s"],
        direct_values["y"],
        direct_values["z"],
        expected_frequency=_direct_request_frequencies(request),
        stage="result_decode",
    )
    return direct_values


def _validate_success_staging(
    staging: Path,
    outcome: Mapping[str, object],
    request: Mapping[str, object],
    plan: Mapping[str, object],
    *,
    request_path: Path,
    optimization_checkpoint: BaselineCheckpoint | None = None,
    expected_julia_threads: int = 1,
    expected_blas_threads: int = 1,
    collect_sweep_observations: bool = False,
) -> dict[str, object]:
    if outcome.get("runtime_semantic") != request.get("runtime_semantic"):
        raise BackendProtocolError(
            "outcome runtime identity does not match the request", stage="outcome"
        )
    result_path = _inside(staging, "result.json")
    if (
        result_path.is_symlink()
        or not result_path.is_file()
        or sha256(result_path.read_bytes()).hexdigest() != outcome.get("result_sha256")
    ):
        raise BackendProtocolError(
            "success outcome does not bind result.json", stage="outcome"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if canonical_json_bytes(result) != result_path.read_bytes():
        raise BackendProtocolError("result.json is not canonical", stage="outcome")
    parameter_source = request.get("parameter_source")
    is_parameter_sweep = isinstance(parameter_source, Mapping) and parameter_source.get(
        "kind"
    ) in {"grid", "points"}
    expected_kind = (
        "parameter_sweep"
        if is_parameter_sweep
        else "direct_response"
        if request.get("operation") == "solve_direct"
        else "hb_batch"
        if request.get("operation") == "solve_hb"
        else "optimization"
        if request.get("operation") == "optimize_direct"
        else request.get("spec", {}).get("type")
        if request.get("operation") == "evaluate_direct"
        and isinstance(request.get("spec"), Mapping)
        else None
    )
    expected_result_fields = (
        {
            "schema",
            "schema_version",
            "result_kind",
            "request_sha256",
            "attempt_sha256",
            "parameter_source_sha256",
            "point_count",
            "chunk_size",
            "manifest",
            "chunks",
        }
        if expected_kind == "parameter_sweep"
        else {
            "schema",
            "schema_version",
            "result_kind",
            "request_sha256",
            "attempt_sha256",
            "parameters",
            "parameters_sha256",
            "ref_lineage",
            "scalar_catalog",
            "array_catalog",
        }
        if expected_kind
        in {
            "direct_response",
            "diagonal_root",
            "operator_element_root",
            "hybridized_pole",
            "transfer_zero",
            "residue_normalized_coupling",
            "response_element",
            "operator",
        }
        else {
            "schema",
            "schema_version",
            "result_kind",
            "request_sha256",
            "attempt_sha256",
            "parameters",
            "parameters_sha256",
            "ref_lineage",
            "baseline",
            "best",
            "completed_generations",
            "unused_evaluations",
            "ledger_artifacts",
        }
        if expected_kind == "optimization"
        else {
            "schema",
            "schema_version",
            "result_kind",
            "request_sha256",
            "attempt_sha256",
            "parameters",
            "parameters_sha256",
            "ref_lineage",
            "lattice",
            "truncation",
            "topology_evidence",
            "cases",
        }
        if expected_kind == "hb_batch"
        else None
    )
    if expected_result_fields is not None and expected_kind != "parameter_sweep" and "discretization" in result:
        expected_result_fields.add("discretization")
    if (
        expected_result_fields is None
        or set(result) != expected_result_fields
        or result.get("schema") != "scnsim.result"
        or result.get("schema_version") != 2
        or result.get("result_kind") != expected_kind
        or result.get("request_sha256") != outcome.get("request_sha256")
        or result.get("attempt_sha256") != outcome.get("attempt_sha256")
    ):
        raise BackendProtocolError(
            "result envelope does not match its request and operation", stage="outcome"
        )
    _verify_result_document(
        result,
        request,
        str(outcome.get("request_sha256")),
        str(outcome.get("attempt_sha256")),
        plan,
        optimization_checkpoint=optimization_checkpoint,
    )
    catalogs: list[Mapping[str, object]] = []
    if expected_kind == "parameter_sweep":
        catalogs.append(result["manifest"])
        expected_links = [dict(result["manifest"])]
    elif expected_kind == "hb_batch":
        expected_links: list[dict[str, object]] = []
        cases = result.get("cases")
        if not isinstance(cases, list):
            raise BackendProtocolError(
                "HB result has no ordered case catalog", stage="outcome"
            )
        for case in cases:
            if not isinstance(case, Mapping):
                raise BackendProtocolError(
                    "HB result case catalog is malformed", stage="outcome"
                )
            if case.get("status") == "failure":
                continue
            artifacts = case.get("artifacts")
            traces = case.get("traces")
            case_id = case.get("case_id")
            if (
                not isinstance(case_id, str)
                or not isinstance(artifacts, Mapping)
                or not isinstance(traces, list)
            ):
                raise BackendProtocolError(
                    "HB success artifact catalog is malformed", stage="outcome"
                )
            ordered = [
                artifacts[name]
                for name in (
                    "s",
                    "y",
                    "z",
                    "backend_native_s",
                    "backend_native_z",
                    "states",
                    "effective_source_vectors",
                )
            ]
            ordered.extend(traces)
            for artifact in ordered:
                if not isinstance(artifact, Mapping):
                    raise BackendProtocolError(
                        "HB success artifact catalog is malformed", stage="outcome"
                    )
                catalogs.append(artifact)
                expected_links.append(
                    {
                        "case_id": case_id,
                        "id": artifact.get("id"),
                        "path": artifact.get("path"),
                        "sha256": artifact.get("sha256"),
                    }
                )
        if outcome.get("artifacts") != expected_links:
            raise BackendProtocolError(
                "HB outcome artifact inventory does not match result.json",
                stage="outcome",
            )
        _verify_artifact_inventory(
            staging, result, {"artifacts": expected_links}, request_path=request_path
        )
    elif expected_kind != "optimization":
        catalogs.extend(result["array_catalog"].values())
    else:
        catalogs.extend(result["ledger_artifacts"])
    sweep_records: list[dict[str, object]] = []
    if expected_kind == "parameter_sweep":
        if outcome.get("artifacts") != expected_links:
            raise BackendProtocolError(
                "batch outcome does not bind its manifest", stage="outcome"
            )
        verified_sweep_records = _verify_artifact_inventory(
            staging,
            result,
            {"artifacts": expected_links},
            request_path=request_path,
            include_sweep_records=collect_sweep_observations,
        )
        if verified_sweep_records is not None:
            sweep_records = verified_sweep_records
    elif expected_kind != "hb_batch":
        expected_links = [
            {"id": artifact["id"], "sha256": artifact["sha256"]}
            for artifact in catalogs
        ]
        if outcome.get("artifacts") != expected_links or len(
            {item["id"] for item in expected_links}
        ) != len(expected_links):
            raise BackendProtocolError(
                "outcome artifact inventory does not match result.json", stage="outcome"
            )
        _verify_artifact_inventory(
            staging, result, {"artifacts": expected_links}, request_path=request_path
        )
    generation_records: list[tuple[int, str, Mapping[str, object]]] = []
    if expected_kind == "optimization":
        generation_records = cast(list[tuple[int, str, Mapping[str, object]]], _verify_generation_artifacts(
            staging,
            expected_links,
            request_path=request_path,
            request_sha256=str(outcome["request_sha256"]),
            attempt_sha256=str(outcome["attempt_sha256"]),
            expected_julia_threads=expected_julia_threads,
            expected_blas_threads=expected_blas_threads,
            include_records=True,
        ))
    for artifact in catalogs:
        path = _inside(staging, str(artifact["path"]))
        if artifact.get("media_type") == "application/vnd+zarr-v2":
            manifest_path = _inside(staging, str(artifact["file_manifest"]))
            rebuilt = zarr_artifact_manifest(
                artifact_directory=path,
                artifact_id=artifact["id"],
                artifact_path=artifact["path"],
            )
            if (
                manifest_path.is_symlink()
                or not manifest_path.is_file()
                or canonical_json_bytes(rebuilt) != manifest_path.read_bytes()
                or sha256(manifest_path.read_bytes()).hexdigest() != artifact["sha256"]
            ):
                raise EvidenceIntegrityError(
                    "Zarr manifest does not match exact artifact bytes",
                    stage="artifact_validation",
                )
            _verify_zarr_catalog_metadata(path, artifact, stage="artifact_validation")
        else:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != artifact["byte_length"]
                or sha256(path.read_bytes()).hexdigest() != artifact["sha256"]
            ):
                raise EvidenceIntegrityError(
                    "file artifact does not match its catalog",
                    stage="artifact_validation",
                )
    direct_arrays: dict[str, np.ndarray] | None = None
    if expected_kind == "direct_response":
        arrays = result["array_catalog"]
        direct_arrays = {
            "frequencies": _read_zarr(staging, arrays["frequencies"], complex_values=False),
            "s": _read_zarr(staging, arrays["s"], complex_values=True),
            "y": _read_zarr(staging, arrays["y"], complex_values=True),
            "z": _read_zarr(staging, arrays["z"], complex_values=True),
        }
        _validate_direct_values(
            direct_arrays["frequencies"],
            direct_arrays["s"],
            direct_arrays["y"],
            direct_arrays["z"],
            expected_frequency=_direct_request_frequencies(request),
            stage="artifact_validation",
        )
    if collect_sweep_observations and expected_kind == "parameter_sweep":
        for verified_point in sweep_records:
            payload = verified_point["payload"]
            if not isinstance(payload, Mapping) or payload.get("result_kind") != "direct_response":
                continue
            verified_point["direct_arrays"] = _decode_direct_point_arrays(
                staging, payload, request,
            )
    return {
        "result": result,
        "generation_records": generation_records,
        "direct_arrays": direct_arrays,
        "sweep_records": sweep_records,
    }
