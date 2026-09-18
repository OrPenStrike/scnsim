"""Independent exact-byte artifact validation shared by execution and decoding."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from ._canonical import (
    canonical_json_bytes,
    float64_from_hex,
    zarr_array_metadata_bytes,
    zarr_artifact_manifest,
)
from ._workspace import VerifiedSuccess, WorkspaceBinding, _inside
from .errors import (
    BackendProtocolError,
    CompilerInvariantError,
    DirectResponseFormationError,
    EliminatedBlockSolveFailure,
    EvidenceIntegrityError,
    InvalidCandidatePhysicalParameter,
    InvalidDiagonalRootHint,
    InvalidOptimizationSpec,
    NumericalResolutionUnresolved,
    PortRealizabilityError,
    RootSlopeUnresolved,
    ScaffoldUnavailableError,
    SCNSimError,
    UnsupportedSingularCapacitanceForDiagonalRootV1,
)

_FAILURES: dict[str, type[SCNSimError]] = {
    "backend_protocol": BackendProtocolError,
    "compiler_invariant": CompilerInvariantError,
    "direct_response_formation": DirectResponseFormationError,
    "eliminated_block_solve_failure": EliminatedBlockSolveFailure,
    "evidence_integrity": EvidenceIntegrityError,
    "invalid_diagonal_root_hint": InvalidDiagonalRootHint,
    "invalid_candidate_physical_parameter": InvalidCandidatePhysicalParameter,
    "invalid_optimization_spec": InvalidOptimizationSpec,
    "numerical_resolution_unresolved": NumericalResolutionUnresolved,
    "port_realizability": PortRealizabilityError,
    "root_slope_unresolved": RootSlopeUnresolved,
    "scaffold_unavailable": ScaffoldUnavailableError,
    "unsupported_singular_capacitance_for_diagonal_root_v1": (
        UnsupportedSingularCapacitanceForDiagonalRootV1
    ),
}


@dataclass(frozen=True)
class _VerifiedEvidenceLease:
    """Revalidate one exact success while protecting deferred artifact reads."""

    binding: WorkspaceBinding
    request_sha256: str
    attempt_sha256: str
    receipt_sha256: str
    result_sha256: str
    attempt_directory: str

    @contextmanager
    def reader(self) -> Iterator[Path]:
        """Yield the current verified attempt path under the binding's read lock."""

        with self.binding.reader():
            attempt = _inside(self.binding.leaf, self.attempt_directory)
            if attempt.is_symlink() or not attempt.is_dir():
                raise EvidenceIntegrityError(
                    "Deferred Result attempt is missing or unsafe.",
                    stage="result_decode",
                    evidence={"request_sha256": self.request_sha256},
                )
            sealed = (
                (
                    _inside(
                        self.binding.leaf,
                        f"requests/{self.request_sha256}/request.json",
                    ),
                    self.request_sha256,
                ),
                (_inside(attempt, "attempt.json"), self.attempt_sha256),
                (_inside(attempt, "receipt.json"), self.receipt_sha256),
                (_inside(attempt, "result.json"), self.result_sha256),
            )
            if any(
                path.is_symlink()
                or not path.is_file()
                or sha256(path.read_bytes()).hexdigest() != expected_sha256
                for path, expected_sha256 in sealed
            ):
                raise EvidenceIntegrityError(
                    "Deferred Result evidence no longer matches its verified success.",
                    stage="result_decode",
                    evidence={"request_sha256": self.request_sha256},
                )
            yield attempt


def _verified_evidence_lease(
    binding: WorkspaceBinding,
    success: VerifiedSuccess,
) -> _VerifiedEvidenceLease:
    """Bind deferred reads to identities from an independently verified success."""

    request_sha256 = success.attempt.get("request_sha256")
    result_sha256 = success.receipt.get("result_sha256")
    attempt_directory = success.attempt.get("directory")
    if (
        not _is_sha256_text(request_sha256)
        or not _is_sha256_text(result_sha256)
        or not isinstance(attempt_directory, str)
        or not attempt_directory
    ):
        raise EvidenceIntegrityError(
            "Verified success lacks exact deferred-read identities.",
            stage="result_decode",
        )
    return _VerifiedEvidenceLease(
        binding=binding,
        request_sha256=request_sha256,
        attempt_sha256=sha256(canonical_json_bytes(success.attempt)).hexdigest(),
        receipt_sha256=sha256(canonical_json_bytes(success.receipt)).hexdigest(),
        result_sha256=result_sha256,
        attempt_directory=attempt_directory,
    )


def _validated_failure_record(
    value: object,
    operation: object,
    *,
    request: Mapping[str, object] | None = None,
    require_optimization_context: bool = False,
    completed_generations: int = 0,
) -> dict[str, object]:
    """Validate a producer failure against the closed public failure taxonomy."""

    if not isinstance(value, Mapping) or set(value) != {
        "category",
        "kind",
        "stage",
        "message",
        "evidence",
    }:
        raise BackendProtocolError(
            "failure outcome lacks a closed typed failure", stage="outcome"
        )
    kind = value.get("kind")
    cls = _FAILURES.get(kind) if isinstance(kind, str) else None
    evidence = value.get("evidence")
    if (
        cls is None
        or value.get("category") != cls.category
        or not isinstance(value.get("stage"), str)
        or not value["stage"]
        or not isinstance(value.get("message"), str)
        or not value["message"]
        or not isinstance(evidence, Mapping)
        or evidence.get("type") != "failure_evidence"
        or evidence.get("operation") != operation
        or not isinstance(evidence.get("context_kind"), str)
        or not evidence["context_kind"]
    ):
        raise BackendProtocolError(
            "failure outcome discriminator or evidence is invalid", stage="outcome"
        )
    try:
        from ._workspace import (
            _optimization_failure_requires_context,
            _verify_failure_document,
            _verify_terminal_optimization_failure,
        )

        _verify_failure_document(value, operation)
        optimization_context = evidence.get("optimization_context")
        if (
            require_optimization_context
            and _optimization_failure_requires_context(value, operation)
            and not isinstance(optimization_context, Mapping)
        ):
            raise BackendProtocolError(
                "optimization execution failure lacks its phase context",
                stage="outcome",
            )
        if (
            operation == "optimize_direct"
            and isinstance(optimization_context, Mapping)
        ):
            if request is None:
                raise BackendProtocolError(
                    "optimization failure context lacks its sealed request",
                    stage="outcome",
                )
            _verify_terminal_optimization_failure(
                value, request.get("spec"),
                completed_generations=completed_generations,
            )
    except EvidenceIntegrityError as error:
        raise BackendProtocolError(
            "failure outcome evidence is inconsistent with its sealed request",
            stage="outcome",
        ) from error
    return dict(value)


def _error_from_record(record: Mapping[str, object]) -> SCNSimError:
    """Reconstruct one typed error after its discriminator was verified."""

    cls = _FAILURES.get(str(record.get("kind")))
    if cls is None or record.get("category") != cls.category:
        raise EvidenceIntegrityError(
            "stored failure discriminator is unknown or inconsistent",
            stage="failure_decode",
            evidence={"kind": record.get("kind"), "category": record.get("category")},
        )
    evidence = record.get("evidence")
    return cls(
        str(record.get("message", "SCNSim backend failure")),
        stage=str(record.get("stage", "backend")),
        evidence=evidence if isinstance(evidence, Mapping) else None,
    )


def _is_sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_canonical_artifact_json(
    attempt: Path,
    path_value: object,
    digest_value: object,
    *,
    stage: str,
) -> Mapping[str, object]:
    if (
        not isinstance(path_value, str)
        or not path_value
        or not _is_sha256_text(digest_value)
    ):
        raise EvidenceIntegrityError(
            "JSON artifact link is malformed",
            stage=stage,
        )
    path = _inside(attempt, path_value)
    if path.is_symlink() or not path.is_file():
        raise EvidenceIntegrityError(
            "JSON artifact is not a regular file",
            stage=stage,
        )
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest_value:
        raise EvidenceIntegrityError(
            "JSON artifact failed exact hash verification",
            stage=stage,
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceIntegrityError(
            "JSON artifact is malformed",
            stage=stage,
        ) from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        raise EvidenceIntegrityError(
            "JSON artifact is not a canonical object",
            stage=stage,
        )
    return value


def _read_zarr(
    attempt: Path, artifact: Mapping[str, object], *, complex_values: bool
) -> np.ndarray:
    root = _inside(attempt, str(artifact["path"]))
    manifest_path = _inside(attempt, str(artifact["file_manifest"]))
    rebuilt = zarr_artifact_manifest(
        artifact_directory=root,
        artifact_id=artifact["id"],
        artifact_path=artifact["path"],
    )
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise EvidenceIntegrityError(
            "stored Zarr manifest is not a regular file", stage="result_decode"
        )
    manifest_bytes = manifest_path.read_bytes()
    if (
        canonical_json_bytes(rebuilt) != manifest_bytes
        or sha256(manifest_bytes).hexdigest() != artifact["sha256"]
    ):
        raise EvidenceIntegrityError(
            "stored Zarr manifest failed exact verification", stage="result_decode"
        )
    _verify_zarr_catalog_metadata(root, artifact, stage="result_decode")
    import zarr

    group = zarr.open_group(root, mode="r")
    if complex_values:
        return np.asarray(group["real"][:], dtype=np.float64) + 1j * np.asarray(
            group["imag"][:], dtype=np.float64
        )
    return np.asarray(group["values"][:], dtype=np.float64)


def _read_json_artifact(
    attempt: Path, artifact: Mapping[str, object]
) -> Mapping[str, object]:
    path = _inside(attempt, str(artifact["path"]))
    if path.is_symlink() or not path.is_file():
        raise EvidenceIntegrityError(
            "optimization ledger is not a regular file", stage="result_decode"
        )
    raw = path.read_bytes()
    if (
        len(raw) != artifact["byte_length"]
        or sha256(raw).hexdigest() != artifact["sha256"]
    ):
        raise EvidenceIntegrityError(
            "optimization ledger artifact failed verification", stage="result_decode"
        )
    value = json.loads(raw)
    if canonical_json_bytes(value) != raw:
        raise EvidenceIntegrityError(
            "optimization ledger is not canonical", stage="result_decode"
        )
    return value


def _validate_direct_values(
    frequency: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    expected_frequency: np.ndarray,
    stage: str,
) -> None:
    if (
        not np.all(np.isfinite(frequency))
        or np.any(frequency <= 0.0)
        or np.any(np.diff(frequency) <= 0.0)
        or frequency.shape != expected_frequency.shape
        or not np.array_equal(
            frequency.view(np.uint64), expected_frequency.view(np.uint64)
        )
        or any(not np.all(np.isfinite(values)) for values in (s, y, z))
    ):
        raise EvidenceIntegrityError(
            "Direct artifacts contain non-finite values or an invalid frequency grid",
            stage=stage,
        )


def _direct_request_frequencies(request: Mapping[str, object]) -> np.ndarray:
    spec = request.get("spec")
    values = spec.get("frequencies") if isinstance(spec, Mapping) else None
    if not isinstance(values, list):
        raise EvidenceIntegrityError(
            "Direct request frequency grid is malformed", stage="request_decode"
        )
    try:
        return np.asarray(
            [float64_from_hex(item["si_value_f64"]) for item in values],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceIntegrityError(
            "Direct request frequency grid is malformed", stage="request_decode"
        ) from error


def _operator_request_frequencies(request: Mapping[str, object]) -> np.ndarray:
    spec = request.get("spec")
    values = (
        spec.get("frequencies")
        if isinstance(spec, Mapping) and spec.get("type") == "operator"
        else None
    )
    if not isinstance(values, list):
        raise EvidenceIntegrityError(
            "Operator request frequency grid is malformed", stage="request_decode"
        )
    try:
        result = np.asarray(
            [float64_from_hex(item["si_value_f64"]) for item in values],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceIntegrityError(
            "Operator request frequency grid is malformed", stage="request_decode"
        ) from error
    if (
        not result.size
        or not np.all(np.isfinite(result))
        or np.any(result <= 0.0)
        or np.any(np.diff(result) <= 0.0)
    ):
        raise EvidenceIntegrityError(
            "Operator request frequency grid is malformed", stage="request_decode"
        )
    return result


def _verify_zarr_catalog_metadata(
    root: Path, artifact: Mapping[str, object], *, stage: str
) -> None:
    datasets = artifact.get("datasets")
    if not isinstance(datasets, list):
        raise EvidenceIntegrityError("Zarr catalog datasets are malformed", stage=stage)
    for dataset in datasets:
        if not isinstance(dataset, Mapping) or not isinstance(dataset.get("path"), str):
            raise EvidenceIntegrityError(
                "Zarr catalog dataset is malformed", stage=stage
            )
        metadata = dataset.get("metadata")
        path = root / str(dataset["path"]) / ".zarray"
        if (
            not isinstance(metadata, Mapping)
            or path.is_symlink()
            or not path.is_file()
            or path.read_bytes()
            != zarr_array_metadata_bytes(
                shape=metadata.get("shape", ()),
                chunks=metadata.get("chunks", ()),
            )
        ):
            raise EvidenceIntegrityError(
                "Zarr catalog metadata disagrees with exact artifact bytes", stage=stage
            )
