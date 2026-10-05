"""Attempt and receipt evidence assembled by execution coordination.

Builders capture observed runtime and source provenance without allocating or
publishing state. Canonical workspace envelopes remain workspace-owned."""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from ..canonical import sha256_hex
from ..workspace.documents import canonical_receipt_document
from ..workspace import AttemptAllocation, BaselineCheckpoint, _required_extrapolation_rows
from ..errors import CompilerInvariantError, SCNSimError
from .protocol import BootstrapReady


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _attempt_document(
    allocation: AttemptAllocation,
    *,
    started: str,
    executable_sha: str,
    state: str,
    ready: BootstrapReady | None = None,
    resume_ledger_sha: str | None = None,
    optimization: bool = False,
    checkpoint: BaselineCheckpoint | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": "scnsim.attempt",
        "schema_version": 2 if optimization else 1,
        "request_sha256": allocation.request_sha256,
        "ordinal": allocation.ordinal,
        "ordinal_text": allocation.ordinal_text,
        "directory": allocation.attempt_directory_text,
        "staging_directory": allocation.staging_directory_text,
        "attempt_state": state,
        "started_at_utc": started,
        "julia_executable_sha256": executable_sha,
        "os": platform.system(),
        "architecture": platform.machine() or "unknown",
        "cpu": platform.processor() or "unknown",
    }
    if ready is not None:
        document.update(
            {
                "julia_threads": ready.julia_threads,
                "blas_threads": ready.blas_threads,
                "blas_vendor": ready.blas_vendor,
            }
        )
        fftw_threads = getattr(ready, "fftw_threads", None)
        if fftw_threads is not None:
            document["fftw_threads"] = fftw_threads
    if resume_ledger_sha is not None:
        document["resume_ledger_sha256"] = resume_ledger_sha
    if checkpoint is not None:
        document["baseline_checkpoint_sha256"] = checkpoint.checkpoint_sha256
        document["baseline_checkpoint_seal_sha256"] = checkpoint.seal_sha256
    return document


def _failure_record(
    error: SCNSimError, operation: object, request_sha: str, attempt_sha: str
) -> dict[str, object]:
    return {
        "category": error.category,
        "kind": error.kind,
        "stage": error.stage,
        "message": str(error),
        "evidence": {
            "type": "failure_evidence",
            "operation": operation
            if operation
            in {"solve_direct", "solve_hb", "evaluate_direct", "optimize_direct"}
            else "backend_protocol",
            "context_kind": "protocol",
            "request_sha256": request_sha,
            "attempt_sha256": attempt_sha,
        },
    }


def _receipt(
    *,
    request: Mapping[str, object],
    plan_document: Mapping[str, object],
    request_sha: str,
    attempt_sha: str,
    outcome: str,
    artifacts: Sequence[object],
    source_units: Sequence[Mapping[str, object]],
    outcome_sha: str | None = None,
    result_sha: object | None = None,
    failure: Mapping[str, object] | None = None,
    interruption: Mapping[str, object] | None = None,
) -> dict[str, object]:
    runtime_sha = sha256_hex(request["runtime_semantic"])
    provenance = sha256_hex(
        {"schema": "scnsim.receipt_provenance", "source_units": list(source_units)}
    )
    evidence: dict[str, object] = {
        "runtime_semantic_sha256": runtime_sha,
        "source_units": list(source_units),
        "extrapolation_evidence": _receipt_extrapolation_evidence(
            request, plan_document, require_authorized=outcome == "success"
        ),
        "provenance_sha256": provenance,
    }
    evidence["evidence_sha256"] = sha256_hex(evidence)
    document: dict[str, object] = {
        "request_sha256": request_sha,
        "attempt_sha256": attempt_sha,
        "outcome": outcome,
        "artifacts": list(artifacts),
        "evidence": evidence,
        "sealed_at_utc": _utc_now(),
    }
    if outcome_sha is not None:
        document["outcome_sha256"] = outcome_sha
    if result_sha is not None:
        document["result_sha256"] = result_sha
    if failure is not None:
        document["failure"] = dict(failure)
    if interruption is not None:
        document["interruption"] = dict(interruption)
    return canonical_receipt_document(document)


def _receipt_extrapolation_evidence(
    request: Mapping[str, object],
    plan_document: Mapping[str, object],
    *,
    require_authorized: bool,
) -> list[dict[str, object]]:
    """Project receipt evidence through the workspace's closed fan-out verifier."""
    if request.get("operation") == "optimize_direct":
        return []
    source = request.get("parameter_source")
    if isinstance(source, Mapping) and source.get("kind") in {"grid", "points"}:
        # Point-local authorization is verified against every chunk entry;
        # the request receipt must not pretend an ordered space is one point.
        return []
    parameters = source.get("parameters") if isinstance(source, Mapping) else None
    if not isinstance(parameters, Mapping):
        raise CompilerInvariantError(
            "receipt request has no ParameterSet", stage="receipt"
        )
    return _required_extrapolation_rows(
        plan_document,
        parameters,
        authorization_source="parameter_set",
        require_authorized=require_authorized,
    )
