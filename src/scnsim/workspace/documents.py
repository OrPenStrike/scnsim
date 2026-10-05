"""Canonical workspace, attempt, Result, and receipt documents.

These builders own stored envelope representation only. Workspace locking,
publication, and evidence verification remain with their lifecycle owners."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from ..canonical import _nfc, _nonempty, _sha256, _validation, canonical_value
from .validation.common import _UUID4


_DIRECT_RESULTS = frozenset({
    "direct_response", "diagonal_root", "operator_element_root", "hybridized_pole", "transfer_zero",
    "residue_normalized_coupling", "response_element", "operator", "optimization", "hb_batch",
})


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
    operation: str | None = None,
    baseline_checkpoint_sha256: str | None = None,
    baseline_checkpoint_seal_sha256: str | None = None,
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
        "schema_version": 2 if operation == "optimize_direct" else 1,
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
    if (baseline_checkpoint_sha256 is None) != (baseline_checkpoint_seal_sha256 is None):
        raise _validation("baseline checkpoint content and seal identities are a pair")
    if baseline_checkpoint_sha256 is not None:
        if operation != "optimize_direct":
            raise _validation("only optimization attempts may bind a baseline checkpoint")
        document["baseline_checkpoint_sha256"] = _sha256(
            baseline_checkpoint_sha256, field="baseline_checkpoint_sha256"
        )
        document["baseline_checkpoint_seal_sha256"] = _sha256(
            baseline_checkpoint_seal_sha256, field="baseline_checkpoint_seal_sha256"
        )
    return canonical_value(document)  # type: ignore[return-value]


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
    *,
    workspace_instance_id: str,
    leaf_instance_id: str,
    plan_sha256: str,
    maintenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    leaf = _uuid(leaf_instance_id)
    fields: dict[str, object] = {
        "kind": "replaceable_workspace",
        "workspace_instance_id": _uuid(workspace_instance_id),
        "active_leaf": {
            "directory": f"leaves/{leaf}",
            "workspace_instance_id": leaf,
            "plan_sha256": _sha256(plan_sha256, field="plan_sha256"),
        },
    }
    if maintenance is not None:
        fields["maintenance"] = dict(maintenance)
    return _workspace_document(fields)


def versioned_workspace_document(
    *,
    workspace_instance_id: str,
    iterations: Iterable[Mapping[str, object]],
    maintenance: Mapping[str, object] | None = None,
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
    fields: dict[str, object] = {
        "kind": "versioned_workspace",
        "workspace_instance_id": _uuid(workspace_instance_id),
        "next_iteration": expected,
        "iterations": index,
    }
    if maintenance is not None:
        fields["maintenance"] = dict(maintenance)
    return _workspace_document(fields)


def _workspace_document(fields: Mapping[str, object]) -> dict[str, object]:
    return canonical_value({"schema": "scnsim.workspace", "schema_version": 1, **fields})  # type: ignore[return-value]


def _uuid(value: object) -> str:
    if not isinstance(value, str) or not _UUID4.fullmatch(value):
        raise _validation("workspace identity must be lowercase UUIDv4")
    return value
