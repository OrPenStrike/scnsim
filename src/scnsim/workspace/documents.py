"""Canonical workspace, attempt, Result, and receipt documents.

These builders own stored envelope representation only. Workspace locking,
publication, and evidence verification remain with their lifecycle owners."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from ..canonical import _sha256, _validation, canonical_value
from .validation.common import _UUID4


_DIRECT_RESULTS = frozenset({
    "direct_response", "diagonal_root", "operator_element_root", "hybridized_pole", "transfer_zero",
    "residue_normalized_coupling", "response_element", "operator", "optimization", "hb_batch",
})


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
