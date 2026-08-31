"""Durable workspace binding and receipt-chain verification.

This module owns filesystem state only.  Public request encoding, result
decoding, and Julia execution remain in their respective modules; callers pass
already-canonical document bytes here and receive verified files back.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import sys
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import (
    EvidenceIntegrityError,
    ResultUnavailableError,
    UnsupportedRuntimePlatformError,
    WorkspacePlanReplacedError,
    WorkspaceVersioningDowngradeForbidden,
)

if sys.platform in {"linux", "darwin"}:
    import fcntl


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_ATTEMPT = re.compile(r"^(?!000000$)(?:[0-9]{6}|[1-9][0-9]{6,})$")
_STAGING = re.compile(
    r"^\.staging-((?!000000-)(?:[0-9]{6}|[1-9][0-9]{6,}))"
    r"-([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$"
)
_LEAF_STAGING = re.compile(
    r"^\.staging-leaf-([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$"
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_IDENTIFIER = re.compile(r"^[^/\\\x00-\x1f\x7f]+$")


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    # Canonical serialization is deliberately centralized in _canonical.
    from ._canonical import canonical_json_bytes

    return canonical_json_bytes(value)


def _sha256(data: bytes) -> str:
    from ._canonical import sha256_hex

    return sha256_hex(data)


def _require_platform() -> None:
    if sys.platform not in {"linux", "darwin"}:
        raise UnsupportedRuntimePlatformError(
            "SCNSim workspace mutation is supported only on Linux and macOS.",
            stage="workspace",
            evidence={"platform": sys.platform},
        )


def _integrity(message: str, **evidence: object) -> EvidenceIntegrityError:
    return EvidenceIntegrityError(message, stage="workspace", evidence=evidence)


def _valid_sha(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _integrity("Expected a lowercase SHA-256 digest.", value=value)
    return value


def _valid_uuid(value: object) -> str:
    if not isinstance(value, str) or _UUID4.fullmatch(value) is None:
        raise _integrity("Expected a lowercase canonical UUIDv4.", value=value)
    return value


def _new_uuid(*, excluding: str | None = None) -> str:
    value = str(uuid.uuid4())
    while value == excluding:
        value = str(uuid.uuid4())
    return value


def _relative_path(value: object) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or _CONTROL.search(value) is not None
        or value.startswith("/")
        or "//" in value
    ):
        raise _integrity("Evidence path is not a normalized POSIX relative path.", path=value)
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise _integrity("Evidence path escapes its attempt root.", path=value)
    return Path(*parts)


def _load_canonical(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise _integrity("Evidence JSON must be a regular file.", path=str(path))
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _integrity("Evidence JSON cannot be read.", path=str(path), error=str(error)) from error
    if not isinstance(value, dict):
        raise _integrity("Evidence JSON envelope must be an object.", path=str(path))
    if _canonical_bytes(value) != raw:
        raise _integrity("Evidence JSON is not the required canonical byte stream.", path=str(path))
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink() or path.parent.is_symlink():
        raise _integrity("Evidence write target must not traverse a symlink.", path=str(path))
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    if path.is_symlink():
        raise _integrity("Evidence directory must not be a symlink.", path=str(path))
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(path: Path) -> None:
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise _integrity("Evidence tree contains a symlink.", path=str(child))
        if child.is_file():
            with child.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted((node for node in path.rglob("*") if node.is_dir()), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(path)


@dataclass(frozen=True)
class AttemptAllocation:
    """Reserved, unsealed sibling staging directory for one immutable attempt."""

    request_sha256: str
    ordinal: int
    ordinal_text: str
    staging_directory: Path
    final_directory: Path

    @property
    def attempt_directory_text(self) -> str:
        return f"requests/{self.request_sha256}/attempts/{self.ordinal_text}"

    @property
    def staging_directory_text(self) -> str:
        return (
            f"requests/{self.request_sha256}/attempts/"
            f"{self.staging_directory.name}"
        )


@dataclass(frozen=True)
class VerifiedSuccess:
    """One verified success chain, ready for result reconstruction."""

    request: Mapping[str, object]
    attempt: Mapping[str, object]
    receipt: Mapping[str, object]
    result: Mapping[str, object]
    directory: Path


@dataclass(frozen=True)
class WorkspaceBinding:
    """One Run's concrete, Plan-bound leaf beneath a stable workspace root."""

    root: Path
    leaf: Path
    plan_sha256: str
    workspace_instance_id: str

    @contextmanager
    def writer(self) -> Iterator[WorkspaceBinding]:
        """Hold the root's exclusive lock for a complete durable operation."""

        with _workspace_lock(self.root, exclusive=True):
            self.assert_current()
            self._cleanup_staging()
            yield self

    @contextmanager
    def reader(self) -> Iterator[WorkspaceBinding]:
        """Read one stable chain without acquiring an execution slot."""

        with _workspace_lock(self.root, exclusive=False):
            self.assert_current()
            yield self

    def assert_current(self) -> None:
        """Reject a stale Run before it can read another Plan's leaf."""

        root = _load_canonical(self.root / "workspace.json")
        _assert_root_envelope(root)
        kind = root.get("kind")
        if kind == "replaceable_workspace":
            active = root.get("active_leaf")
            if not isinstance(active, dict):
                raise _integrity("Replaceable workspace lacks an active leaf.")
            expected_directory = f"leaves/{self.workspace_instance_id}"
            matches = (
                active.get("workspace_instance_id") == self.workspace_instance_id
                and active.get("plan_sha256") == self.plan_sha256
                and active.get("directory") == expected_directory
                and self.leaf == self.root / expected_directory
            )
        elif kind == "versioned_workspace":
            iterations = root.get("iterations")
            next_iteration = root.get("next_iteration")
            if not isinstance(iterations, list) or not isinstance(next_iteration, int) or next_iteration < 1:
                raise _integrity("Versioned workspace lacks its iteration index.")
            plans: set[str] = set()
            ordinals: set[int] = set()
            leaf_ids: set[str] = set()
            match: Mapping[str, object] | None = None
            for expected_ordinal, entry in enumerate(iterations, 1):
                if not isinstance(entry, dict) or set(entry) != {"ordinal", "directory", "workspace_instance_id", "plan_sha256"}:
                    raise _integrity("Versioned workspace has an open iteration entry.")
                ordinal = entry.get("ordinal")
                identity = _valid_uuid(entry.get("workspace_instance_id"))
                plan = _valid_sha(entry.get("plan_sha256"))
                if (
                    not isinstance(ordinal, int)
                    or ordinal != expected_ordinal
                    or ordinal in ordinals
                    or plan in plans
                    or identity in leaf_ids
                    or identity == root.get("workspace_instance_id")
                    or entry.get("directory") != f"iteration{ordinal:02d}"
                ):
                    raise _integrity("Versioned workspace iteration index is inconsistent.")
                ordinals.add(ordinal)
                plans.add(plan)
                leaf_ids.add(identity)
                # The root index is shared authority, so its own shape and
                # uniqueness are still checked.  A sibling's leaf evidence is
                # not: a Run is bound to exactly one versioned iteration and
                # read-only APIs must not turn unrelated historical damage
                # into a latest/current selector or a failure of this leaf.
                if identity == self.workspace_instance_id and plan == self.plan_sha256:
                    match = entry
            if sorted(ordinals) != list(range(1, len(ordinals) + 1)) or next_iteration != len(ordinals) + 1:
                raise _integrity("Versioned workspace next_iteration is not canonical.")
            matches = match is not None
            if match is not None:
                directory = match.get("directory")
                ordinal = match.get("ordinal")
                if not isinstance(ordinal, int) or not isinstance(directory, str) or directory != f"iteration{ordinal:02d}":
                    raise _integrity("Versioned workspace points outside an iteration.")
                expected_leaf = self.root / directory
                pre_upgrade_leaf = self.root / "leaves" / self.workspace_instance_id
                if self.leaf == pre_upgrade_leaf:
                    object.__setattr__(self, "leaf", expected_leaf)
                elif self.leaf != expected_leaf:
                    raise WorkspacePlanReplacedError(
                        "This Run's versioned workspace leaf no longer matches its bound iteration.",
                        stage="workspace",
                        evidence={"workspace": str(self.root), "workspace_instance_id": self.workspace_instance_id},
                    )
        else:
            raise _integrity("Workspace root has an unknown kind.", kind=kind)
        if not matches:
            raise WorkspacePlanReplacedError(
                "This Run's workspace leaf was replaced by another topology.",
                stage="workspace",
                evidence={
                    "workspace": str(self.root),
                    "plan_sha256": self.plan_sha256,
                    "workspace_instance_id": self.workspace_instance_id,
                },
            )
        self._verify_leaf()

    def _verify_leaf(self) -> None:
        if self.leaf.is_symlink() or not self.leaf.is_dir():
            raise _integrity("Workspace leaf path is missing or symlinked.", leaf=str(self.leaf))
        leaf = _load_canonical(self.leaf / "workspace.json")
        if (
            set(leaf) != {"schema", "schema_version", "kind", "workspace_instance_id", "plan_sha256"}
            or
            leaf.get("schema") != "scnsim.workspace"
            or leaf.get("schema_version") != 1
            or leaf.get("kind") != "plan_workspace"
            or leaf.get("workspace_instance_id") != self.workspace_instance_id
            or leaf.get("plan_sha256") != self.plan_sha256
        ):
            raise _integrity("Leaf workspace binding disagrees with its Run.", leaf=str(self.leaf))
        plan = self.leaf / "plan.json"
        if plan.is_symlink() or not plan.is_file() or _sha256(plan.read_bytes()) != self.plan_sha256:
            raise _integrity("Leaf plan bytes do not match its sealed identity.", leaf=str(self.leaf))
        requests = self.leaf / "requests"
        if requests.is_symlink() or (requests.exists() and not requests.is_dir()):
            raise _integrity("Workspace requests path is unsafe.", path=str(requests))

    def ensure_request(self, request_sha256: str, request_bytes: bytes) -> Path:
        """Store one immutable canonical request or verify its existing bytes."""

        _valid_sha(request_sha256)
        if _sha256(request_bytes) != request_sha256:
            raise _integrity("Request bytes do not match the supplied request identity.")
        request = _decode_bytes(request_bytes, "request")
        _verify_request_document(request, self.plan_sha256, _load_canonical(self.leaf / "plan.json"))
        if request.get("plan_sha256") != self.plan_sha256:
            raise _integrity("Request Plan identity does not match the bound workspace leaf.")
        requests = self.leaf / "requests"
        if requests.exists() and (requests.is_symlink() or not requests.is_dir()):
            raise _integrity("Workspace requests path is unsafe.", path=str(requests))
        directory = requests / request_sha256
        path = directory / "request.json"
        if path.exists():
            if directory.is_symlink() or not directory.is_dir() or path.is_symlink() or not path.is_file() or path.read_bytes() != request_bytes:
                raise _integrity("Existing request directory contains different evidence.", request_sha256=request_sha256)
            return directory
        directory.mkdir(parents=True, exist_ok=False)
        _atomic_write(path, request_bytes)
        _fsync_directory(directory)
        return directory

    def allocate_attempt(self, request_sha256: str) -> AttemptAllocation:
        """Reserve the next append-only attempt staging directory.

        The child is still blocked: no ``attempt.json`` exists until bootstrap
        observation supplies its truthful allocated or launched evidence.
        """

        request_directory = self.leaf / "requests" / request_sha256
        if request_directory.is_symlink() or not request_directory.is_dir():
            raise _integrity("Request directory is missing or symlinked.", path=str(request_directory))
        request = request_directory / "request.json"
        if request.is_symlink() or not request.is_file() or _sha256(request.read_bytes()) != request_sha256:
            raise _integrity("Attempt allocation requires an exact stored request.", request_sha256=request_sha256)
        attempts = request_directory / "attempts"
        if attempts.exists() and (attempts.is_symlink() or not attempts.is_dir()):
            raise _integrity("Request attempts path is unsafe.", path=str(attempts))
        attempts.mkdir(exist_ok=True)
        ordinal = self._next_attempt_ordinal(attempts)
        text = str(ordinal).zfill(6)
        final = attempts / text
        staging = attempts / f".staging-{text}-{uuid.uuid4()}"
        staging.mkdir()
        _fsync_directory(attempts)
        return AttemptAllocation(request_sha256, ordinal, text, staging, final)

    def seal_attempt(self, allocation: AttemptAllocation, attempt: Mapping[str, object]) -> str:
        """Seal the one allocated/launched attempt envelope before authorization."""

        self._require_allocation(allocation)
        expected = {
            "schema": "scnsim.attempt",
            "schema_version": 1,
            "request_sha256": allocation.request_sha256,
            "ordinal": allocation.ordinal,
            "ordinal_text": allocation.ordinal_text,
            "directory": allocation.attempt_directory_text,
            "staging_directory": allocation.staging_directory_text,
        }
        for key, value in expected.items():
            if attempt.get(key) != value:
                raise _integrity("Attempt envelope disagrees with its allocated path.", field=key)
        if attempt.get("attempt_state") not in {"allocated", "launched"}:
            raise _integrity("Attempt envelope has an invalid state.")
        path = allocation.staging_directory / "attempt.json"
        if path.exists():
            raise _integrity("Attempt envelope is immutable once sealed.", path=str(path))
        raw = _canonical_bytes(dict(attempt))
        _atomic_write(path, raw)
        return _sha256(raw)

    def promote_attempt(self, allocation: AttemptAllocation, receipt: Mapping[str, object]) -> None:
        """Write ``receipt.json`` last, fsync, and atomically publish one attempt."""

        self._require_allocation(allocation)
        attempt_path = allocation.staging_directory / "attempt.json"
        attempt_sha256 = _sha256(_canonical_bytes(_load_canonical(attempt_path)))
        if (
            receipt.get("schema") != "scnsim.receipt"
            or receipt.get("schema_version") != 1
            or receipt.get("request_sha256") != allocation.request_sha256
            or receipt.get("attempt_sha256") != attempt_sha256
            or receipt.get("outcome") not in {"success", "failure", "interrupted"}
        ):
            raise _integrity("Receipt does not bind the sealed attempt.")
        receipt_path = allocation.staging_directory / "receipt.json"
        if receipt_path.exists():
            raise _integrity("Receipt is immutable and must be written last.", path=str(receipt_path))
        _atomic_write(receipt_path, _canonical_bytes(dict(receipt)))
        self._verify_attempt(
            allocation.staging_directory,
            allocation.request_sha256,
            allocation.ordinal_text,
            require_final_name=False,
        )
        _fsync_tree(allocation.staging_directory)
        if allocation.final_directory.exists():
            raise _integrity("Attempt promotion would overwrite final evidence.")
        os.replace(allocation.staging_directory, allocation.final_directory)
        _fsync_directory(allocation.final_directory.parent)
        self._verify_attempt(
            allocation.final_directory,
            allocation.request_sha256,
            allocation.ordinal_text,
            require_final_name=True,
        )

    def find_success(self, request_sha256: str) -> VerifiedSuccess | None:
        """Verify every final attempt and return its sole reusable success."""

        requests = self.leaf / "requests"
        request_directory = requests / request_sha256
        request = request_directory / "request.json"
        if requests.is_symlink() or request_directory.is_symlink() or request.is_symlink():
            raise _integrity("Request evidence is symlinked.", path=str(request))
        if not request.is_file():
            if request.exists() or request_directory.exists() and not request_directory.is_dir():
                raise _integrity("Request evidence is not a regular file.", path=str(request))
            return None
        if _sha256(request.read_bytes()) != request_sha256:
            raise _integrity("Request file hash does not match its directory.", request_sha256=request_sha256)
        request_document = _load_canonical(request)
        if request_document.get("plan_sha256") != self.plan_sha256:
            raise _integrity("Request belongs to another Plan leaf.", request_sha256=request_sha256)
        attempts = request.parent / "attempts"
        if attempts.is_symlink():
            raise _integrity("Request attempts path is symlinked.", path=str(attempts))
        if not attempts.exists():
            return None
        if not attempts.is_dir():
            raise _integrity("Request attempts path is unsafe.", path=str(attempts))
        successful: list[VerifiedSuccess] = []
        for final in self._final_attempt_directories(attempts):
            attempt, receipt, result = self._verify_attempt(final, request_sha256, final.name, require_final_name=True)
            if receipt["outcome"] == "success":
                if result is None:
                    raise _integrity("Successful receipt lacks a Result.", attempt=str(final))
                successful.append(VerifiedSuccess(request_document, attempt, receipt, result, final))
        if len(successful) > 1:
            raise _integrity("One request has competing verified successes.", request_sha256=request_sha256)
        return successful[0] if successful else None

    def resolve_success(self, request_sha256: str) -> VerifiedSuccess:
        """Return exact verified evidence; never start, retry, or select latest."""

        success = self.find_success(request_sha256)
        if success is None:
            raise ResultUnavailableError(
                "No verified success exists for this exact request.",
                stage="resolve",
                evidence={"request_sha256": request_sha256, "workspace": str(self.root)},
            )
        return success

    def resolve_matching_success(
        self,
        *,
        operation: str,
        spec: Mapping[str, object],
        parameters: Mapping[str, object],
        runtime_semantic: Mapping[str, object],
        lazy_lineage: Mapping[str, object],
    ) -> VerifiedSuccess:
        """Read one exact success without compiling, allocating, or selecting latest.

        A View declaration deliberately does not contain compiler-realized
        matrices.  This reader therefore compares its declarative projection
        with each stored realized lineage, while every stored request and final
        attempt remains fully chain-verified.  It is intentionally leaf-local
        and cannot be repurposed as a workspace-wide ``current`` selector.
        """

        if not isinstance(operation, str) or not operation:
            raise TypeError("operation must be a nonempty string")
        for name, value in (
            ("spec", spec),
            ("parameters", parameters),
            ("runtime_semantic", runtime_semantic),
            ("lazy_lineage", lazy_lineage),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")

        # The lock spans enumeration and all artifact reads.  No writer path,
        # cleanup, request construction, or Julia preparation is reachable.
        with self.reader():
            plan = _load_canonical(self.leaf / "plan.json")
            wanted_lineage = _lazy_lineage_projection(lazy_lineage, plan)
            wanted_spec = _canonical_bytes(dict(spec))
            wanted_parameters = _canonical_bytes(dict(parameters))
            wanted_runtime = _canonical_bytes(dict(runtime_semantic))
            requests = self.leaf / "requests"
            if requests.is_symlink() or (requests.exists() and not requests.is_dir()):
                raise _integrity("Workspace requests path is unsafe.", path=str(requests))

            successes: list[VerifiedSuccess] = []
            if requests.exists():
                for request_directory in sorted(requests.iterdir(), key=lambda item: item.name):
                    if (
                        request_directory.is_symlink()
                        or not request_directory.is_dir()
                        or _SHA256.fullmatch(request_directory.name) is None
                    ):
                        raise _integrity("Workspace contains a malformed request directory.", path=str(request_directory))
                    request_sha256 = request_directory.name
                    request_path = request_directory / "request.json"
                    if (
                        request_path.is_symlink()
                        or not request_path.is_file()
                        or _sha256(request_path.read_bytes()) != request_sha256
                    ):
                        raise _integrity("Request file hash does not match its directory.", request_sha256=request_sha256)
                    request = _load_canonical(request_path)
                    _verify_request_document(request, self.plan_sha256, plan)
                    attempts = request_directory / "attempts"
                    if attempts.is_symlink() or not attempts.is_dir():
                        raise _integrity("Resolve request has no final attempts.", request_sha256=request_sha256)
                    finals = self._final_attempt_directories(attempts)
                    if not finals:
                        raise _integrity("Resolve request has no final attempts.", request_sha256=request_sha256)

                    matches = (
                        request.get("operation") == operation
                        and _canonical_bytes(request["spec"]) == wanted_spec
                        and _canonical_bytes(request["parameters"]) == wanted_parameters
                        and _canonical_bytes(request["runtime_semantic"]) == wanted_runtime
                        and _lazy_lineage_projection(request["ref_lineage"], plan) == wanted_lineage
                    )
                    for final in finals:
                        attempt, receipt, result = self._verify_attempt(
                            final, request_sha256, final.name, require_final_name=True
                        )
                        if matches and receipt["outcome"] == "success":
                            if result is None:
                                raise _integrity("Successful receipt lacks a Result.", attempt=str(final))
                            successes.append(VerifiedSuccess(request, attempt, receipt, result, final))

            if not successes:
                raise ResultUnavailableError(
                    "No verified success exists for this exact declared request.",
                    stage="resolve",
                    evidence={"operation": operation, "workspace": str(self.root)},
                )
            if len(successes) != 1:
                raise _integrity(
                    "Multiple verified successes match one declared request.",
                    operation=operation,
                    workspace=str(self.root),
                )
            return successes[0]

    def inventory_document(self) -> dict[str, object]:
        """Verify and summarize only this Run's bound immutable leaf.

        This is intentionally a leaf-local reader: callers must already hold
        :meth:`reader`, so it neither cleans staging nor chooses a result for
        any later operation.
        """

        self.assert_current()
        requests = self.leaf / "requests"
        if requests.is_symlink() or (requests.exists() and not requests.is_dir()):
            raise _integrity("Workspace requests path is unsafe.", path=str(requests))
        rows: list[dict[str, object]] = []
        if requests.exists():
            for request_directory in sorted(requests.iterdir(), key=lambda item: item.name):
                if (
                    request_directory.is_symlink()
                    or not request_directory.is_dir()
                    or _SHA256.fullmatch(request_directory.name) is None
                ):
                    raise _integrity("Workspace contains a malformed request directory.", path=str(request_directory))
                request_sha256 = request_directory.name
                request_path = request_directory / "request.json"
                if request_path.is_symlink() or not request_path.is_file() or _sha256(request_path.read_bytes()) != request_sha256:
                    raise _integrity("Request file hash does not match its directory.", request_sha256=request_sha256)
                request = _load_canonical(request_path)
                _verify_request_document(request, self.plan_sha256, _load_canonical(self.leaf / "plan.json"))
                attempts = request_directory / "attempts"
                if attempts.is_symlink() or not attempts.is_dir():
                    raise _integrity("Inventory request has no final attempts.", request_sha256=request_sha256)
                finals = self._final_attempt_directories(attempts)
                if not finals:
                    raise _integrity("Inventory request has no final attempts.", request_sha256=request_sha256)
                outcomes: list[str] = []
                for final in finals:
                    _attempt, receipt, _result = self._verify_attempt(
                        final, request_sha256, final.name, require_final_name=True
                    )
                    outcome = receipt.get("outcome")
                    if outcome not in {"success", "failure", "interrupted"}:
                        raise _integrity("Inventory final attempt lacks a terminal outcome.", attempt=str(final))
                    outcomes.append(outcome)
                status = "succeeded" if "success" in outcomes else "failed" if outcomes[-1] == "failure" else "interrupted"
                if status not in {"succeeded", "failed", "interrupted"}:
                    raise _integrity("Inventory status cannot be determined.", request_sha256=request_sha256)
                rows.append({
                    "request_sha256": request_sha256,
                    "operation": request["operation"],
                    "status": status,
                    "attempts": [final.name for final in finals],
                })
        return {
            "schema": "scnsim.inventory",
            "schema_version": 1,
            "workspace_instance_id": self.workspace_instance_id,
            "plan_sha256": self.plan_sha256,
            "requests": rows,
        }

    def resume_ledger_sha256(self, request_sha256: str) -> str | None:
        """Return the latest attempt's highest verified CMA generation ledger."""

        attempts = self.leaf / "requests" / request_sha256 / "attempts"
        if attempts.is_symlink():
            raise _integrity("Request attempts path is symlinked.", path=str(attempts))
        if not attempts.exists():
            return None
        if not attempts.is_dir():
            raise _integrity("Request attempts path is unsafe.", path=str(attempts))
        candidates: list[tuple[int, str]] = []
        for final in self._final_attempt_directories(attempts):
            attempt, receipt, _ = self._verify_attempt(
                final, request_sha256, final.name, require_final_name=True
            )
            if receipt["outcome"] == "success":
                continue
            ledgers = _verify_generation_artifacts(
                final,
                receipt["artifacts"],
                request_sha256=request_sha256,
                attempt_sha256=_sha256(_canonical_bytes(attempt)),
            )
            if ledgers:
                generation, digest = ledgers[-1]
                candidates.append((generation, digest))
        if not candidates:
            return None
        highest = max(generation for generation, _ in candidates)
        digests = {digest for generation, digest in candidates if generation == highest}
        if len(digests) != 1:
            raise _integrity(
                "Equal-generation resume ledgers have different identities.",
                generation=highest,
            )
        return digests.pop()

    def _cleanup_staging(self) -> None:
        """Remove only canonical crash leftovers while holding the exclusive lock."""

        requests = self.leaf / "requests"
        if requests.is_symlink():
            raise _integrity("Workspace requests path is symlinked.", path=str(requests))
        if not requests.exists():
            return
        for request in requests.iterdir():
            if request.is_symlink() or not request.is_dir() or _SHA256.fullmatch(request.name) is None:
                raise _integrity("Workspace contains a malformed request directory.", path=str(request))
            attempts = request / "attempts"
            if attempts.is_symlink():
                raise _integrity("Workspace contains an unsafe attempts path.", path=str(attempts))
            if not attempts.exists():
                continue
            if not attempts.is_dir():
                raise _integrity("Workspace contains an unsafe attempts path.", path=str(attempts))
            for child in attempts.iterdir():
                if child.name.startswith(".staging-"):
                    match = _STAGING.fullmatch(child.name)
                    if match is None or not child.is_dir() or child.is_symlink():
                        raise _integrity("Workspace contains malformed staging evidence.", path=str(child))
                    shutil.rmtree(child)
                    _fsync_directory(attempts)

    def _next_attempt_ordinal(self, attempts: Path) -> int:
        ordinals: list[int] = []
        for child in attempts.iterdir():
            if child.name.startswith(".staging-"):
                raise _integrity("Staging cleanup must run before attempt allocation.", path=str(child))
            if child.is_symlink() or not child.is_dir() or _ATTEMPT.fullmatch(child.name) is None:
                raise _integrity("Workspace contains malformed final attempt evidence.", path=str(child))
            ordinals.append(int(child.name))
        if sorted(ordinals) != list(range(1, len(ordinals) + 1)):
            raise _integrity("Attempt ordinals are not a contiguous append-only sequence.")
        return len(ordinals) + 1

    def _require_allocation(self, allocation: AttemptAllocation) -> None:
        if allocation.request_sha256 == "" or allocation.staging_directory.parent != allocation.final_directory.parent:
            raise _integrity("Attempt allocation does not belong to one request directory.")
        if allocation.staging_directory.parent != self.leaf / "requests" / allocation.request_sha256 / "attempts":
            raise _integrity("Attempt allocation belongs to another workspace leaf.")
        if (
            allocation.staging_directory.parent.is_symlink()
            or allocation.staging_directory.is_symlink()
            or not allocation.staging_directory.is_dir()
            or allocation.final_directory.is_symlink()
            or allocation.final_directory.exists()
        ):
            raise _integrity("Attempt allocation is no longer a writable staging directory.")

    def _final_attempt_directories(self, attempts: Path) -> list[Path]:
        if attempts.is_symlink() or not attempts.is_dir():
            raise _integrity("Request attempts path is unsafe.", path=str(attempts))
        result: list[Path] = []
        for child in attempts.iterdir():
            if child.name.startswith(".staging-"):
                if _STAGING.fullmatch(child.name) is None:
                    raise _integrity("Workspace contains malformed staging evidence.", path=str(child))
                continue
            if not child.is_dir() or child.is_symlink() or _ATTEMPT.fullmatch(child.name) is None:
                raise _integrity("Workspace contains malformed final attempt evidence.", path=str(child))
            result.append(child)
        result.sort(key=lambda path: int(path.name))
        if [int(path.name) for path in result] != list(range(1, len(result) + 1)):
            raise _integrity("Attempt ordinals are not a contiguous append-only sequence.")
        return result

    def _verify_attempt(
        self,
        directory: Path,
        request_sha256: str,
        ordinal_text: str,
        *,
        require_final_name: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        if directory.is_symlink() or not directory.is_dir() or directory.parent.is_symlink() or directory.parent.parent.is_symlink():
            raise _integrity("Attempt path traverses a symlink.", path=str(directory))
        if require_final_name and directory.name != ordinal_text:
            raise _integrity("Final attempt directory does not match its ordinal.", path=str(directory))
        request_path = directory.parent.parent / "request.json"
        if request_path.is_symlink() or not request_path.is_file() or _sha256(request_path.read_bytes()) != request_sha256:
            raise _integrity("Attempt request bytes do not match their identity.", attempt=str(directory))
        request_document = _load_canonical(request_path)
        plan_document = _load_canonical(self.leaf / "plan.json")
        _verify_request_document(request_document, self.plan_sha256, plan_document)
        attempt_path = directory / "attempt.json"
        receipt_path = directory / "receipt.json"
        attempt = _load_canonical(attempt_path)
        receipt = _load_canonical(receipt_path)
        expected_directory = f"requests/{request_sha256}/attempts/{ordinal_text}"
        staging_directory = attempt.get("staging_directory")
        if (
            attempt.get("schema") != "scnsim.attempt"
            or attempt.get("schema_version") != 1
            or attempt.get("request_sha256") != request_sha256
            or attempt.get("ordinal_text") != ordinal_text
            or attempt.get("ordinal") != int(ordinal_text)
            or attempt.get("directory") != expected_directory
            or not isinstance(staging_directory, str)
            or _STAGING.fullmatch(Path(staging_directory).name) is None
            or staging_directory != f"requests/{request_sha256}/attempts/{Path(staging_directory).name}"
        ):
            raise _integrity("Attempt envelope has inconsistent path evidence.", attempt=str(directory))
        state = attempt.get("attempt_state")
        attempt_fields = {
            "schema", "schema_version", "request_sha256", "ordinal", "ordinal_text",
            "directory", "staging_directory", "attempt_state", "started_at_utc",
            "julia_executable_sha256", "os", "architecture", "cpu",
        }
        if state == "launched":
            attempt_fields.update({"julia_threads", "blas_threads", "blas_vendor"})
        if attempt.get("resume_ledger_sha256") is not None:
            attempt_fields.add("resume_ledger_sha256")
        if set(attempt) != attempt_fields:
            raise _integrity("Attempt envelope is open or has state-incompatible fields.", attempt=str(directory))
        if (
            state not in {"allocated", "launched"}
            or not isinstance(attempt.get("started_at_utc"), str)
            or not str(attempt["started_at_utc"]).endswith("Z")
            or _SHA256.fullmatch(str(attempt.get("julia_executable_sha256", ""))) is None
            or any(not isinstance(attempt.get(key), str) or not attempt[key] for key in ("os", "architecture", "cpu"))
        ):
            raise _integrity("Attempt envelope lacks required machine evidence.", attempt=str(directory))
        if state == "launched":
            if any(not isinstance(attempt.get(key), int) or attempt[key] < 1 for key in ("julia_threads", "blas_threads")) or not isinstance(attempt.get("blas_vendor"), str) or not attempt["blas_vendor"]:
                raise _integrity("Launched attempt lacks child runtime evidence.", attempt=str(directory))
        elif any(key in attempt for key in ("julia_threads", "blas_threads", "blas_vendor")):
            raise _integrity("Allocated attempt must not claim child runtime evidence.", attempt=str(directory))
        attempt_sha256 = _sha256(_canonical_bytes(attempt))
        resume = attempt.get("resume_ledger_sha256")
        if resume is not None:
            resume = _valid_sha(resume)
            found = False
            for sibling in self._final_attempt_directories(directory.parent):
                if int(sibling.name) >= int(ordinal_text):
                    continue
                producer_attempt, producer_receipt, _ = self._verify_attempt(
                    sibling,
                    request_sha256,
                    sibling.name,
                    require_final_name=True,
                )
                ledgers = _verify_generation_artifacts(
                    sibling,
                    producer_receipt["artifacts"],
                    request_sha256=request_sha256,
                    attempt_sha256=_sha256(_canonical_bytes(producer_attempt)),
                )
                if any(digest == resume for _, digest in ledgers):
                    found = True
                    break
            if not found:
                raise _integrity("Attempt resume ledger is absent from prior finalized evidence.")
        if (
            receipt.get("schema") != "scnsim.receipt"
            or receipt.get("schema_version") != 1
            or receipt.get("request_sha256") != request_sha256
            or receipt.get("attempt_sha256") != attempt_sha256
        ):
            raise _integrity("Receipt does not bind its request and attempt.", attempt=str(directory))
        outcome = receipt.get("outcome")
        if outcome not in {"success", "failure", "interrupted"}:
            raise _integrity("Receipt has no terminal outcome.", attempt=str(directory))
        if not isinstance(receipt.get("sealed_at_utc"), str) or not str(receipt["sealed_at_utc"]).endswith("Z") or not isinstance(receipt.get("artifacts"), list) or not isinstance(receipt.get("evidence"), dict):
            raise _integrity("Receipt lacks required terminal evidence.", attempt=str(directory))
        receipt_fields = {
            "schema", "schema_version", "request_sha256", "attempt_sha256", "outcome",
            "artifacts", "evidence", "sealed_at_utc",
        }
        if outcome == "success":
            receipt_fields.update({"outcome_sha256", "result_sha256"})
        elif outcome == "failure":
            receipt_fields.add("failure")
            if receipt.get("outcome_sha256") is not None:
                receipt_fields.add("outcome_sha256")
        else:
            receipt_fields.add("interruption")
            if receipt.get("outcome_sha256") is not None:
                receipt_fields.add("outcome_sha256")
        if set(receipt) != receipt_fields:
            raise _integrity("Receipt envelope is open or has outcome-incompatible fields.", attempt=str(directory))
        evidence = receipt["evidence"]
        if set(evidence) != {"runtime_semantic_sha256", "source_units", "extrapolation_evidence", "provenance_sha256", "evidence_sha256"}:
            raise _integrity("Receipt evidence envelope is open.", attempt=str(directory))
        evidence_without_hash = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
        source_units = evidence.get("source_units")
        if not isinstance(source_units, list) or any(
            not isinstance(item, dict)
            or set(item) != {"identity", "source_unit", "canonical_si_unit", "canonical_dimensionality"}
            or any(not isinstance(item[field], str) or not item[field] for field in item)
            for item in source_units
        ):
            raise _integrity("Receipt source-unit evidence is malformed.", attempt=str(directory))
        if [item["identity"] for item in source_units] != sorted({item["identity"] for item in source_units}):
            raise _integrity("Receipt source-unit evidence is not sorted and unique.", attempt=str(directory))
        _verify_extrapolation_evidence(
            evidence.get("extrapolation_evidence"),
            allowed_sources={"parameter_set"},
            required_rows=[] if request_document.get("operation") == "optimize_direct" else _required_extrapolation_rows(
                plan_document, request_document["parameters"],
                authorization_source="parameter_set", require_authorized=outcome == "success",
            ),
        )
        expected_provenance = _sha256(_canonical_bytes({"schema": "scnsim.receipt_provenance", "source_units": source_units}))
        if (
            evidence.get("runtime_semantic_sha256") != _sha256(_canonical_bytes(request_document.get("runtime_semantic")))
            or not isinstance(source_units, list)
            or evidence.get("provenance_sha256") != expected_provenance
            or evidence.get("evidence_sha256") != _sha256(_canonical_bytes(evidence_without_hash))
        ):
            raise _integrity("Receipt evidence hashes do not match their exact sources.", attempt=str(directory))
        result: dict[str, Any] | None = None
        outcome_document: dict[str, Any] | None = None
        outcome_path = directory / "outcome.json"
        outcome_sha = receipt.get("outcome_sha256")
        if outcome_sha is not None:
            _valid_sha(outcome_sha)
            outcome_document = _load_canonical(outcome_path)
            if _sha256(_canonical_bytes(outcome_document)) != outcome_sha:
                raise _integrity("Receipt outcome hash does not match outcome bytes.", attempt=str(directory))
            outcome_fields = {
                "schema", "schema_version", "request_sha256", "attempt_sha256",
                "runtime_semantic", "status", "artifacts",
                "result_sha256" if outcome == "success" else "failure" if outcome == "failure" else "interruption",
            }
            if (
                outcome_document.get("schema") != "scnsim.outcome"
                or outcome_document.get("schema_version") != 1
                or set(outcome_document) != outcome_fields
                or outcome_document.get("request_sha256") != request_sha256
                or outcome_document.get("attempt_sha256") != attempt_sha256
                or outcome_document.get("status") != outcome
                or outcome_document.get("runtime_semantic") != request_document.get("runtime_semantic")
            ):
                raise _integrity("Outcome envelope does not match its receipt.", attempt=str(directory))
            _compare_artifacts(outcome_document.get("artifacts"), receipt.get("artifacts"))
        elif outcome_path.exists():
            raise _integrity("An unlinked outcome.json is not authoritative evidence.", attempt=str(directory))
        if outcome == "success":
            if outcome_document is None:
                raise _integrity("Successful receipt lacks its authoritative outcome.", attempt=str(directory))
            result_path = directory / "result.json"
            result_sha = receipt.get("result_sha256")
            if not isinstance(result_sha, str):
                raise _integrity("Success receipt result hash does not match result bytes.", attempt=str(directory))
            result = _load_canonical(result_path)
            if _sha256(_canonical_bytes(result)) != result_sha:
                raise _integrity("Success receipt result hash does not match result bytes.", attempt=str(directory))
            expected_result_kind = (
                "direct_response" if request_document.get("operation") == "solve_direct"
                else "optimization" if request_document.get("operation") == "optimize_direct"
                else request_document.get("spec", {}).get("type") if request_document.get("operation") == "evaluate_direct" and isinstance(request_document.get("spec"), dict)
                else None
            )
            if (
                result.get("schema") != "scnsim.result"
                or result.get("request_sha256") != request_sha256
                or result.get("attempt_sha256") != attempt_sha256
                or result.get("result_kind") != expected_result_kind
            ):
                raise _integrity("Result envelope does not match its success receipt.", attempt=str(directory))
            if outcome_document.get("result_sha256") != result_sha:
                raise _integrity("Outcome and receipt bind different Result identities.", attempt=str(directory))
            _verify_result_document(result, request_document, request_sha256, attempt_sha256, plan_document)
            _verify_artifact_inventory(directory, result, receipt)
            if result.get("result_kind") == "optimization":
                _verify_generation_artifacts(
                    directory,
                    receipt["artifacts"],
                    request_sha256=request_sha256,
                    attempt_sha256=attempt_sha256,
                )
        elif receipt.get("result_sha256") is not None or (directory / "result.json").exists():
            raise _integrity("Non-success evidence must not retain a Result.", attempt=str(directory))
        else:
            _verify_generation_artifacts(
                directory,
                receipt["artifacts"],
                request_sha256=request_sha256,
                attempt_sha256=attempt_sha256,
            )
            if outcome_sha is not None:
                linked = "failure" if outcome == "failure" else "interruption"
                if receipt.get(linked) != outcome_document.get(linked):
                    raise _integrity(
                        f"{linked.capitalize()} receipt does not match its authoritative outcome.",
                        attempt=str(directory),
                    )
        if outcome == "failure":
            _verify_failure_document(receipt.get("failure"), request_document.get("operation"))
        elif outcome == "interrupted":
            interruption = receipt.get("interruption")
            if (
                not isinstance(interruption, dict)
                or set(interruption) != {"kind", "termination", "interrupted_at_utc"}
                or interruption.get("kind") != "keyboard_interrupt"
                or interruption.get("termination") not in {"terminated", "killed_after_grace"}
                or not isinstance(interruption.get("interrupted_at_utc"), str)
                or not interruption["interrupted_at_utc"].endswith("Z")
            ):
                raise _integrity("Interruption evidence is open or malformed.", attempt=str(directory))
        if outcome in {"success", "failure"} and outcome_sha is None:
            failure = receipt.get("failure")
            if not (outcome == "failure" and isinstance(failure, dict) and failure.get("kind") == "backend_protocol"):
                raise _integrity("Completed terminal evidence requires a valid outcome envelope.", attempt=str(directory))
        _verify_attempt_layout(directory, outcome=outcome, has_authoritative_outcome=outcome_sha is not None)
        return attempt, receipt, result


@contextmanager
def _workspace_lock(root: Path, *, exclusive: bool) -> Iterator[None]:
    _require_platform()
    if root.is_symlink() or not root.is_dir():
        raise _integrity("Workspace root is missing or symlinked.", path=str(root))
    lock_path = root / ".scnsim.lock"
    if lock_path.is_symlink():
        raise _integrity("Workspace lock path must not be a symlink.", path=str(lock_path))
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)  # type: ignore[name-defined]
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[name-defined]


def bind_workspace(
    workspace: str | os.PathLike[str],
    *,
    plan_sha256: str,
    plan_bytes: bytes,
    versioned: bool,
) -> WorkspaceBinding:
    """Bind one sealed Plan to replacement or history-preserving evidence.

    The caller must have sealed the Plan before calling this function.  This is
    the only operation that creates/replaces a workspace leaf.
    """

    _require_platform()
    _valid_sha(plan_sha256)
    if _sha256(plan_bytes) != plan_sha256:
        raise _integrity("Sealed Plan bytes do not match their identity.")
    plan = _decode_bytes(plan_bytes, "plan")
    if plan.get("schema") != "scnsim.plan" or plan.get("schema_version") != 1:
        raise _integrity("Plan bytes are not a V1 Plan envelope.")
    root = Path(workspace).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    with _workspace_lock(root, exclusive=True):
        state_path = root / "workspace.json"
        if not state_path.exists():
            return _recover_or_create_root(root, plan_sha256, plan_bytes, versioned)
        root_state = _load_canonical(state_path)
        _assert_root_envelope(root_state)
        kind = root_state.get("kind")
        if kind == "replaceable_workspace":
            current = _binding_from_replaceable(root, root_state)
            current._verify_leaf()
            if not versioned and current.plan_sha256 != plan_sha256:
                recovered = _adopt_replaceable_orphan(root, root_state, current, plan_sha256)
                if recovered is not None:
                    return recovered
            _cleanup_replaceable_staging(root, current)
            if versioned:
                return _upgrade_to_versioned(root, root_state, current)
            if current.plan_sha256 == plan_sha256:
                current.assert_current()
                return current
            new = _create_leaf(root / "leaves", plan_sha256, plan_bytes, excluding=str(root_state["workspace_instance_id"]))
            root_state["active_leaf"] = _leaf_pointer(new, root)
            _atomic_write(state_path, _canonical_bytes(root_state))
            _remove_leaf(current.leaf)
            return new
        if kind == "versioned_workspace":
            if not versioned:
                raise WorkspaceVersioningDowngradeForbidden(
                    "This workspace preserves topology history; choose another workspace for replacement mode.",
                    stage="workspace",
                    evidence={"workspace": str(root)},
                )
            return _bind_versioned(root, root_state, plan_sha256, plan_bytes)
        raise _integrity("Workspace root has an unknown kind.", kind=kind)


def _create_root(root: Path, plan_sha256: str, plan_bytes: bytes, versioned: bool) -> WorkspaceBinding:
    if versioned:
        leaf = _create_leaf(root, plan_sha256, plan_bytes, directory="iteration01")
        state: dict[str, object] = {
            "schema": "scnsim.workspace",
            "schema_version": 1,
            "kind": "versioned_workspace",
            "workspace_instance_id": _new_uuid(excluding=leaf.workspace_instance_id),
            "next_iteration": 2,
            "iterations": [{"ordinal": 1, **_leaf_pointer(leaf, root)}],
        }
    else:
        leaf = _create_leaf(root / "leaves", plan_sha256, plan_bytes)
        state = {
            "schema": "scnsim.workspace",
            "schema_version": 1,
            "kind": "replaceable_workspace",
            "workspace_instance_id": _new_uuid(excluding=leaf.workspace_instance_id),
            "active_leaf": _leaf_pointer(leaf, root),
        }
    _atomic_write(root / "workspace.json", _canonical_bytes(state))
    return leaf


def _recover_or_create_root(
    root: Path,
    plan_sha256: str,
    plan_bytes: bytes,
    versioned: bool,
) -> WorkspaceBinding:
    """Finish only a recognizable interrupted initial root creation.

    A final leaf without its root pointer is retained and either adopted when it
    exactly matches this bind or rejected untouched.  Hidden leaf staging is
    the sole disposable initial-creation state.
    """

    _cleanup_initial_leaf_staging(root)
    entries = {child.name: child for child in root.iterdir() if child.name != ".scnsim.lock"}
    if not entries:
        return _create_root(root, plan_sha256, plan_bytes, versioned)
    if versioned and set(entries) == {"iteration01"}:
        leaf = _recover_leaf(root, entries["iteration01"], plan_sha256, expected_directory="iteration01")
        state: dict[str, object] = {
            "schema": "scnsim.workspace",
            "schema_version": 1,
            "kind": "versioned_workspace",
            "workspace_instance_id": _new_uuid(excluding=leaf.workspace_instance_id),
            "next_iteration": 2,
            "iterations": [{"ordinal": 1, **_leaf_pointer(leaf, root)}],
        }
        _atomic_write(root / "workspace.json", _canonical_bytes(state))
        return leaf
    if not versioned and set(entries) == {"leaves"}:
        leaves = entries["leaves"]
        if leaves.is_symlink() or not leaves.is_dir():
            raise _integrity("Interrupted replacement workspace has an unsafe leaves directory.")
        children = list(leaves.iterdir())
        if len(children) == 1 and _UUID4.fullmatch(children[0].name) is not None:
            leaf = _recover_leaf(root, children[0], plan_sha256, expected_directory=f"leaves/{children[0].name}")
            state = {
                "schema": "scnsim.workspace",
                "schema_version": 1,
                "kind": "replaceable_workspace",
                "workspace_instance_id": _new_uuid(excluding=leaf.workspace_instance_id),
                "active_leaf": _leaf_pointer(leaf, root),
            }
            _atomic_write(root / "workspace.json", _canonical_bytes(state))
            return leaf
    raise _integrity("Workspace has unbound initial-creation evidence; refusing to delete it.", workspace=str(root))


def _recover_leaf(root: Path, leaf_path: Path, plan_sha256: str, *, expected_directory: str) -> WorkspaceBinding:
    if leaf_path.is_symlink() or not leaf_path.is_dir():
        raise _integrity("Interrupted workspace leaf is unsafe.", leaf=str(leaf_path))
    leaf = _load_canonical(leaf_path / "workspace.json")
    if (
        set(leaf) != {"schema", "schema_version", "kind", "workspace_instance_id", "plan_sha256"}
        or leaf.get("schema") != "scnsim.workspace"
        or leaf.get("schema_version") != 1
        or leaf.get("kind") != "plan_workspace"
        or leaf.get("plan_sha256") != plan_sha256
    ):
        raise _integrity("Interrupted workspace leaf cannot be safely adopted.", leaf=str(leaf_path))
    identity = _valid_uuid(leaf.get("workspace_instance_id"))
    binding = WorkspaceBinding(root, leaf_path, plan_sha256, identity)
    if _leaf_pointer(binding, root).get("directory") != expected_directory:
        raise _integrity("Interrupted workspace leaf has an unexpected path.", leaf=str(leaf_path))
    binding._verify_leaf()
    return binding


def _cleanup_initial_leaf_staging(root: Path) -> None:
    parents = [root]
    leaves = root / "leaves"
    if leaves.exists():
        if leaves.is_symlink() or not leaves.is_dir():
            raise _integrity("Initial workspace leaves directory is unsafe.")
        parents.append(leaves)
    for parent in parents:
        for child in parent.iterdir():
            if not child.name.startswith(".staging-leaf-"):
                continue
            if child.is_symlink() or not child.is_dir() or _LEAF_STAGING.fullmatch(child.name) is None:
                raise _integrity("Initial workspace staging path is malformed.", path=str(child))
            shutil.rmtree(child)
        _fsync_directory(parent)
    if leaves.exists() and not any(leaves.iterdir()):
        leaves.rmdir()
    _fsync_directory(root)


def _create_leaf(
    parent: Path,
    plan_sha256: str,
    plan_bytes: bytes,
    *,
    directory: str | None = None,
    excluding: str | None = None,
) -> WorkspaceBinding:
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise _integrity("Workspace leaf parent is unsafe.", path=str(parent))
    parent.mkdir(parents=True, exist_ok=True)
    leaf_id = _new_uuid(excluding=excluding)
    target = parent / (directory or leaf_id)
    staging = parent / f".staging-leaf-{leaf_id}"
    if target.exists() or staging.exists():
        raise _integrity("Workspace leaf allocation would overwrite evidence.", target=str(target))
    staging.mkdir()
    leaf_state: dict[str, object] = {
        "schema": "scnsim.workspace",
        "schema_version": 1,
        "kind": "plan_workspace",
        "workspace_instance_id": leaf_id,
        "plan_sha256": plan_sha256,
    }
    _atomic_write(staging / "workspace.json", _canonical_bytes(leaf_state))
    _atomic_write(staging / "plan.json", plan_bytes)
    _fsync_tree(staging)
    os.replace(staging, target)
    _fsync_directory(parent)
    return WorkspaceBinding(target.parents[1] if target.parent.name == "leaves" else target.parent, target, plan_sha256, leaf_id)


def _leaf_pointer(binding: WorkspaceBinding, root: Path) -> dict[str, object]:
    return {
        "directory": binding.leaf.relative_to(root).as_posix(),
        "workspace_instance_id": binding.workspace_instance_id,
        "plan_sha256": binding.plan_sha256,
    }


def _binding_from_replaceable(root: Path, state: Mapping[str, object]) -> WorkspaceBinding:
    active = state.get("active_leaf")
    if not isinstance(active, dict):
        raise _integrity("Replaceable workspace has no active leaf.")
    directory = active.get("directory")
    leaf_id = _valid_uuid(active.get("workspace_instance_id"))
    plan_sha256 = _valid_sha(active.get("plan_sha256"))
    if directory != f"leaves/{leaf_id}":
        raise _integrity("Replaceable workspace points outside its leaves directory.", directory=directory)
    return WorkspaceBinding(root, root / _relative_path(directory), plan_sha256, leaf_id)


def _cleanup_replaceable_staging(root: Path, current: WorkspaceBinding) -> None:
    leaves = root / "leaves"
    if leaves.is_symlink() or not leaves.is_dir():
        raise _integrity("Replaceable workspace leaves directory is absent.")
    for child in leaves.iterdir():
        name = child.name
        if child == current.leaf:
            continue
        if child.is_symlink() or not child.is_dir():
            raise _integrity("Replaceable workspace contains a malformed leaf path.", path=str(child))
        if _LEAF_STAGING.fullmatch(name) is not None:
            shutil.rmtree(child)
            continue
        if _UUID4.fullmatch(name) is None:
            raise _integrity("Replaceable workspace has a malformed leaf name.", path=str(child))
        orphan = _verified_unbound_leaf(root, child, f"leaves/{name}")
        # A valid unbound leaf is the old side of a pointer-switch crash.  The
        # active leaf has already been verified, so completing replacement is safe.
        _remove_leaf(orphan.leaf)
    _fsync_directory(leaves)


def _adopt_replaceable_orphan(
    root: Path,
    state: Mapping[str, object],
    current: WorkspaceBinding,
    requested_plan_sha256: str,
) -> WorkspaceBinding | None:
    leaves = root / "leaves"
    matches: list[WorkspaceBinding] = []
    for child in leaves.iterdir():
        if child == current.leaf or _LEAF_STAGING.fullmatch(child.name) is not None:
            continue
        if child.is_symlink() or not child.is_dir() or _UUID4.fullmatch(child.name) is None:
            raise _integrity("Replaceable workspace contains malformed orphan evidence.", path=str(child))
        orphan = _verified_unbound_leaf(root, child, f"leaves/{child.name}")
        requests = child / "requests"
        if requests.exists() and (requests.is_symlink() or not requests.is_dir() or any(requests.iterdir())):
            continue
        if orphan.plan_sha256 == requested_plan_sha256:
            matches.append(orphan)
    if not matches:
        return None
    if len(matches) != 1:
        raise _integrity("Replaceable workspace has competing replacement leaves.")
    recovered = matches[0]
    updated = dict(state)
    updated["active_leaf"] = _leaf_pointer(recovered, root)
    _atomic_write(root / "workspace.json", _canonical_bytes(updated))
    _remove_leaf(current.leaf)
    _cleanup_replaceable_staging(root, recovered)
    return recovered


def _verified_unbound_leaf(root: Path, path: Path, expected_directory: str) -> WorkspaceBinding:
    state = _load_canonical(path / "workspace.json")
    return _recover_leaf(
        root,
        path,
        _valid_sha(state.get("plan_sha256")),
        expected_directory=expected_directory,
    )


def _assert_root_envelope(state: Mapping[str, object]) -> None:
    root_id = state.get("workspace_instance_id")
    kind = state.get("kind")
    if (
        state.get("schema") != "scnsim.workspace"
        or state.get("schema_version") != 1
        or kind not in {"replaceable_workspace", "versioned_workspace"}
        or not isinstance(root_id, str)
        or _UUID4.fullmatch(root_id) is None
    ):
        raise _integrity("Workspace root is not a valid V1 workspace envelope.")
    if kind == "replaceable_workspace":
        active = state.get("active_leaf")
        if set(state) != {"schema", "schema_version", "kind", "workspace_instance_id", "active_leaf"} or not isinstance(active, dict) or set(active) != {"directory", "workspace_instance_id", "plan_sha256"}:
            raise _integrity("Replaceable workspace envelope is open or malformed.")
        leaf_id = _valid_uuid(active.get("workspace_instance_id"))
        if leaf_id == root_id or active.get("directory") != f"leaves/{leaf_id}":
            raise _integrity("Replaceable workspace active pointer is not canonical.")
        _valid_sha(active.get("plan_sha256"))
    else:
        if set(state) != {"schema", "schema_version", "kind", "workspace_instance_id", "next_iteration", "iterations"}:
            raise _integrity("Versioned workspace envelope is open or malformed.")


def _upgrade_to_versioned(root: Path, state: Mapping[str, object], current: WorkspaceBinding) -> WorkspaceBinding:
    current._verify_leaf()
    if any(path.is_symlink() for path in current.leaf.rglob("*")):
        raise _integrity("Workspace conversion refuses symlinked evidence.")
    destination = root / "iteration01"
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir() or any(path.is_symlink() for path in destination.rglob("*")):
            raise _integrity("Interrupted workspace upgrade left unsafe iteration01 evidence.")
        # The replaceable active leaf remains authoritative until the root
        # index switch.  An unindexed iteration01 is therefore a disposable,
        # possibly partial copy from that interrupted conversion.
        shutil.rmtree(destination)
        _fsync_directory(root)
    shutil.copytree(current.leaf, destination)
    _fsync_tree(destination)
    upgraded = WorkspaceBinding(root, destination, current.plan_sha256, current.workspace_instance_id)
    upgraded._verify_leaf()
    index: dict[str, object] = {
        "schema": "scnsim.workspace",
        "schema_version": 1,
        "kind": "versioned_workspace",
        "workspace_instance_id": _new_uuid(excluding=upgraded.workspace_instance_id),
        "next_iteration": 2,
        "iterations": [{"ordinal": 1, **_leaf_pointer(upgraded, root)}],
    }
    _atomic_write(root / "workspace.json", _canonical_bytes(index))
    _remove_leaf(current.leaf)
    leaves = root / "leaves"
    if leaves.exists() and not any(leaves.iterdir()):
        leaves.rmdir()
        _fsync_directory(root)
    return upgraded


def _bind_versioned(root: Path, state: Mapping[str, object], plan_sha256: str, plan_bytes: bytes) -> WorkspaceBinding:
    iterations = state.get("iterations")
    next_iteration = state.get("next_iteration")
    if not isinstance(iterations, list) or not isinstance(next_iteration, int) or next_iteration < 1:
        raise _integrity("Versioned workspace index is malformed.")
    seen_plans: set[str] = set()
    seen_ordinals: set[int] = set()
    seen_leaf_ids: set[str] = set()
    root_id = _valid_uuid(state.get("workspace_instance_id"))
    existing: WorkspaceBinding | None = None
    for expected_ordinal, entry in enumerate(iterations, 1):
        if not isinstance(entry, dict) or set(entry) != {"ordinal", "directory", "workspace_instance_id", "plan_sha256"}:
            raise _integrity("Versioned workspace contains a malformed iteration entry.")
        ordinal = entry.get("ordinal")
        directory = entry.get("directory")
        identity = _valid_uuid(entry.get("workspace_instance_id"))
        plan = _valid_sha(entry.get("plan_sha256"))
        if not isinstance(ordinal, int) or ordinal != expected_ordinal or ordinal in seen_ordinals or plan in seen_plans or identity in seen_leaf_ids or identity == root_id:
            raise _integrity("Versioned workspace has duplicate iteration identity.")
        if not isinstance(directory, str) or directory != f"iteration{ordinal:02d}":
            raise _integrity("Versioned workspace iteration directory disagrees with its ordinal.")
        seen_ordinals.add(ordinal)
        seen_plans.add(plan)
        seen_leaf_ids.add(identity)
        binding = WorkspaceBinding(root, root / directory, plan, identity)
        binding._verify_leaf()
        if plan == plan_sha256:
            existing = binding
    if sorted(seen_ordinals) != list(range(1, len(seen_ordinals) + 1)) or next_iteration != len(seen_ordinals) + 1:
        raise _integrity("Versioned workspace next_iteration is not canonical.")
    indexed = {f"iteration{ordinal:02d}" for ordinal in seen_ordinals}
    _cleanup_upgrade_duplicate(root, indexed)
    recovered = _adopt_versioned_orphan(root, state, plan_sha256, next_iteration, indexed)
    if recovered is not None:
        return recovered
    _cleanup_versioned_staging(root, indexed)
    if existing is not None:
        return existing
    directory = f"iteration{next_iteration:02d}"
    leaf = _create_leaf(root, plan_sha256, plan_bytes, directory=directory, excluding=root_id)
    state = dict(state)
    state["iterations"] = [*iterations, {"ordinal": next_iteration, **_leaf_pointer(leaf, root)}]
    state["next_iteration"] = next_iteration + 1
    _atomic_write(root / "workspace.json", _canonical_bytes(state))
    return leaf


def _cleanup_upgrade_duplicate(root: Path, indexed: set[str]) -> None:
    leaves = root / "leaves"
    if leaves.is_symlink():
        raise _integrity("Versioned workspace retains unsafe replacement evidence.", path=str(leaves))
    if not leaves.exists():
        return
    if not leaves.is_dir() or "iteration01" not in indexed:
        raise _integrity("Versioned workspace retains unsafe replacement evidence.", path=str(leaves))
    children = list(leaves.iterdir())
    if len(children) != 1 or children[0].is_symlink() or _UUID4.fullmatch(children[0].name) is None:
        raise _integrity("Versioned workspace upgrade duplicate is malformed.", path=str(leaves))
    old = _verified_unbound_leaf(root, children[0], f"leaves/{children[0].name}")
    upgraded = _verified_unbound_leaf(root, root / "iteration01", "iteration01")
    if old.workspace_instance_id != upgraded.workspace_instance_id or old.plan_sha256 != upgraded.plan_sha256:
        raise _integrity("Versioned workspace upgrade copies disagree.")
    _remove_leaf(old.leaf)
    leaves.rmdir()
    _fsync_directory(root)


def _adopt_versioned_orphan(
    root: Path,
    state: Mapping[str, object],
    requested_plan_sha256: str,
    next_iteration: int,
    indexed: set[str],
) -> WorkspaceBinding | None:
    directory = f"iteration{next_iteration:02d}"
    candidate = root / directory
    if not candidate.exists():
        return None
    if candidate.name in indexed or candidate.is_symlink() or not candidate.is_dir():
        raise _integrity("Versioned workspace has unsafe unindexed iteration evidence.", path=str(candidate))
    leaf = _verified_unbound_leaf(root, candidate, directory)
    requests = candidate / "requests"
    if leaf.plan_sha256 != requested_plan_sha256:
        if requests.exists() and (requests.is_symlink() or not requests.is_dir() or any(requests.iterdir())):
            raise _integrity("Unindexed iteration contains request evidence and cannot be discarded.")
        shutil.rmtree(candidate)
        _fsync_directory(root)
        return None
    updated = dict(state)
    iterations = updated.get("iterations")
    if not isinstance(iterations, list):
        raise _integrity("Versioned workspace index is malformed.")
    updated["iterations"] = [*iterations, {"ordinal": next_iteration, **_leaf_pointer(leaf, root)}]
    updated["next_iteration"] = next_iteration + 1
    _atomic_write(root / "workspace.json", _canonical_bytes(updated))
    return leaf


def _cleanup_versioned_staging(root: Path, indexed: set[str]) -> None:
    leaves = root / "leaves"
    if leaves.is_symlink() or leaves.exists():
        raise _integrity("Versioned workspace retains unbound replacement leaf evidence.", path=str(leaves))
    for child in root.iterdir():
        staging = _LEAF_STAGING.fullmatch(child.name) is not None
        iteration = re.fullmatch(r"iteration[0-9]{2,}", child.name) is not None
        if staging:
            if child.is_symlink() or not child.is_dir():
                raise _integrity("Versioned workspace contains a malformed staging path.", path=str(child))
            shutil.rmtree(child)
        elif iteration and child.name not in indexed:
            raise _integrity("Versioned workspace has unindexed iteration evidence.", path=str(child))
    _fsync_directory(root)


def _remove_leaf(leaf: Path) -> None:
    if leaf.parent.name != "leaves" or leaf.parent.is_symlink() or leaf.is_symlink():
        raise _integrity("Refusing to remove a non-leaf workspace path.", leaf=str(leaf))
    shutil.rmtree(leaf)
    _fsync_directory(leaf.parent)


def _decode_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _integrity(f"{label.capitalize()} bytes are not JSON.", error=str(error)) from error
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise _integrity(f"{label.capitalize()} bytes are not canonical JSON.")
    return value


def _verify_request_document(
    request: Mapping[str, object],
    plan_sha256: str,
    plan: Mapping[str, object],
) -> None:
    operation = request.get("operation")
    spec = request.get("spec")
    runtime = request.get("runtime_semantic")
    algorithms = {
        "solve_direct": {"direct_solve": "scnsim.direct_response.v1"},
        "evaluate_direct": {
            "diagonal_root": "scnsim.diagonal_root.newton32.v1",
            "hybridized_pole": "scnsim.hybridized_pole.newton32.v1",
            "transfer_zero": "scnsim.transfer_zero.newton32.v1",
            "residue_normalized_coupling": "scnsim.residue_normalized_coupling.v1",
            "response_element": "scnsim.response_element.v1",
            "operator": "scnsim.direct_operator.v1",
        },
        "optimize_direct": {"optimization": "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v2"},
    }
    expected_algorithm = algorithms.get(operation, {}).get(spec.get("type") if isinstance(spec, dict) else None)
    if (
        set(request) != {"schema", "schema_version", "plan_sha256", "operation", "ref_lineage", "spec", "parameters", "runtime_semantic"}
        or request.get("schema") != "scnsim.request"
        or request.get("schema_version") != 1
        or request.get("plan_sha256") != plan_sha256
        or expected_algorithm is None
        or any(not isinstance(request.get(field), dict) for field in ("ref_lineage", "spec", "parameters", "runtime_semantic"))
        or runtime.get("algorithm_id") != expected_algorithm
    ):
        raise _integrity("Stored request envelope is open or inconsistent.")
    runtime_fields = {
        "algorithm_id", "python_source_sha256", "julia_source_sha256",
        "julia_version", "project_sha256", "manifest_sha256",
    }
    if set(runtime) != runtime_fields or not isinstance(runtime.get("julia_version"), str) or not runtime["julia_version"]:
        raise _integrity("Stored runtime semantic identity is open or malformed.")
    for field in ("python_source_sha256", "julia_source_sha256", "project_sha256", "manifest_sha256"):
        _valid_sha(runtime.get(field))
    _verify_parameter_set_document(request["parameters"])
    terminal, port_realizable = _verify_v1_lineage(request.get("ref_lineage"), plan)
    if operation == "solve_direct":
        _verify_v1_direct_spec(spec, terminal, port_realizable)
    elif operation == "evaluate_direct":
        _verify_v1_evaluation_spec(spec, terminal, port_realizable)
    else:
        _verify_v1_optimization_spec(spec, terminal, port_realizable)
        if request["parameters"].get("allow_extrapolation") != spec.get("allow_extrapolation"):
            raise _integrity("Optimization request has inconsistent extrapolation authorities.")


def _identifiers(value: object, *, field: str, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value) or any(
        not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in value
    ):
        raise _integrity(f"{field} is not an ordered identifier array.")
    if len(set(value)) != len(value):
        raise _integrity(f"{field} repeats an identifier.")
    return list(value)


def _lazy_lineage_projection(
    lineage: Mapping[str, object] | object,
    plan: Mapping[str, object],
) -> dict[str, object]:
    """Project lazy and realized lineage into the resolve declaration key.

    The lazy ``NetworkViewRef`` deliberately has only user declarations for
    PTC, transforms, and retain.  Its compiler-realized counterpart includes
    matrices, branch evidence, and reconstruction checks.  Resolve may bind
    only the declaration the user actually made: original identity, selected
    PTC ports, ordered transform inputs plus generated identities, and the
    retained coordinate list.  This is not a second View implementation.
    """

    outer_fields = {
        "type", "original", "ptc", "transforms", "retain",
        "terminal_coordinates", "port_realizable", "lineage_sha256",
    }
    if not isinstance(lineage, Mapping) or set(lineage) != outer_fields or lineage.get("type") != "network_view_lineage":
        raise _integrity("Resolve View declaration is open or malformed.")

    original = lineage.get("original")
    original_fields = {
        "type", "compiled_graph_sha256", "coordinate_order", "port_order", "port_realizable",
    }
    if not isinstance(original, Mapping) or set(original) != original_fields or original.get("type") != "original":
        raise _integrity("Resolve original View identity is malformed.")
    _valid_sha(original.get("compiled_graph_sha256"))
    coordinates = _identifiers(original.get("coordinate_order"), field="Resolve original coordinate order")
    ports = _identifiers(original.get("port_order"), field="Resolve original Port order", nonempty=False)
    plan_ports = plan.get("ports")
    if (
        not isinstance(plan_ports, list)
        or ports != [item.get("port_id") for item in plan_ports if isinstance(item, Mapping)]
        or not isinstance(original.get("port_realizable"), bool)
    ):
        raise _integrity("Resolve original View identity disagrees with the sealed Plan.")

    ptc = lineage.get("ptc")
    selected_ports: list[str] | None = None
    if ptc is not None:
        if not isinstance(ptc, Mapping) or ptc.get("type") != "ptc" or "selected_ports" not in ptc:
            raise _integrity("Resolve PTC declaration is malformed.")
        # Lazy declarations contain exactly these two fields; realized steps
        # are separately validated by _verify_request_document before they
        # reach this projection.
        selected_ports = _identifiers(ptc.get("selected_ports"), field="Resolve PTC selected Ports")
        if any(port not in ports for port in selected_ports):
            raise _integrity("Resolve PTC selects a Port outside the original View.")

    transforms = lineage.get("transforms")
    if not isinstance(transforms, list):
        raise _integrity("Resolve transform declaration is malformed.")
    projected_transforms: list[dict[str, object]] = []
    current = list(coordinates)
    for step in transforms:
        if not isinstance(step, Mapping) or step.get("type") != "transform_pair":
            raise _integrity("Resolve transform declaration is malformed.")
        inputs = _identifiers(step.get("input_coordinates"), field="Resolve transform input coordinates")
        if len(inputs) != 2 or any(value not in current for value in inputs):
            raise _integrity("Resolve transform inputs are not a current ordered pair.")
        if "id" in step:
            # This is the lazy Python declaration.  The generated names are
            # authoritative rather than a reversible parsing convention.
            identifier = step.get("id")
            output = step.get("output_coordinates")
            if (
                set(step) != {"type", "id", "input_coordinates", "output_coordinates"}
                or not isinstance(identifier, str)
                or not identifier
                or any(character.isspace() for character in identifier)
                or output != [f"{identifier}.common", f"{identifier}.differential"]
            ):
                raise _integrity("Resolve lazy transform declaration is malformed.")
            common, differential = output
        else:
            # This is the stored realized evidence.  Its closed shape is
            # verified before projection, so select only the declared outputs.
            common, differential = step.get("common_id"), step.get("differential_id")
            if not isinstance(common, str) or not common or not isinstance(differential, str) or not differential:
                raise _integrity("Resolve realized transform declaration is malformed.")
        if common == differential or common in current or differential in current:
            raise _integrity("Resolve generated transform identities collide with the current basis.")
        current = [value for value in current if value not in inputs]
        current.extend((common, differential))
        projected_transforms.append(
            {
                "input_coordinates": inputs,
                "common_id": common,
                "differential_id": differential,
            }
        )

    retain = lineage.get("retain")
    retained: list[str] | None = None
    if retain is not None:
        if not isinstance(retain, Mapping) or retain.get("type") != "retain":
            raise _integrity("Resolve retain declaration is malformed.")
        retained = _identifiers(retain.get("retained_coordinates"), field="Resolve retained coordinates")
        if any(value not in current for value in retained):
            raise _integrity("Resolve retain declaration selects an unavailable coordinate.")

    return {
        "original": dict(original),
        "ptc_selected_ports": selected_ports,
        "transforms": projected_transforms,
        "retained_coordinates": retained,
    }


def _verify_v1_lineage(lineage: object, plan: Mapping[str, object] | None) -> tuple[list[str], bool]:
    """Close the full dev5 View grammar without reimplementing compilation.

    Matrix bytes are compiler-owned evidence; the workspace binds their hashes,
    ordering and applicability rather than manufacturing a second compiler in
    Python.
    """

    if not isinstance(lineage, dict) or set(lineage) != {
        "type", "original", "ptc", "transforms", "retain", "terminal_coordinates", "port_realizable", "lineage_sha256",
    } or lineage.get("type") != "network_view_lineage":
        raise _integrity("View lineage envelope is open or malformed.")
    if lineage.get("lineage_sha256") != _sha256(_canonical_bytes({key: value for key, value in lineage.items() if key != "lineage_sha256"})):
        raise _integrity("View lineage hash does not bind its contents.")
    original = lineage.get("original")
    if not isinstance(original, dict) or set(original) != {"type", "compiled_graph_sha256", "coordinate_order", "port_order", "port_realizable"} or original.get("type") != "original":
        raise _integrity("Original View lineage is malformed.")
    _valid_sha(original.get("compiled_graph_sha256"))
    original_coordinates = _identifiers(original.get("coordinate_order"), field="Original coordinate order")
    if plan is not None and original_coordinates != _plan_coordinates(plan)[0]:
        raise _integrity("Original View coordinate order disagrees with the sealed Plan.")
    plan_ports = plan.get("ports") if plan is not None else None
    if plan is not None and not isinstance(plan_ports, list):
        raise _integrity("Sealed Plan ports are malformed.")
    expected_ports = [port.get("port_id") for port in plan_ports if isinstance(port, dict)] if isinstance(plan_ports, list) else None
    port_roles = {
        port.get("port_id"): port.get("role")
        for port in plan_ports or ()
        if isinstance(port, dict)
    }
    port_order = _identifiers(original.get("port_order"), field="Original Port order", nonempty=False)
    if (expected_ports is not None and port_order != expected_ports) or not isinstance(original.get("port_realizable"), bool):
        raise _integrity("Original View Port identity disagrees with the sealed Plan.")
    coordinates = list(original_coordinates)
    ptc = lineage.get("ptc")
    if ptc is not None:
        if not isinstance(ptc, dict) or set(ptc) != {"type", "selected_ports", "load_mask_sha256", "loads", "reconstruction_residual_f64", "output_coordinate_order", "evidence_sha256"} or ptc.get("type") != "ptc":
            raise _integrity("PTC lineage step is malformed.")
        selected = _identifiers(ptc.get("selected_ports"), field="PTC selected Ports")
        if any(port not in port_order for port in selected) or ptc.get("output_coordinate_order") != coordinates:
            raise _integrity("PTC lineage does not preserve the original coordinate basis.")
        if any(port_roles.get(port) != "nonloading_probe" for port in selected):
            raise _integrity("PTC selects a Port that is not a nonloading probe.")
        if not isinstance(ptc.get("loads"), list) or [item.get("port_id") if isinstance(item, dict) else None for item in ptc["loads"]] != selected:
            raise _integrity("PTC load evidence does not match its selected Port order.")
        for item in ptc["loads"]:
            if not isinstance(item, dict) or set(item) != {"port_id", "reference_impedance", "before", "after"} or item.get("before") != "raw" or item.get("after") != "compensated":
                raise _integrity("PTC load evidence is malformed.")
            _verify_quantity_role(item.get("reference_impedance"), complex_value=False, unit="ohm", dimensionality="resistance")
        _valid_sha(ptc.get("load_mask_sha256")); _valid_sha(ptc.get("evidence_sha256")); _f64_value(ptc.get("reconstruction_residual_f64"))
    transforms = lineage.get("transforms")
    if not isinstance(transforms, list):
        raise _integrity("Transform lineage must be an array.")
    for step in transforms:
        fields = {"type", "input_coordinates", "weights_f64", "differential_id", "common_id", "included_external_cut_branches", "excluded_direct_mutual_branches", "reference_matrix", "principal_root", "reconstruction_residual_f64", "output_coordinate_order", "evidence_sha256"}
        if not isinstance(step, dict) or set(step) != fields or step.get("type") != "transform_pair":
            raise _integrity("Transform lineage step is malformed.")
        pair = _identifiers(step.get("input_coordinates"), field="Transform input coordinates")
        if len(pair) != 2 or any(item not in coordinates for item in pair):
            raise _integrity("Transform input coordinates are not an ordered current pair.")
        weights = step.get("weights_f64")
        if not isinstance(weights, list) or len(weights) != 2 or any(not _finite_f64(item) for item in weights):
            raise _integrity("Transform weights are malformed.")
        common, differential = step.get("common_id"), step.get("differential_id")
        if any(not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in (common, differential)) or differential == common or differential in coordinates or common in coordinates:
            raise _integrity("Transform output coordinate identity is malformed.")
        expected = [item for item in coordinates if item not in pair] + [common, differential]
        if step.get("output_coordinate_order") != expected:
            raise _integrity("Transform output ordering is not canonical.")
        for matrix in ("reference_matrix", "principal_root"):
            evidence = step.get(matrix)
            if not isinstance(evidence, dict) or set(evidence) != {"rows", "columns", "sha256"} or any(not isinstance(evidence.get(field), int) or evidence[field] < 0 for field in ("rows", "columns")):
                raise _integrity("Transform matrix evidence is malformed.")
            _valid_sha(evidence.get("sha256"))
        _verify_branch_refs(step.get("included_external_cut_branches"), field="Transform included cut branches", nonempty=True)
        _verify_branch_refs(step.get("excluded_direct_mutual_branches"), field="Transform excluded direct-mutual branches", nonempty=False)
        _valid_sha(step.get("evidence_sha256")); _f64_value(step.get("reconstruction_residual_f64"))
        coordinates = expected
    retain = lineage.get("retain")
    if retain is not None:
        fields = {"type", "retained_coordinates", "eliminated_coordinates", "output_coordinate_order", "a_matrix", "b_matrix", "r_matrix", "d_matrix", "q_matrix", "selected_projector", "omitted_projector", "omitted_matched_loads", "source_boundary_sha256", "deembedding_evidence_sha256"}
        if not isinstance(retain, dict) or set(retain) != fields or retain.get("type") != "retain":
            raise _integrity("Retain lineage step is malformed.")
        retained = _identifiers(retain.get("retained_coordinates"), field="Retained coordinates")
        eliminated = _identifiers(retain.get("eliminated_coordinates"), field="Eliminated coordinates", nonempty=False)
        if set(retained) | set(eliminated) != set(coordinates) or set(retained) & set(eliminated) or retain.get("output_coordinate_order") != retained:
            raise _integrity("Retain lineage is not an exact partition of its input basis.")
        for matrix in ("a_matrix", "b_matrix", "r_matrix", "d_matrix", "q_matrix", "selected_projector", "omitted_projector", "omitted_matched_loads"):
            evidence = retain.get(matrix)
            if not isinstance(evidence, dict) or set(evidence) != {"rows", "columns", "sha256"}:
                raise _integrity("Retain matrix evidence is malformed.")
            _valid_sha(evidence.get("sha256"))
        _valid_sha(retain.get("source_boundary_sha256")); _valid_sha(retain.get("deembedding_evidence_sha256"))
        coordinates = retained
    terminal = _identifiers(lineage.get("terminal_coordinates"), field="Terminal channel order")
    port_realizable = lineage.get("port_realizable")
    # The compiler's original basis is physical-node ordered, whereas the raw
    # public Direct boundary is the declared logical-Port order.  Transforms
    # alter quantity coordinates but do not themselves create a wave boundary;
    # only terminal retain() selects transformed channel IDs.
    expected_terminal = coordinates if retain is not None else port_order
    if terminal != expected_terminal or not isinstance(port_realizable, bool):
        raise _integrity("Terminal View capability disagrees with its lineage.")
    return terminal, port_realizable


def _verify_v1_direct_spec(spec: object, terminal: list[str], port_realizable: bool) -> None:
    if not port_realizable:
        raise _integrity("Direct S/Y/Z request is not Port-realizable.")
    if not isinstance(spec, dict) or set(spec) != {"type", "frequencies", "traces"} or spec.get("type") != "direct_solve":
        raise _integrity("Direct solve Spec is malformed.")
    frequencies = spec.get("frequencies")
    if not isinstance(frequencies, list) or not frequencies:
        raise _integrity("Direct solve frequency grid is malformed.")
    previous = 0.0
    for frequency in frequencies:
        _verify_quantity_role(frequency, complex_value=False, unit="hertz", dimensionality="inverse_time")
        value = _f64_value(frequency["si_value_f64"])
        if value <= previous:
            raise _integrity("Direct solve frequency grid is not strictly positive and increasing.")
        previous = value
    traces = spec.get("traces")
    if not isinstance(traces, list):
        raise _integrity("Direct trace declarations are malformed.")
    trace_ids: set[str] = set()
    for trace in traces:
        if (
            not isinstance(trace, dict)
            or set(trace) != {"id", "input_port", "input_mode", "output_port", "output_mode"}
            or not isinstance(trace.get("id"), str)
            or _IDENTIFIER.fullmatch(trace["id"]) is None
            or trace["id"] in trace_ids
            or trace.get("input_port") not in terminal
            or trace.get("output_port") not in terminal
            or trace.get("input_mode") != []
            or trace.get("output_mode") != []
        ):
            raise _integrity("Direct trace declaration is malformed.")
        trace_ids.add(trace["id"])


def _verify_v1_evaluation_spec(
    spec: object,
    terminal: list[str],
    port_realizable: bool,
    *,
    residue_branch: bool = False,
) -> None:
    if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
        raise _integrity("Direct evaluation Spec is malformed.")
    kind = spec["type"]
    if kind == "diagonal_root":
        coordinate = spec.get("coordinate")
        invalid = coordinate not in terminal or len(terminal) < 2 if residue_branch else terminal != [coordinate]
        if set(spec) != {"type", "coordinate", "root_hint"} or invalid:
            raise _integrity("Diagonal-root Spec is incompatible with its retained View.")
        _verify_quantity_role(spec.get("root_hint"), complex_value=False, unit="hertz", dimensionality="inverse_time")
    elif kind == "hybridized_pole":
        if set(spec) != {"type", "coordinates", "anchor"} or len(terminal) < 2 or _identifiers(spec.get("coordinates"), field="Hybridized-pole coordinates") != terminal:
            raise _integrity("Hybridized-pole coordinates must equal the complete retained View.")
        _verify_frequency_anchor(spec.get("anchor"))
    elif kind == "transfer_zero":
        if set(spec) != {"type", "anchor", "family", "input_coordinate", "output_coordinate"} or spec.get("family") not in {"S", "Y", "Z"} or spec.get("input_coordinate") not in terminal or spec.get("output_coordinate") not in terminal:
            raise _integrity("Transfer-zero Spec is malformed.")
        if spec.get("family") == "S" and not port_realizable:
            raise _integrity("S-family transfer-zero evaluation is not Port-realizable.")
        _verify_frequency_anchor(spec.get("anchor"))
    elif kind == "residue_normalized_coupling":
        branches = (spec.get("branch_a"), spec.get("branch_b"))
        if (
            set(spec) != {"type", "branch_a", "branch_b", "frequency"}
            or any(not isinstance(branch, dict) or branch.get("type") not in {"diagonal_root", "hybridized_pole"} for branch in branches)
        ):
            raise _integrity("Residue-normalized coupling Spec is malformed.")
        _verify_v1_evaluation_spec(branches[0], terminal, port_realizable, residue_branch=True)
        _verify_v1_evaluation_spec(branches[1], terminal, port_realizable, residue_branch=True)
        _verify_quantity_role(spec.get("frequency"), complex_value=False, unit="hertz", dimensionality="inverse_time")
    elif kind == "response_element":
        if set(spec) != {"type", "family", "input_coordinate", "output_coordinate", "frequency"} or spec.get("family") not in {"S", "Y", "Z"} or spec.get("input_coordinate") not in terminal or spec.get("output_coordinate") not in terminal:
            raise _integrity("Response-element Spec is malformed.")
        if spec.get("family") == "S" and not port_realizable:
            raise _integrity("S-family response evaluation is not Port-realizable.")
        _verify_quantity_role(spec.get("frequency"), complex_value=False, unit="hertz", dimensionality="inverse_time")
    elif kind == "operator":
        if set(spec) != {"type", "frequencies"}:
            raise _integrity("Operator Spec is malformed.")
        _verify_v1_direct_spec({"type": "direct_solve", "frequencies": spec.get("frequencies"), "traces": []}, terminal, True)
    else:
        raise _integrity("Direct evaluation Spec is outside dev5.")


def _verify_frequency_anchor(value: object) -> None:
    if isinstance(value, dict) and value.get("type") == "quantity_f64":
        _verify_quantity_role(value, complex_value=False, unit="hertz", dimensionality="inverse_time")
    else:
        _verify_quantity_role(value, complex_value=True, unit="hertz", dimensionality="inverse_time")


def _verify_v1_optimization_spec(spec: object, terminal: list[str], port_realizable: bool) -> None:
    if not isinstance(spec, dict) or set(spec) != {"type", "variables", "objectives", "optimizer", "allow_extrapolation"} or spec.get("type") != "optimization":
        raise _integrity("Optimization Spec is malformed.")
    variables, objectives, optimizer, authorizations = spec.get("variables"), spec.get("objectives"), spec.get("optimizer"), spec.get("allow_extrapolation")
    if not isinstance(variables, list) or not variables or not isinstance(objectives, list) or not objectives or not isinstance(optimizer, dict) or not isinstance(authorizations, list):
        raise _integrity("Optimization Spec has malformed collections.")
    variable_keys: list[tuple[tuple[str, ...], str]] = []
    for variable in variables:
        if not isinstance(variable, dict) or set(variable) != {"parameter", "model_default_bounds", "consumer_override_bounds", "lower", "upper", "transform"} or variable.get("transform") not in {"linear", "log"}:
            raise _integrity("Optimization variable is malformed.")
        key = _parameter_key_integrity(variable.get("parameter")); variable_keys.append(key)
        for name in ("model_default_bounds", "consumer_override_bounds"):
            bounds = variable.get(name)
            if bounds is None and name == "consumer_override_bounds":
                continue
            _verify_bounds(bounds)
        _verify_quantity_compatible(variable.get("lower"), variable.get("upper"))
        if variable.get("consumer_override_bounds") is None:
            if variable.get("model_default_bounds") != [variable.get("lower"), variable.get("upper")]:
                raise _integrity("Optimization resolved bounds do not preserve model defaults.")
        elif variable.get("consumer_override_bounds") != [variable.get("lower"), variable.get("upper")]:
            raise _integrity("Optimization resolved bounds do not match consumer override.")
    if len(set(variable_keys)) != len(variable_keys):
        raise _integrity("Optimization variables are not unique.")
    authorization_keys = [_parameter_key_integrity(item) for item in authorizations]
    if authorization_keys != sorted(set(authorization_keys)) or any(key not in variable_keys for key in authorization_keys):
        raise _integrity("Optimization extrapolation authorization is not sorted active variables.")
    objective_ids: set[str] = set()
    for objective in objectives:
        if not isinstance(objective, dict) or set(objective) != {"id", "quantity", "target", "weight_f64", "resolved_scale", "scale_source"} or not isinstance(objective.get("id"), str) or _IDENTIFIER.fullmatch(objective["id"]) is None or objective["id"] in objective_ids:
            raise _integrity("Optimization objective is malformed.")
        objective_ids.add(objective["id"])
        role = _verify_selector(objective.get("quantity"), terminal, port_realizable)
        _verify_quantity_role(objective.get("target"), complex_value=False, unit=role[0], dimensionality=role[1])
        _verify_quantity_role(objective.get("resolved_scale"), complex_value=False, unit=role[0], dimensionality=role[1])
        if not _finite_f64(objective.get("weight_f64")) or objective.get("scale_source") not in {"relative_target", "dimensionless_unity", "explicit"}:
            raise _integrity("Optimization objective scale is malformed.")
    required_optimizer = {"type", "seed", "max_evaluations", "population_size", "resolved_population_size", "initial_sigma_f64", "box_transform_id", "complete_generations", "unused_evaluations", "hidden_stops"}
    if set(optimizer) != required_optimizer or optimizer.get("type") != "cma_es" or optimizer.get("box_transform_id") != "cmaes-jl-0.2.6-linquad-unit-box.v1" or optimizer.get("hidden_stops") != "disabled":
        raise _integrity("Optimization controls are malformed.")


def _parameter_key_integrity(value: object) -> tuple[tuple[str, ...], str]:
    if not isinstance(value, dict) or set(value) != {"component_path", "parameter_id"}:
        raise _integrity("ParameterRef is malformed.")
    path = value.get("component_path"); identifier = value.get("parameter_id")
    if not isinstance(path, list) or not path or any(not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in path) or not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
        raise _integrity("ParameterRef identity is malformed.")
    return tuple(path), identifier


def _verify_branch_refs(value: object, *, field: str, nonempty: bool) -> None:
    if not isinstance(value, list) or (nonempty and not value):
        raise _integrity(f"{field} is malformed.")
    keys: list[tuple[tuple[str, ...], str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"component_path", "branch_id"}:
            raise _integrity(f"{field} has an open branch identity.")
        path, branch = item.get("component_path"), item.get("branch_id")
        if not isinstance(path, list) or not path or any(not isinstance(segment, str) or _IDENTIFIER.fullmatch(segment) is None for segment in path) or not isinstance(branch, str) or _IDENTIFIER.fullmatch(branch) is None:
            raise _integrity(f"{field} has a malformed branch identity.")
        keys.append((tuple(path), branch))
    if keys != sorted(set(keys)):
        raise _integrity(f"{field} is not sorted and unique.")


def _verify_bounds(value: object) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise _integrity("Optimization bounds are malformed.")
    _verify_quantity_compatible(value[0], value[1])
    if _f64_value(value[0]["si_value_f64"]) >= _f64_value(value[1]["si_value_f64"]):
        raise _integrity("Optimization bounds are not ordered.")


def _verify_quantity_compatible(left: object, right: object) -> None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise _integrity("Quantity pair is malformed.")
    unit, dimensionality = left.get("si_unit"), left.get("dimensionality")
    _verify_quantity_role(left, complex_value=False, unit=unit, dimensionality=dimensionality)
    _verify_quantity_role(right, complex_value=False, unit=unit, dimensionality=dimensionality)


def _verify_selector(value: object, terminal: list[str], port_realizable: bool) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise _integrity("Optimization selector is malformed.")
    if value.get("type") == "quantity_sum":
        if set(value) != {"type", "terms"} or not isinstance(value.get("terms"), list) or not value["terms"]:
            raise _integrity("QuantitySum is malformed.")
        roles = [_verify_selector(item, terminal, port_realizable) for item in value["terms"]]
        if any(role[1] != roles[0][1] for role in roles[1:]):
            raise _integrity("QuantitySum terms have incompatible physical roles.")
        return roles[0]
    fields = {"type", "spec", "projection"}
    kind = value.get("type")
    expected = {
        "diagonal_root_projection": ("diagonal_root", {"frequency", "linewidth"}, ("hertz", "inverse_time")),
        "hybridized_pole_projection": ("hybridized_pole", {"frequency", "linewidth"}, ("hertz", "inverse_time")),
        "transfer_zero_projection": ("transfer_zero", {"frequency"}, ("hertz", "inverse_time")),
        "residue_coupling_projection": ("residue_normalized_coupling", {"magnitude"}, ("radian / second", "inverse_time")),
        "response_element_projection": ("response_element", {"magnitude", "real", "imag"}, None),
    }.get(kind)
    if set(value) != fields or expected is None or value.get("projection") not in expected[1] or not isinstance(value.get("spec"), dict) or value["spec"].get("type") != expected[0]:
        raise _integrity("Optimization selector is outside the Direct catalog.")
    _verify_v1_evaluation_spec(value["spec"], terminal, port_realizable)
    if expected[2] is not None:
        return expected[2]
    family = value["spec"].get("family")
    return {"S": ("dimensionless", "dimensionless"), "Y": ("siemens", "conductance"), "Z": ("ohm", "resistance")}[family]


def _plan_coordinates(plan: Mapping[str, object]) -> tuple[list[str], set[str]]:
    """Return the compiler basis order and the public selection subset.

    The Python lineage must name the same full basis that the recursive Julia
    compiler uses.  Public selection deliberately remains narrower: Plan
    public nodes and coordinates exposed by immediate Composite children only.
    """

    from ._canonical import internal_node_id

    nodes = plan.get("nodes")
    components = plan.get("components")
    if not isinstance(nodes, list) or not isinstance(components, list):
        raise _integrity("Sealed Plan coordinate inventory is malformed.")
    order: list[str] = []
    public: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("node_id"), str) or not node["node_id"]:
            raise _integrity("Sealed Plan node inventory is malformed.")
        order.append(node["node_id"])
        if node.get("visibility") in {"public", "port_promoted"}:
            public.add(node["node_id"])

    def component_path(component: Mapping[str, object]) -> tuple[str, ...]:
        path = component.get("component_path")
        if not isinstance(path, list) or not path or any(not isinstance(item, str) or not item for item in path):
            raise _integrity("Sealed Component path is malformed.")
        return tuple(path)

    def realization(component: Mapping[str, object]) -> Mapping[str, object]:
        value = component.get("realization")
        if not isinstance(value, Mapping) or not isinstance(value.get("kind"), str):
            raise _integrity("Sealed Component realization is malformed.")
        return value

    def records(value: object, *, label: str) -> list[Mapping[str, object]]:
        if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
            raise _integrity(f"{label} is malformed.")
        return list(value)

    def endpoint(value: object, *, label: str) -> tuple[tuple[str, ...], str]:
        if not isinstance(value, Mapping):
            raise _integrity(f"{label} is malformed.")
        path = value.get("component_path")
        pin = value.get("pin_id")
        if not isinstance(path, list) or not path or any(not isinstance(item, str) or not item for item in path) or not isinstance(pin, str) or not pin:
            raise _integrity(f"{label} is malformed.")
        return tuple(path), pin

    def private_nodes(container: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
        values = records(realization(container).get("private_nodes"), label="Composite private-node inventory")
        mapped: dict[str, Mapping[str, object]] = {}
        for value in values:
            identifier = value.get("id")
            endpoints = value.get("endpoints")
            if not isinstance(identifier, str) or not identifier or not isinstance(endpoints, list):
                raise _integrity("Composite private-node record is malformed.")
            if identifier in mapped:
                raise _integrity("Composite private-node IDs are not unique.")
            mapped[identifier] = value
        return mapped

    def expanded_node_id(container: Mapping[str, object], private_node: Mapping[str, object]) -> str:
        leaves: list[dict[str, object]] = []

        def expand(current: Mapping[str, object], value: object, ancestry: set[tuple[tuple[str, ...], str]]) -> None:
            path, pin = endpoint(value, label="Composite private endpoint")
            key = (path, pin)
            if key in ancestry:
                raise _integrity("Composite private-node expansion is cyclic or duplicates an endpoint.")
            children = records(realization(current).get("children"), label="Composite child inventory")
            matches = [child for child in children if component_path(child) == path]
            if len(matches) != 1:
                raise _integrity("Composite endpoint does not resolve to one immediate child.")
            child = matches[0]
            child_realization = realization(child)
            if child_realization.get("kind") != "composite":
                leaves.append({"component_path": list(path), "pin_id": pin})
                return
            maps = records(child_realization.get("public_pin_map"), label="Composite public-pin map")
            mappings = [item for item in maps if item.get("public_id") == pin]
            if len(mappings) != 1 or not isinstance(mappings[0].get("private_node_id"), str):
                raise _integrity("Composite child endpoint lacks one public-pin map.")
            nested = private_nodes(child).get(mappings[0]["private_node_id"])
            if nested is None:
                raise _integrity("Composite public pin targets no private node.")
            nested_endpoints = nested.get("endpoints")
            if not isinstance(nested_endpoints, list):
                raise _integrity("Composite private-node record is malformed.")
            next_ancestry = set(ancestry)
            next_ancestry.add(key)
            for nested_endpoint in nested_endpoints:
                expand(child, nested_endpoint, next_ancestry)

        endpoints = private_node.get("endpoints")
        if not isinstance(endpoints, list) or not endpoints:
            raise _integrity("Composite private-node record is malformed.")
        for item in endpoints:
            expand(container, item, set())
        try:
            return internal_node_id(leaves)
        except Exception as error:
            raise _integrity("Composite private-node expansion is malformed.") from error

    def visit(component: object, *, top_level: bool) -> None:
        if not isinstance(component, Mapping):
            raise _integrity("Sealed Plan Component inventory is malformed.")
        path = component_path(component)
        component_realization = realization(component)
        if component_realization.get("kind") == "transmission_line":
            conductors = component_realization.get("pin_conductors")
            sections = component_realization.get("n_sections")
            if (
                not isinstance(conductors, list)
                or not conductors
                or any(not isinstance(item, str) or not item for item in conductors)
                or len(set(conductors)) != len(conductors)
                or not isinstance(sections, int)
                or isinstance(sections, bool)
                or sections < 1
            ):
                raise _integrity("Transmission-line station declaration is malformed.")
            for station in range(1, sections):
                for conductor in conductors:
                    order.append(
                        "internal-"
                        + _sha256(
                            _canonical_bytes(
                                {
                                    "schema": "scnsim.line_station",
                                    "schema_version": 1,
                                    "component_path": list(path),
                                    "station": station,
                                    "conductor": conductor,
                                }
                            )
                        )
                    )
            return
        if component_realization.get("kind") != "composite":
            return
        node_map = private_nodes(component)
        pins = records(component_realization.get("public_pin_map"), label="Composite public-pin map")
        pin_targets: set[str] = set()
        for record in pins:
            public_id = record.get("public_id")
            private_id = record.get("private_node_id")
            if not isinstance(public_id, str) or not public_id or not isinstance(private_id, str) or private_id not in node_map:
                raise _integrity("Composite public-pin map is malformed.")
            pin_targets.add(private_id)
        coordinate_targets: dict[str, str] = {}
        coordinates = records(component_realization.get("public_coordinate_map"), label="Composite coordinate inventory")
        for record in coordinates:
            public_id = record.get("public_id")
            private_id = record.get("private_node_id")
            if not isinstance(public_id, str) or not public_id or not isinstance(private_id, str) or private_id not in node_map:
                raise _integrity("Composite public-coordinate map is malformed.")
            if top_level:
                coordinate_targets[private_id] = public_id
                public.add(public_id)
        for private_id, private_node in node_map.items():
            if private_id not in pin_targets:
                order.append(coordinate_targets.get(private_id, expanded_node_id(component, private_node)))
        for child in records(component_realization.get("children"), label="Composite child inventory"):
            visit(child, top_level=False)

    for component in components:
        visit(component, top_level=True)
    return sorted(set(order)), public


def _verify_direct_request(
    request: Mapping[str, object],
    plan: Mapping[str, object] | None = None,
) -> tuple[list[str], int]:
    lineage = request.get("ref_lineage")
    spec = request.get("spec")
    if not isinstance(lineage, dict) or not isinstance(spec, dict):
        raise _integrity("Direct request View or Spec is malformed.")
    original = lineage.get("original")
    lineage_without_hash = {key: value for key, value in lineage.items() if key != "lineage_sha256"}
    if (
        set(lineage) != {"type", "original", "ptc", "transforms", "retain", "terminal_coordinates", "port_realizable", "lineage_sha256"}
        or lineage.get("type") != "network_view_lineage"
        or lineage.get("ptc") is not None
        or lineage.get("transforms") != []
        or lineage.get("retain") is not None
        or lineage.get("port_realizable") is not True
        or lineage.get("lineage_sha256") != _sha256(_canonical_bytes(lineage_without_hash))
        or not isinstance(original, dict)
        or set(original) != {"type", "compiled_graph_sha256", "coordinate_order", "port_order", "port_realizable"}
        or original.get("type") != "original"
        or original.get("port_realizable") is not True
    ):
        raise _integrity("Dev3 Direct request does not name the exact original View.")
    _valid_sha(original.get("compiled_graph_sha256"))
    coordinates = original.get("coordinate_order")
    ports = original.get("port_order")
    plan_ports = plan.get("ports") if plan is not None else None
    expected_coordinates = _plan_coordinates(plan)[0] if plan is not None else coordinates
    expected_ports = [port.get("port_id") for port in plan_ports if isinstance(port, dict)] if isinstance(plan_ports, list) else ports
    if (
        not isinstance(coordinates, list)
        or not coordinates
        or any(not isinstance(item, str) or not item for item in coordinates)
        or len(set(coordinates)) != len(coordinates)
        or not isinstance(ports, list)
        or len(ports) != 1
        or any(not isinstance(item, str) or not item for item in ports)
        or lineage.get("terminal_coordinates") != ports
        or coordinates != expected_coordinates
        or ports != expected_ports
    ):
        raise _integrity("Dev3 Direct original coordinate order is malformed.")
    frequencies = spec.get("frequencies")
    if set(spec) != {"type", "frequencies", "traces"} or spec.get("type") != "direct_solve" or spec.get("traces") != [] or not isinstance(frequencies, list) or not frequencies:
        raise _integrity("Dev3 Direct Spec is open or malformed.")
    values: list[float] = []
    for frequency in frequencies:
        _verify_quantity_role(frequency, complex_value=False, unit="hertz", dimensionality="inverse_time")
        values.append(_f64_value(frequency["si_value_f64"]))
    if any(value <= 0.0 for value in values) or any(right <= left for left, right in zip(values, values[1:])):
        raise _integrity("Dev3 Direct frequency grid is not positive and strictly increasing.")
    return ports, len(values)


def _lineage_matrix(label: str, values: list[list[float]], applicability: str) -> dict[str, object]:
    rows = len(values)
    columns = len(values[0]) if rows else 0
    bits = [struct.pack(">d", value).hex() for row in values for value in row]
    digest = _sha256(_canonical_bytes({
        "schema": "scnsim.lineage_matrix",
        "schema_version": 1,
        "label": label,
        "applicability": applicability,
        "shape": [rows, columns],
        "row_major_f64": bits,
    }))
    return {"rows": rows, "columns": columns, "sha256": digest}


def _verify_retained_request(request: Mapping[str, object], plan: Mapping[str, object]) -> str:
    lineage = request.get("ref_lineage")
    if not isinstance(lineage, dict):
        raise _integrity("Retained request View is malformed.")
    original = lineage.get("original")
    retain = lineage.get("retain")
    ports = plan.get("ports")
    if not isinstance(ports, list) or len(ports) != 1:
        raise _integrity("Dev3 retained request Plan is not one-Port.")
    node_order, public_coordinates = _plan_coordinates(plan)
    port_order = [port.get("port_id") for port in ports if isinstance(port, dict)]
    if (
        not node_order
        or len(port_order) != 1
        or not isinstance(original, dict)
        or set(original) != {"type", "compiled_graph_sha256", "coordinate_order", "port_order", "port_realizable"}
        or original.get("type") != "original"
        or original.get("coordinate_order") != node_order
        or original.get("port_order") != port_order
        or original.get("port_realizable") is not True
    ):
        raise _integrity("Retained request original lineage disagrees with the sealed Plan.")
    _valid_sha(original.get("compiled_graph_sha256"))
    if (
        set(lineage) != {"type", "original", "ptc", "transforms", "retain", "terminal_coordinates", "port_realizable", "lineage_sha256"}
        or lineage.get("type") != "network_view_lineage"
        or lineage.get("ptc") is not None
        or lineage.get("transforms") != []
        or not isinstance(retain, dict)
        or lineage.get("lineage_sha256") != _sha256(_canonical_bytes({key: value for key, value in lineage.items() if key != "lineage_sha256"}))
    ):
        raise _integrity("Dev3 retained View lineage is open or inconsistent.")
    fields = {
        "type", "retained_coordinates", "eliminated_coordinates", "output_coordinate_order",
        "a_matrix", "b_matrix", "r_matrix", "d_matrix", "q_matrix",
        "selected_projector", "omitted_projector", "omitted_matched_loads",
        "source_boundary_sha256", "deembedding_evidence_sha256",
    }
    retained = retain.get("retained_coordinates")
    if not isinstance(retained, list) or len(retained) != 1 or not isinstance(retained[0], str):
        raise _integrity("Dev3 retain() must name exactly one coordinate.")
    coordinate = retained[0]
    if (
        set(retain) != fields
        or retain.get("type") != "retain"
        or coordinate not in public_coordinates
        or retain.get("eliminated_coordinates") != [item for item in node_order if item != coordinate]
        or retain.get("output_coordinate_order") != retained
        or lineage.get("terminal_coordinates") != retained
    ):
        raise _integrity("Dev3 retained coordinate boundary is inconsistent with the sealed Plan.")
    matching = [index for index, port in enumerate(ports) if isinstance(port, dict) and port.get("node_id") == coordinate]
    port_realizable = len(matching) == 1
    if lineage.get("port_realizable") is not port_realizable:
        raise _integrity("Retained View port-realizability disagrees with its Plan attachment.")
    labels = {
        "a_matrix": "a", "b_matrix": "b", "r_matrix": "r", "d_matrix": "d", "q_matrix": "q",
        "selected_projector": "selected_projector", "omitted_projector": "omitted_projector",
        "omitted_matched_loads": "omitted_matched_loads",
    }
    if port_realizable:
        _verify_quantity_role(ports[0].get("reference_impedance"), complex_value=False, unit="ohm", dimensionality="resistance")
        impedance = _f64_value(ports[0]["reference_impedance"]["si_value_f64"])
        if impedance <= 0.0:
            raise _integrity("Logical Port reference impedance is not strictly positive.")
        root = math.sqrt(impedance)
        b = [[1.0] if item == coordinate else [0.0] for item in node_order]
        matrices = {
            "a_matrix": _lineage_matrix("a", [[1.0]], "port_realizable"),
            "b_matrix": _lineage_matrix("b", b, "port_realizable"),
            "r_matrix": _lineage_matrix("r", [[impedance]], "port_realizable"),
            "d_matrix": _lineage_matrix("d", [[root]], "port_realizable"),
            "q_matrix": _lineage_matrix("q", [[1.0]], "port_realizable"),
            "selected_projector": _lineage_matrix("selected_projector", [[1.0]], "port_realizable"),
            "omitted_projector": _lineage_matrix("omitted_projector", [[0.0]], "port_realizable"),
            "omitted_matched_loads": _lineage_matrix("omitted_matched_loads", [[0.0]], "port_realizable"),
        }
        source = {"schema": "scnsim.source_boundary", "schema_version": 1, "applicability": "port_realizable", "b": matrices["b_matrix"], "r": matrices["r_matrix"]}
        deembedding = {"schema": "scnsim.deembedding", "schema_version": 1, "applicability": "port_realizable", "d": matrices["d_matrix"], "q": matrices["q_matrix"]}
    else:
        matrices = {field: _lineage_matrix(label, [], "not_port_realizable") for field, label in labels.items()}
        source = {"schema": "scnsim.source_boundary", "schema_version": 1, "applicability": "not_port_realizable"}
        deembedding = {"schema": "scnsim.deembedding", "schema_version": 1, "applicability": "not_port_realizable"}
    if (
        any(retain.get(field) != value for field, value in matrices.items())
        or retain.get("source_boundary_sha256") != _sha256(_canonical_bytes(source))
        or retain.get("deembedding_evidence_sha256") != _sha256(_canonical_bytes(deembedding))
    ):
        raise _integrity("Retained View wave-boundary evidence is not canonical for its capability.")
    return coordinate


def _verify_diagonal_root_spec(spec: object, coordinate: str) -> None:
    if not isinstance(spec, dict) or set(spec) != {"type", "coordinate", "root_hint"} or spec.get("type") != "diagonal_root" or spec.get("coordinate") != coordinate:
        raise _integrity("Diagonal-root Spec does not match its retained coordinate.")
    _verify_quantity_role(spec.get("root_hint"), complex_value=False, unit="hertz", dimensionality="inverse_time")
    if _f64_value(spec["root_hint"]["si_value_f64"]) <= 0.0:
        raise _integrity("Diagonal-root hint is not strictly positive.")


def _verify_optimization_selector_coordinates(spec: object, coordinate: str) -> None:
    if not isinstance(spec, dict) or spec.get("type") != "optimization" or not isinstance(spec.get("objectives"), list):
        raise _integrity("Optimization Spec is malformed.")

    def verify(value: object) -> None:
        if not isinstance(value, dict):
            raise _integrity("Optimization scalar selector is malformed.")
        if value.get("type") == "quantity_sum":
            terms = value.get("terms")
            if not isinstance(terms, list) or not terms:
                raise _integrity("Optimization QuantitySum is empty.")
            for term in terms:
                verify(term)
            return
        if set(value) != {"type", "spec", "projection"} or value.get("type") != "diagonal_root_projection" or value.get("projection") not in {"frequency", "linewidth"}:
            raise _integrity("Optimization selector is outside the Direct quantity family.")
        _verify_diagonal_root_spec(value.get("spec"), coordinate)

    for objective in spec["objectives"]:
        if not isinstance(objective, dict):
            raise _integrity("Optimization objective is malformed.")
        verify(objective.get("quantity"))


def _verify_failure_document(value: object, operation: object) -> None:
    if not isinstance(value, dict) or set(value) != {"category", "kind", "stage", "message", "evidence"}:
        raise _integrity("Failure envelope is open or malformed.")
    evidence = value.get("evidence")
    allowed = {
        "type", "operation", "context_kind", "plan_sha256", "request_sha256",
        "attempt_sha256", "workspace_instance_id", "component_path", "parameter",
        "coordinate_id", "port_id", "case_id", "candidate_ordinal", "artifact_id",
        "artifact_path", "expected_sha256", "actual_sha256", "backend_exit_code",
        "evidence_sha256",
    }
    categories = {
        "plan_sealed": "state",
        "workspace_plan_replaced": "state",
        "workspace_versioning_downgrade_forbidden": "state",
        "unsupported_runtime_platform": "capability",
        "unsupported_singular_capacitance_for_diagonal_root_v1": "capability",
        "scaffold_unavailable": "capability",
        "port_realizability": "validation",
        "invalid_diagonal_root_hint": "validation",
        "invalid_optimization_spec": "validation",
        "direct_response_formation": "execution",
        "invalid_candidate_physical_parameter": "execution",
        "compiler_invariant": "execution",
        "eliminated_block_solve_failure": "execution",
        "root_slope_unresolved": "execution",
        "numerical_resolution_unresolved": "execution",
        "hb_case_failure": "execution",
        "runtime_preparation": "execution",
        "backend_protocol": "execution",
        "result_unavailable": "evidence",
        "evidence_integrity": "evidence",
    }
    kind = value.get("kind")
    contexts = {
        "authoring", "workspace", "runtime", "compile", "direct_response",
        "direct_quantity", "optimization_candidate", "hb_case", "protocol",
        "artifact", "resolution", "scaffold",
    }
    if (
        not isinstance(evidence, dict)
        or not {"type", "operation", "context_kind"}.issubset(evidence)
        or not set(evidence).issubset(allowed)
        or evidence.get("type") != "failure_evidence"
        or evidence.get("operation") not in {operation, "backend_protocol"}
        or evidence.get("context_kind") not in contexts
        or kind not in categories
        or value.get("category") != categories.get(kind)
        or not isinstance(value.get("stage"), str)
        or not value["stage"]
        or not isinstance(value.get("message"), str)
        or not value["message"]
    ):
        raise _integrity("Failure discriminator or evidence is malformed.")


def _verify_result_document(
    result: Mapping[str, object],
    request: Mapping[str, object],
    request_sha256: str,
    attempt_sha256: str,
    plan: Mapping[str, object],
) -> None:
    """Close every dev5 Direct Result before typed public decoding."""

    spec = request.get("spec")
    kind = (
        "direct_response" if request.get("operation") == "solve_direct"
        else "optimization" if request.get("operation") == "optimize_direct"
        else spec.get("type") if request.get("operation") == "evaluate_direct" and isinstance(spec, dict)
        else None
    )
    common = {"schema", "schema_version", "result_kind", "request_sha256", "attempt_sha256"}
    if (
        result.get("schema") != "scnsim.result"
        or result.get("schema_version") != 1
        or result.get("result_kind") != kind
        or result.get("request_sha256") != request_sha256
        or result.get("attempt_sha256") != attempt_sha256
    ):
        raise _integrity("Result envelope does not match its request and attempt.")
    if kind == "direct_response":
        if set(result) != common | {"scalar_catalog", "array_catalog"} or result.get("scalar_catalog") != {}:
            raise _integrity("Direct Result envelope is open or has scalar payloads.")
        catalog = result.get("array_catalog")
        if not isinstance(catalog, dict) or set(catalog) != {"frequencies", "s", "y", "z"}:
            raise _integrity("Direct Result array catalog is incomplete.")
        terminal, port_realizable = _verify_v1_lineage(request.get("ref_lineage"), plan)
        expected_probes = _expected_probe_load_state(request.get("ref_lineage"))
        _verify_v1_direct_spec(request.get("spec"), terminal, port_realizable)
        frequencies = request["spec"]["frequencies"]
        expected_frequency_count = len(frequencies)
        frequency_count = _verify_direct_artifact(catalog["frequencies"], "frequencies")
        if frequency_count != expected_frequency_count:
            raise _integrity("Direct artifacts disagree with the requested frequency grid length.")
        for role in ("s", "y", "z"):
            if _verify_direct_artifact(catalog[role], role) != frequency_count:
                raise _integrity("Direct artifacts disagree on frequency-axis length.")
            if (
                catalog[role].get("coordinate_ids") != terminal
                or catalog[role].get("coordinate_ids") != catalog["s"].get("coordinate_ids")
                or catalog[role].get("probe_load_state") != expected_probes
            ):
                raise _integrity("Direct artifacts disagree with the request View or each other.")
    elif kind == "diagonal_root":
        if set(result) != common | {"scalar_catalog", "array_catalog"} or result.get("array_catalog") != {}:
            raise _integrity("Diagonal-root Result envelope is open or has array payloads.")
        scalars = result.get("scalar_catalog")
        if not isinstance(scalars, dict) or set(scalars) != {"root", "frequency", "linewidth", "slope"}:
            raise _integrity("Diagonal-root scalar catalog is incomplete.")
        _verify_quantity_role(scalars["root"], complex_value=True, unit="radian / second", dimensionality="inverse_time")
        _verify_quantity_role(scalars["frequency"], complex_value=False, unit="hertz", dimensionality="inverse_time")
        _verify_quantity_role(scalars["linewidth"], complex_value=False, unit="hertz", dimensionality="inverse_time")
        _verify_quantity_role(scalars["slope"], complex_value=True, unit="siemens", dimensionality="conductance")
    elif kind == "hybridized_pole":
        _verify_root_like_result(result, {"root", "frequency", "linewidth", "slope", "evidence_sha256"})
        arrays = result.get("array_catalog")
        if not isinstance(arrays, dict) or set(arrays) != {"null_vector"}:
            raise _integrity("Hybridized-pole artifact catalog is incomplete.")
        terminal, _ = _verify_v1_lineage(request.get("ref_lineage"), plan)
        _verify_null_vector_artifact(arrays["null_vector"], terminal)
    elif kind == "transfer_zero":
        if set(result) != common | {"scalar_catalog", "array_catalog"} or result.get("array_catalog") != {}:
            raise _integrity("Transfer-zero Result envelope is malformed.")
        scalars = result.get("scalar_catalog")
        if not isinstance(scalars, dict) or set(scalars) != {"zero", "frequency", "numerator_slope", "denominator", "evidence_sha256"}:
            raise _integrity("Transfer-zero scalar catalog is incomplete.")
        _verify_quantity_role(scalars["zero"], complex_value=True, unit="radian / second", dimensionality="inverse_time")
        _verify_quantity_role(scalars["frequency"], complex_value=False, unit="hertz", dimensionality="inverse_time")
        for field in ("numerator_slope", "denominator"):
            _verify_quantity_role(scalars[field], complex_value=True, unit="dimensionless", dimensionality="dimensionless")
        _valid_sha(scalars["evidence_sha256"])
    elif kind == "residue_normalized_coupling":
        if set(result) != common | {"scalar_catalog", "array_catalog"} or result.get("array_catalog") != {}:
            raise _integrity("Residue coupling Result envelope is malformed.")
        scalars = result.get("scalar_catalog")
        if not isinstance(scalars, dict) or set(scalars) != {"coupling", "magnitude", "branch_a_residue", "branch_b_residue", "evidence_sha256"}:
            raise _integrity("Residue coupling scalar catalog is incomplete.")
        _verify_quantity_role(scalars["coupling"], complex_value=True, unit="radian / second", dimensionality="inverse_time")
        _verify_quantity_role(scalars["magnitude"], complex_value=False, unit="radian / second", dimensionality="inverse_time")
        for field in ("branch_a_residue", "branch_b_residue"):
            _verify_quantity_role(scalars[field], complex_value=True, unit="ohm", dimensionality="resistance")
        _valid_sha(scalars["evidence_sha256"])
    elif kind == "response_element":
        if set(result) != common | {"scalar_catalog", "array_catalog"} or result.get("array_catalog") != {}:
            raise _integrity("Response-element Result envelope is malformed.")
        scalars = result.get("scalar_catalog")
        if not isinstance(scalars, dict) or set(scalars) != {"family", "value", "magnitude", "real", "imag", "evidence_sha256"}:
            raise _integrity("Response-element scalar catalog is incomplete.")
        role = {"S": ("dimensionless", "dimensionless"), "Y": ("siemens", "conductance"), "Z": ("ohm", "resistance")}.get(scalars.get("family"))
        if role is None:
            raise _integrity("Response-element family is malformed.")
        _verify_quantity_role(scalars["value"], complex_value=True, unit=role[0], dimensionality=role[1])
        for field in ("magnitude", "real", "imag"):
            _verify_quantity_role(scalars[field], complex_value=False, unit=role[0], dimensionality=role[1])
        _valid_sha(scalars["evidence_sha256"])
    elif kind == "operator":
        if set(result) != common | {"scalar_catalog", "array_catalog"} or result.get("scalar_catalog") != {}:
            raise _integrity("Operator Result envelope is malformed.")
        catalog = result.get("array_catalog")
        if not isinstance(catalog, dict) or set(catalog) != {"frequencies", "operator"}:
            raise _integrity("Operator artifact catalog is incomplete.")
        count = _verify_direct_artifact(catalog["frequencies"], "frequencies")
        spec_frequencies = request.get("spec", {}).get("frequencies") if isinstance(request.get("spec"), dict) else None
        terminal, _ = _verify_v1_lineage(request.get("ref_lineage"), plan)
        if not isinstance(spec_frequencies, list) or count != len(spec_frequencies):
            raise _integrity("Operator frequency artifact disagrees with its request grid.")
        _verify_operator_artifact(
            catalog["operator"], count, terminal,
            _expected_probe_load_state(request.get("ref_lineage")),
        )
    elif kind == "optimization":
        expected = common | {"baseline", "best", "completed_generations", "unused_evaluations", "ledger_artifacts"}
        if set(result) != expected or not isinstance(result.get("baseline"), dict) or not isinstance(result.get("best"), dict):
            raise _integrity("Optimization Result envelope is open or incomplete.")
        best = result["best"]
        if (
            set(best) != {"evaluation_ordinal", "cost_f64", "parameters"}
            or not isinstance(best.get("evaluation_ordinal"), int)
            or isinstance(best.get("evaluation_ordinal"), bool)
            or best["evaluation_ordinal"] < 0
            or not _finite_f64(best.get("cost_f64"))
        ):
            raise _integrity("Optimization winner envelope is open.")
        _verify_parameter_set_document(best["parameters"], require_empty_authorization=True)
        baseline = result["baseline"]
        expected_baseline = {
            "evaluation_ordinal", "origin", "generation", "population_column",
            "optimizer_coordinates_f64", "parameters", "cache_hit",
            "extrapolation_evidence", "outcome",
        }
        baseline_outcome = baseline.get("outcome")
        if (
            set(baseline) != expected_baseline
            or baseline.get("evaluation_ordinal") != 0
            or baseline.get("origin") != "baseline"
            or baseline.get("generation") != 0
            or baseline.get("population_column") is not None
            or baseline.get("cache_hit") is not False
            or not isinstance(baseline.get("optimizer_coordinates_f64"), list)
            or not baseline["optimizer_coordinates_f64"]
            or any(not _finite_f64(value) for value in baseline["optimizer_coordinates_f64"])
            or not isinstance(baseline_outcome, dict)
            or set(baseline_outcome) != {"status", "cost_f64", "objective_components"}
            or baseline_outcome.get("status") != "success"
            or not _finite_f64(baseline_outcome.get("cost_f64"))
            or not isinstance(baseline_outcome.get("objective_components"), list)
        ):
            raise _integrity("Optimization baseline envelope is open or malformed.")
        _verify_parameter_set_document(baseline["parameters"], require_empty_authorization=True)
        _verify_extrapolation_evidence(
            baseline.get("extrapolation_evidence"),
            allowed_sources={"none", "optimization_spec"},
            required_rows=_required_extrapolation_rows(
                plan,
                baseline["parameters"],
                authorization_source="optimization_spec",
                optimization_authorizations=request.get("spec", {}).get("allow_extrapolation", [])
                if isinstance(request.get("spec"), dict) else [],
            ),
        )
        generations = result.get("completed_generations")
        unused = result.get("unused_evaluations")
        ledgers = result.get("ledger_artifacts")
        if (
            not isinstance(generations, int)
            or isinstance(generations, bool)
            or generations < 1
            or not isinstance(unused, int)
            or isinstance(unused, bool)
            or unused < 0
            or not isinstance(ledgers, list)
            or len(ledgers) != generations
        ):
            raise _integrity("Optimization Result has no generation ledger catalog.")
        for generation, ledger in enumerate(ledgers, 1):
            text = str(generation).zfill(6)
            if (
                not isinstance(ledger, dict)
                or set(ledger) != {"id", "path", "sha256", "media_type", "byte_length"}
                or ledger.get("id") != f"generation_{text}"
                or ledger.get("path") != f"artifacts/generations/{text}.json"
                or ledger.get("media_type") != "application/json"
                or not isinstance(ledger.get("byte_length"), int)
                or isinstance(ledger.get("byte_length"), bool)
                or ledger["byte_length"] < 1
                or _SHA256.fullmatch(str(ledger.get("sha256", ""))) is None
            ):
                raise _integrity("Optimization ledger catalog entry is open or malformed.")
    else:
        raise _integrity("Result operation is outside the Direct runtime.")


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


def _verify_quantity_role(value: object, *, complex_value: bool, unit: str, dimensionality: str) -> None:
    if not isinstance(value, dict):
        raise _integrity("Typed quantity Result field is not an object.")
    magnitude_fields = {"real_si_f64", "imag_si_f64"} if complex_value else {"si_value_f64"}
    expected_type = "complex_quantity_f64" if complex_value else "quantity_f64"
    if (
        set(value) != {"type", "si_unit", "dimensionality"} | magnitude_fields
        or value.get("type") != expected_type
        or value.get("si_unit") != unit
        or value.get("dimensionality") != dimensionality
        or any(not _finite_f64(value[field]) for field in magnitude_fields)
    ):
        raise _integrity("Typed quantity Result field has the wrong physical role.")


def _verify_quantity_any(value: object) -> None:
    if not isinstance(value, dict) or value.get("type") != "quantity_f64":
        raise _integrity("Typed quantity is malformed.")
    unit, dimensionality = value.get("si_unit"), value.get("dimensionality")
    if not isinstance(unit, str) or not isinstance(dimensionality, str):
        raise _integrity("Typed quantity has no physical role.")
    if (unit, dimensionality) not in _canonical_quantity_roles():
        raise _integrity("Typed quantity uses a closed-vocabulary-invalid physical role.")
    _verify_quantity_role(value, complex_value=False, unit=unit, dimensionality=dimensionality)


def _finite_f64(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{16}", value) is not None
        and math.isfinite(struct.unpack(">d", bytes.fromhex(value))[0])
    )


def _canonical_quantity_roles() -> frozenset[tuple[str, str]]:
    """Reuse the identity schema's closed SI-unit/dimensionality vocabulary."""

    from ._canonical import _UNITS

    return frozenset(_UNITS.items())


def _verify_parameter_set_document(value: object, *, require_empty_authorization: bool = False) -> None:
    if not isinstance(value, dict) or set(value) != {"type", "bindings", "allow_extrapolation"} or value.get("type") != "parameter_set":
        raise _integrity("ParameterSet envelope is open or malformed.")
    bindings = value.get("bindings")
    authorizations = value.get("allow_extrapolation")
    if not isinstance(bindings, list) or not isinstance(authorizations, list):
        raise _integrity("ParameterSet arrays are malformed.")
    keys: list[tuple[tuple[str, ...], str]] = []
    units = _canonical_quantity_roles()
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {"parameter", "value"}:
            raise _integrity("ParameterSet binding is open or malformed.")
        reference = binding.get("parameter")
        quantity = binding.get("value")
        if not isinstance(reference, dict) or set(reference) != {"component_path", "parameter_id"}:
            raise _integrity("ParameterSet reference is malformed.")
        path = reference.get("component_path")
        parameter_id = reference.get("parameter_id")
        if not isinstance(path, list) or not path or any(not isinstance(item, str) or not item for item in path) or not isinstance(parameter_id, str) or not parameter_id:
            raise _integrity("ParameterSet reference identity is malformed.")
        if not isinstance(quantity, dict) or set(quantity) != {"type", "si_value_f64", "si_unit", "dimensionality"} or quantity.get("type") != "quantity_f64" or not _finite_f64(quantity.get("si_value_f64")) or (quantity.get("si_unit"), quantity.get("dimensionality")) not in units:
            raise _integrity("ParameterSet value has an unsupported physical role.")
        keys.append((tuple(path), parameter_id))
    if keys != sorted(set(keys)):
        raise _integrity("ParameterSet bindings are not sorted and unique.")
    authorization_keys: list[tuple[tuple[str, ...], str]] = []
    for reference in authorizations:
        if not isinstance(reference, dict) or set(reference) != {"component_path", "parameter_id"}:
            raise _integrity("ParameterSet authorization is malformed.")
        path = reference.get("component_path")
        parameter_id = reference.get("parameter_id")
        if not isinstance(path, list) or not path or any(not isinstance(item, str) or not item for item in path) or not isinstance(parameter_id, str) or not parameter_id:
            raise _integrity("ParameterSet authorization identity is malformed.")
        authorization_keys.append((tuple(path), parameter_id))
    if authorization_keys != sorted(set(authorization_keys)) or any(key not in keys for key in authorization_keys):
        raise _integrity("ParameterSet authorizations are not sorted active references.")
    if require_empty_authorization and authorization_keys:
        raise _integrity("Optimization candidate ParameterSet inherited extrapolation authorization.")


def _verify_extrapolation_evidence(
    value: object,
    *,
    allowed_sources: set[str],
    required_rows: list[dict[str, object]] | None = None,
) -> None:
    """Validate one evidence row per explicitly out-of-support fan-out edge."""

    if not isinstance(value, list):
        raise _integrity("Extrapolation evidence is not an array.")
    keys: list[tuple[tuple[tuple[str, ...], str], tuple[tuple[str, ...], str]]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"parameter", "consumer_target", "support", "input_value", "side", "distance", "authorization_source"}:
            raise _integrity("Extrapolation evidence row is malformed.")
        parameter = _parameter_key_integrity(row.get("parameter"))
        target = _parameter_key_integrity(row.get("consumer_target"))
        support = row.get("support")
        if not isinstance(support, list) or len(support) != 2:
            raise _integrity("Extrapolation support interval is malformed.")
        _verify_quantity_compatible(support[0], support[1])
        _verify_quantity_compatible(support[0], row.get("input_value"))
        _verify_quantity_compatible(support[0], row.get("distance"))
        lower, upper = _f64_value(support[0]["si_value_f64"]), _f64_value(support[1]["si_value_f64"])
        input_value = _f64_value(row["input_value"]["si_value_f64"])
        distance = _f64_value(row["distance"]["si_value_f64"])
        side = row.get("side")
        expected = lower - input_value if side == "lower" else input_value - upper if side == "upper" else None
        if lower >= upper or expected is None or expected <= 0.0 or distance <= 0.0 or struct.pack(">d", expected).hex() != row["distance"].get("si_value_f64"):
            raise _integrity("Extrapolation evidence does not reproduce its canonical distance.")
        if row.get("authorization_source") not in allowed_sources:
            raise _integrity("Extrapolation evidence has an unauthorized source.")
        keys.append((parameter, target))
    if keys != sorted(set(keys)):
        raise _integrity("Extrapolation evidence is not sorted and unique per fan-out edge.")
    if required_rows is not None and value != required_rows:
        raise _integrity("Extrapolation evidence omits or alters a required affine fan-out edge.")


def _required_extrapolation_rows(
    plan: Mapping[str, object],
    parameters: object,
    *,
    authorization_source: str,
    optimization_authorizations: object | None = None,
    require_authorized: bool = True,
) -> list[dict[str, object]]:
    """Derive every out-of-support affine edge from sealed authorities.

    This mirrors the fixed binding order without becoming a second compiler:
    it resolves only scalar parameter bindings needed to identify support
    crossings.  Receipt evidence uses the request's ParameterSet authority;
    candidate evidence uses the complete OptimizationSpec authorization set.
    An unauthorized candidate still owns a ``none`` row, so a typed ``+Inf``
    outcome cannot silently erase the rejected fan-out that caused it.
    """

    if authorization_source not in {"parameter_set", "optimization_spec"}:
        raise _integrity("Extrapolation evidence authority is unknown.")
    _verify_parameter_set_document(parameters)
    assert isinstance(parameters, Mapping)  # narrowed by the closed verifier
    bindings = parameters.get("bindings")
    if not isinstance(bindings, list):
        raise _integrity("ParameterSet bindings are malformed.")
    values: dict[tuple[tuple[str, ...], str], dict[str, object]] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise _integrity("ParameterSet binding is malformed.")
        key = _parameter_key_integrity(binding.get("parameter"))
        raw_value = binding.get("value")
        _verify_quantity_any(raw_value)
        values[key] = dict(raw_value)

    if authorization_source == "parameter_set":
        raw_authorizations = parameters.get("allow_extrapolation")
    else:
        raw_authorizations = optimization_authorizations
    if not isinstance(raw_authorizations, list):
        raise _integrity("Extrapolation authorization collection is malformed.")
    authorized = {_parameter_key_integrity(reference) for reference in raw_authorizations}
    rows: list[dict[str, object]] = []
    resolved_targets: set[tuple[tuple[str, ...], str]] = set()

    def resolve(binding: object, target: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(binding, Mapping) or not isinstance(binding.get("kind"), str):
            raise _integrity("Sealed parameter binding is malformed.")
        kind = binding["kind"]
        if kind == "constant":
            if set(binding) != {"kind", "value"}:
                raise _integrity("Constant parameter binding is malformed.")
            value = binding.get("value")
            _verify_quantity_any(value)
            return dict(value)
        input_reference = binding.get("input")
        input_key = _parameter_key_integrity(input_reference)
        input_value = values.get(input_key)
        if input_value is None:
            raise _integrity("Sealed affine input has no resolved public value.")
        if kind == "identity":
            if set(binding) != {"kind", "input"}:
                raise _integrity("Identity parameter binding is malformed.")
            return dict(input_value)
        if kind != "affine" or set(binding) != {"kind", "input", "slope", "intercept", "support"}:
            raise _integrity("Affine parameter binding is malformed.")
        support = binding.get("support")
        if not isinstance(support, list) or len(support) != 2:
            raise _integrity("Affine support interval is malformed.")
        _verify_quantity_compatible(support[0], support[1])
        _verify_quantity_compatible(support[0], input_value)
        slope, intercept = binding.get("slope"), binding.get("intercept")
        _verify_quantity_any(slope); _verify_quantity_any(intercept)
        lower = _f64_value(support[0]["si_value_f64"])
        upper = _f64_value(support[1]["si_value_f64"])
        input_scalar = _f64_value(input_value["si_value_f64"])
        if lower >= upper:
            raise _integrity("Affine support interval is not ordered.")
        if input_scalar < lower or input_scalar > upper:
            source = authorization_source if input_key in authorized else "none"
            if authorization_source == "parameter_set" and source == "none":
                if require_authorized:
                    raise _integrity("Successful request has unauthorized affine extrapolation.")
            side, distance = (
                ("lower", lower - input_scalar)
                if input_scalar < lower else ("upper", input_scalar - upper)
            )
            if distance <= 0.0:
                raise _integrity("Affine extrapolation distance is not positive.")
            from ._canonical import float64_hex

            distance_record = dict(input_value)
            distance_record["si_value_f64"] = float64_hex(distance)
            if source != "none" or authorization_source == "optimization_spec":
                rows.append(
                    {
                        "parameter": dict(input_reference),
                        "consumer_target": dict(target),
                        "support": [dict(support[0]), dict(support[1])],
                        "input_value": dict(input_value),
                        "side": side,
                        "distance": distance_record,
                        "authorization_source": source,
                    }
                )
        from ._canonical import float64_hex

        mapped = _f64_value(slope["si_value_f64"]) * input_scalar + _f64_value(intercept["si_value_f64"])
        if not math.isfinite(mapped):
            raise _integrity("Affine mapping is non-finite.")
        output = dict(intercept)
        output["si_value_f64"] = float64_hex(mapped)
        return output

    def apply(target: Mapping[str, object], binding: object) -> None:
        target_key = _parameter_key_integrity(target)
        if target_key in values:
            return
        if target_key in resolved_targets:
            raise _integrity("Sealed parameter graph repeats one consumer target.")
        resolved_targets.add(target_key)
        values[target_key] = resolve(binding, target)

    def visit(component: object) -> None:
        if not isinstance(component, Mapping):
            raise _integrity("Sealed Plan component is malformed.")
        path = component.get("component_path")
        if not isinstance(path, list) or not path or any(not isinstance(item, str) or not item for item in path):
            raise _integrity("Sealed component path is malformed.")
        entries = component.get("parameter_bindings")
        realization = component.get("realization")
        if not isinstance(entries, list) or not isinstance(realization, Mapping):
            raise _integrity("Sealed component parameter inventory is malformed.")
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {"id", "binding"} or not isinstance(entry.get("id"), str) or not entry["id"]:
                raise _integrity("Sealed bound parameter is malformed.")
            apply({"component_path": list(path), "parameter_id": entry["id"]}, entry.get("binding"))
        if realization.get("kind") != "composite":
            return
        maps = realization.get("public_parameter_maps")
        children = realization.get("children")
        if not isinstance(maps, list) or not isinstance(children, list):
            raise _integrity("Sealed Composite parameter graph is malformed.")
        for parameter_map in maps:
            if not isinstance(parameter_map, Mapping) or set(parameter_map) != {"parameter", "consumers"}:
                raise _integrity("Sealed Composite public parameter map is malformed.")
            source = _parameter_key_integrity(parameter_map.get("parameter"))
            if source not in values:
                raise _integrity("Composite public parameter map has no resolved source.")
            consumers = parameter_map.get("consumers")
            if not isinstance(consumers, list):
                raise _integrity("Composite public parameter consumers are malformed.")
            for consumer in consumers:
                if not isinstance(consumer, Mapping) or set(consumer) != {"target", "binding"}:
                    raise _integrity("Composite public parameter consumer is malformed.")
                apply(consumer.get("target"), consumer.get("binding"))
        for child in children:
            visit(child)

    components = plan.get("components")
    if not isinstance(components, list):
        raise _integrity("Sealed Plan components are malformed.")
    for component in components:
        visit(component)
    rows.sort(key=lambda row: (_parameter_key_integrity(row["parameter"]), _parameter_key_integrity(row["consumer_target"])))
    if len({(_parameter_key_integrity(row["parameter"]), _parameter_key_integrity(row["consumer_target"])) for row in rows}) != len(rows):
        raise _integrity("Sealed affine fan-out evidence is ambiguous.")
    return rows


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


def _compare_artifacts(left: object, right: object) -> None:
    if not isinstance(left, list) or not isinstance(right, list):
        raise _integrity("Outcome and receipt require artifact inventories.")
    normalized: list[list[tuple[str, str]]] = []
    for inventory in (left, right):
        entries: list[tuple[str, str]] = []
        identifiers: set[str] = set()
        for entry in inventory:
            if not isinstance(entry, dict) or set(entry) != {"id", "sha256"}:
                raise _integrity("Artifact inventory entry is malformed.")
            identifier = entry.get("id")
            digest = entry.get("sha256")
            if not isinstance(identifier, str) or not identifier:
                raise _integrity("Artifact inventory ID is malformed.")
            if identifier in identifiers:
                raise _integrity("Artifact inventory contains a duplicate ID.", artifact_id=identifier)
            identifiers.add(identifier)
            entries.append((identifier, _valid_sha(digest)))
        normalized.append(entries)
    if normalized[0] != normalized[1]:
        raise _integrity("Outcome and receipt artifact inventories disagree.")


def _verify_artifact_inventory(directory: Path, result: Mapping[str, object], receipt: Mapping[str, object]) -> None:
    catalog = result.get("array_catalog")
    if catalog is None:
        catalog = {}
    if not isinstance(catalog, dict):
        raise _integrity("Result has no typed array catalog.")
    receipt_artifacts = receipt.get("artifacts")
    if not isinstance(receipt_artifacts, list):
        raise _integrity("Receipt has no artifact inventory.")
    declared: set[tuple[str, str]] = set()
    declared_ids: set[str] = set()
    for entry in receipt_artifacts:
        if not isinstance(entry, dict):
            raise _integrity("Receipt artifact inventory entry is malformed.")
        identifier = entry.get("id")
        digest = entry.get("sha256")
        if not isinstance(identifier, str) or not identifier or identifier in declared_ids:
            raise _integrity("Receipt artifact inventory has a duplicate or invalid ID.")
        declared_ids.add(identifier)
        declared.add((identifier, _valid_sha(digest)))
    resolved: set[tuple[object, object]] = set()
    resolved_ids: set[str] = set()
    resolved_paths: set[str] = set()
    for artifact in catalog.values():
        if not isinstance(artifact, dict):
            raise _integrity("Result array catalog entry is malformed.")
        identifier = artifact.get("id")
        digest = artifact.get("sha256")
        path = artifact.get("path")
        manifest = artifact.get("file_manifest")
        if not isinstance(identifier, str) or not isinstance(digest, str) or not isinstance(path, str) or not isinstance(manifest, str):
            raise _integrity("Result array catalog lacks required artifact identity.")
        if identifier in resolved_ids or path in resolved_paths or manifest in resolved_paths:
            raise _integrity("Result artifact catalog contains duplicate IDs or paths.")
        resolved_ids.add(identifier)
        resolved_paths.update({path, manifest})
        pair = (identifier, _valid_sha(digest))
        resolved.add(pair)
        artifact_path = _inside(directory, path)
        manifest_path = _inside(directory, manifest)
        if not artifact_path.is_dir() or artifact_path.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
            raise _integrity("Result artifact path is missing or unsafe.", artifact_id=identifier)
        manifest_bytes = manifest_path.read_bytes()
        if _sha256(manifest_bytes) != digest:
            raise _integrity("Artifact manifest hash disagrees with result catalog.", artifact_id=identifier)
        manifest_doc = _decode_bytes(manifest_bytes, "artifact manifest")
        if (
            manifest_doc.get("schema") != "scnsim.artifact_manifest"
            or manifest_doc.get("artifact_id") != identifier
            or manifest_doc.get("artifact_path") != path
        ):
            raise _integrity("Artifact manifest identity disagrees with result catalog.", artifact_id=identifier)
        _verify_manifest_tree(artifact_path, manifest_doc)
    ledgers = result.get("ledger_artifacts", [])
    if not isinstance(ledgers, list):
        raise _integrity("Optimization Result ledger catalog is malformed.")
    for artifact in ledgers:
        if not isinstance(artifact, dict):
            raise _integrity("Optimization ledger catalog entry is malformed.")
        identifier = artifact.get("id")
        digest = artifact.get("sha256")
        path = artifact.get("path")
        length = artifact.get("byte_length")
        if not isinstance(identifier, str) or not isinstance(digest, str) or not isinstance(path, str) or not isinstance(length, int):
            raise _integrity("Optimization ledger lacks its file artifact identity.")
        if identifier in resolved_ids or path in resolved_paths:
            raise _integrity("Result artifact catalog contains duplicate IDs or paths.")
        resolved_ids.add(identifier)
        resolved_paths.add(path)
        file_path = _inside(directory, path)
        if not file_path.is_file() or file_path.is_symlink() or file_path.stat().st_size != length:
            raise _integrity("Optimization ledger artifact path is missing or unsafe.", artifact_id=identifier)
        if _sha256(file_path.read_bytes()) != _valid_sha(digest):
            raise _integrity("Optimization ledger hash disagrees with its result catalog.", artifact_id=identifier)
        resolved.add((identifier, digest))
    if declared != resolved:
        raise _integrity("Receipt artifact inventory does not exactly match Result catalog.")
    artifact_root = directory / "artifacts"
    if artifact_root.is_symlink():
        raise _integrity("Result artifact directory is unsafe.")
    expected_top = {
        "/".join(path.split("/")[:2])
        for path in resolved_paths
    }
    if artifact_root.exists():
        if not artifact_root.is_dir():
            raise _integrity("Result artifact directory is unsafe.")
        actual_top = {
            child.relative_to(directory).as_posix()
            for child in artifact_root.iterdir()
        }
        if actual_top != expected_top:
            raise _integrity("Result artifact directory contains undeclared entries.")
    elif expected_top:
        raise _integrity("Result artifact directory is missing.")


def _verify_attempt_layout(directory: Path, *, outcome: str, has_authoritative_outcome: bool) -> None:
    allowed = {"attempt.json", "receipt.json"}
    if has_authoritative_outcome:
        allowed.add("outcome.json")
    if outcome == "success":
        allowed.add("result.json")
    if (directory / "logs").exists():
        allowed.add("logs")
        logs = directory / "logs"
        if logs.is_symlink() or not logs.is_dir():
            raise _integrity("Attempt log directory is open or unsafe.")
        names = {path.name for path in logs.iterdir()}
        if not names or not names.issubset({"stdout.log", "stderr.log", "untrusted-outcome.json"}):
            raise _integrity("Attempt log directory is open or unsafe.")
        if any(path.is_symlink() or not path.is_file() for path in logs.iterdir()):
            raise _integrity("Attempt log entry is unsafe.")
    if (directory / "artifacts").exists():
        allowed.add("artifacts")
    children = {path.name for path in directory.iterdir()}
    if children != allowed:
        raise _integrity("Attempt directory contains undeclared entries.", entries=sorted(children - allowed))


def _verify_generation_artifacts(
    directory: Path,
    artifacts: object,
    *,
    request_sha256: str,
    attempt_sha256: str,
    allow_other_artifacts: bool = False,
) -> list[tuple[int, str]]:
    if not isinstance(artifacts, list):
        raise _integrity("Attempt has no artifact inventory.")
    request_path = directory.parent.parent / "request.json"
    if request_path.is_symlink() or not request_path.is_file() or _sha256(request_path.read_bytes()) != request_sha256:
        raise _integrity("Optimization ledgers lack their exact request envelope.")
    request = _load_canonical(request_path)
    plan_path = directory.parents[3] / "plan.json"
    plan = _load_canonical(plan_path)
    plan_sha256 = request.get("plan_sha256")
    if not isinstance(plan_sha256, str) or _sha256(plan_path.read_bytes()) != plan_sha256:
        raise _integrity("Optimization ledger request does not bind its leaf Plan.")
    _verify_request_document(request, plan_sha256, plan)
    spec = request.get("spec")
    if request.get("operation") != "optimize_direct" and artifacts:
        raise _integrity("Only optimization attempts may retain generation ledgers.")
    if artifacts and (not isinstance(spec, dict) or spec.get("type") != "optimization"):
        raise _integrity("Optimization ledger request spec is malformed.")
    ledgers: list[tuple[int, str, Mapping[str, object]]] = []
    identifiers: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"id", "sha256"}:
            raise _integrity("Optimization ledger artifact is malformed.")
        identifier = artifact.get("id")
        digest = artifact.get("sha256")
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"generation_[0-9]{6,}", identifier) is None
        ):
            raise _integrity("Non-success attempts may retain only generation ledgers.")
        if identifier in identifiers:
            raise _integrity("Optimization ledger inventory repeats an artifact ID.", artifact_id=identifier)
        identifiers.add(identifier)
        generation = int(identifier.removeprefix("generation_"))
        path = f"artifacts/generations/{generation:06d}.json"
        file_path = _inside(directory, path)
        if not file_path.is_file() or file_path.is_symlink():
            raise _integrity("Optimization ledger file is missing.", path=path)
        raw = file_path.read_bytes()
        if _sha256(raw) != _valid_sha(digest):
            raise _integrity("Optimization ledger digest does not match its bytes.", path=path)
        ledger = _decode_bytes(raw, "optimization ledger")
        if (
            ledger.get("schema") != "scnsim.optimization_ledger"
            or ledger.get("schema_version") != 1
            or ledger.get("request_sha256") != request_sha256
            or ledger.get("generation") != generation
        ):
            raise _integrity("Optimization ledger identity is inconsistent.", path=path)
        _verify_generation_ledger(ledger, spec, plan, generation)
        producer = ledger.get("attempt_sha256")
        if producer != attempt_sha256 and not _prior_ledger_is_receipt_backed(
            directory,
            request_sha256=request_sha256,
            attempt_sha256=producer,
            artifact_id=identifier,
            digest=digest,
        ):
            raise _integrity("Replayed ledger lacks its producing attempt evidence.", path=path)
        ledgers.append((generation, digest, ledger))
    ledgers.sort(key=lambda item: item[0])
    if [item[0] for item in ledgers] != list(range(1, len(ledgers) + 1)):
        raise _integrity("Optimization ledger generations are not contiguous.")
    previous: str | None = None
    complete_generations = None
    if ledgers:
        complete_generations = spec.get("optimizer", {}).get("complete_generations") if isinstance(spec.get("optimizer"), dict) else None
        if not isinstance(complete_generations, int) or isinstance(complete_generations, bool) or complete_generations < 1:
            raise _integrity("Optimization request has an invalid complete-generation count.")
    for index, (generation, digest, ledger) in enumerate(ledgers):
        if ledger.get("previous_ledger_sha256") != previous:
            raise _integrity("Optimization ledger hash chain is broken.")
        certificate = ledger["continuation_certificate"]
        expected_boundary = "terminal_post_update" if generation == complete_generations else "post_update_post_next_sample_pre_next_update"
        if generation > complete_generations or certificate.get("boundary") != expected_boundary:
            raise _integrity("Optimization ledger continuation boundary is inconsistent with its requested generation.")
        if index + 1 < len(ledgers):
            following = ledgers[index + 1][2]
            if (
                certificate.get("next_raw_optimizer_population_sha256") != following.get("raw_optimizer_population_sha256")
                or certificate.get("next_transformed_optimizer_population_sha256") != following.get("transformed_optimizer_population_sha256")
            ):
                raise _integrity("Optimization continuation certificate does not bind the next generation population.")
        previous = digest
    generation_root = _inside(directory, "artifacts/generations")
    if generation_root.is_symlink() or (generation_root.exists() and not generation_root.is_dir()):
        raise _integrity("Generation artifact directory is unsafe.")
    children = list(generation_root.iterdir()) if generation_root.exists() else []
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise _integrity("Generation artifact directory contains a non-regular entry.")
    actual = {path.relative_to(directory).as_posix() for path in children}
    declared = {str(artifact[2]["generation"]).zfill(6) for artifact in ledgers}
    expected = {f"artifacts/generations/{name}.json" for name in declared}
    if actual != expected:
        raise _integrity("Generation artifact directory contains undeclared files.")
    artifact_root = directory / "artifacts"
    if artifact_root.exists():
        if artifact_root.is_symlink() or not artifact_root.is_dir():
            raise _integrity("Attempt artifact directory is unsafe.")
        if not allow_other_artifacts and any(child.name != "generations" for child in artifact_root.iterdir()):
            raise _integrity("Non-success attempt contains undeclared solver artifacts.")
    result_path = directory / "result.json"
    if result_path.exists():
        result = _load_canonical(result_path)
        if result.get("result_kind") == "optimization":
            _verify_optimization_winner(result, spec, plan, [ledger for _, _, ledger in ledgers])
    return [(generation, digest) for generation, digest, _ in ledgers]


def _verify_generation_ledger(
    ledger: Mapping[str, object],
    spec: Mapping[str, object],
    plan: Mapping[str, object],
    generation: int,
) -> None:
    expected = {
        "schema", "schema_version", "request_sha256", "attempt_sha256",
        "algorithm_id", "generation", "previous_ledger_sha256", "population_size",
        "raw_optimizer_population_sha256", "transformed_optimizer_population_sha256",
        "continuation_certificate", "candidates",
    }
    optimizer = spec.get("optimizer")
    variables = spec.get("variables")
    objectives = spec.get("objectives")
    if not isinstance(optimizer, dict) or not isinstance(variables, list) or not isinstance(objectives, list):
        raise _integrity("Optimization request controls are malformed.")
    population_size = optimizer.get("resolved_population_size")
    candidates = ledger.get("candidates")
    if (
        set(ledger) != expected
        or ledger.get("algorithm_id") != "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v2"
        or not isinstance(population_size, int)
        or isinstance(population_size, bool)
        or population_size < 2
        or ledger.get("population_size") != population_size
        or _SHA256.fullmatch(str(ledger.get("raw_optimizer_population_sha256", ""))) is None
        or _SHA256.fullmatch(str(ledger.get("transformed_optimizer_population_sha256", ""))) is None
        or not isinstance(candidates, list)
        or len(candidates) != population_size
    ):
        raise _integrity("Optimization ledger envelope is open or inconsistent.")
    _verify_continuation_certificate(ledger.get("continuation_certificate"), generation)
    for column, candidate in enumerate(candidates, 1):
        expected_ordinal = 1 + (generation - 1) * population_size + (column - 1)
        _verify_candidate_outcome(
            candidate,
            variables=len(variables),
            objective_ids=[str(objective.get("id")) for objective in objectives if isinstance(objective, dict)],
            plan=plan,
            optimization_authorizations=spec.get("allow_extrapolation", []),
            generation=generation,
            column=column,
            evaluation_ordinal=expected_ordinal,
            baseline=False,
        )
    if (
        ledger.get("raw_optimizer_population_sha256")
        != _candidate_population_sha256(candidates, "optimizer_latent_coordinates_f64", len(variables))
        or ledger.get("transformed_optimizer_population_sha256")
        != _candidate_population_sha256(candidates, "optimizer_coordinates_f64", len(variables))
    ):
        raise _integrity("Optimization population hashes do not reproduce their candidate coordinate arrays.")


def _candidate_population_sha256(
    candidates: list[object],
    field: str,
    variables: int,
) -> str:
    values = [
        candidate[field][row]
        for row in range(variables)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    if len(values) != variables * len(candidates):
        raise _integrity("Optimization population matrix is incomplete.")
    return _sha256(_canonical_bytes({
        "shape": [variables, len(candidates)],
        "values_f64": values,
    }))


def _verify_continuation_certificate(value: object, generation: int) -> None:
    if not isinstance(value, dict):
        raise _integrity("CMA continuation certificate is missing.")
    common = {
        "schema", "schema_version", "projection_id", "boundary",
        "completed_generation", "state_sha256",
    }
    boundary = value.get("boundary")
    expected = common | (
        {"next_raw_optimizer_population_sha256", "next_transformed_optimizer_population_sha256"}
        if boundary == "post_update_post_next_sample_pre_next_update"
        else set()
    )
    if (
        set(value) != expected
        or value.get("schema") != "scnsim.cmaes_continuation_certificate"
        or value.get("schema_version") != 1
        or value.get("projection_id") != "cmaes-jl-0.2.6-julia-1.12.6-continuation-state.v1"
        or boundary not in {"post_update_post_next_sample_pre_next_update", "terminal_post_update"}
        or value.get("completed_generation") != generation
        or any(_SHA256.fullmatch(str(value.get(field, ""))) is None for field in expected if field.endswith("sha256"))
    ):
        raise _integrity("CMA continuation certificate is open or malformed.")


def _verify_candidate_outcome(
    value: object,
    *,
    variables: int,
    objective_ids: list[str],
    plan: Mapping[str, object],
    optimization_authorizations: object,
    generation: int,
    column: int | None,
    evaluation_ordinal: int,
    baseline: bool,
) -> None:
    if not isinstance(value, dict):
        raise _integrity("Optimization candidate is not an object.")
    expected = {
        "evaluation_ordinal", "origin", "generation", "population_column",
        "optimizer_coordinates_f64", "parameters", "cache_hit",
        "extrapolation_evidence", "outcome",
    }
    if not baseline:
        expected.add("optimizer_latent_coordinates_f64")
    coordinates = value.get("optimizer_coordinates_f64")
    latent = value.get("optimizer_latent_coordinates_f64")
    if (
        set(value) != expected
        or value.get("evaluation_ordinal") != evaluation_ordinal
        or value.get("origin") != ("baseline" if baseline else "population")
        or value.get("generation") != generation
        or value.get("population_column") != column
        or not isinstance(value.get("cache_hit"), bool)
        or not isinstance(coordinates, list)
        or len(coordinates) != variables
        or any(not _finite_f64(item) or not 0.0 <= _f64_value(item) <= 1.0 for item in coordinates)
        or (not baseline and (not isinstance(latent, list) or len(latent) != variables or any(not _finite_f64(item) for item in latent)))
    ):
        raise _integrity("Optimization candidate envelope is open or malformed.")
    _verify_parameter_set_document(value.get("parameters"), require_empty_authorization=True)
    _verify_extrapolation_evidence(
        value.get("extrapolation_evidence"),
        allowed_sources={"none", "optimization_spec"},
        required_rows=_required_extrapolation_rows(
            plan,
            value["parameters"],
            authorization_source="optimization_spec",
            optimization_authorizations=optimization_authorizations,
        ),
    )
    outcome = value.get("outcome")
    if not isinstance(outcome, dict):
        raise _integrity("Optimization candidate outcome is malformed.")
    if outcome.get("status") == "success":
        components = outcome.get("objective_components")
        if (
            set(outcome) != {"status", "cost_f64", "objective_components"}
            or not _finite_f64(outcome.get("cost_f64"))
            or _f64_value(outcome["cost_f64"]) < 0.0
            or not isinstance(components, list)
            or len(components) != len(objective_ids)
        ):
            raise _integrity("Successful optimization candidate is malformed.")
        total = 0.0
        for objective_id, component in zip(objective_ids, components):
            if (
                not isinstance(component, dict)
                or set(component) != {"objective_id", "value", "normalized_residual_f64", "weighted_cost_f64"}
                or component.get("objective_id") != objective_id
                or not _finite_f64(component.get("normalized_residual_f64"))
                or not _finite_f64(component.get("weighted_cost_f64"))
                or _f64_value(component["weighted_cost_f64"]) < 0.0
            ):
                raise _integrity("Optimization objective component is malformed.")
            _verify_quantity_any(component.get("value"))
            total += _f64_value(component["weighted_cost_f64"])
        if struct.pack(">d", total).hex() != outcome.get("cost_f64"):
            raise _integrity("Optimization candidate cost does not equal its ordered components.")
    elif outcome.get("status") == "failure":
        if set(outcome) != {"status", "penalty", "failure"} or outcome.get("penalty") != "positive_infinity":
            raise _integrity("Failed optimization candidate is malformed.")
        _verify_failure_document(outcome.get("failure"), "optimize_direct")
        if outcome["failure"].get("kind") not in {
            "invalid_candidate_physical_parameter", "eliminated_block_solve_failure",
            "root_slope_unresolved", "numerical_resolution_unresolved",
        }:
            raise _integrity("Optimization candidate uses a request-level failure kind.")
    else:
        raise _integrity("Optimization candidate outcome discriminator is unknown.")


def _verify_optimization_winner(
    result: Mapping[str, object],
    spec: Mapping[str, object],
    plan: Mapping[str, object],
    ledgers: list[Mapping[str, object]],
) -> None:
    variables = spec.get("variables")
    objectives = spec.get("objectives")
    optimizer = spec.get("optimizer")
    if not isinstance(variables, list) or not isinstance(objectives, list) or not isinstance(optimizer, dict):
        raise _integrity("Optimization Result request spec is malformed.")
    objective_ids = [str(objective.get("id")) for objective in objectives if isinstance(objective, dict)]
    baseline = result.get("baseline")
    _verify_candidate_outcome(
        baseline,
        variables=len(variables),
        objective_ids=objective_ids,
        plan=plan,
        optimization_authorizations=spec.get("allow_extrapolation", []),
        generation=0,
        column=None,
        evaluation_ordinal=0,
        baseline=True,
    )
    if (
        result.get("completed_generations") != len(ledgers)
        or result.get("completed_generations") != optimizer.get("complete_generations")
        or result.get("unused_evaluations") != optimizer.get("unused_evaluations")
        or not ledgers
        or ledgers[-1]["continuation_certificate"].get("boundary") != "terminal_post_update"
    ):
        raise _integrity("Optimization Result does not close its requested complete generations.")
    records = [baseline, *(candidate for ledger in ledgers for candidate in ledger["candidates"])]
    seen: dict[bytes, object] = {}
    winners: list[tuple[float, int, Mapping[str, object]]] = []
    for record in records:
        parameters = _canonical_bytes(record["parameters"])
        cached = record.get("cache_hit")
        if cached is True and (parameters not in seen or seen[parameters] != record.get("outcome")):
            raise _integrity("Optimization cache hit does not match its earlier candidate.")
        if cached is False and parameters in seen:
            raise _integrity("Repeated optimization parameters were not marked as a cache hit.")
        seen.setdefault(parameters, record.get("outcome"))
        outcome = record["outcome"]
        if outcome.get("status") == "success":
            winners.append((_f64_value(outcome["cost_f64"]), record["evaluation_ordinal"], record))
    winner = min(winners, key=lambda item: (item[0], item[1]))[2]
    best = result.get("best")
    if (
        not isinstance(best, dict)
        or best.get("evaluation_ordinal") != winner.get("evaluation_ordinal")
        or best.get("cost_f64") != winner["outcome"].get("cost_f64")
        or best.get("parameters") != winner.get("parameters")
    ):
        raise _integrity("Optimization winner does not match the earliest lowest finite candidate.")


def _f64_value(value: object) -> float:
    if not _finite_f64(value):
        raise _integrity("Expected one finite Float64 bit string.")
    return struct.unpack(">d", bytes.fromhex(str(value)))[0]


def _prior_ledger_is_receipt_backed(
    directory: Path,
    *,
    request_sha256: str,
    attempt_sha256: object,
    artifact_id: str,
    digest: object,
) -> bool:
    if not isinstance(attempt_sha256, str) or _SHA256.fullmatch(attempt_sha256) is None:
        return False
    current_match = _ATTEMPT.fullmatch(directory.name)
    staging_match = _STAGING.fullmatch(directory.name)
    if current_match is not None:
        current_ordinal = int(directory.name)
    elif staging_match is not None:
        current_ordinal = int(staging_match.group(1))
    else:
        return False
    for sibling in directory.parent.iterdir():
        if (
            sibling.is_symlink()
            or not sibling.is_dir()
            or _ATTEMPT.fullmatch(sibling.name) is None
            or int(sibling.name) >= current_ordinal
        ):
            continue
        attempt_path = sibling / "attempt.json"
        receipt_path = sibling / "receipt.json"
        if attempt_path.is_symlink() or receipt_path.is_symlink():
            raise _integrity("Prior attempt evidence is symlinked.", path=str(sibling))
        if not attempt_path.is_file() or not receipt_path.is_file():
            continue
        prior_attempt = _load_canonical(attempt_path)
        if _sha256(_canonical_bytes(prior_attempt)) != attempt_sha256:
            continue
        receipt = _load_canonical(receipt_path)
        if (
            receipt.get("schema") != "scnsim.receipt"
            or receipt.get("schema_version") != 1
            or receipt.get("request_sha256") != request_sha256
            or receipt.get("attempt_sha256") != attempt_sha256
            or receipt.get("outcome") not in {"success", "failure", "interrupted"}
            or not isinstance(receipt.get("artifacts"), list)
        ):
            continue
        links = receipt.get("artifacts")
        if isinstance(links, list) and any(
            isinstance(link, dict)
            and link.get("id") == artifact_id
            and link.get("sha256") == digest
            for link in links
        ):
            return True
    return False


def verified_generation_links(
    directory: Path,
    *,
    request_sha256: str,
    attempt_sha256: str,
    allow_other_artifacts: bool = False,
) -> list[dict[str, str]]:
    """Build and verify the receipt links for completed staged generations."""

    root = _inside(directory, "artifacts/generations")
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise _integrity("Generation artifact directory is unsafe.")
    children = sorted(root.iterdir()) if root.exists() else []
    if any(
        path.is_symlink()
        or not path.is_file()
        or re.fullmatch(r"[0-9]{6,}\.json", path.name) is None
        for path in children
    ):
        raise _integrity("Generation artifact directory contains an unsafe entry.")
    links = [
        {
            "id": f"generation_{path.stem}",
            "sha256": _sha256(path.read_bytes()),
        }
        for path in children
    ]
    _verify_generation_artifacts(
        directory,
        links,
        request_sha256=request_sha256,
        attempt_sha256=attempt_sha256,
        allow_other_artifacts=allow_other_artifacts,
    )
    return links


def _inside(root: Path, relative: str) -> Path:
    path = root / _relative_path(relative)
    current = root
    if current.is_symlink():
        raise _integrity("Evidence root is symlinked.", path=str(root))
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise _integrity("Evidence path traverses a symlink.", path=relative)
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise _integrity("Artifact path escapes its attempt directory.", path=relative) from error
    return path


def _verify_manifest_tree(artifact: Path, manifest: Mapping[str, object]) -> None:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise _integrity("Artifact manifest lacks file inventory.")
    declared: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise _integrity("Artifact manifest file entry is malformed.")
        relative = entry.get("path")
        digest = entry.get("sha256")
        length = entry.get("byte_length")
        if not isinstance(relative, str) or not isinstance(length, int) or length < 0:
            raise _integrity("Artifact manifest file entry has invalid path or length.")
        if relative in declared:
            raise _integrity("Artifact manifest repeats a file path.", path=relative)
        path = _inside(artifact, relative)
        if not path.is_file() or path.is_symlink() or path.stat().st_size != length or _sha256(path.read_bytes()) != _valid_sha(digest):
            raise _integrity("Artifact file disagrees with its manifest.", path=relative)
        declared.add(relative)
    actual: set[str] = set()
    for child in artifact.rglob("*"):
        if child.is_symlink():
            raise _integrity("Artifact tree contains a symlink.", path=str(child))
        if child.is_file():
            actual.add(child.relative_to(artifact).as_posix())
    if actual != declared:
        raise _integrity("Artifact manifest does not enumerate exactly its regular files.")


__all__ = [
    "AttemptAllocation",
    "VerifiedSuccess",
    "WorkspaceBinding",
    "bind_workspace",
    "verified_generation_links",
]
