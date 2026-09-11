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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from .errors import (
    EvidenceIntegrityError,
    ResultUnavailableError,
    UnsupportedRuntimePlatformError,
    WorkspacePlanReplacedError,
    WorkspaceCommitIndeterminateError,
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


class _WorkspacePublishIndeterminate(Exception):
    """The active-pointer replace succeeded but its directory fsync did not."""


def _publish_workspace_state(path: Path, data: bytes) -> None:
    """Publish the logical workspace commit with a distinct uncertain tail."""

    if path.is_symlink() or path.parent.is_symlink():
        raise _integrity("Evidence write target must not traverse a symlink.", path=str(path))
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4()}")
    replaced = False
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        replaced = True
        try:
            _fsync_directory(path.parent)
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise _integrity(
                    "Published workspace pointer failed exact state confirmation.",
                    path=str(path),
                )
        except BaseException as exc:
            raise _WorkspacePublishIndeterminate from exc
    finally:
        if not replaced and temporary.exists():
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
            state = _load_canonical(self.root / "workspace.json")
            _assert_root_envelope(state)
            self.assert_current()
            _validate_active_evidence(self.root, state)
            _finish_root_maintenance(self.root, state)
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
        root_state = _load_canonical(self.root / "workspace.json")
        _assert_root_envelope(root_state)
        maintenance = root_state.get("maintenance")
        return {
            "schema": "scnsim.inventory",
            "schema_version": 2,
            "workspace_instance_id": self.workspace_instance_id,
            "plan_sha256": self.plan_sha256,
            "requests": rows,
            "maintenance": [] if maintenance is None else [dict(maintenance)],
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
        is_hb = request_document.get("operation") == "solve_hb"
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
            if is_hb:
                attempt_fields.add("fftw_threads")
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
            if is_hb and attempt.get("fftw_threads") != 1:
                raise _integrity("HB launched attempt lacks fixed FFTW thread evidence.", attempt=str(directory))
            if not is_hb and "fftw_threads" in attempt:
                raise _integrity("Direct launched attempt must not claim HB FFTW evidence.", attempt=str(directory))
        elif any(key in attempt for key in ("julia_threads", "blas_threads", "blas_vendor", "fftw_threads")):
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
        parameter_source = request_document.get("parameter_source")
        point_parameters = (
            parameter_source.get("parameters")
            if isinstance(parameter_source, Mapping) and parameter_source.get("kind") == "point"
            else None
        )
        required_extrapolation = (
            []
            if request_document.get("operation") == "optimize_direct" or point_parameters is None
            else _required_extrapolation_rows(
                plan_document, point_parameters,
                authorization_source="parameter_set", require_authorized=outcome == "success",
            )
        )
        _verify_extrapolation_evidence(
            evidence.get("extrapolation_evidence"),
            allowed_sources={"parameter_set"},
            required_rows=required_extrapolation,
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
        completed_generation_count = 0
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
            _compare_artifacts(
                outcome_document.get("artifacts"),
                receipt.get("artifacts"),
                operation=request_document.get("operation"),
            )
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
            parameter_source = request_document.get("parameter_source")
            is_parameter_sweep = (
                isinstance(parameter_source, Mapping)
                and parameter_source.get("kind") in {"grid", "points"}
            )
            expected_result_kind = (
                "parameter_sweep" if is_parameter_sweep
                else "direct_response" if request_document.get("operation") == "solve_direct"
                else "hb_batch" if request_document.get("operation") == "solve_hb"
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
                verified_generations = _verify_generation_artifacts(
                    directory,
                    receipt["artifacts"],
                    request_sha256=request_sha256,
                    attempt_sha256=attempt_sha256,
                )
                completed_generation_count = len(verified_generations)
        elif receipt.get("result_sha256") is not None or (directory / "result.json").exists():
            raise _integrity("Non-success evidence must not retain a Result.", attempt=str(directory))
        else:
            verified_generations = _verify_generation_artifacts(
                directory,
                receipt["artifacts"],
                request_sha256=request_sha256,
                attempt_sha256=attempt_sha256,
            )
            completed_generation_count = len(verified_generations)
            if outcome_sha is not None:
                linked = "failure" if outcome == "failure" else "interruption"
                if receipt.get(linked) != outcome_document.get(linked):
                    raise _integrity(
                        f"{linked.capitalize()} receipt does not match its authoritative outcome.",
                        attempt=str(directory),
                    )
        if outcome == "failure":
            _verify_failure_document(receipt.get("failure"), request_document.get("operation"))
            failure_evidence = receipt.get("failure", {}).get("evidence") if isinstance(receipt.get("failure"), Mapping) else None
            if request_document.get("operation") == "optimize_direct" and isinstance(failure_evidence, Mapping) and failure_evidence.get("optimization_context") is not None:
                _verify_terminal_optimization_failure(
                    receipt["failure"], request_document.get("spec"),
                    completed_generations=completed_generation_count,
                )
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
    commit: Callable[[], None],
) -> WorkspaceBinding:
    """Bind prepared Plan evidence, then invoke its non-failing seal commit."""

    _require_platform()
    _valid_sha(plan_sha256)
    if _sha256(plan_bytes) != plan_sha256:
        raise _integrity("Sealed Plan bytes do not match their identity.")
    plan = _decode_bytes(plan_bytes, "plan")
    if plan.get("schema") != "scnsim.plan" or plan.get("schema_version") != 2:
        raise _integrity("Plan bytes are not a schema-version 2 Plan envelope.")
    if not callable(commit):
        raise TypeError("workspace commit must be callable")
    root = Path(workspace).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    with _workspace_lock(root, exclusive=True):
        try:
            state_path = root / "workspace.json"
            if not state_path.exists():
                binding = _recover_or_create_root(root, plan_sha256, plan_bytes, versioned)
            else:
                root_state = _load_canonical(state_path)
                _assert_root_envelope(root_state)
                _validate_active_evidence(root, root_state)
                _finish_root_maintenance(root, root_state)
                root_state = _load_canonical(state_path)
                _assert_root_envelope(root_state)
                kind = root_state.get("kind")
                if kind == "replaceable_workspace":
                    current = _binding_from_replaceable(root, root_state)
                    current._verify_leaf()
                    if not versioned and current.plan_sha256 != plan_sha256:
                        recovered = _adopt_replaceable_orphan(root, root_state, current, plan_sha256)
                        if recovered is not None:
                            binding = recovered
                        else:
                            _cleanup_replaceable_staging(root, current)
                            new = _create_leaf(
                                root / "leaves",
                                plan_sha256,
                                plan_bytes,
                                excluding=str(root_state["workspace_instance_id"]),
                            )
                            updated = dict(root_state)
                            updated["active_leaf"] = _leaf_pointer(new, root)
                            updated["maintenance"] = _retired_leaf_maintenance(current, root)
                            _publish_workspace_state(state_path, _canonical_bytes(updated))
                            binding = new
                    elif versioned:
                        _cleanup_replaceable_staging(root, current)
                        binding = _upgrade_to_versioned(root, root_state, current)
                    else:
                        _cleanup_replaceable_staging(root, current)
                        current.assert_current()
                        binding = current
                elif kind == "versioned_workspace":
                    if not versioned:
                        raise WorkspaceVersioningDowngradeForbidden(
                            "This workspace preserves topology history; choose another workspace for replacement mode.",
                            stage="workspace",
                            evidence={"workspace": str(root)},
                        )
                    binding = _bind_versioned(root, root_state, plan_sha256, plan_bytes)
                else:
                    raise _integrity("Workspace root has an unknown kind.", kind=kind)
        except _WorkspacePublishIndeterminate as exc:
            commit()
            raise WorkspaceCommitIndeterminateError(
                "The workspace pointer may be committed, but durable confirmation failed.",
                stage="workspace_commit",
                evidence={
                    "workspace": str(root),
                    "plan_sha256": plan_sha256,
                    "plan_sealed": True,
                    "workspace_may_have_changed": True,
                },
            ) from exc
        commit()
        try:
            state = _load_canonical(root / "workspace.json")
            _assert_root_envelope(state)
            _finish_root_maintenance(root, state)
        except BaseException:
            # Publication and Plan sealing are already committed. Exact pending
            # cleanup remains visible in the root inventory for the next lock.
            pass
        return binding


def _retired_leaf_maintenance(binding: WorkspaceBinding, root: Path) -> dict[str, object]:
    pointer = _leaf_pointer(binding, root)
    if binding.leaf.parent != root / "leaves":
        raise _integrity("Only a replaceable leaf may become cleanup-pending.")
    return {"kind": "retired_leaf_cleanup", **pointer}


def _finish_root_maintenance(root: Path, state: Mapping[str, object]) -> None:
    maintenance = state.get("maintenance")
    if maintenance is None:
        return
    if not isinstance(maintenance, dict) or set(maintenance) != {
        "kind", "directory", "workspace_instance_id", "plan_sha256"
    } or maintenance.get("kind") != "retired_leaf_cleanup":
        raise _integrity("Workspace maintenance record is open or malformed.")
    identity = _valid_uuid(maintenance.get("workspace_instance_id"))
    plan_sha256 = _valid_sha(maintenance.get("plan_sha256"))
    directory = maintenance.get("directory")
    if directory != f"leaves/{identity}":
        raise _integrity("Workspace maintenance target is not a replaceable leaf.")
    active = state.get("active_leaf")
    if isinstance(active, dict) and active.get("workspace_instance_id") == identity:
        raise _integrity("Workspace maintenance cannot retire the active leaf.")
    leaves = root / "leaves"
    if leaves.exists() and (leaves.is_symlink() or not leaves.is_dir()):
        raise _integrity("Workspace maintenance leaf parent is unsafe.", path=str(leaves))
    target = root / _relative_path(directory)
    if target.exists():
        retired = _recover_leaf(root, target, plan_sha256, expected_directory=str(directory))
        _remove_leaf(retired.leaf)
    if leaves.exists() and not any(leaves.iterdir()):
        leaves.rmdir()
        _fsync_directory(root)
    cleared = dict(state)
    cleared.pop("maintenance", None)
    _atomic_write(root / "workspace.json", _canonical_bytes(cleared))


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
    _publish_workspace_state(root / "workspace.json", _canonical_bytes(state))
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
        _publish_workspace_state(root / "workspace.json", _canonical_bytes(state))
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
            _publish_workspace_state(root / "workspace.json", _canonical_bytes(state))
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


def _validate_active_evidence(root: Path, state: Mapping[str, object]) -> None:
    """Verify indexed active evidence before any retired-leaf maintenance."""

    if state.get("kind") == "replaceable_workspace":
        _binding_from_replaceable(root, state)._verify_leaf()
        return
    _validated_versioned_index(root, state)


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
        _verified_unbound_leaf(root, child, f"leaves/{name}")
        raise _integrity(
            "Replaceable workspace contains preserved unbound leaf evidence.",
            path=str(child),
        )
    _fsync_directory(leaves)


def _adopt_replaceable_orphan(
    root: Path,
    state: Mapping[str, object],
    current: WorkspaceBinding,
    requested_plan_sha256: str,
) -> WorkspaceBinding | None:
    leaves = root / "leaves"
    matches: list[WorkspaceBinding] = []
    unbound: list[WorkspaceBinding] = []
    for child in leaves.iterdir():
        if child == current.leaf or _LEAF_STAGING.fullmatch(child.name) is not None:
            continue
        if child.is_symlink() or not child.is_dir() or _UUID4.fullmatch(child.name) is None:
            raise _integrity("Replaceable workspace contains malformed orphan evidence.", path=str(child))
        orphan = _verified_unbound_leaf(root, child, f"leaves/{child.name}")
        unbound.append(orphan)
        requests = child / "requests"
        if requests.exists() and (requests.is_symlink() or not requests.is_dir() or any(requests.iterdir())):
            continue
        if orphan.plan_sha256 == requested_plan_sha256:
            matches.append(orphan)
    if not matches:
        return None
    if len(matches) != 1 or len(unbound) != 1:
        raise _integrity("Replaceable workspace has competing unbound leaf evidence.")
    recovered = matches[0]
    updated = dict(state)
    updated["active_leaf"] = _leaf_pointer(recovered, root)
    updated["maintenance"] = _retired_leaf_maintenance(current, root)
    _publish_workspace_state(root / "workspace.json", _canonical_bytes(updated))
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
        if set(state) not in (
            {"schema", "schema_version", "kind", "workspace_instance_id", "active_leaf"},
            {"schema", "schema_version", "kind", "workspace_instance_id", "active_leaf", "maintenance"},
        ) or not isinstance(active, dict) or set(active) != {"directory", "workspace_instance_id", "plan_sha256"}:
            raise _integrity("Replaceable workspace envelope is open or malformed.")
        leaf_id = _valid_uuid(active.get("workspace_instance_id"))
        if leaf_id == root_id or active.get("directory") != f"leaves/{leaf_id}":
            raise _integrity("Replaceable workspace active pointer is not canonical.")
        _valid_sha(active.get("plan_sha256"))
    else:
        if set(state) not in (
            {"schema", "schema_version", "kind", "workspace_instance_id", "next_iteration", "iterations"},
            {"schema", "schema_version", "kind", "workspace_instance_id", "next_iteration", "iterations", "maintenance"},
        ):
            raise _integrity("Versioned workspace envelope is open or malformed.")
    maintenance = state.get("maintenance")
    if maintenance is not None and (
        not isinstance(maintenance, dict)
        or set(maintenance) != {"kind", "directory", "workspace_instance_id", "plan_sha256"}
        or maintenance.get("kind") != "retired_leaf_cleanup"
        or maintenance.get("directory") != f"leaves/{_valid_uuid(maintenance.get('workspace_instance_id'))}"
    ):
        raise _integrity("Workspace maintenance record is open or malformed.")
    if isinstance(maintenance, dict):
        _valid_sha(maintenance.get("plan_sha256"))
        active_pointer = state.get("active_leaf")
        if (
            kind == "replaceable_workspace"
            and isinstance(active_pointer, dict)
            and active_pointer.get("workspace_instance_id")
            == maintenance.get("workspace_instance_id")
        ):
            raise _integrity("Workspace maintenance cannot retire the active leaf.")


def _upgrade_to_versioned(root: Path, state: Mapping[str, object], current: WorkspaceBinding) -> WorkspaceBinding:
    current._verify_leaf()
    if any(path.is_symlink() for path in current.leaf.rglob("*")):
        raise _integrity("Workspace conversion refuses symlinked evidence.")
    destination = root / "iteration01"
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir() or any(path.is_symlink() for path in destination.rglob("*")):
            raise _integrity("Interrupted workspace upgrade left unsafe iteration01 evidence.")
        upgraded = _recover_leaf(
            root,
            destination,
            current.plan_sha256,
            expected_directory="iteration01",
        )
        if upgraded.workspace_instance_id != current.workspace_instance_id:
            raise _integrity("Interrupted workspace upgrade copy has the wrong identity.")
    else:
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
        "maintenance": _retired_leaf_maintenance(current, root),
    }
    _publish_workspace_state(root / "workspace.json", _canonical_bytes(index))
    return upgraded


def _validated_versioned_index(
    root: Path,
    state: Mapping[str, object],
) -> tuple[
    list[Mapping[str, object]],
    int,
    str,
    tuple[tuple[str, WorkspaceBinding], ...],
]:
    """Validate a complete versioned index and each indexed leaf."""

    iterations = state.get("iterations")
    next_iteration = state.get("next_iteration")
    if not isinstance(iterations, list) or not isinstance(next_iteration, int) or next_iteration < 1:
        raise _integrity("Versioned workspace index is malformed.")
    checked_iterations: list[Mapping[str, object]] = []
    bindings: list[tuple[str, WorkspaceBinding]] = []
    seen_plans: set[str] = set()
    seen_ordinals: set[int] = set()
    seen_leaf_ids: set[str] = set()
    root_id = _valid_uuid(state.get("workspace_instance_id"))
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
        checked_iterations.append(entry)
        bindings.append((plan, binding))
    if sorted(seen_ordinals) != list(range(1, len(seen_ordinals) + 1)) or next_iteration != len(seen_ordinals) + 1:
        raise _integrity("Versioned workspace next_iteration is not canonical.")
    return checked_iterations, next_iteration, root_id, tuple(bindings)


def _bind_versioned(root: Path, state: Mapping[str, object], plan_sha256: str, plan_bytes: bytes) -> WorkspaceBinding:
    iterations, next_iteration, root_id, bindings = _validated_versioned_index(root, state)
    existing = next(
        (binding for plan, binding in bindings if plan == plan_sha256),
        None,
    )
    indexed = {binding.leaf.name for _, binding in bindings}
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
    _publish_workspace_state(root / "workspace.json", _canonical_bytes(state))
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
        raise _integrity("Unindexed iteration belongs to another Plan and is preserved.")
    updated = dict(state)
    iterations = updated.get("iterations")
    if not isinstance(iterations, list):
        raise _integrity("Versioned workspace index is malformed.")
    updated["iterations"] = [*iterations, {"ordinal": next_iteration, **_leaf_pointer(leaf, root)}]
    updated["next_iteration"] = next_iteration + 1
    _publish_workspace_state(root / "workspace.json", _canonical_bytes(updated))
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
        "solve_hb": {"hb_solve": "scnsim.hb_response.josephsoncircuits.v1"},
        "evaluate_direct": {
            "diagonal_root": "scnsim.diagonal_root.newton32.v1",
            "hybridized_pole": "scnsim.hybridized_pole.newton32.v1",
            "transfer_zero": "scnsim.transfer_zero.newton32.v1",
            "residue_normalized_coupling": "scnsim.residue_normalized_coupling.v1",
            "response_element": "scnsim.response_element.v1",
            "operator": "scnsim.direct_operator.v1",
        },
        "optimize_direct": {"optimization": "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v4"},
    }
    expected_algorithm = algorithms.get(operation, {}).get(spec.get("type") if isinstance(spec, dict) else None)
    if (
        set(request) != {"schema", "schema_version", "plan_sha256", "operation", "view", "spec", "parameter_source", "runtime_semantic"}
        or request.get("schema") != "scnsim.request"
        or request.get("schema_version") != 2
        or request.get("plan_sha256") != plan_sha256
        or expected_algorithm is None
        or any(not isinstance(request.get(field), dict) for field in ("view", "spec", "parameter_source", "runtime_semantic"))
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
    _verify_parameter_source(request["parameter_source"], plan)
    terminal, port_realizable = _verify_view_declaration(request["view"], plan)
    if operation == "solve_direct":
        _verify_v1_direct_spec(spec, terminal, port_realizable)
    elif operation == "solve_hb":
        logical_ports = [port["id"] for port in plan["connectivity"]["ports"]]
        _verify_v1_hb_spec(spec, terminal, port_realizable, logical_ports)
    elif operation == "evaluate_direct":
        _verify_v1_evaluation_spec(spec, terminal, port_realizable)
    else:
        _verify_v1_optimization_spec(spec, plan)
        leaves = _optimization_selector_leaves(spec)
        if not leaves or request["view"] != leaves[0].get("view"):
            raise _integrity("Optimization primary View is not its first normalized selector View.")
        parameters = request["parameter_source"].get("parameters")
        if not isinstance(parameters, Mapping):
            raise _integrity("Optimization request requires one complete parameter point.")
        active = {_parameter_key_integrity(item["parameter"]) for item in spec["variables"]}
        declared = {_parameter_key_integrity(item) for item in spec["allow_extrapolation"]}
        if declared != ({_parameter_key_integrity(item) for item in parameters["allow_extrapolation"]} & active):
            raise _integrity("Optimization request has inconsistent active extrapolation authorities.")


def _verify_parameter_source(source: object, plan: Mapping[str, object]) -> None:
    if not isinstance(source, dict):
        raise _integrity("Parameter source is malformed.")
    try:
        from ._canonical import canonical_parameter_source

        if canonical_parameter_source(source) != source:
            raise ValueError("noncanonical parameter source")
    except Exception as error:
        raise _integrity("Parameter source is open or noncanonical.") from error
    definitions = plan.get("parameter_closure", {}).get("definitions")
    if not isinstance(definitions, list):
        raise _integrity("Plan parameter closure is malformed.")
    expected = {
        _parameter_key_integrity({
            "definitions_id": item.get("definitions_id"),
            "parameter_id": item.get("parameter_id"),
        })
        for item in definitions
        if isinstance(item, Mapping)
    }
    if len(expected) != len(definitions):
        raise _integrity("Plan parameter definitions are malformed.")

    def parameter_set(value: object, *, complete: bool) -> None:
        _verify_parameter_set_document(value)
        keys = {_parameter_key_integrity(item["parameter"]) for item in value["bindings"]}
        if not keys <= expected or (complete and keys != expected):
            raise _integrity("Parameter source does not bind the Plan's consumed definitions.")

    kind = source["kind"]
    if kind == "point":
        parameter_set(source["parameters"], complete=True)
    elif kind == "grid":
        parameter_set(source["base_parameters"], complete=True)
        axis_keys = []
        for axis in source["axes"]:
            key = _parameter_key_integrity(axis["parameter"])
            if key not in expected:
                raise _integrity("Grid axis does not bind a consumed Plan parameter.")
            axis_keys.append(key)
            for value in axis["values"]:
                _verify_parameter_value(value)
        if len(set(axis_keys)) != len(axis_keys):
            raise _integrity("Grid axes repeat a parameter.")
    elif kind == "points":
        parameter_set(source["baseline_parameters"], complete=True)
        for point in source["points"]:
            parameter_set(point, complete=False)


def _verify_view_declaration(
    view: object,
    plan: Mapping[str, object],
) -> tuple[list[str], bool]:
    if not isinstance(view, dict) or set(view) != {"type", "ptc", "transforms", "retain"} or view.get("type") != "network_view":
        raise _integrity("Declarative View is malformed.")
    _, public = _plan_coordinates(plan)
    available = list(sorted(public))
    connectivity = plan.get("connectivity")
    ports = connectivity.get("ports") if isinstance(connectivity, Mapping) else None
    if not isinstance(ports, list):
        raise _integrity("Plan Port inventory is malformed.")
    port_ids = [port.get("id") for port in ports]
    if any(not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in port_ids) or len(set(port_ids)) != len(port_ids):
        raise _integrity("Plan Port IDs are malformed.")
    net_to_compiler = {
        node["final_net"]: node["compiler_node_id"]
        for node in connectivity["node_coordinates"]
    }
    port_coordinates = {
        net_to_compiler[port["net"]]
        for port in ports
        if port.get("net") in net_to_compiler
    }
    ptc = view.get("ptc")
    if ptc is not None:
        if not isinstance(ptc, dict) or set(ptc) != {"selected_ports"}:
            raise _integrity("View PTC declaration is malformed.")
        selected = ptc["selected_ports"]
        if not isinstance(selected, list) or not selected or selected != [item for item in port_ids if item in selected] or len(set(selected)) != len(selected):
            raise _integrity("View PTC Port order is malformed.")
        by_id = {port["id"]: port for port in ports}
        if any(by_id[item].get("role") != "nonloading_probe" for item in selected):
            raise _integrity("View PTC selects a loading Port.")
    transforms = view.get("transforms")
    if not isinstance(transforms, list):
        raise _integrity("View transforms are malformed.")
    transform_ids: set[str] = set()
    for transform in transforms:
        if not isinstance(transform, dict) or set(transform) != {"id", "input_coordinates", "output_coordinates"}:
            raise _integrity("View transform is malformed.")
        identifier = transform.get("id")
        inputs = transform.get("input_coordinates")
        outputs = transform.get("output_coordinates")
        expected_outputs = [f"{identifier}.common", f"{identifier}.differential"]
        if (
            not isinstance(identifier, str)
            or _IDENTIFIER.fullmatch(identifier) is None
            or identifier in transform_ids
            or not isinstance(inputs, list)
            or len(inputs) != 2
            or len(set(inputs)) != 2
            or any(item not in available for item in inputs)
            or outputs != expected_outputs
            or any(item in available for item in expected_outputs)
        ):
            raise _integrity("View transform basis transition is malformed.")
        transform_ids.add(identifier)
        index = min(available.index(item) for item in inputs)
        both_ports = all(item in port_coordinates for item in inputs)
        available = [item for item in available if item not in inputs]
        available[index:index] = expected_outputs
        port_coordinates.difference_update(inputs)
        if both_ports:
            port_coordinates.update(expected_outputs)
    retain = view.get("retain")
    if retain is None:
        return list(port_ids), bool(port_ids)
    if not isinstance(retain, dict) or set(retain) != {"retained_coordinates"}:
        raise _integrity("View retain declaration is malformed.")
    retained = retain["retained_coordinates"]
    if not isinstance(retained, list) or not retained or len(set(retained)) != len(retained) or any(item not in available for item in retained):
        raise _integrity("View retained basis is malformed.")
    return list(retained), all(item in port_coordinates for item in retained)


def _identifiers(value: object, *, field: str, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value) or any(
        not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in value
    ):
        raise _integrity(f"{field} is not an ordered identifier array.")
    if len(set(value)) != len(value):
        raise _integrity(f"{field} repeats an identifier.")
    return list(value)


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
    connectivity = plan.get("connectivity") if plan is not None else None
    plan_ports = connectivity.get("ports") if isinstance(connectivity, Mapping) else None
    if plan is not None and not isinstance(plan_ports, list):
        raise _integrity("Sealed Plan ports are malformed.")
    expected_ports = [port.get("id") for port in plan_ports if isinstance(port, dict)] if isinstance(plan_ports, list) else None
    port_roles = {
        port.get("id"): port.get("role")
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


def _verify_v1_hb_spec(
    spec: object,
    terminal: list[str],
    port_realizable: bool,
    logical_ports: list[str],
) -> None:
    """Close the declared HB request before workspace evidence is allocated.

    Lattice realization remains Julia-owned, but a sealed request must already
    bind a complete, unique authoring declaration to a port-realizable View.
    """

    if not port_realizable:
        raise _integrity("HB S/Y/Z request is not Port-realizable.")
    expected = {
        "type", "pump_axes", "drives", "frequencies", "cases", "truncation",
        "traces", "allow_driven_ptc",
    }
    if not isinstance(spec, dict) or set(spec) != expected or spec.get("type") != "hb_solve":
        raise _integrity("HB solve Spec is malformed.")
    axes = spec.get("pump_axes")
    drives = spec.get("drives")
    cases = spec.get("cases")
    traces = spec.get("traces")
    frequencies = spec.get("frequencies")
    truncation = spec.get("truncation")
    if not isinstance(axes, list) or not isinstance(drives, list) or not isinstance(cases, list) or not cases or not isinstance(traces, list) or not isinstance(frequencies, list) or not frequencies or not isinstance(truncation, dict) or not isinstance(spec.get("allow_driven_ptc"), bool):
        raise _integrity("HB solve Spec has incomplete declarations.")
    axis_ids: set[str] = set()
    for axis in axes:
        if not isinstance(axis, dict) or set(axis) != {"id", "frequency"} or not isinstance(axis.get("id"), str) or _IDENTIFIER.fullmatch(axis["id"]) is None or axis["id"] in axis_ids:
            raise _integrity("HB pump-axis declaration is malformed.")
        _verify_quantity_role(axis.get("frequency"), complex_value=False, unit="hertz", dimensionality="inverse_time")
        if _f64_value(axis["frequency"]["si_value_f64"]) <= 0.0:
            raise _integrity("HB pump-axis frequency is not strictly positive.")
        axis_ids.add(axis["id"])
    previous = 0.0
    for frequency in frequencies:
        _verify_quantity_role(frequency, complex_value=False, unit="hertz", dimensionality="inverse_time")
        value = _f64_value(frequency["si_value_f64"])
        if value <= previous:
            raise _integrity("HB response frequency grid is not strictly positive and increasing.")
        previous = value
    drive_ids: set[str] = set()
    for drive in drives:
        if (
            not isinstance(drive, dict)
            or set(drive) != {"id", "port_id", "mode", "orientation"}
            or not isinstance(drive.get("id"), str)
            or _IDENTIFIER.fullmatch(drive["id"]) is None
            or drive["id"] in drive_ids
            or drive.get("port_id") not in logical_ports
            or drive.get("orientation") != "port_node_to_reference"
            or not _valid_mode_tuple(drive.get("mode"), len(axis_ids))
        ):
            raise _integrity("HB current-drive declaration is malformed.")
        drive_ids.add(drive["id"])
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "currents"} or not isinstance(case.get("id"), str) or _IDENTIFIER.fullmatch(case["id"]) is None or case["id"] in case_ids or not isinstance(case.get("currents"), list):
            raise _integrity("HB case declaration is malformed.")
        current_ids: set[str] = set()
        for current in case["currents"]:
            if (
                not isinstance(current, dict)
                or set(current) != {"drive_id", "coefficient", "coefficient_convention"}
                or current.get("drive_id") not in drive_ids
                or current["drive_id"] in current_ids
                or current.get("coefficient_convention") != "exp_minus_i_m_dot_omega_t_fourier_coefficient"
            ):
                raise _integrity("HB case current binding is malformed.")
            _verify_quantity_role(current.get("coefficient"), complex_value=True, unit="ampere", dimensionality="current")
            current_ids.add(current["drive_id"])
        case_ids.add(case["id"])
    expected_truncation = {"pump_harmonics", "modulation_harmonics", "max_intermodulation_order", "three_wave_mixing", "four_wave_mixing"}
    if set(truncation) != expected_truncation or not isinstance(truncation.get("pump_harmonics"), list) or not isinstance(truncation.get("modulation_harmonics"), list) or len(truncation["pump_harmonics"]) != len(axis_ids) or len(truncation["modulation_harmonics"]) != len(axis_ids) or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in [*truncation["pump_harmonics"], *truncation["modulation_harmonics"]]) or (truncation.get("max_intermodulation_order") is not None and (not isinstance(truncation["max_intermodulation_order"], int) or isinstance(truncation["max_intermodulation_order"], bool) or truncation["max_intermodulation_order"] < 0)) or not isinstance(truncation.get("three_wave_mixing"), bool) or not isinstance(truncation.get("four_wave_mixing"), bool):
        raise _integrity("HB truncation declaration is malformed.")
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
            or not _valid_mode_tuple(trace.get("input_mode"), len(axis_ids))
            or not _valid_mode_tuple(trace.get("output_mode"), len(axis_ids))
        ):
            raise _integrity("HB trace declaration is malformed.")
        trace_ids.add(trace["id"])


def _valid_mode_tuple(value: object, rank: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == rank
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _hb_operating_lattice_is_vacuous(spec: Mapping[str, object]) -> bool:
    """Reproduce the pinned JC empty operating-basis condition from the request.

    This delegates to the full pinned ordering reconstruction below, so its
    vacuity rule cannot drift from the actual RFFT/parity/crop construction.
    """

    return not _hb_declared_modes_from_spec(spec, response=False)


def _hb_declared_modes_from_spec(spec: Mapping[str, object], *, response: bool) -> list[list[int]]:
    """Mirror the pinned JC 0.5.4 Fourier construction and its ordering.

    The receipt does not merely attest that a returned set fits the requested
    bounds.  `calcfreqsrdft`/`calcfreqsdft`, `truncfreqs`, and (for the
    operating basis) `removeconjfreqs` determine the ordered public channel
    basis.  Reconstructing it here closes result reuse against a backend that
    has silently permuted an otherwise valid lattice.
    """

    axes = spec.get("pump_axes")
    truncation = spec.get("truncation")
    drives = spec.get("drives")
    if not isinstance(axes, list) or not isinstance(truncation, Mapping) or not isinstance(drives, list):
        raise _integrity("HB request cannot reproduce its pinned JC lattice.")
    rank = len(axes)
    limits_key = "modulation_harmonics" if response else "pump_harmonics"
    limits = truncation.get(limits_key)
    if (
        not isinstance(limits, list)
        or len(limits) != rank
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in limits)
        or not isinstance(truncation.get("three_wave_mixing"), bool)
        or not isinstance(truncation.get("four_wave_mixing"), bool)
    ):
        raise _integrity("HB request has an invalid JC lattice truncation.")
    crop = truncation.get("max_intermodulation_order")
    if crop is not None and (not isinstance(crop, int) or isinstance(crop, bool) or crop < 0):
        raise _integrity("HB request has an invalid JC intermodulation crop.")
    declared_dc = any(
        isinstance(drive, Mapping)
        and isinstance(drive.get("mode"), list)
        and len(drive["mode"]) == rank
        and all(value == 0 for value in drive["mode"])
        for drive in drives
    )
    if rank == 0:
        # This is SCNSim's documented private JC adapter: the backend uses an
        # inert `(0,)`, while the public rank-zero basis is `()`.
        return [[]] if response or declared_dc else []

    # Julia CartesianIndices is column-major: the first pump axis advances
    # first.  Iterate reversed Python products to preserve that exact order.
    dimensions = [2 * limit + 1 for limit in limits] if response else [limits[0] + 1, *[2 * limit + 1 for limit in limits[1:]]]
    modes: list[tuple[int, ...]] = []
    for reversed_indices in product(*(range(1, dimension + 1) for dimension in reversed(dimensions))):
        indices = tuple(reversed(reversed_indices))
        mode = tuple(
            index - 1 if index <= limit + 1 else -dimension + index - 1
            for index, limit, dimension in zip(indices, limits, dimensions)
        )
        absolute_order = sum(abs(value) for value in mode)
        criterion = (
            (response and all(value == 0 for value in mode))
            or ((truncation["four_wave_mixing"] if response else truncation["three_wave_mixing"]) and absolute_order > 0 and absolute_order % 2 == 0)
            or ((truncation["three_wave_mixing"] if response else truncation["four_wave_mixing"]) and absolute_order % 2 == 1)
            or (not response and declared_dc and all(value == 0 for value in mode))
        )
        if criterion and (sum(value != 0 for value in mode) == 1 or crop is None or absolute_order <= crop):
            modes.append(mode)
    if response:
        return [list(mode) for mode in modes]

    # `removeconjfreqs` removes the lexicographically greater coordinate of
    # every JC-conjugate pair and retains original Cartesian order.
    nw = tuple(dimensions)
    nt = (2 * nw[0] - 1, *nw[1:])
    removed: set[tuple[int, ...]] = set()
    for reversed_indices in product(*(range(1, dimension + 1) for dimension in reversed(nw))):
        coordinate = tuple(reversed(reversed_indices))
        target = tuple((length - (index - 1)) % length + 1 for index, length in zip(coordinate, nt))
        if coordinate != target and all(index <= dimension for index, dimension in zip(target, nw)):
            removed.add(max(coordinate, target))
    retained: list[list[int]] = []
    for reversed_indices in product(*(range(1, dimension + 1) for dimension in reversed(nw))):
        coordinate = tuple(reversed(reversed_indices))
        if coordinate in removed:
            continue
        mode = tuple(
            index - 1 if index <= limit + 1 else -dimension + index - 1
            for index, limit, dimension in zip(coordinate, limits, dimensions)
        )
        if mode in modes:
            retained.append(list(mode))
    return retained


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


def _verify_v1_optimization_spec(spec: object, plan: Mapping[str, object]) -> None:
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
        role = _verify_selector(objective.get("quantity"), plan)
        _verify_quantity_role(objective.get("target"), complex_value=False, unit=role[0], dimensionality=role[1])
        _verify_quantity_role(objective.get("resolved_scale"), complex_value=False, unit=role[0], dimensionality=role[1])
        if not _finite_f64(objective.get("weight_f64")) or objective.get("scale_source") not in {"relative_target", "dimensionless_unity", "explicit"}:
            raise _integrity("Optimization objective scale is malformed.")
    required_optimizer = {"type", "seed", "max_evaluations", "population_size", "resolved_population_size", "initial_sigma_f64", "box_transform_id", "complete_generations", "unused_evaluations", "hidden_stops"}
    if set(optimizer) != required_optimizer or optimizer.get("type") != "cma_es" or optimizer.get("box_transform_id") != "cmaes-jl-0.2.6-linquad-unit-box.v1" or optimizer.get("hidden_stops") != "disabled":
        raise _integrity("Optimization controls are malformed.")


def _parameter_key_integrity(value: object) -> tuple[str, str]:
    if not isinstance(value, dict) or set(value) != {"definitions_id", "parameter_id"}:
        raise _integrity("ParameterRef is malformed.")
    definitions = value.get("definitions_id"); identifier = value.get("parameter_id")
    if not isinstance(definitions, str) or _IDENTIFIER.fullmatch(definitions) is None or not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
        raise _integrity("ParameterRef identity is malformed.")
    return definitions, identifier


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


def _optimization_selector_leaves(spec: object) -> list[Mapping[str, object]]:
    if not isinstance(spec, Mapping) or not isinstance(spec.get("objectives"), list):
        return []
    leaves: list[Mapping[str, object]] = []
    for objective in spec["objectives"]:
        quantity = objective.get("quantity") if isinstance(objective, Mapping) else None
        if isinstance(quantity, Mapping) and quantity.get("type") == "quantity_sum":
            terms = quantity.get("terms")
            if isinstance(terms, list):
                leaves.extend(term for term in terms if isinstance(term, Mapping))
        elif isinstance(quantity, Mapping):
            leaves.append(quantity)
    return leaves


def _verify_selector(value: object, plan: Mapping[str, object]) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise _integrity("Optimization selector is malformed.")
    if value.get("type") == "quantity_sum":
        if set(value) != {"type", "terms"} or not isinstance(value.get("terms"), list) or not value["terms"]:
            raise _integrity("QuantitySum is malformed.")
        roles = [_verify_selector(item, plan) for item in value["terms"]]
        if any(role[1] != roles[0][1] for role in roles[1:]):
            raise _integrity("QuantitySum terms have incompatible physical roles.")
        return roles[0]
    fields = {"type", "spec", "projection", "view"}
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
    terminal, port_realizable = _verify_view_declaration(value.get("view"), plan)
    _verify_v1_evaluation_spec(value["spec"], terminal, port_realizable)
    if expected[2] is not None:
        return expected[2]
    family = value["spec"].get("family")
    return {"S": ("dimensionless", "dimensionless"), "Y": ("siemens", "conductance"), "Z": ("ohm", "resistance")}[family]


def _plan_coordinates(plan: Mapping[str, object]) -> tuple[list[str], set[str]]:
    """Return the snapshot-owned compiler basis and public subset."""

    connectivity = plan.get("connectivity")
    nodes = connectivity.get("node_coordinates") if isinstance(connectivity, Mapping) else None
    if not isinstance(nodes, list) or not nodes:
        raise _integrity("Sealed Plan coordinate inventory is malformed.")
    order: list[str] = []
    public: set[str] = set()
    for node in nodes:
        if (
            not isinstance(node, Mapping)
            or set(node) != {"final_net", "compiler_node_id", "visibility", "public_aliases"}
            or not isinstance(node.get("compiler_node_id"), str)
            or not node["compiler_node_id"]
            or node.get("compiler_node_id") != node.get("final_net")
            or node.get("visibility") not in {"public", "internal"}
            or not isinstance(node.get("public_aliases"), list)
        ):
            raise _integrity("Sealed Plan node inventory is malformed.")
        compiler_id = node["compiler_node_id"]
        if compiler_id in order:
            raise _integrity("Sealed Plan compiler coordinates are not unique.")
        order.append(compiler_id)
        if node["visibility"] == "public":
            if not node["public_aliases"]:
                raise _integrity("Public compiler coordinate has no public alias.")
            public.add(compiler_id)
        elif node["public_aliases"]:
            raise _integrity("Internal compiler coordinate exposes a public alias.")
    return order, public


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


def _optimization_leaf_catalog(objectives: object) -> list[tuple[dict[str, object], Mapping[str, object]]]:
    if not isinstance(objectives, list):
        raise _integrity("Optimization objectives are malformed.")
    leaves: list[tuple[dict[str, object], Mapping[str, object]]] = []
    for objective in objectives:
        if not isinstance(objective, Mapping) or not isinstance(objective.get("id"), str):
            raise _integrity("Optimization objective identity is malformed.")
        for ordinal, selector in enumerate(_selector_terms(objective.get("quantity")), 1):
            leaves.append(({"objective_id": objective["id"], "term_ordinal": ordinal}, selector))
    return leaves


def _optimization_dependency(selector: Mapping[str, object], *, kind: str = "quantity") -> dict[str, object]:
    view = selector.get("view")
    if not isinstance(view, Mapping):
        raise _integrity("Optimization selector View is malformed.")
    view_sha = _sha256(_canonical_bytes(view))
    dependency_sha = view_sha if kind == "view" else _sha256(_canonical_bytes({
        "type": selector.get("type"), "spec": selector.get("spec"), "view": view,
    }))
    return {"kind": kind, "view_sha256": view_sha, "dependency_sha256": dependency_sha}


def _verify_optimization_context_shape(value: object) -> Mapping[str, object]:
    required = {"schema", "schema_version", "phase", "candidate", "owner", "affected_leaves"}
    if not isinstance(value, Mapping) or not required.issubset(value) or not set(value).issubset(required | {"dependency"}):
        raise _integrity("Optimization failure context is open or malformed.")
    if value.get("schema") != "scnsim.optimization_failure_context" or value.get("schema_version") != 1 or value.get("phase") not in {
        "candidate_prepare", "candidate_compile", "view_realization", "baseline_root_anchor",
        "quantity_evaluation", "objective_aggregation", "total_aggregation",
    }:
        raise _integrity("Optimization failure phase is malformed.")
    candidate = value.get("candidate")
    if not isinstance(candidate, Mapping) or set(candidate) != {"evaluation_ordinal", "origin", "generation", "population_column"}:
        raise _integrity("Optimization failure candidate position is malformed.")
    ordinal, origin, generation, column = (candidate.get(key) for key in ("evaluation_ordinal", "origin", "generation", "population_column"))
    integer = lambda item: isinstance(item, int) and not isinstance(item, bool)
    if not integer(ordinal) or not integer(generation) or ordinal < 0 or generation < 0:
        raise _integrity("Optimization failure candidate ordinals are malformed.")
    if origin == "baseline":
        if (ordinal, generation, column) != (0, 0, None):
            raise _integrity("Optimization baseline failure position is inconsistent.")
    elif origin == "population":
        if ordinal < 1 or generation < 1 or not integer(column) or column < 1:
            raise _integrity("Optimization population failure position is inconsistent.")
    else:
        raise _integrity("Optimization failure candidate origin is unknown.")
    owner = value.get("owner")
    valid_owner = (
        isinstance(owner, Mapping)
        and (
            set(owner) == {"kind"} and owner.get("kind") in {"candidate", "dependency"}
            or set(owner) == {"kind", "leaf"} and owner.get("kind") == "leaf"
            or set(owner) == {"kind", "objective_id"} and owner.get("kind") == "objective" and isinstance(owner.get("objective_id"), str)
        )
    )
    if not valid_owner:
        raise _integrity("Optimization failure owner is malformed.")
    affected = value.get("affected_leaves")
    locators = [owner.get("leaf")] if owner.get("kind") == "leaf" else []
    if not isinstance(affected, list) or any(not isinstance(item, Mapping) for item in [*affected, *locators]):
        raise _integrity("Optimization failure leaf locators are malformed.")
    for locator in [*affected, *locators]:
        if set(locator) != {"objective_id", "term_ordinal"} or not isinstance(locator.get("objective_id"), str) or not integer(locator.get("term_ordinal")) or locator["term_ordinal"] < 1:
            raise _integrity("Optimization failure leaf locator is malformed.")
    if len({_canonical_bytes(item) for item in affected}) != len(affected):
        raise _integrity("Optimization affected leaves repeat a locator.")
    dependency = value.get("dependency")
    if dependency is not None and (
        not isinstance(dependency, Mapping)
        or set(dependency) != {"kind", "view_sha256", "dependency_sha256"}
        or dependency.get("kind") not in {"view", "quantity"}
        or _SHA256.fullmatch(str(dependency.get("view_sha256", ""))) is None
        or _SHA256.fullmatch(str(dependency.get("dependency_sha256", ""))) is None
    ):
        raise _integrity("Optimization failure dependency is malformed.")
    if owner.get("kind") == "dependency" and dependency is None:
        raise _integrity("Optimization dependency failure has no dependency identity.")
    return value


def _is_projection_only_optimization_failure(value: object) -> bool:
    """Classify the closed selector failures that own no shared dependency."""

    if not isinstance(value, Mapping):
        return False
    evidence = value.get("evidence")
    return (
        value.get("kind") == "invalid_optimization_spec"
        and value.get("stage") in {"selector", "quantity_sum"}
        and isinstance(evidence, Mapping)
        and evidence.get("operation") == "optimize_direct"
        and evidence.get("context_kind") == "optimization_candidate"
    )


def _optimization_failure_requires_context(value: object, operation: object) -> bool:
    """Return whether a sealed optimization failure is execution-owned."""

    if operation != "optimize_direct" or not isinstance(value, Mapping):
        return False
    evidence = value.get("evidence")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("operation") != "optimize_direct"
        or evidence.get("context_kind")
        not in {"optimization_candidate", "direct_quantity", "direct_response"}
    ):
        return False
    kind = value.get("kind")
    return (
        kind
        in {
            "direct_response_formation",
            "invalid_candidate_physical_parameter",
            "eliminated_block_solve_failure",
            "root_slope_unresolved",
            "numerical_resolution_unresolved",
            "unsupported_singular_capacitance_for_diagonal_root_v1",
            "port_realizability",
        }
        or kind == "compiler_invariant"
        and value.get("stage") not in {"optimization", "optimization_replay"}
        or _is_projection_only_optimization_failure(value)
    )


def _verify_failure_document(value: object, operation: object) -> None:
    if not isinstance(value, dict) or set(value) != {"category", "kind", "stage", "message", "evidence"}:
        raise _integrity("Failure envelope is open or malformed.")
    evidence = value.get("evidence")
    allowed = {
        "type", "operation", "context_kind", "plan_sha256", "request_sha256",
        "attempt_sha256", "workspace_instance_id", "component_path", "parameter",
        "coordinate_id", "port_id", "case_id", "candidate_ordinal", "artifact_id",
        "artifact_path", "expected_sha256", "actual_sha256", "backend_exit_code",
        "evidence_sha256", "optimization_context",
    }
    categories = {
        "plan_sealed": "state",
        "workspace_plan_replaced": "state",
        "workspace_commit_indeterminate": "state",
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
    context = evidence.get("optimization_context")
    if _optimization_failure_requires_context(value, operation) and context is None:
        raise _integrity("Optimization execution failure lacks its phase context.")
    if context is not None:
        if operation != "optimize_direct" or evidence.get("operation") != "optimize_direct":
            raise _integrity("Non-optimization failure carries optimization context.")
        _verify_optimization_context_shape(context)


def _verify_result_document(
    result: Mapping[str, object],
    request: Mapping[str, object],
    request_sha256: str,
    attempt_sha256: str,
    plan: Mapping[str, object],
) -> None:
    """Close one schema-version 2 single or batch result."""

    if result.get("result_kind") == "parameter_sweep":
        _verify_parameter_sweep_result(result, request, request_sha256, attempt_sha256)
        return
    common = {
        "schema", "schema_version", "result_kind", "request_sha256", "attempt_sha256",
        "parameters", "parameters_sha256", "ref_lineage",
    }
    if (
        result.get("schema") != "scnsim.result"
        or result.get("schema_version") != 2
        or result.get("request_sha256") != request_sha256
        or result.get("attempt_sha256") != attempt_sha256
        or not common <= set(result)
    ):
        raise _integrity("Result envelope does not bind its request and attempt.")
    parameters = result.get("parameters")
    _verify_parameter_set_document(parameters)
    from ._canonical import canonical_parameters_sha256

    if result.get("parameters_sha256") != canonical_parameters_sha256(parameters):
        raise _integrity("Result parameter identity is malformed.")
    source = request.get("parameter_source")
    if not isinstance(source, Mapping) or source.get("kind") != "point" or parameters != source.get("parameters"):
        raise _integrity("Single-point Result does not bind its requested point.")
    _verify_v1_lineage(result.get("ref_lineage"), plan)
    scientific_result = dict(result)
    for field in ("parameters", "parameters_sha256", "ref_lineage"):
        scientific_result.pop(field)
    scientific_result["schema_version"] = 1
    scientific_request = dict(request)
    scientific_request["schema_version"] = 1
    scientific_request["ref_lineage"] = result["ref_lineage"]
    scientific_request["parameters"] = parameters
    scientific_request.pop("view", None)
    scientific_request.pop("parameter_source", None)
    _verify_single_result_document(
        scientific_result, scientific_request, request_sha256, attempt_sha256, plan
    )


def _verify_parameter_sweep_result(
    result: Mapping[str, object],
    request: Mapping[str, object],
    request_sha256: str,
    attempt_sha256: str,
) -> None:
    expected = {
        "schema", "schema_version", "result_kind", "request_sha256", "attempt_sha256",
        "parameter_source_sha256", "point_count", "chunk_size", "manifest", "chunks",
    }
    source = request.get("parameter_source")
    from ._canonical import sha256_hex

    if (
        set(result) != expected
        or result.get("schema") != "scnsim.result"
        or result.get("schema_version") != 2
        or result.get("result_kind") != "parameter_sweep"
        or result.get("request_sha256") != request_sha256
        or result.get("attempt_sha256") != attempt_sha256
        or not isinstance(source, Mapping)
        or source.get("kind") not in {"grid", "points"}
        or result.get("parameter_source_sha256") != sha256_hex(source)
        or result.get("chunk_size") != 64
        or not isinstance(result.get("point_count"), int)
        or isinstance(result.get("point_count"), bool)
        or result["point_count"] < 1
        or not isinstance(result.get("manifest"), Mapping)
        or not isinstance(result.get("chunks"), list)
    ):
        raise _integrity("Parameter-sweep Result envelope is malformed.")
    expected_count = (
        math.prod(source["shape"])
        if source["kind"] == "grid"
        else len(source["points"])
    )
    if result["point_count"] != expected_count:
        raise _integrity("Parameter-sweep point count disagrees with its source.")
    manifest = result["manifest"]
    if (
        set(manifest) != {"id", "path", "sha256", "media_type", "byte_length"}
        or manifest.get("id") != "parameter_points"
        or manifest.get("path") != "artifacts/parameter_points.manifest.json"
        or manifest.get("media_type") != "application/json"
        or not isinstance(manifest.get("byte_length"), int)
        or isinstance(manifest.get("byte_length"), bool)
        or manifest["byte_length"] < 1
    ):
        raise _integrity("Parameter-sweep manifest link is malformed.")
    _valid_sha(manifest.get("sha256"))
    chunks = result["chunks"]
    expected_chunks = (expected_count + 63) // 64
    if len(chunks) != expected_chunks:
        raise _integrity("Parameter-sweep chunk count is malformed.")
    for index, chunk in enumerate(chunks):
        first = index * 64
        count = min(64, expected_count - first)
        if (
            not isinstance(chunk, Mapping)
            or set(chunk) != {"chunk_ordinal", "first_point", "point_count", "path", "sha256"}
            or chunk.get("chunk_ordinal") != index
            or chunk.get("first_point") != first
            or chunk.get("point_count") != count
            or chunk.get("path") != f"artifacts/parameter_points/chunks/{index:06d}.json"
        ):
            raise _integrity("Parameter-sweep chunk link is malformed.")
        _valid_sha(chunk.get("sha256"))


def _verify_single_result_document(
    result: Mapping[str, object],
    request: Mapping[str, object],
    request_sha256: str,
    attempt_sha256: str,
    plan: Mapping[str, object],
) -> None:
    """Verify the unchanged inner scientific result records."""

    spec = request.get("spec")
    kind = (
        "direct_response" if request.get("operation") == "solve_direct"
        else "hb_batch" if request.get("operation") == "solve_hb"
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
    if kind == "hb_batch":
        _verify_hb_batch_result(result, request, plan)
    elif kind == "direct_response":
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
        raise _integrity("Result operation is outside the supported runtime.")


def _verify_hb_batch_result(
    result: Mapping[str, object],
    request: Mapping[str, object],
    plan: Mapping[str, object],
) -> None:
    """Verify the case-local HB Result catalog before receipt promotion.

    HB artifacts are semantic catalog records on successful cases.  Their
    manifest hashes attest only to the byte trees; receipt links retain the
    case-local semantic key so repeated role names in separate cases cannot
    collapse into a global artifact namespace.
    """

    common = {"schema", "schema_version", "result_kind", "request_sha256", "attempt_sha256"}
    expected = common | {"lattice", "truncation", "topology_evidence", "cases"}
    if set(result) != expected:
        raise _integrity("HB batch Result envelope is open or incomplete.")
    spec = request.get("spec")
    if not isinstance(spec, dict):
        raise _integrity("HB batch Result has no solve Spec.")
    if result.get("truncation") != spec.get("truncation"):
        raise _integrity("HB Result truncation disagrees with its request.")
    lattice = result.get("lattice")
    lattice_fields = {
        "pump_axes", "operating_point_modes", "input_modes", "output_modes",
        "matrix_order", "tuple_frequency_collision_check_sha256",
    }
    if not isinstance(lattice, dict) or set(lattice) != lattice_fields or lattice.get("pump_axes") != spec.get("pump_axes") or lattice.get("matrix_order") != "port_major_mode_minor":
        raise _integrity("HB lattice evidence is malformed or disagrees with its request.")
    from ._canonical import float64_hex

    pump_rank = len(spec.get("pump_axes", []))
    pump_axes = spec.get("pump_axes")
    frequencies = spec.get("frequencies")
    if not isinstance(pump_axes, list) or not isinstance(frequencies, list):
        raise _integrity("HB request cannot reproduce its lattice frequencies.")
    pump_frequencies = [_f64_value(axis["frequency"]["si_value_f64"]) for axis in pump_axes]
    response_frequencies = [_f64_value(frequency["si_value_f64"]) for frequency in frequencies]
    vacuous_operating_lattice = _hb_operating_lattice_is_vacuous(spec)
    for field, is_response_lattice in (("operating_point_modes", False), ("input_modes", True), ("output_modes", True)):
        modes = lattice.get(field)
        if not isinstance(modes, list):
            raise _integrity("HB lattice is missing an ordered mode basis.", field=field)
        expected_modes = _hb_declared_modes_from_spec(spec, response=is_response_lattice)
        if field == "operating_point_modes" and vacuous_operating_lattice:
            if modes:
                raise _integrity("HB operating lattice is nonempty for a vacuous pinned JC basis.")
            continue
        if not modes:
            raise _integrity("HB lattice is missing an ordered mode basis.", field=field)
        seen: set[tuple[int, ...]] = set()
        for order, item in enumerate(modes):
            keys = {"mode", "signed_frequency", "order"} if not is_response_lattice else {"mode", "signed_frequency_grid", "order"}
            if not isinstance(item, dict) or set(item) != keys or not _valid_mode_tuple(item.get("mode"), pump_rank) or item.get("order") != order:
                raise _integrity("HB lattice mode row is malformed.", field=field)
            mode = tuple(item["mode"])
            if mode in seen:
                raise _integrity("HB lattice repeats a mode tuple.", field=field)
            seen.add(mode)
        if [item["mode"] for item in modes] != expected_modes:
            raise _integrity("HB lattice mode order disagrees with pinned JosephsonCircuits construction.", field=field)
        _verify_hb_lattice_injectivity(modes, response=is_response_lattice, field=field)
        for item in modes:
            mode = tuple(item["mode"])
            values = item.get("signed_frequency_grid") if is_response_lattice else [item.get("signed_frequency")]
            expected_values = (
                [frequency + sum((float(coefficient) * pump for coefficient, pump in zip(mode, pump_frequencies)), 0.0) for frequency in response_frequencies]
                if is_response_lattice else [sum((float(coefficient) * pump for coefficient, pump in zip(mode, pump_frequencies)), 0.0)]
            )
            if not isinstance(values, list) or len(values) != len(expected_values):
                raise _integrity("HB lattice frequency evidence is malformed.", field=field)
            for value, expected_value in zip(values, expected_values):
                _verify_quantity_role(value, complex_value=False, unit="hertz", dimensionality="inverse_time")
                if value["si_value_f64"] != float64_hex(expected_value):
                    raise _integrity("HB lattice signed frequency disagrees with its sealed axes.", field=field)
                if is_response_lattice and expected_value == 0.0:
                    raise _integrity("HB response lattice contains a zero-frequency sideband.", field=field)
    input_modes = lattice["input_modes"]
    if lattice.get("output_modes") != input_modes:
        raise _integrity("HB input and output response lattices disagree.")
    collision_entries = [
        {"mode": row["mode"], "frequency": value["si_value_f64"]}
        for row in input_modes
        for value in row["signed_frequency_grid"]
    ]
    expected_collision = _sha256(_canonical_bytes({
        "schema": "scnsim.hb_tuple_frequency_collision", "schema_version": 1,
        "entries": collision_entries,
    }))
    if lattice.get("tuple_frequency_collision_check_sha256") != expected_collision:
        raise _integrity("HB tuple-frequency collision evidence disagrees with the sealed lattice.")
    cases = result.get("cases")
    declared_cases = spec.get("cases")
    if not isinstance(cases, list) or not isinstance(declared_cases, list) or len(cases) != len(declared_cases):
        raise _integrity("HB Result case inventory disagrees with its declaration.")
    lineage = request.get("ref_lineage")
    if not isinstance(lineage, Mapping):
        raise _integrity("HB batch Result has no realized View lineage.")
    terminal, _ = _verify_v1_lineage(lineage, plan)
    _verify_hb_topology_evidence(result.get("topology_evidence"), spec, lineage)
    original = lineage.get("original") if isinstance(lineage, Mapping) else None
    native_ports = _identifiers(original.get("port_order"), field="HB original Port order", nonempty=False) if isinstance(original, Mapping) else []
    original_coordinates = _identifiers(
        original.get("coordinate_order"), field="HB original coordinate order"
    ) if isinstance(original, Mapping) else []
    connectivity = plan.get("connectivity")
    plan_ports = connectivity.get("ports") if isinstance(connectivity, Mapping) else None
    if not isinstance(plan_ports, list):
        raise _integrity("HB sealed Plan has no Port inventory.")
    expected_injection_sha256: dict[str, str] = {}
    for port in plan_ports:
        if (
            not isinstance(port, Mapping)
            or not isinstance(port.get("id"), str)
            or not isinstance(port.get("net"), str)
            or port["id"] in expected_injection_sha256
            or port["net"] not in original_coordinates
        ):
            raise _integrity("HB sealed Port cannot reproduce its compiler injection map.")
        incidence = [0.0] * len(original_coordinates)
        incidence[original_coordinates.index(port["net"])] = 1.0
        expected_injection_sha256[port["id"]] = _sha256(
            _canonical_bytes(
                {
                    "schema": "scnsim.hb_injection_map",
                    "schema_version": 1,
                    "port_id": port["id"],
                    "incidence_f64": [float64_hex(item) for item in incidence],
                }
            )
        )
    expected_probe = _expected_probe_load_state(lineage)
    native_probe = [{"port_id": port, "state": "raw"} for port in native_ports]
    operating_modes = [item["mode"] for item in lattice["operating_point_modes"]]
    for ordinal, (outcome, declared) in enumerate(zip(cases, declared_cases), 1):
        if not isinstance(outcome, dict) or not isinstance(declared, dict) or outcome.get("case_ordinal") != ordinal or outcome.get("case_id") != declared.get("id"):
            raise _integrity("HB Result cases are not declaration ordered.")
        _verify_hb_effective_sources(
            outcome.get("effective_sources"),
            pump_rank,
            spec=spec,
            declared_case=declared,
            expected_injection_sha256=expected_injection_sha256,
            operating_modes=operating_modes,
        )
        status = outcome.get("status")
        if status == "failure":
            if set(outcome) != {"case_ordinal", "case_id", "status", "effective_sources", "failure"}:
                raise _integrity("Failed HB outcome leaks success-only evidence.")
            failure = outcome.get("failure")
            if not isinstance(failure, dict) or set(failure) != {"kind", "stage", "message", "evidence_sha256"} or failure.get("kind") != "hb_case_failure" or failure.get("stage") not in {"operating_point", "linearization", "response_formation"} or not isinstance(failure.get("message"), str) or not failure["message"]:
                raise _integrity("HB case failure is malformed.")
            expected_failure_evidence = _sha256(
                _canonical_bytes(
                    {
                        "schema": "scnsim.hb_case_failure",
                        "schema_version": 1,
                        "case_ordinal": ordinal,
                        "case_id": declared["id"],
                        "stage": failure["stage"],
                        "message": failure["message"],
                        "effective_sources": outcome["effective_sources"],
                    }
                )
            )
            if failure.get("evidence_sha256") != expected_failure_evidence:
                raise _integrity("HB case failure evidence disagrees with its sealed outcome.")
            continue
        if status != "success":
            raise _integrity("HB case has an unknown terminal status.")
        success_fields = {
            "case_ordinal", "case_id", "status", "bias_state", "pump_state",
            "effective_sources", "operating_point_closure", "artifacts", "traces", "reconciliation",
            "backend_normalization_evidence_sha256", "state_node_map",
        }
        if set(outcome) != success_fields or outcome.get("bias_state") not in {"off", "on"} or outcome.get("pump_state") not in {"off", "on"}:
            raise _integrity("Successful HB outcome is open or malformed.")
        expected_normalization_evidence = _sha256(
            _canonical_bytes(
                {"normalization": "backend_photon_flux_to_scnsim_power_wave"}
            )
        )
        if outcome.get("backend_normalization_evidence_sha256") != expected_normalization_evidence:
            raise _integrity("HB backend normalization evidence is not reproducible.")
        _verify_hb_operating_point_closure(outcome.get("operating_point_closure"), operating_modes)
        _verify_hb_reconciliation(outcome.get("reconciliation"), lineage)
        _verify_hb_state_node_map(outcome.get("state_node_map"))
        _verify_hb_case_catalog(
            outcome.get("artifacts"), outcome.get("traces"), ordinal,
            spec, lattice, outcome["state_node_map"], terminal,
            expected_probe, native_ports, native_probe,
        )


def _verify_hb_effective_sources(
    value: object,
    pump_rank: int,
    *,
    spec: Mapping[str, object],
    declared_case: Mapping[str, object],
    expected_injection_sha256: Mapping[str, str],
    operating_modes: list[list[int]],
) -> None:
    drives = spec.get("drives")
    bindings = declared_case.get("currents")
    if not isinstance(value, list) or not isinstance(drives, list) or not isinstance(bindings, list) or len(value) != len(drives):
        raise _integrity("HB outcome has no effective-source evidence.")
    by_drive: dict[str, Mapping[str, object]] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping) or not isinstance(binding.get("drive_id"), str) or binding["drive_id"] in by_drive:
            raise _integrity("HB case current declaration is malformed.")
        by_drive[binding["drive_id"]] = binding
    from ._canonical import float64_hex

    for source, drive in zip(value, drives):
        if not isinstance(source, dict) or set(source) != {"drive_id", "mode", "coefficient", "generated_conjugate", "backend_binding", "injection_map_sha256"} or not isinstance(source.get("drive_id"), str) or _IDENTIFIER.fullmatch(source["drive_id"]) is None or not _valid_mode_tuple(source.get("mode"), pump_rank):
            raise _integrity("HB effective-source row is malformed.")
        if not isinstance(drive, Mapping) or source.get("drive_id") != drive.get("id") or source.get("mode") != drive.get("mode"):
            raise _integrity("HB effective sources are not in drive declaration order.")
        _verify_quantity_role(source.get("coefficient"), complex_value=True, unit="ampere", dimensionality="current")
        binding = by_drive.pop(source["drive_id"], None)
        coefficient = source["coefficient"]
        if binding is None:
            if _f64_value(coefficient["real_si_f64"]) != 0.0 or _f64_value(coefficient["imag_si_f64"]) != 0.0:
                raise _integrity("An omitted HB current did not materialize as exact zero.")
        elif coefficient != binding.get("coefficient"):
            raise _integrity("HB effective-source coefficient disagrees with its case declaration.")
        source_mode = list(source["mode"])
        inverse_mode = [-value for value in source_mode]
        is_dc = all(value == 0 for value in source_mode)
        expected_generated = (
            {"mode": source_mode, "coefficient": coefficient}
            if is_dc else {
                "mode": inverse_mode,
                "coefficient": {
                    "type": "complex_quantity_f64",
                    "real_si_f64": coefficient["real_si_f64"],
                    "imag_si_f64": float64_hex(-_f64_value(coefficient["imag_si_f64"])),
                    "si_unit": "ampere",
                    "dimensionality": "current",
                },
            }
        )
        if source.get("generated_conjugate") != expected_generated:
            raise _integrity("HB generated conjugate disagrees with its declared coefficient.")
        if is_dc:
            expected_representative = [0] if pump_rank == 0 else source_mode
            backend_coefficient = coefficient
        elif source_mode in operating_modes:
            expected_representative = source_mode
            backend_coefficient = {
                "type": "complex_quantity_f64",
                "real_si_f64": coefficient["real_si_f64"],
                "imag_si_f64": float64_hex(-_f64_value(coefficient["imag_si_f64"])),
                "si_unit": "ampere",
                "dimensionality": "current",
            }
        elif inverse_mode in operating_modes:
            expected_representative = inverse_mode
            backend_coefficient = coefficient
        else:
            raise _integrity("HB source mode and its generated conjugate are absent from the operating lattice.")
        try:
            representative_index = operating_modes.index(source_mode if pump_rank == 0 else expected_representative)
        except ValueError as error:
            raise _integrity("HB source representative is absent from the operating lattice.") from error
        expected_backend = {
            "representative_mode": expected_representative,
            "representative_index": representative_index,
            "coefficient": backend_coefficient,
            "coefficient_convention": "exp_plus_i_m_dot_omega_t_josephsoncircuits_source",
        }
        if source.get("backend_binding") != expected_backend:
            raise _integrity("HB backend source binding disagrees with the sealed case and lattice.")
        expected_injection = expected_injection_sha256.get(str(drive.get("port_id")))
        if expected_injection is None or source.get("injection_map_sha256") != expected_injection:
            raise _integrity("HB effective-source injection map disagrees with the sealed compiler basis.")
    if by_drive:
        raise _integrity("HB case current names a drive absent from effective sources.")


def _verify_hb_lattice_injectivity(
    modes: list[object], *, response: bool, field: str,
) -> None:
    """Reject a non-injective tuple/frequency channel basis by exact bits."""

    if response:
        grid_length: int | None = None
        for item in modes:
            grid = item.get("signed_frequency_grid") if isinstance(item, Mapping) else None
            if not isinstance(grid, list):
                raise _integrity("HB response lattice frequency grid is malformed.", field=field)
            if grid_length is None:
                grid_length = len(grid)
            elif len(grid) != grid_length:
                raise _integrity("HB response lattice frequency grids disagree in length.", field=field)
        if grid_length is None:
            raise _integrity("HB response lattice has no mode rows.", field=field)
        for frequency_ordinal in range(grid_length):
            seen_frequencies: set[str] = set()
            for item in modes:
                grid = item["signed_frequency_grid"]
                frequency = grid[frequency_ordinal]
                if not isinstance(frequency, Mapping) or not isinstance(frequency.get("si_value_f64"), str):
                    raise _integrity("HB response lattice frequency evidence is malformed.", field=field)
                bits = frequency["si_value_f64"]
                if bits in seen_frequencies:
                    raise _integrity("HB response lattice has a duplicate signed frequency at one declared grid ordinal.", field=field)
                seen_frequencies.add(bits)
        return
    seen_frequencies: set[str] = set()
    for item in modes:
        frequency = item.get("signed_frequency") if isinstance(item, Mapping) else None
        if not isinstance(frequency, Mapping) or not isinstance(frequency.get("si_value_f64"), str):
            raise _integrity("HB operating lattice frequency evidence is malformed.", field=field)
        bits = frequency["si_value_f64"]
        if bits in seen_frequencies:
            raise _integrity("HB operating lattice has a duplicate signed frequency.", field=field)
        seen_frequencies.add(bits)


def _verify_hb_topology_evidence(
    value: object,
    spec: Mapping[str, object],
    lineage: Mapping[str, object],
) -> None:
    """Bind HB's loaded nonlinear and selected response topologies to one View."""

    original = lineage.get("original")
    if not isinstance(original, Mapping):
        raise _integrity("HB topology evidence has no original compiler lineage.")
    intrinsic = _valid_sha(original.get("compiled_graph_sha256"))
    full_lineage = _valid_sha(lineage.get("lineage_sha256"))
    balance_lineage = _hb_lineage_prefix_sha(lineage, "load_or_ptc")
    expected = {
        "allow_driven_ptc": spec.get("allow_driven_ptc"),
        "intrinsic_compiled_graph_sha256": intrinsic,
        "nonlinear_balance": {
            "load_state": "loaded",
            "lineage_sha256": balance_lineage,
        },
        "response_linearization": {
            "load_state": "compensated" if lineage.get("ptc") is not None else "raw",
            "lineage_sha256": full_lineage,
        },
    }
    if value != expected:
        raise _integrity("HB topology evidence disagrees with its sealed View and driven-PTC authorization.")


def _verify_hb_operating_point_closure(value: object, operating_modes: list[list[int]]) -> None:
    """Verify the fixed HB residual disjunction or the exact vacuous exception."""

    if not operating_modes:
        if value != {"status": "not_applicable", "reason": "no_operating_point_lattice"}:
            raise _integrity("Vacuous HB operating lattice has the wrong closure evidence.")
        return
    if not isinstance(value, Mapping) or set(value) != {
        "status", "absolute_residual_f64", "relative_residual", "successful_disjunct",
    } or value.get("status") != "satisfied":
        raise _integrity("HB operating-point closure is malformed.")
    absolute = value.get("absolute_residual_f64")
    if not _finite_f64(absolute) or _f64_value(absolute) < 0.0:
        raise _integrity("HB operating-point absolute residual is malformed.")
    absolute_passes = _f64_value(absolute) <= 1.0e-8
    relative = value.get("relative_residual")
    relative_passes = False
    if isinstance(relative, Mapping) and set(relative) == {"status", "value_f64"} and relative.get("status") == "value":
        relative_value = relative.get("value_f64")
        if not _finite_f64(relative_value) or _f64_value(relative_value) < 0.0:
            raise _integrity("HB operating-point relative residual is malformed.")
        relative_passes = _f64_value(relative_value) < 1.0e-8
    elif not (
        isinstance(relative, Mapping)
        and relative == {"status": "not_applicable", "reason": "zero_state_norm"}
    ):
        raise _integrity("HB operating-point relative residual is malformed.")
    disjunct = value.get("successful_disjunct")
    expected_disjunct = (
        "both" if absolute_passes and relative_passes
        else "absolute" if absolute_passes
        else "relative" if relative_passes
        else None
    )
    if disjunct != expected_disjunct:
        raise _integrity("HB operating-point closure does not satisfy the fixed residual disjunction.")


def _verify_hb_reconciliation(value: object, lineage: Mapping[str, object]) -> None:
    fields = {"comparable", "reason", "last_comparable_ancestor", "normalization", "evidence_sha256"}
    if not isinstance(value, dict) or not fields.issubset(value) or not set(value).issubset(fields | {"residual_f64", "coordinate_projection"}) or not isinstance(value.get("comparable"), bool) or value.get("normalization") != "backend_photon_flux_to_scnsim_power_wave":
        raise _integrity("HB reconciliation evidence is malformed.")
    _valid_sha(value.get("last_comparable_ancestor")); _valid_sha(value.get("evidence_sha256"))
    comparable = value["comparable"]
    expected_reason = _hb_reconciliation_reason(lineage)
    expected_ancestor = _hb_lineage_prefix_sha(lineage, expected_reason)
    if value.get("last_comparable_ancestor") != expected_ancestor:
        raise _integrity("HB reconciliation ancestor does not bind the actual lineage prefix.")
    if comparable:
        residual = value.get("residual_f64")
        projection = _verify_hb_coordinate_projection(value.get("coordinate_projection"), lineage)
        if expected_reason is not None or value.get("reason") is not None or not _finite_f64(residual) or _f64_value(residual) < 0.0:
            raise _integrity("Comparable HB reconciliation lacks its normalized residual.")
        expected_evidence = _sha256(_canonical_bytes({
            "coordinate_producer_sha256": _hb_coordinate_producer_sha(lineage),
            "coordinate_projection": projection,
            "residual_f64": residual,
        }))
        if value.get("evidence_sha256") != expected_evidence:
            raise _integrity("HB comparable reconciliation evidence does not bind its coordinate producer and residual.")
    else:
        if "residual_f64" in value or "coordinate_projection" in value or value.get("reason") != expected_reason:
            raise _integrity("Incomparable HB reconciliation is malformed.")
        expected_evidence = _sha256(_canonical_bytes({
            "reason": expected_reason,
            "last_comparable_ancestor": expected_ancestor,
        }))
        if value.get("evidence_sha256") != expected_evidence:
            raise _integrity("HB incomparable reconciliation evidence does not bind its reason and lineage prefix.")


def _hb_reconciliation_reason(lineage: Mapping[str, object]) -> str | None:
    original = lineage.get("original")
    retain = lineage.get("retain")
    ptc = lineage.get("ptc")
    transforms = lineage.get("transforms")
    if not isinstance(original, Mapping) or not isinstance(transforms, list):
        raise _integrity("HB reconciliation cannot reconstruct its lineage.")
    ports = _identifiers(original.get("port_order"), field="HB reconciliation original Ports", nonempty=False)
    plain_port_subset = (
        isinstance(retain, Mapping)
        and ptc is None
        and not transforms
        and isinstance(retain.get("retained_coordinates"), list)
        and all(value in ports for value in retain["retained_coordinates"])
    )
    if ptc is not None:
        return "load_or_ptc"
    if transforms:
        return "reference_plane"
    if retain is not None and not plain_port_subset:
        return "channel_basis"
    return None


def _hb_lineage_prefix_sha(lineage: Mapping[str, object], reason: str | None) -> str:
    """Rebuild Julia's longest-comparable canonical lineage prefix exactly."""

    original = lineage.get("original")
    if not isinstance(original, Mapping):
        raise _integrity("HB reconciliation lineage has no original step.")
    if reason is None or reason in {"reference_matrix", "normalization", "signed_frequency_grid"}:
        return _valid_sha(lineage.get("lineage_sha256"))
    terminal = _identifiers(original.get("port_order"), field="HB reconciliation original Ports", nonempty=False)
    prefix: dict[str, object] = {
        "type": "network_view_lineage",
        "original": dict(original),
        "ptc": lineage.get("ptc") if reason in {"reference_plane", "channel_basis"} else None,
        "transforms": list(lineage.get("transforms", [])) if reason == "channel_basis" else [],
        "retain": None,
        "terminal_coordinates": terminal,
        "port_realizable": original.get("port_realizable"),
    }
    prefix["lineage_sha256"] = _sha256(_canonical_bytes(prefix))
    return str(prefix["lineage_sha256"])


def _hb_coordinate_producer_sha(lineage: Mapping[str, object]) -> str:
    """Return the sealed source identity of a comparable selected Port map."""

    retain = lineage.get("retain")
    if isinstance(retain, Mapping):
        q_matrix = retain.get("q_matrix")
        if not isinstance(q_matrix, Mapping):
            raise _integrity("HB comparable retain lineage has no Q-matrix evidence.")
        return _valid_sha(q_matrix.get("sha256"))
    original = lineage.get("original")
    if not isinstance(original, Mapping):
        raise _integrity("HB comparable lineage has no original mapping identity.")
    return _valid_sha(original.get("compiled_graph_sha256"))


def _verify_hb_coordinate_projection(value: object, lineage: Mapping[str, object]) -> dict[str, object]:
    """Bind the response-side Q row map to its selected and native bases."""

    if not isinstance(value, Mapping) or set(value) != {"shape", "values_f64"}:
        raise _integrity("HB comparable reconciliation has no closed coordinate projection.")
    shape = value.get("shape")
    bits = value.get("values_f64")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in shape)
        or not isinstance(bits, list)
        or len(bits) != shape[0] * shape[1]
        or any(not _finite_f64(item) for item in bits)
    ):
        raise _integrity("HB comparable coordinate projection shape or values are malformed.")
    from ._canonical import float64_hex

    if any(float64_hex(_f64_value(item)) != item for item in bits):
        raise _integrity("HB comparable coordinate projection has noncanonical Float64 values.")
    original = lineage.get("original")
    if not isinstance(original, Mapping):
        raise _integrity("HB comparable reconciliation has no original View basis.")
    ports = _identifiers(original.get("port_order"), field="HB comparable original Ports", nonempty=False)
    retain = lineage.get("retain")
    terminal = _identifiers(lineage.get("terminal_coordinates"), field="HB comparable terminal coordinates")
    expected_shape = [len(terminal), len(ports)]
    if shape != expected_shape:
        raise _integrity("HB comparable coordinate projection does not span its selected and original Port bases.")
    normalized = {"shape": list(shape), "values_f64": list(bits)}
    if retain is None:
        expected_identity = [float64_hex(1.0 if row == column else 0.0) for row in range(len(ports)) for column in range(len(ports))]
        if terminal != ports or normalized != {"shape": [len(ports), len(ports)], "values_f64": expected_identity}:
            raise _integrity("HB comparable original Port projection is not the canonical identity.")
        return normalized
    if not isinstance(retain, Mapping):
        raise _integrity("HB comparable retain projection has malformed lineage.")
    q_matrix = retain.get("q_matrix")
    if not isinstance(q_matrix, Mapping) or q_matrix.get("rows") != shape[0] or q_matrix.get("columns") != shape[1]:
        raise _integrity("HB comparable projection disagrees with retain Q-matrix dimensions.")
    values = [
        [_f64_value(bits[row * shape[1] + column]) for column in range(shape[1])]
        for row in range(shape[0])
    ]
    expected_q = _lineage_matrix("q", values, "port_realizable")
    if q_matrix.get("sha256") != expected_q["sha256"]:
        raise _integrity("HB comparable projection does not reproduce retain Q-matrix evidence.")
    return normalized


def _verify_hb_state_node_map(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise _integrity("HB success lacks its state-node map.")
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != {"state_index", "compiler_node_id", "source"} or row.get("state_index") != index or not isinstance(row.get("compiler_node_id"), str) or _IDENTIFIER.fullmatch(row["compiler_node_id"]) is None or not isinstance(row.get("source"), dict):
            raise _integrity("HB state-node map is malformed.")
        source = row["source"]
        kind = source.get("kind")
        valid = (
            kind == "plan_node"
            and set(source) == {"kind", "plan_node_id", "visibility"}
            and isinstance(source.get("plan_node_id"), str)
            and _IDENTIFIER.fullmatch(source["plan_node_id"]) is not None
            and source.get("visibility") in {"public", "port_promoted"}
        ) or (
            kind == "component_private"
            and set(source) == {"kind", "component_path", "private_node_id"}
            and isinstance(source.get("component_path"), list)
            and bool(source["component_path"])
            and all(isinstance(segment, str) and _IDENTIFIER.fullmatch(segment) is not None for segment in source["component_path"])
            and isinstance(source.get("private_node_id"), str)
            and _IDENTIFIER.fullmatch(source["private_node_id"]) is not None
        ) or (
            kind == "anonymous_internal"
            and set(source) == {"kind", "internal_node_id"}
            and isinstance(source.get("internal_node_id"), str)
            and re.fullmatch(r"internal-[0-9a-f]{64}", source["internal_node_id"]) is not None
        )
        if not valid:
            raise _integrity("HB state-node source mapping is malformed.")


def _verify_hb_case_catalog(
    artifacts: object,
    traces: object,
    ordinal: int,
    spec: Mapping[str, object],
    lattice: Mapping[str, object],
    state_node_map: list[dict[str, object]],
    terminal: list[str],
    expected_probe: list[dict[str, str]],
    native_ports: list[str],
    native_probe: list[dict[str, str]],
) -> None:
    roles = ("s", "y", "z", "backend_native_s", "backend_native_z", "states", "effective_source_vectors")
    if not isinstance(artifacts, dict) or set(artifacts) != set(roles) or not isinstance(traces, list):
        raise _integrity("HB case artifact catalog is incomplete.")
    input_modes = [item["mode"] for item in lattice["input_modes"]]
    output_modes = [item["mode"] for item in lattice["output_modes"]]
    operating_modes = [item["mode"] for item in lattice["operating_point_modes"]]
    compiler_nodes = [item["compiler_node_id"] for item in state_node_map]
    for role in roles:
        native = role in {"backend_native_s", "backend_native_z"}
        _verify_hb_catalog_artifact(
            artifacts[role], ordinal, role, spec,
            native_ports if native else terminal,
            native_probe if native else expected_probe,
            input_modes=input_modes,
            output_modes=output_modes,
            operating_modes=operating_modes,
            compiler_nodes=compiler_nodes,
        )
    declared_traces = spec.get("traces")
    if not isinstance(declared_traces, list) or len(traces) != len(declared_traces):
        raise _integrity("HB trace catalog disagrees with declaration.")
    for artifact, declaration in zip(traces, declared_traces):
        if not isinstance(declaration, dict) or not isinstance(artifact, dict) or artifact.get("id") != declaration.get("id"):
            raise _integrity("HB trace catalog is not declaration ordered.")
        _verify_hb_catalog_artifact(
            artifact, ordinal, str(declaration["id"]), spec, terminal,
            expected_probe, trace=True, input_modes=input_modes,
            output_modes=output_modes, operating_modes=operating_modes,
            compiler_nodes=compiler_nodes,
        )


def _verify_hb_catalog_artifact(
    artifact: object,
    ordinal: int,
    role: str,
    spec: Mapping[str, object],
    terminal: list[str],
    expected_probe: list[dict[str, str]],
    *,
    trace: bool = False,
    input_modes: list[list[int]],
    output_modes: list[list[int]],
    operating_modes: list[list[int]],
    compiler_nodes: list[str],
) -> None:
    if not isinstance(artifact, dict):
        raise _integrity("HB artifact catalog entry is malformed.", artifact_id=role)
    base = {"id", "path", "sha256", "media_type", "file_manifest", "dtype", "shape", "chunks", "complex_storage", "group_metadata", "datasets", "axes", "unit", "dimensionality", "chunk_policy"}
    matrix = not trace and role in {"s", "y", "z", "backend_native_s", "backend_native_z"}
    expected_fields = base | ({"coordinate_ids", "probe_load_state", "output_channels", "input_channels"} if matrix else set())
    if set(artifact) != expected_fields or artifact.get("id") != role or artifact.get("media_type") != "application/vnd+zarr-v2" or artifact.get("group_metadata") != {"zarr_format": 2}:
        raise _integrity("HB artifact catalog entry has the wrong semantic role.", artifact_id=role)
    prefix = f"artifacts/cases/{ordinal:06d}/"
    expected_path = f"{prefix}traces/{role}.zarr" if trace else f"{prefix}{role}.zarr"
    expected_manifest = expected_path.removesuffix(".zarr") + ".manifest.json"
    if artifact.get("path") != expected_path or artifact.get("file_manifest") != expected_manifest:
        raise _integrity("HB artifact path does not match its case ordinal.", artifact_id=role)
    _valid_sha(artifact.get("sha256"))
    shape = artifact.get("shape")
    chunks = artifact.get("chunks")
    allow_empty_leading = not trace and role in {"states", "effective_source_vectors"}
    if (
        not isinstance(shape, list)
        or not isinstance(chunks, list)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in chunks)
        or any(value == 0 and (not allow_empty_leading or index != 0) for index, value in enumerate(shape))
    ):
        raise _integrity("HB artifact has invalid shape or chunks.", artifact_id=role)
    if matrix:
        if len(shape) != 3 or len(chunks) != 3 or chunks != [min(shape[0], 1024), shape[1], shape[2]] or artifact.get("dtype") != "complex128" or artifact.get("complex_storage") != "paired_float64_real_imag" or artifact.get("chunk_policy") != "frequency_slab_full_matrix_v1":
            raise _integrity("HB matrix artifact storage is malformed.", artifact_id=role)
        units = {"s": ("dimensionless", "dimensionless"), "y": ("siemens", "conductance"), "z": ("ohm", "resistance"), "backend_native_s": ("dimensionless", "dimensionless"), "backend_native_z": ("ohm", "resistance")}
        if (artifact.get("unit"), artifact.get("dimensionality")) != units[role] or artifact.get("coordinate_ids") != terminal or artifact.get("probe_load_state") != expected_probe:
            raise _integrity("HB matrix artifact semantic metadata disagrees with the View.", artifact_id=role)
        channels = artifact.get("output_channels"), artifact.get("input_channels")
        expected_output_channels = [
            {"coordinate": coordinate, "mode": mode}
            for coordinate in terminal for mode in output_modes
        ]
        expected_input_channels = [
            {"coordinate": coordinate, "mode": mode}
            for coordinate in terminal for mode in input_modes
        ]
        if (
            not all(isinstance(channels_value, list) for channels_value in channels)
            or channels[0] != expected_output_channels
            or channels[1] != expected_input_channels
            or len(channels[0]) != shape[1]
            or len(channels[1]) != shape[2]
        ):
            raise _integrity("HB matrix channel catalog disagrees with its shape.", artifact_id=role)
        rank = len(spec.get("pump_axes", []))
        for channel_list in channels:
            seen: set[tuple[str, tuple[int, ...]]] = set()
            for channel in channel_list:
                if not isinstance(channel, dict) or set(channel) != {"coordinate", "mode"} or channel.get("coordinate") not in terminal or not _valid_mode_tuple(channel.get("mode"), rank):
                    raise _integrity("HB matrix channel label is malformed.", artifact_id=role)
                key = (str(channel["coordinate"]), tuple(channel["mode"]))
                if key in seen:
                    raise _integrity("HB matrix channel labels repeat.", artifact_id=role)
                seen.add(key)
        expected_axes = [
            {"id": "frequency", "kind": "frequency", "request_field": "spec.frequencies"},
            {"id": "output_channel", "kind": "output_channel", "values": channels[0]},
            {"id": "input_channel", "kind": "input_channel", "values": channels[1]},
        ]
        if artifact.get("axes") != expected_axes:
            raise _integrity("HB matrix axes disagree with its channel catalog.", artifact_id=role)
    else:
        is_state = role == "states"
        if trace:
            if len(shape) != 1 or len(chunks) != 1 or chunks != [min(shape[0], 1024)] or artifact.get("unit") != "dimensionless" or artifact.get("dimensionality") != "dimensionless" or artifact.get("chunk_policy") != "frequency_capped_1024_v1":
                raise _integrity("HB trace artifact storage is malformed.", artifact_id=role)
            expected_axes = [{"id": "frequency", "kind": "frequency", "request_field": "spec.frequencies"}]
        else:
            expected_chunks = [max(1, shape[0]), shape[1]] if len(shape) == 2 else None
            if len(shape) != 2 or len(chunks) != 2 or shape[1] < 1 or chunks != expected_chunks or artifact.get("unit") != ("weber" if is_state else "ampere") or artifact.get("dimensionality") != ("magnetic_flux" if is_state else "current") or artifact.get("chunk_policy") != "single_complete_array_v1":
                raise _integrity("HB state/source artifact storage is malformed.", artifact_id=role)
            axes = artifact.get("axes")
            if not isinstance(axes, list) or len(axes) != 2 or not all(isinstance(axis, dict) for axis in axes) or axes[0].get("kind") != "pump_mode" or axes[1].get("kind") != "node_coordinate":
                raise _integrity("HB state/source axes are malformed.", artifact_id=role)
            pump_modes, nodes = axes[0].get("values"), axes[1].get("values")
            if (
                not isinstance(pump_modes, list)
                or not isinstance(nodes, list)
                or len(pump_modes) != shape[0]
                or len(nodes) != shape[1]
                or any(not _valid_mode_tuple(mode, len(spec.get("pump_axes", []))) for mode in pump_modes)
                or any(not isinstance(node, str) or _IDENTIFIER.fullmatch(node) is None for node in nodes)
                or len({tuple(mode) for mode in pump_modes}) != len(pump_modes)
                or len(set(nodes)) != len(nodes)
                or pump_modes != operating_modes
                or nodes != compiler_nodes
            ):
                raise _integrity("HB state/source axis values disagree with its shape.", artifact_id=role)
            expected_axes = axes
        if artifact.get("dtype") != "complex128" or artifact.get("complex_storage") != "paired_float64_real_imag" or artifact.get("axes") != expected_axes:
            raise _integrity("HB non-matrix artifact has invalid representation.", artifact_id=role)
    _verify_zarr_datasets(artifact.get("datasets"), shape=shape, chunks=chunks, names=["real", "imag"])


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
    if not isinstance(value, dict) or set(value) != {"type", "bindings", "allow_extrapolation"} or value.get("type") != "parameter_set_v2":
        raise _integrity("ParameterSet envelope is open or malformed.")
    bindings = value.get("bindings")
    authorizations = value.get("allow_extrapolation")
    if not isinstance(bindings, list) or not isinstance(authorizations, list):
        raise _integrity("ParameterSet arrays are malformed.")
    keys: list[tuple[str, str]] = []
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {"parameter", "value"}:
            raise _integrity("ParameterSet binding is open or malformed.")
        reference = binding.get("parameter")
        keys.append(_parameter_key_integrity(reference))
        _verify_parameter_value(binding.get("value"))
    if keys != sorted(set(keys)):
        raise _integrity("ParameterSet bindings are not sorted and unique.")
    authorization_keys: list[tuple[str, str]] = []
    for reference in authorizations:
        authorization_keys.append(_parameter_key_integrity(reference))
    if authorization_keys != sorted(set(authorization_keys)) or any(key not in keys for key in authorization_keys):
        raise _integrity("ParameterSet authorizations are not sorted active references.")
    if require_empty_authorization and authorization_keys:
        raise _integrity("Optimization candidate ParameterSet inherited extrapolation authorization.")


def _verify_parameter_value(value: object) -> None:
    if isinstance(value, dict) and value.get("type") == "quantity_f64":
        _verify_quantity_any(value)
        return
    matrix_fields = {
        "resistance_per_length": ("ohm / meter", "resistance_per_length"),
        "inductance_per_length": ("henry / meter", "inductance_per_length"),
        "conductance_per_length": ("siemens / meter", "conductance_per_length"),
        "capacitance_per_length": ("farad / meter", "capacitance_per_length"),
    }
    required = {
        "type", "conductors", "reference_conductor", "orientation", "source",
        *matrix_fields, "extraction_frequency",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("type") != "rlgc":
        raise _integrity("ParameterSet value has an unsupported physical role.")
    conductors = value.get("conductors")
    if (
        not isinstance(conductors, list)
        or not conductors
        or any(not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in conductors)
        or len(set(conductors)) != len(conductors)
        or not isinstance(value.get("reference_conductor"), str)
        or _IDENTIFIER.fullmatch(value["reference_conductor"]) is None
        or value["reference_conductor"] in conductors
        or value.get("orientation") != "extractor_positive_z_is_head_to_tail"
        or not isinstance(value.get("source"), dict)
    ):
        raise _integrity("RLGC parameter basis or provenance is malformed.")
    size = len(conductors)
    for field, (unit, dimensionality) in matrix_fields.items():
        matrix = value[field]
        if (
            not isinstance(matrix, dict)
            or set(matrix) != {"type", "shape", "values_f64", "si_unit", "dimensionality"}
            or matrix.get("type") != "quantity_matrix_f64"
            or matrix.get("shape") != [size, size]
            or matrix.get("si_unit") != unit
            or matrix.get("dimensionality") != dimensionality
            or not isinstance(matrix.get("values_f64"), list)
            or len(matrix["values_f64"]) != size * size
            or any(not _finite_f64(item) for item in matrix["values_f64"])
        ):
            raise _integrity("RLGC parameter matrix is malformed.")
    frequency = value.get("extraction_frequency")
    if frequency is not None:
        _verify_quantity_role(
            frequency, complex_value=False, unit="hertz", dimensionality="inverse_time"
        )


def _verify_extrapolation_evidence(
    value: object,
    *,
    allowed_sources: set[str],
    required_rows: list[dict[str, object]] | None = None,
) -> None:
    """Validate one evidence row per explicitly out-of-support fan-out edge."""

    if not isinstance(value, list):
        raise _integrity("Extrapolation evidence is not an array.")
    keys: list[tuple[tuple[str, str], tuple[tuple[str, ...], str]]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"parameter", "consumer_target", "support", "input_value", "side", "distance", "authorization_source"}:
            raise _integrity("Extrapolation evidence row is malformed.")
        parameter = _parameter_key_integrity(row.get("parameter"))
        target_record = row.get("consumer_target")
        if not isinstance(target_record, Mapping) or set(target_record) != {"path", "field"}:
            raise _integrity("Extrapolation consumer target is malformed.")
        path, field = target_record.get("path"), target_record.get("field")
        if not isinstance(path, list) or not path or any(not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in path) or not isinstance(field, str) or _IDENTIFIER.fullmatch(field) is None:
            raise _integrity("Extrapolation consumer target is malformed.")
        target = (tuple(path), field)
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
    """Derive affine support crossings from normalized physical-field bindings."""

    if authorization_source not in {"parameter_set", "optimization_spec"}:
        raise _integrity("Extrapolation evidence authority is unknown.")
    _verify_parameter_set_document(parameters)
    assert isinstance(parameters, Mapping)
    values = {
        _parameter_key_integrity(binding["parameter"]): binding["value"]
        for binding in parameters["bindings"]
    }
    raw_authorizations = (
        parameters["allow_extrapolation"]
        if authorization_source == "parameter_set"
        else optimization_authorizations
    )
    if not isinstance(raw_authorizations, list):
        raise _integrity("Extrapolation authorization collection is malformed.")
    authorized = {_parameter_key_integrity(item) for item in raw_authorizations}
    leaves = plan.get("physical_leaves")
    if not isinstance(leaves, list):
        raise _integrity("Plan physical-field inventory is malformed.")
    rows: list[dict[str, object]] = []
    from ._canonical import float64_hex

    for leaf in leaves:
        if not isinstance(leaf, Mapping) or not isinstance(leaf.get("path"), list) or not isinstance(leaf.get("fields"), list):
            raise _integrity("Plan physical leaf is malformed.")
        path = list(leaf["path"])
        for field in leaf["fields"]:
            binding = field.get("binding") if isinstance(field, Mapping) else None
            if not isinstance(binding, Mapping) or binding.get("kind") != "affine":
                continue
            if set(binding) != {"kind", "input", "slope", "intercept", "support"}:
                raise _integrity("Affine physical-field binding is malformed.")
            parameter = _parameter_key_integrity(binding["input"])
            input_value = values.get(parameter)
            if not isinstance(input_value, Mapping) or input_value.get("type") != "quantity_f64":
                raise _integrity("Affine input has no scalar resolved parameter value.")
            support = binding["support"]
            if not isinstance(support, list) or len(support) != 2:
                raise _integrity("Affine support interval is malformed.")
            _verify_quantity_compatible(support[0], support[1])
            _verify_quantity_compatible(support[0], input_value)
            lower = _f64_value(support[0]["si_value_f64"])
            upper = _f64_value(support[1]["si_value_f64"])
            selected = _f64_value(input_value["si_value_f64"])
            if lower >= upper:
                raise _integrity("Affine support interval is not ordered.")
            if lower <= selected <= upper:
                continue
            authority = authorization_source if parameter in authorized else "none"
            if require_authorized and authority == "none":
                raise _integrity("Successful request has unauthorized affine extrapolation.")
            side, distance = (
                ("lower", lower - selected)
                if selected < lower
                else ("upper", selected - upper)
            )
            distance_record = dict(input_value)
            distance_record["si_value_f64"] = float64_hex(distance)
            rows.append({
                "parameter": dict(binding["input"]),
                "consumer_target": {"path": path, "field": field["id"]},
                "support": [dict(support[0]), dict(support[1])],
                "input_value": dict(input_value),
                "side": side,
                "distance": distance_record,
                "authorization_source": authority,
            })
    rows.sort(key=lambda row: (
        _parameter_key_integrity(row["parameter"]),
        (tuple(row["consumer_target"]["path"]), row["consumer_target"]["field"]),
    ))
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


def _compare_artifacts(left: object, right: object, *, operation: object) -> None:
    if not isinstance(left, list) or not isinstance(right, list):
        raise _integrity("Outcome and receipt require artifact inventories.")
    normalized: list[list[tuple[object, ...]]] = []
    for inventory in (left, right):
        entries: list[tuple[object, ...]] = []
        identities: set[tuple[object, ...]] = set()
        paths: set[str] = set()
        for entry in inventory:
            if not isinstance(entry, dict):
                raise _integrity("Artifact inventory entry is malformed.")
            if operation == "solve_hb" and set(entry) != {
                "id", "path", "sha256", "media_type", "byte_length"
            }:
                if set(entry) != {"case_id", "id", "path", "sha256"} or not isinstance(entry.get("case_id"), str) or _IDENTIFIER.fullmatch(entry["case_id"]) is None or not isinstance(entry.get("id"), str) or _IDENTIFIER.fullmatch(entry["id"]) is None or not isinstance(entry.get("path"), str):
                    raise _integrity("HB artifact reference is malformed.")
                identity = (entry["case_id"], entry["id"], entry["path"])
                if identity in identities or entry["path"] in paths:
                    raise _integrity("HB artifact references repeat an identity or path.")
                identities.add(identity); paths.add(entry["path"])
                entries.append((*identity, _valid_sha(entry.get("sha256"))))
            elif set(entry) == {"id", "path", "sha256", "media_type", "byte_length"}:
                if (
                    not isinstance(entry.get("id"), str)
                    or not entry["id"]
                    or not isinstance(entry.get("path"), str)
                    or not entry["path"]
                    or not isinstance(entry.get("media_type"), str)
                    or not entry["media_type"]
                    or not isinstance(entry.get("byte_length"), int)
                    or isinstance(entry.get("byte_length"), bool)
                    or entry["byte_length"] < 1
                ):
                    raise _integrity("Artifact inventory entry is malformed.")
                identity = (entry["id"], entry["path"])
                if identity in identities or entry["path"] in paths:
                    raise _integrity("Artifact inventory contains a duplicate identity or path.")
                identities.add(identity)
                paths.add(entry["path"])
                entries.append((
                    *identity,
                    _valid_sha(entry.get("sha256")),
                    entry["media_type"],
                    entry["byte_length"],
                ))
            else:
                if set(entry) != {"id", "sha256"} or not isinstance(entry.get("id"), str) or not entry["id"]:
                    raise _integrity("Artifact inventory entry is malformed.")
                identity = (entry["id"],)
                if identity in identities:
                    raise _integrity("Artifact inventory contains a duplicate ID.", artifact_id=entry["id"])
                identities.add(identity)
                entries.append((*identity, _valid_sha(entry.get("sha256"))))
        normalized.append(entries)
    if normalized[0] != normalized[1]:
        raise _integrity("Outcome and receipt artifact inventories disagree.")


def _verify_hb_artifact_inventory(directory: Path, result: Mapping[str, object], receipt: Mapping[str, object]) -> None:
    """Cross-check HB's case-local semantic catalog against receipt bytes."""

    cases = result.get("cases")
    links = receipt.get("artifacts")
    if not isinstance(cases, list) or not isinstance(links, list):
        raise _integrity("HB Result or receipt lacks its artifact inventory.")
    expected: list[dict[str, str]] = []
    seen_identity: set[tuple[str, str, str]] = set()
    seen_paths: set[str] = set()
    roles = ("s", "y", "z", "backend_native_s", "backend_native_z", "states", "effective_source_vectors")
    for outcome in cases:
        if not isinstance(outcome, dict) or outcome.get("status") == "failure":
            continue
        if outcome.get("status") != "success" or not isinstance(outcome.get("case_id"), str) or not isinstance(outcome.get("artifacts"), dict) or not isinstance(outcome.get("traces"), list):
            raise _integrity("HB success catalog is malformed.")
        catalog = outcome["artifacts"]
        artifacts = [catalog[role] for role in roles]
        artifacts.extend(outcome["traces"])
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise _integrity("HB artifact catalog entry is malformed.")
            artifact_id, path, digest = artifact.get("id"), artifact.get("path"), artifact.get("sha256")
            if not isinstance(artifact_id, str) or _IDENTIFIER.fullmatch(artifact_id) is None or not isinstance(path, str):
                raise _integrity("HB artifact catalog has an invalid semantic identity.")
            # A trace may deliberately reuse a fixed role ID (for example
            # ``s``); the canonical HB reference is case + local ID + path.
            identity = (outcome["case_id"], artifact_id, path)
            if identity in seen_identity or path in seen_paths:
                raise _integrity("HB artifact catalog repeats a case-local identity or path.")
            seen_identity.add(identity); seen_paths.add(path)
            _valid_sha(digest)
            expected.append({"case_id": outcome["case_id"], "id": artifact_id, "path": path, "sha256": digest})
            manifest_path = artifact.get("file_manifest")
            if not isinstance(manifest_path, str):
                raise _integrity("HB artifact catalog has no manifest path.", artifact_id=artifact_id)
            artifact_path = _inside(directory, path)
            manifest = _inside(directory, manifest_path)
            if not artifact_path.is_dir() or artifact_path.is_symlink() or not manifest.is_file() or manifest.is_symlink():
                raise _integrity("HB artifact path is missing or unsafe.", artifact_id=artifact_id)
            manifest_bytes = manifest.read_bytes()
            if _sha256(manifest_bytes) != digest:
                raise _integrity("HB artifact manifest hash disagrees with its catalog.", artifact_id=artifact_id)
            manifest_doc = _decode_bytes(manifest_bytes, "artifact manifest")
            if manifest_doc.get("schema") != "scnsim.artifact_manifest" or manifest_doc.get("artifact_id") != artifact_id or manifest_doc.get("artifact_path") != path:
                raise _integrity("HB artifact manifest identity disagrees with its catalog.", artifact_id=artifact_id)
            _verify_manifest_tree(artifact_path, manifest_doc)
    _compare_artifacts(expected, links, operation="solve_hb")
    artifact_root = directory / "artifacts"
    if not expected:
        if artifact_root.exists():
            raise _integrity("All-failed HB batch must not retain an artifact directory.")
        return
    if artifact_root.is_symlink() or not artifact_root.is_dir() or (artifact_root / "cases").is_symlink() or not (artifact_root / "cases").is_dir():
        raise _integrity("HB artifact root is missing or unsafe.")
    if any(child.is_symlink() or not child.is_dir() for child in (artifact_root / "cases").iterdir()):
        raise _integrity("HB case artifact root contains an unsafe entry.")
    actual_ordinals = {
        child.name for child in (artifact_root / "cases").iterdir()
        if child.is_dir() and not child.is_symlink()
    }
    expected_ordinals = {
        path.split("/")[2] for path in seen_paths
    }
    if actual_ordinals != expected_ordinals:
        raise _integrity("HB case artifact directories disagree with successful outcomes.")
    for ordinal in expected_ordinals:
        case_root = artifact_root / "cases" / ordinal
        expected_case_entries: set[str] = set()
        expected_trace_entries: set[str] = set()
        for path in seen_paths:
            parts = path.split("/")
            if parts[2] != ordinal:
                continue
            if len(parts) == 4:
                stem = parts[3].removesuffix(".zarr")
                expected_case_entries.update({parts[3], f"{stem}.manifest.json"})
            else:
                stem = parts[4].removesuffix(".zarr")
                expected_case_entries.add("traces")
                expected_trace_entries.update({parts[4], f"{stem}.manifest.json"})
        entries = {child.name: child for child in case_root.iterdir()}
        if set(entries) != expected_case_entries:
            raise _integrity("HB case directory contains undeclared entries.", case_ordinal=ordinal)
        for name, child in entries.items():
            if child.is_symlink() or (name == "traces" and not child.is_dir()) or (name != "traces" and (name.endswith(".zarr") != child.is_dir() or name.endswith(".manifest.json") != child.is_file())):
                raise _integrity("HB case directory contains an unsafe entry.", case_ordinal=ordinal)
        if expected_trace_entries:
            trace_root = case_root / "traces"
            trace_entries = {child.name: child for child in trace_root.iterdir()}
            if set(trace_entries) != expected_trace_entries or any(child.is_symlink() or (name.endswith(".zarr") != child.is_dir() or name.endswith(".manifest.json") != child.is_file()) for name, child in trace_entries.items()):
                raise _integrity("HB trace directory contains undeclared or unsafe entries.", case_ordinal=ordinal)


def _verify_artifact_inventory(directory: Path, result: Mapping[str, object], receipt: Mapping[str, object]) -> None:
    if result.get("result_kind") == "parameter_sweep":
        _verify_parameter_sweep_artifacts(directory, result, receipt)
        return
    if result.get("result_kind") == "hb_batch":
        _verify_hb_artifact_inventory(directory, result, receipt)
        return
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


def _verify_parameter_sweep_artifacts(
    directory: Path,
    result: Mapping[str, object],
    receipt: Mapping[str, object],
) -> None:
    from ._canonical import canonical_parameters_sha256

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
            common = {"ordinal", "source_index", "parameters", "parameters_sha256", "status"}
            if (
                not isinstance(point, Mapping)
                or point.get("ordinal") != point_ordinal
                or point.get("source_index") != expected_source_index
                or point.get("parameters") != expected_parameters
                or point.get("parameters_sha256") != canonical_parameters_sha256(expected_parameters)
                or point.get("status") not in {"success", "failure"}
            ):
                raise _integrity("Parameter-sweep point identity is malformed.")
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
                point_result["attempt_sha256"] = result["attempt_sha256"]
                point_result["parameters"] = point["parameters"]
                point_result["parameters_sha256"] = point["parameters_sha256"]
                point_result["ref_lineage"] = point["ref_lineage"]
                _verify_result_document(
                    point_result,
                    point_request,
                    result["request_sha256"],
                    result["attempt_sha256"],
                    plan,
                )
            point_ordinal += 1
    if point_ordinal != result["point_count"]:
        raise _integrity("Parameter-sweep chunks do not cover every point.")


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
            or ledger.get("schema_version") != 3
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
        or ledger.get("algorithm_id") != "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v4"
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
            objectives=objectives,
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


def _verify_selector_lineage(
    selector: Mapping[str, object],
    lineage: object,
    plan: Mapping[str, object],
) -> None:
    declaration = selector.get("view")
    declared_terminal, declared_port_realizable = _verify_view_declaration(declaration, plan)
    terminal, port_realizable = _verify_v1_lineage(lineage, plan)
    if terminal != declared_terminal or port_realizable != declared_port_realizable:
        raise _integrity("Optimization term lineage disagrees with its declared View.")
    if not isinstance(declaration, Mapping) or not isinstance(lineage, Mapping):
        raise _integrity("Optimization term View evidence is malformed.")
    declared_ptc = declaration.get("ptc")
    actual_ptc = lineage.get("ptc")
    if (None if actual_ptc is None else {"selected_ports": actual_ptc.get("selected_ports")}) != declared_ptc:
        raise _integrity("Optimization term PTC lineage disagrees with its declaration.")
    declared_transforms = declaration.get("transforms")
    actual_transforms = lineage.get("transforms")
    if not isinstance(declared_transforms, list) or not isinstance(actual_transforms, list) or len(actual_transforms) != len(declared_transforms):
        raise _integrity("Optimization term transform lineage count is inconsistent.")
    for declared, actual in zip(declared_transforms, actual_transforms):
        if (
            not isinstance(declared, Mapping)
            or not isinstance(actual, Mapping)
            or actual.get("input_coordinates") != declared.get("input_coordinates")
            or [actual.get("common_id"), actual.get("differential_id")] != declared.get("output_coordinates")
        ):
            raise _integrity("Optimization term transform lineage disagrees with its declaration.")
    declared_retain = declaration.get("retain")
    actual_retain = lineage.get("retain")
    if (None if actual_retain is None else {"retained_coordinates": actual_retain.get("retained_coordinates")}) != declared_retain:
        raise _integrity("Optimization term retain lineage disagrees with its declaration.")


def _selector_terms(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Mapping):
        raise _integrity("Optimization objective quantity is malformed.")
    if value.get("type") == "quantity_sum":
        terms = value.get("terms")
        if not isinstance(terms, list) or not terms or any(not isinstance(term, Mapping) for term in terms):
            raise _integrity("Optimization QuantitySum terms are malformed.")
        return list(terms)
    return [value]


def _convert_selector_value(value: float, source_unit: object, target_unit: object) -> float:
    if source_unit == target_unit:
        return value
    if source_unit == "radian / second" and target_unit == "hertz":
        return value / (2.0 * math.pi)
    if source_unit == "hertz" and target_unit == "radian / second":
        return value * (2.0 * math.pi)
    raise _integrity("Optimization term unit conversion is unsupported.")


def _optimization_failure_context(failure: object) -> Mapping[str, object]:
    if not isinstance(failure, Mapping) or not isinstance(failure.get("evidence"), Mapping):
        raise _integrity("Optimization failure lacks evidence.")
    context = failure["evidence"].get("optimization_context")
    return _verify_optimization_context_shape(context)


def _verify_candidate_failure_context(
    failure: object,
    *,
    objectives: list[object],
    candidate: Mapping[str, object],
    phase: str,
    owner: Mapping[str, object],
    affected: list[Mapping[str, object]],
    dependency: Mapping[str, object] | None = None,
) -> None:
    context = _optimization_failure_context(failure)
    expected_candidate = {
        "evaluation_ordinal": candidate["evaluation_ordinal"],
        "origin": candidate["origin"],
        "generation": candidate["generation"],
        "population_column": candidate["population_column"],
    }
    catalog = _optimization_leaf_catalog(objectives)
    known = [locator for locator, _ in catalog]
    if any(locator not in known for locator in affected):
        raise _integrity("Optimization failure names a leaf absent from its request.")
    expected = {
        "schema": "scnsim.optimization_failure_context",
        "schema_version": 1,
        "phase": phase,
        "candidate": expected_candidate,
        "owner": dict(owner),
        "affected_leaves": [dict(item) for item in affected],
    }
    if dependency is not None:
        expected["dependency"] = dict(dependency)
    if context != expected:
        raise _integrity("Optimization failure context disagrees with evaluated request order and dependencies.")


def _optimization_root_selectors(selector: Mapping[str, object]) -> list[Mapping[str, object]]:
    kind = selector.get("type")
    if kind in {"diagonal_root_projection", "hybridized_pole_projection", "transfer_zero_projection"}:
        return [selector]
    if kind == "residue_coupling_projection":
        spec = selector.get("spec")
        view = selector.get("view")
        if not isinstance(spec, Mapping) or not isinstance(view, Mapping):
            raise _integrity("Residue selector root dependencies are malformed.")
        roots: list[Mapping[str, object]] = []
        for name in ("branch_a", "branch_b"):
            branch = spec.get(name)
            if not isinstance(branch, Mapping):
                raise _integrity("Residue selector branch is malformed.")
            branch_type = branch.get("type")
            selector_type = (
                "residue_diagonal_root_projection" if branch_type == "diagonal_root"
                else "hybridized_pole_projection" if branch_type == "hybridized_pole"
                else None
            )
            if selector_type is None:
                raise _integrity("Residue selector branch type is unsupported.")
            roots.append({"type": selector_type, "spec": branch, "projection": "frequency", "view": view})
        return roots
    return []


def _optimization_quantity_failure_consumers(
    catalog: list[tuple[dict[str, object], Mapping[str, object]]],
    start: int,
    selector: Mapping[str, object],
    dependency: Mapping[str, object],
) -> list[dict[str, object]]:
    """Derive later public consumers of the selector or its failed private root."""

    selector_dependency = _optimization_dependency(selector)
    root_dependencies = [
        _optimization_dependency(root)
        for root in _optimization_root_selectors(selector)
    ]
    if dependency == selector_dependency:
        return [
            locator for locator, item in catalog[start:]
            if _optimization_dependency(item) == dependency
        ]
    if dependency in root_dependencies:
        return [
            locator for locator, item in catalog[start:]
            if any(
                _optimization_dependency(root) == dependency
                for root in _optimization_root_selectors(item)
            )
        ]
    raise _integrity("Quantity failure dependency disagrees with its failed selector.")


def _verify_terminal_optimization_failure(
    failure: object,
    spec: object,
    *,
    completed_generations: int = 0,
) -> None:
    if not isinstance(spec, Mapping) or spec.get("type") != "optimization":
        raise _integrity("Optimization terminal failure lacks its request spec.")
    context = _optimization_failure_context(failure)
    candidate = context.get("candidate")
    if not isinstance(candidate, Mapping):
        raise _integrity("Terminal optimization failure has no candidate position.")
    if (
        not isinstance(completed_generations, int)
        or isinstance(completed_generations, bool)
        or completed_generations < 0
    ):
        raise _integrity("Terminal optimization ledger prefix is malformed.")
    if candidate.get("origin") == "population":
        optimizer = spec.get("optimizer")
        population = optimizer.get("resolved_population_size") if isinstance(optimizer, Mapping) else None
        complete = optimizer.get("complete_generations") if isinstance(optimizer, Mapping) else None
        generation = candidate.get("generation")
        column = candidate.get("population_column")
        if (
            not isinstance(population, int) or isinstance(population, bool) or population < 2
            or not isinstance(complete, int) or isinstance(complete, bool) or complete < 1
            or not isinstance(generation, int) or isinstance(generation, bool)
            or not isinstance(column, int) or isinstance(column, bool)
            or generation < 1 or generation > complete
            or column < 1 or column > population
            or generation != completed_generations + 1
            or candidate.get("evaluation_ordinal") != 1 + (generation - 1) * population + (column - 1)
        ):
            raise _integrity("Terminal optimization population position is inconsistent with its request.")
    elif completed_generations != 0:
        raise _integrity("Baseline failure follows completed population ledgers.")
    objectives = spec.get("objectives")
    catalog = _optimization_leaf_catalog(objectives)
    all_leaves = [locator for locator, _ in catalog]
    phase = context.get("phase")
    dependency = context.get("dependency")
    owner = context.get("owner")
    affected = context.get("affected_leaves")
    if phase in {"candidate_prepare", "candidate_compile"}:
        expected = ({"kind": "candidate"}, all_leaves, None)
    elif phase == "view_realization":
        matches = [(locator, selector) for locator, selector in catalog if _optimization_dependency(selector, kind="view") == dependency]
        if not matches:
            raise _integrity("Terminal View failure dependency is absent from the request.")
        expected = ({"kind": "dependency"}, [locator for locator, _ in matches], dependency)
    elif phase == "baseline_root_anchor":
        if candidate.get("origin") != "baseline":
            raise _integrity("Population failure claims baseline root-anchor ownership.")
        matches: list[Mapping[str, object]] = []
        for locator, selector in catalog:
            if any(_optimization_dependency(root) == dependency for root in _optimization_root_selectors(selector)):
                matches.append(locator)
        if not matches:
            raise _integrity("Baseline root-anchor dependency is absent from the request.")
        expected = ({"kind": "dependency"}, matches, dependency)
    elif phase == "quantity_evaluation":
        leaf = owner.get("leaf") if isinstance(owner, Mapping) else None
        if leaf not in all_leaves:
            raise _integrity("Baseline quantity failure owner is absent from the request.")
        index = all_leaves.index(leaf)
        selector = catalog[index][1]
        if dependency is None:
            if not _is_projection_only_optimization_failure(failure):
                raise _integrity("Shared quantity failure omits its dependency identity.")
            expected_affected = [leaf]
        else:
            if _is_projection_only_optimization_failure(failure):
                raise _integrity("Projection-only failure claims a shared dependency.")
            expected_affected = _optimization_quantity_failure_consumers(
                catalog, index, selector, dependency,
            )
        expected = ({"kind": "leaf", "leaf": leaf}, expected_affected, dependency)
    elif phase == "objective_aggregation":
        objective_id = owner.get("objective_id") if isinstance(owner, Mapping) else None
        if objective_id not in [item.get("id") for item in objectives if isinstance(item, Mapping)]:
            raise _integrity("Baseline objective aggregation owner is absent from the request.")
        expected = ({"kind": "objective", "objective_id": objective_id}, [], None)
    elif phase == "total_aggregation":
        expected = ({"kind": "candidate"}, [], None)
    else:
        raise _integrity("Terminal optimization failure uses an invalid baseline phase.")
    if owner != expected[0] or affected != expected[1] or dependency != expected[2]:
        raise _integrity("Terminal optimization failure context disagrees with its sealed request.")


def _verify_objective_component(
    component: object,
    objective: object,
    plan: Mapping[str, object],
    *,
    expected_status: str,
) -> None:
    if not isinstance(component, dict) or not isinstance(objective, dict):
        raise _integrity("Optimization objective component is malformed.")
    quantity = objective.get("quantity")
    expected_terms = _selector_terms(quantity)
    terms = component.get("terms")
    common = {"objective_id", "quantity", "status", "terms"}
    expected_fields = (
        common | {"value", "normalized_residual_f64", "weighted_cost_f64"}
        if expected_status == "success"
        else common | {"failure"}
    )
    if (
        set(component) != expected_fields
        or component.get("objective_id") != objective.get("id")
        or component.get("quantity") != quantity
        or component.get("status") != expected_status
        or not isinstance(terms, list)
        or len(terms) != len(expected_terms)
    ):
        raise _integrity("Optimization objective component is open or inconsistent.")
    term_statuses: list[str] = []
    total = 0.0
    target_unit = objective.get("target", {}).get("si_unit") if isinstance(objective.get("target"), Mapping) else None
    for ordinal, (term, selector) in enumerate(zip(terms, expected_terms), 1):
        if not isinstance(term, dict) or term.get("term_ordinal") != ordinal or term.get("selector") != selector:
            raise _integrity("Optimization term evidence is out of order or names another selector.")
        status = term.get("status")
        term_statuses.append(str(status))
        if status == "success":
            if set(term) != {"term_ordinal", "selector", "status", "ref_lineage", "value"}:
                raise _integrity("Successful optimization term is open or malformed.")
            _verify_selector_lineage(selector, term.get("ref_lineage"), plan)
            role = _verify_selector(selector, plan)
            _verify_quantity_role(term.get("value"), complex_value=False, unit=role[0], dimensionality=role[1])
            total += _convert_selector_value(_f64_value(term["value"]["si_value_f64"]), role[0], target_unit)
        elif status == "failure":
            if set(term) != {"term_ordinal", "selector", "status", "ref_lineage", "failure"}:
                raise _integrity("Failed optimization term is open or malformed.")
            _verify_selector_lineage(selector, term.get("ref_lineage"), plan)
            _verify_failure_document(term.get("failure"), "optimize_direct")
        elif status == "not_evaluated":
            if set(term) != {"term_ordinal", "selector", "status", "failure"}:
                raise _integrity("Unevaluated optimization term is open or malformed.")
            _verify_failure_document(term.get("failure"), "optimize_direct")
        else:
            raise _integrity("Optimization term status is unknown.")
    if expected_status == "success":
        if any(status != "success" for status in term_statuses):
            raise _integrity("Successful objective contains an unevaluated term.")
        _verify_quantity_role(
            component.get("value"), complex_value=False,
            unit=objective["target"]["si_unit"], dimensionality=objective["target"]["dimensionality"],
        )
        if struct.pack(">d", total).hex() != component["value"]["si_value_f64"]:
            raise _integrity("Optimization objective value does not equal its ordered terms.")
        residual = (total - _f64_value(objective["target"]["si_value_f64"])) / _f64_value(objective["resolved_scale"]["si_value_f64"])
        weighted = _f64_value(objective["weight_f64"]) * residual * residual
        if (
            not _finite_f64(component.get("normalized_residual_f64"))
            or not _finite_f64(component.get("weighted_cost_f64"))
            or struct.pack(">d", residual).hex() != component["normalized_residual_f64"]
            or struct.pack(">d", weighted).hex() != component["weighted_cost_f64"]
            or weighted < 0.0
        ):
            raise _integrity("Optimization objective normalization does not reproduce.")
    else:
        _verify_failure_document(component.get("failure"), "optimize_direct")
        if any(
            term.get("status") in {"failure", "not_evaluated"}
            and term.get("failure") != component.get("failure")
            for term in terms
            if isinstance(term, Mapping)
        ):
            raise _integrity("Optimization term failure disagrees with its objective failure.")
        if expected_status == "failure":
            failure_indexes = [
                index for index, status in enumerate(term_statuses)
                if status == "failure"
            ]
            ordinary_failure = (
                len(failure_indexes) == 1
                and term_statuses[:failure_indexes[0]] == ["success"] * failure_indexes[0]
                and term_statuses[failure_indexes[0] + 1:]
                == ["not_evaluated"] * (len(term_statuses) - failure_indexes[0] - 1)
            )
            if not ordinary_failure and any(status != "success" for status in term_statuses):
                raise _integrity("Failed objective term status order is inconsistent.")
        elif any(status != "not_evaluated" for status in term_statuses):
            raise _integrity("Unevaluated objective contains evaluated terms.")


def _verify_candidate_outcome(
    value: object,
    *,
    variables: int,
    objectives: list[object],
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
            or len(components) != len(objectives)
        ):
            raise _integrity("Successful optimization candidate is malformed.")
        total = 0.0
        for objective, component in zip(objectives, components):
            _verify_objective_component(component, objective, plan, expected_status="success")
            total += _f64_value(component["weighted_cost_f64"])
        if struct.pack(">d", total).hex() != outcome.get("cost_f64"):
            raise _integrity("Optimization candidate cost does not equal its ordered components.")
    elif outcome.get("status") == "failure":
        components = outcome.get("objective_components")
        if (
            set(outcome) != {"status", "penalty", "failure", "objective_components"}
            or outcome.get("penalty") != "positive_infinity"
            or not isinstance(components, list)
            or len(components) != len(objectives)
        ):
            raise _integrity("Failed optimization candidate is malformed.")
        _verify_failure_document(outcome.get("failure"), "optimize_direct")
        if outcome["failure"].get("kind") not in {
            "invalid_candidate_physical_parameter", "eliminated_block_solve_failure",
            "root_slope_unresolved", "numerical_resolution_unresolved",
        }:
            raise _integrity("Optimization candidate uses a request-level failure kind.")
        context = _optimization_failure_context(outcome["failure"])
        phase = context["phase"]
        statuses: list[str] = []
        for objective, component in zip(objectives, components):
            status = component.get("status") if isinstance(component, Mapping) else None
            statuses.append(str(status))
            if status == "success":
                _verify_objective_component(component, objective, plan, expected_status="success")
            elif status == "failure":
                _verify_objective_component(component, objective, plan, expected_status="failure")
                if component.get("failure") != outcome["failure"]:
                    raise _integrity("Failed objective does not bind the candidate failure.")
            elif status == "not_evaluated":
                _verify_objective_component(component, objective, plan, expected_status="not_evaluated")
                if component.get("failure") != outcome["failure"]:
                    raise _integrity("Unevaluated objective does not bind the candidate failure.")
            else:
                raise _integrity("Failed candidate objective status order is inconsistent.")
        catalog = _optimization_leaf_catalog(objectives)
        all_leaves = [locator for locator, _ in catalog]
        if phase in {"candidate_prepare", "candidate_compile"}:
            if any(status != "not_evaluated" for status in statuses):
                raise _integrity("Candidate preparation failure contains evaluated objectives.")
            _verify_candidate_failure_context(
                outcome["failure"], objectives=objectives, candidate=value,
                phase=phase, owner={"kind": "candidate"}, affected=all_leaves,
            )
        elif phase == "view_realization":
            if any(status != "not_evaluated" for status in statuses):
                raise _integrity("View-realization failure contains evaluated objectives.")
            dependency = context.get("dependency")
            matches = [
                (locator, selector) for locator, selector in catalog
                if _optimization_dependency(selector, kind="view") == dependency
            ]
            if not matches:
                raise _integrity("View-realization failure dependency is absent from the request.")
            _verify_candidate_failure_context(
                outcome["failure"], objectives=objectives, candidate=value,
                phase=phase, owner={"kind": "dependency"},
                affected=[locator for locator, _ in matches], dependency=dependency,
            )
        elif phase == "quantity_evaluation":
            failed_objectives = [index for index, status in enumerate(statuses) if status == "failure"]
            if len(failed_objectives) != 1:
                raise _integrity("Quantity failure has no unique failed objective.")
            failed_objective = failed_objectives[0]
            if statuses[:failed_objective] != ["success"] * failed_objective or statuses[failed_objective + 1:] != ["not_evaluated"] * (len(statuses) - failed_objective - 1):
                raise _integrity("Quantity failure objective status order is inconsistent.")
            failed_terms = [
                ({"objective_id": objective["id"], "term_ordinal": term["term_ordinal"]}, selector)
                for objective, component in zip(objectives, components)
                if isinstance(objective, Mapping) and isinstance(component, Mapping)
                for term, selector in zip(component.get("terms", []), _selector_terms(objective.get("quantity")))
                if isinstance(term, Mapping) and term.get("status") == "failure"
            ]
            if len(failed_terms) != 1:
                raise _integrity("Quantity failure does not identify exactly one failed leaf.")
            locator, selector = failed_terms[0]
            start = all_leaves.index(locator)
            dependency = context.get("dependency")
            if dependency is None:
                if not _is_projection_only_optimization_failure(outcome["failure"]):
                    raise _integrity("Shared quantity failure omits its dependency identity.")
                affected = [locator]
            else:
                if _is_projection_only_optimization_failure(outcome["failure"]):
                    raise _integrity("Projection-only failure claims a shared dependency.")
                affected = _optimization_quantity_failure_consumers(
                    catalog, start, selector, dependency,
                )
            _verify_candidate_failure_context(
                outcome["failure"], objectives=objectives, candidate=value,
                phase=phase, owner={"kind": "leaf", "leaf": locator},
                affected=affected, dependency=dependency,
            )
        elif phase == "objective_aggregation":
            failed_indexes = [index for index, status in enumerate(statuses) if status == "failure"]
            if len(failed_indexes) != 1:
                raise _integrity("Objective aggregation failure has no unique owner.")
            failed_index = failed_indexes[0]
            if statuses[:failed_index] != ["success"] * failed_index or statuses[failed_index + 1:] != ["not_evaluated"] * (len(statuses) - failed_index - 1):
                raise _integrity("Objective aggregation failure status order is inconsistent.")
            objective = objectives[failed_index]
            if not isinstance(objective, Mapping):
                raise _integrity("Objective aggregation owner is malformed.")
            terms = components[failed_index].get("terms") if isinstance(components[failed_index], Mapping) else None
            if (
                not isinstance(terms, list)
                or any(
                    not isinstance(term, Mapping) or term.get("status") != "success"
                    for term in terms
                )
            ):
                raise _integrity("Objective aggregation failure does not preserve successful terms.")
            _verify_candidate_failure_context(
                outcome["failure"], objectives=objectives, candidate=value,
                phase=phase, owner={"kind": "objective", "objective_id": objective["id"]},
                affected=[],
            )
        elif phase == "total_aggregation":
            if any(status != "success" for status in statuses):
                raise _integrity("Total aggregation failure does not preserve successful objectives.")
            _verify_candidate_failure_context(
                outcome["failure"], objectives=objectives, candidate=value,
                phase=phase, owner={"kind": "candidate"}, affected=[],
            )
        else:
            raise _integrity("Population candidate uses a baseline-only or unknown failure phase.")
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
    baseline = result.get("baseline")
    _verify_candidate_outcome(
        baseline,
        variables=len(variables),
        objectives=objectives,
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
        comparable_outcome = _optimization_outcome_without_candidate_position(record.get("outcome"))
        if cached is True and (parameters not in seen or seen[parameters] != comparable_outcome):
            raise _integrity("Optimization cache hit does not match its earlier candidate.")
        if cached is False and parameters in seen:
            raise _integrity("Repeated optimization parameters were not marked as a cache hit.")
        seen.setdefault(parameters, comparable_outcome)
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


def _optimization_outcome_without_candidate_position(value: object) -> object:
    if isinstance(value, Mapping):
        if value.get("schema") == "scnsim.optimization_failure_context":
            return {key: ("<candidate-position>" if key == "candidate" else _optimization_outcome_without_candidate_position(item)) for key, item in value.items()}
        return {key: _optimization_outcome_without_candidate_position(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_optimization_outcome_without_candidate_position(item) for item in value]
    return value


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
