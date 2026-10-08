"""Workspace binding, locking, attempt transactions, and receipt lifecycle."""

from __future__ import annotations

import os
import re
import shutil
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace as _dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

if sys.platform in {"linux", "darwin"}:
    import fcntl

from ..canonical import canonical_json_bytes as _canonical_bytes, sha256_hex as _sha256
from ..errors import (
    EvidenceIntegrityError,
    ResultUnavailableError,
    UnsupportedRuntimePlatformError,
    WorkspaceCommitIndeterminateError,
    WorkspacePlanReplacedError,
    WorkspaceVersioningDowngradeForbidden,
)
from .documents import (
    plan_workspace_document,
    replaceable_workspace_document,
    versioned_workspace_document,
)
from .primitives import _relative_path
from .records import (
    AttemptAllocation,
    BaselineCheckpoint,
    PointCheckpoint,
    VerifiedSuccess,
    _IncomingCheckpointEvidenceError,
)
from .storage import (
    _ATTEMPT,
    _CHECKPOINT_STAGING,
    _LEAF_STAGING,
    _STAGING,
    _WorkspacePublishIndeterminate,
    _atomic_write,
    _decode_bytes,
    _fsync_directory,
    _fsync_tree,
    _load_canonical,
    _path_entry_exists,
    _publish_workspace_state,
    _remove_leaf,
)
from .validation.common import (
    _SHA256,
    _UUID4,
    _integrity,
    _required_extrapolation_rows,
    _valid_sha,
    _valid_uuid,
    _verify_extrapolation_evidence,
)
from .validation.inventory import _compare_artifacts, _verify_artifact_inventory
from .validation.optimization import (
    _verify_attempt_checkpoint_consumption,
    _verify_baseline_checkpoint_directory,
    _verify_baseline_checkpoint_document,
    _verify_generation_artifacts,
    _verify_terminal_optimization_failure,
)
from .validation.requests import _verify_failure_document, _verify_request_document
from .validation.results import _verify_result_document
from .validation.sweeps import (
    _parameter_source_points,
    _verify_point_checkpoint_record,
    _verify_point_checkpoints,
)

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

def _require_platform() -> None:
    if sys.platform not in {"linux", "darwin"}:
        raise UnsupportedRuntimePlatformError(
            "SCNSim workspace mutation is supported only on Linux and macOS.",
            stage="workspace",
            evidence={"platform": sys.platform},
        )

def _new_uuid(*, excluding: str | None = None) -> str:
    value = str(uuid.uuid4())
    while value == excluding:
        value = str(uuid.uuid4())
    return value

@dataclass(frozen=True)
class WorkspaceBinding:
    """One Run's concrete, Plan-bound leaf beneath a stable workspace root."""

    root: Path
    leaf: Path
    plan_sha256: str
    workspace_instance_id: str
    _expected_julia_threads: int = field(default=1, repr=False, compare=False)
    _expected_blas_threads: int = field(default=1, repr=False, compare=False)

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
        if request["parameter_source"]["kind"] in {"grid", "points"}:
            _atomic_write(directory / "point-checkpoint-anchor.json", _canonical_bytes({
                "schema": "scnsim.point_checkpoint_anchor", "schema_version": 1,
                "request_sha256": request_sha256}))
            point_root = directory / "point-checkpoints"
            point_root.mkdir()
            _atomic_write(point_root / "index.json", _canonical_bytes({
                "schema": "scnsim.point_checkpoint_index", "schema_version": 1,
                "request_sha256": request_sha256, "entries": []}))
            _fsync_directory(point_root)
        _fsync_directory(directory)
        return directory

    def baseline_checkpoint(self, request_sha256: str) -> BaselineCheckpoint | None:
        """Verify and return one request-owned baseline checkpoint, if present."""

        request_directory = self.leaf / "requests" / _valid_sha(request_sha256)
        checkpoint_directory = request_directory / "baseline-checkpoint"
        if not _path_entry_exists(checkpoint_directory):
            return None
        request = _load_canonical(request_directory / "request.json")
        plan = _load_canonical(self.leaf / "plan.json")
        return _verify_baseline_checkpoint_directory(
            checkpoint_directory, request_sha256=request_sha256,
            request=request, plan=plan,
            expected_julia_threads=self._expected_julia_threads,
            expected_blas_threads=self._expected_blas_threads,
        )

    def point_checkpoints(self, request_sha256: str) -> tuple[PointCheckpoint, ...]:
        return cast(
            tuple[PointCheckpoint, ...],
            self._verified_point_checkpoints(request_sha256),
        )

    def _verified_point_checkpoints(
        self, request_sha256: str, *, include_payloads: bool = False,
    ) -> tuple[PointCheckpoint | tuple[PointCheckpoint, Mapping[str, object] | None], ...]:
        request_directory = self.leaf / "requests" / _valid_sha(request_sha256)
        return _verify_point_checkpoints(request_directory,
            _load_canonical(request_directory / "request.json"),
            _load_canonical(self.leaf / "plan.json"),
            include_payloads=include_payloads,
        )

    def _point_checkpoints_with_payloads(
        self, request_sha256: str,
    ) -> tuple[tuple[PointCheckpoint, Mapping[str, object] | None], ...]:
        return cast(
            tuple[tuple[PointCheckpoint, Mapping[str, object] | None], ...],
            self._verified_point_checkpoints(request_sha256, include_payloads=True),
        )

    def publish_point_checkpoint(self, request_sha256: str, attempt_sha256: str,
            staging: Path, ready: Mapping[str, object]) -> PointCheckpoint:
        request_directory = self.leaf / "requests" / _valid_sha(request_sha256)
        request = _load_canonical(request_directory / "request.json")
        plan = _load_canonical(self.leaf / "plan.json")
        previous = _verify_point_checkpoints(request_directory, request, plan)
        ordinal = len(previous)
        if ready.get("ordinal") != ordinal:
            raise _integrity("Point checkpoint publication is not in request order.")
        ready_path = staging / "point-ready.json"
        if ready_path.is_symlink() or not ready_path.is_file():
            raise _integrity("Point checkpoint record is absent from child staging.")
        raw = ready_path.read_bytes()
        if len(raw) != ready.get("byte_length") or _sha256(raw) != ready.get("record_sha256"):
            raise _integrity("Point checkpoint ready frame does not bind its bytes.")
        record = _decode_bytes(raw, "point checkpoint record")
        point_root = staging / "artifacts" / "parameter_points" / "points" / f"{ordinal:06d}"
        attempt_path = staging / "attempt.json"
        if attempt_path.is_symlink() or not attempt_path.is_file() or _sha256(attempt_path.read_bytes()) != attempt_sha256:
            raise _integrity("Point checkpoint source attempt is absent or changed.")
        source_attempt = attempt_path.read_bytes()
        _verify_point_checkpoint_record(record, point_root, request, plan, ordinal, attempt_sha256)
        root = request_directory / "point-checkpoints"
        if not root.exists():
            root.mkdir()
            _atomic_write(root / "index.json", _canonical_bytes({
                "schema": "scnsim.point_checkpoint_index", "schema_version": 1,
                "request_sha256": request_sha256, "entries": []}))
            _fsync_directory(request_directory)
        stage = root / f".staging-{uuid.uuid4()}"
        stage.mkdir()
        try:
            _atomic_write(stage / "record.json", raw)
            _atomic_write(stage / "source-attempt.json", source_attempt)
            shutil.copytree(point_root, stage / "point")
            seal = {"schema": "scnsim.point_checkpoint_seal", "schema_version": 1,
                "request_sha256": request_sha256, "ordinal": ordinal,
                "record_sha256": _sha256(raw), "source_attempt_sha256": attempt_sha256,
                "published_at_utc": _utc_now()}
            _atomic_write(stage / "seal.json", _canonical_bytes(seal))
            _fsync_tree(stage)
            final = root / f"{ordinal:06d}"
            if _path_entry_exists(final):
                raise _integrity("Point checkpoint publication target already exists.")
            os.replace(stage, final)
            _fsync_directory(root)
            entries = [{"ordinal": index, "seal_sha256": item.seal_sha256}
                for index, item in enumerate(previous)]
            entries.append({"ordinal": ordinal, "seal_sha256": _sha256(_canonical_bytes(seal))})
            _atomic_write(root / "index.json", _canonical_bytes({
                "schema": "scnsim.point_checkpoint_index", "schema_version": 1,
                "request_sha256": request_sha256, "entries": entries}))
            return _verify_point_checkpoints(request_directory, request, plan)[-1]
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def publish_baseline_checkpoint(
        self,
        request_sha256: str,
        attempt_sha256: str,
        producer_path: Path,
        *,
        expected_sha256: str,
        expected_byte_length: int,
    ) -> BaselineCheckpoint:
        """Own, validate, seal, and atomically publish exact producer bytes."""

        request_sha256 = _valid_sha(request_sha256)
        attempt_sha256 = _valid_sha(attempt_sha256)
        request_directory = self.leaf / "requests" / request_sha256
        request = _load_canonical(request_directory / "request.json")
        plan = _load_canonical(self.leaf / "plan.json")
        if request.get("operation") != "optimize_direct":
            raise _integrity("Only optimization may publish a baseline checkpoint.")
        try:
            if (
                producer_path.is_symlink() or not producer_path.is_file()
                or producer_path.parent.is_symlink()
            ):
                raise _integrity("Producer checkpoint is not a regular staged file.")
            # One read creates the bytes that are both validated and published.
            checkpoint_bytes = producer_path.read_bytes()
            if len(checkpoint_bytes) != expected_byte_length or _sha256(checkpoint_bytes) != expected_sha256:
                raise _integrity("Producer checkpoint bytes disagree with the ready frame.")
            checkpoint = _decode_bytes(checkpoint_bytes, "baseline checkpoint")
            _verify_baseline_checkpoint_document(
                checkpoint, request_sha256=request_sha256, request=request, plan=plan,
            )
            attempt_path = producer_path.parent / "attempt.json"
            if attempt_path.is_symlink() or not attempt_path.is_file():
                raise _integrity("Baseline checkpoint source attempt is not a regular file.")
            source_attempt_bytes = attempt_path.read_bytes()
            if _sha256(source_attempt_bytes) != attempt_sha256:
                raise _integrity("Baseline checkpoint source attempt identity is invalid.")
            source_attempt = _decode_bytes(source_attempt_bytes, "source attempt")
            if source_attempt.get("request_sha256") != request_sha256 or source_attempt.get("schema_version") != 2:
                raise _integrity("Baseline checkpoint source attempt is not optimization attempt v2.")
        except (EvidenceIntegrityError, OSError) as error:
            raise _IncomingCheckpointEvidenceError(
                "Incoming baseline checkpoint evidence is invalid.",
                stage="optimization_checkpoint",
                evidence={"source": "child_staging"},
            ) from error

        final = request_directory / "baseline-checkpoint"
        if _path_entry_exists(final):
            existing = _verify_baseline_checkpoint_directory(
                final, request_sha256=request_sha256, request=request, plan=plan,
                expected_julia_threads=self._expected_julia_threads,
                expected_blas_threads=self._expected_blas_threads,
            )
            if existing.checkpoint_sha256 != expected_sha256:
                raise _integrity("A different baseline checkpoint is already published.")
            producer_path.unlink()
            _fsync_directory(producer_path.parent)
            return existing

        staging = request_directory / f".staging-baseline-checkpoint-{uuid.uuid4()}"
        staging.mkdir()
        try:
            _atomic_write(staging / "checkpoint.json", checkpoint_bytes)
            _atomic_write(staging / "source-attempt.json", source_attempt_bytes)
            seal = {
                "schema": "scnsim.optimization_checkpoint_seal",
                "schema_version": 1,
                "request_sha256": request_sha256,
                "checkpoint_sha256": expected_sha256,
                "checkpoint_byte_length": len(checkpoint_bytes),
                "source_attempt_sha256": attempt_sha256,
                "source_attempt_byte_length": len(source_attempt_bytes),
                "published_at_utc": _utc_now(),
            }
            _atomic_write(staging / "seal.json", _canonical_bytes(seal))
            _fsync_tree(staging)
            verified = _verify_baseline_checkpoint_directory(
                staging, request_sha256=request_sha256, request=request, plan=plan,
                expected_julia_threads=self._expected_julia_threads,
                expected_blas_threads=self._expected_blas_threads,
            )
            if _path_entry_exists(final):
                raise _integrity(
                    "Baseline checkpoint publication target appeared during publication."
                )
            os.replace(staging, final)
            _fsync_directory(request_directory)
            published = _verify_baseline_checkpoint_directory(
                final, request_sha256=request_sha256, request=request, plan=plan,
                expected_julia_threads=self._expected_julia_threads,
                expected_blas_threads=self._expected_blas_threads,
            )
            producer_path.unlink()
            _fsync_directory(producer_path.parent)
            return published
        finally:
            if staging.exists():
                shutil.rmtree(staging)

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
        request = _load_canonical(
            self.leaf / "requests" / allocation.request_sha256 / "request.json"
        )
        expected = {
            "schema": "scnsim.attempt",
            "schema_version": 2 if request.get("operation") == "optimize_direct" else 1,
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
                checkpoint = self.baseline_checkpoint(request_sha256)
                point_checkpoints = (self.point_checkpoints(request_sha256)
                    if request["parameter_source"]["kind"] in {"grid", "points"} else ())
                point_counts = None
                if request["parameter_source"]["kind"] in {"grid", "points"}:
                    total = len(list(_parameter_source_points(request["parameter_source"])))
                    point_counts = {"published_points": len(point_checkpoints),
                        "successful_points": sum(item.record["metadata"]["status"] == "success" for item in point_checkpoints),
                        "failed_points": sum(item.record["metadata"]["status"] == "failure" for item in point_checkpoints),
                        "remaining_points": total - len(point_checkpoints)}
                attempts = request_directory / "attempts"
                if attempts.is_symlink() or (attempts.exists() and not attempts.is_dir()):
                    raise _integrity("Inventory request attempts path is unsafe.", request_sha256=request_sha256)
                if not attempts.exists():
                    if checkpoint is None and not point_checkpoints:
                        raise _integrity("Inventory request has no final attempts.", request_sha256=request_sha256)
                    rows.append({
                        "request_sha256": request_sha256,
                        "operation": request["operation"],
                        "status": "partial_points" if point_checkpoints else "partial_baseline",
                        "attempts": [],
                        **({"baseline_checkpoint_sha256": checkpoint.checkpoint_sha256,
                        "baseline_checkpoint_seal_sha256": checkpoint.seal_sha256} if checkpoint else {}),
                        **(point_counts or {}),
                    })
                    continue
                finals = self._final_attempt_directories(attempts)
                if not finals:
                    if checkpoint is None and not point_checkpoints:
                        raise _integrity("Inventory request has no final attempts.", request_sha256=request_sha256)
                    rows.append({
                        "request_sha256": request_sha256,
                        "operation": request["operation"],
                        "status": "partial_points" if point_checkpoints else "partial_baseline",
                        "attempts": [],
                        **({"baseline_checkpoint_sha256": checkpoint.checkpoint_sha256,
                        "baseline_checkpoint_seal_sha256": checkpoint.seal_sha256} if checkpoint else {}),
                        **(point_counts or {}),
                    })
                    continue
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
                row: dict[str, object] = {
                    "request_sha256": request_sha256,
                    "operation": request["operation"],
                    "status": status,
                    "attempts": [final.name for final in finals],
                }
                if checkpoint is not None:
                    row.update({
                        "baseline_checkpoint_sha256": checkpoint.checkpoint_sha256,
                        "baseline_checkpoint_seal_sha256": checkpoint.seal_sha256,
                    })
                if point_counts is not None:
                    row.update(point_counts)
                rows.append(row)
        root_state = _load_canonical(self.root / "workspace.json")
        _assert_root_envelope(root_state)
        maintenance = root_state.get("maintenance")
        return {
            "schema": "scnsim.inventory",
            "schema_version": 3,
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
                expected_julia_threads=self._expected_julia_threads,
                expected_blas_threads=self._expected_blas_threads,
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
            point_root = request / "point-checkpoints"
            if point_root.exists():
                if point_root.is_symlink() or not point_root.is_dir():
                    raise _integrity("Workspace contains unsafe point checkpoint directory.")
                for child in point_root.iterdir():
                    if child.name.startswith(".staging-"):
                        if child.is_symlink() or not child.is_dir() or _UUID4.fullmatch(child.name[len(".staging-"):]) is None:
                            raise _integrity("Workspace contains malformed point checkpoint staging.")
                        shutil.rmtree(child)
                        _fsync_directory(point_root)
            for child in request.iterdir():
                if not child.name.startswith(".staging-baseline-checkpoint-"):
                    continue
                if child.is_symlink() or not child.is_dir() or _CHECKPOINT_STAGING.fullmatch(child.name) is None:
                    raise _integrity("Workspace contains malformed checkpoint staging evidence.", path=str(child))
                shutil.rmtree(child)
                _fsync_directory(request)
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
        expected_attempt_version = 2 if request_document.get("operation") == "optimize_direct" else 1
        if (
            attempt.get("schema") != "scnsim.attempt"
            or attempt.get("schema_version") != expected_attempt_version
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
        checkpoint_sha = attempt.get("baseline_checkpoint_sha256")
        checkpoint_seal_sha = attempt.get("baseline_checkpoint_seal_sha256")
        if checkpoint_sha is not None or checkpoint_seal_sha is not None:
            attempt_fields.update({"baseline_checkpoint_sha256", "baseline_checkpoint_seal_sha256"})
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
        checkpoint: BaselineCheckpoint | None = None
        if checkpoint_sha is not None or checkpoint_seal_sha is not None:
            checkpoint = self.baseline_checkpoint(request_sha256)
            if (
                checkpoint is None
                or _valid_sha(checkpoint_sha) != checkpoint.checkpoint_sha256
                or _valid_sha(checkpoint_seal_sha) != checkpoint.seal_sha256
            ):
                raise _integrity("Attempt baseline checkpoint reference is absent or corrupt.")
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
                    expected_julia_threads=self._expected_julia_threads,
                    expected_blas_threads=self._expected_blas_threads,
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
            if not (set(receipt) == receipt_fields | {"point_checkpoint_count"} and
                    request_document.get("parameter_source", {}).get("kind") in {"grid", "points"}):
                raise _integrity("Receipt envelope is open or has outcome-incompatible fields.", attempt=str(directory))
        if "point_checkpoint_count" in receipt:
            count = receipt["point_checkpoint_count"]
            checkpoints = self.point_checkpoints(request_sha256)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0 or count > len(checkpoints):
                raise _integrity("Sweep receipt references missing point checkpoints.")
        if request_document.get("operation") == "optimize_direct":
            if checkpoint is None:
                checkpoint = self.baseline_checkpoint(request_sha256)
            _verify_attempt_checkpoint_consumption(
                attempt,
                receipt,
                attempt_sha256=attempt_sha256,
                checkpoint=checkpoint,
            )
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
            _verify_result_document(
                result,
                request_document,
                request_sha256,
                attempt_sha256,
                plan_document,
                optimization_checkpoint=checkpoint,
            )
            _verify_artifact_inventory(directory, result, receipt)
            if result.get("result_kind") == "optimization":
                verified_generations = _verify_generation_artifacts(
                    directory,
                    receipt["artifacts"],
                    request_sha256=request_sha256,
                    attempt_sha256=attempt_sha256,
                    expected_julia_threads=self._expected_julia_threads,
                    expected_blas_threads=self._expected_blas_threads,
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
                expected_julia_threads=self._expected_julia_threads,
                expected_blas_threads=self._expected_blas_threads,
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
                    receipt["failure"], request_document.get("spec"), plan_document,
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
            if not (outcome == "failure" and isinstance(failure, dict) and failure.get("kind") in {"backend_protocol", "optimization_progress_callback"}):
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
    _expected_julia_threads: int = 1,
    _expected_blas_threads: int = 1,
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
        return _dataclass_replace(
            binding,
            _expected_julia_threads=_expected_julia_threads,
            _expected_blas_threads=_expected_blas_threads,
        )

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
        state = versioned_workspace_document(
            workspace_instance_id=_new_uuid(excluding=leaf.workspace_instance_id),
            iterations=[{"ordinal": 1, **_leaf_pointer(leaf, root)}],
        )
    else:
        leaf = _create_leaf(root / "leaves", plan_sha256, plan_bytes)
        state = replaceable_workspace_document(
            workspace_instance_id=_new_uuid(excluding=leaf.workspace_instance_id),
            leaf_instance_id=leaf.workspace_instance_id,
            plan_sha256=leaf.plan_sha256,
        )
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
        state = versioned_workspace_document(
            workspace_instance_id=_new_uuid(excluding=leaf.workspace_instance_id),
            iterations=[{"ordinal": 1, **_leaf_pointer(leaf, root)}],
        )
        _publish_workspace_state(root / "workspace.json", _canonical_bytes(state))
        return leaf
    if not versioned and set(entries) == {"leaves"}:
        leaves = entries["leaves"]
        if leaves.is_symlink() or not leaves.is_dir():
            raise _integrity("Interrupted replacement workspace has an unsafe leaves directory.")
        children = list(leaves.iterdir())
        if len(children) == 1 and _UUID4.fullmatch(children[0].name) is not None:
            leaf = _recover_leaf(root, children[0], plan_sha256, expected_directory=f"leaves/{children[0].name}")
            state = replaceable_workspace_document(
                workspace_instance_id=_new_uuid(excluding=leaf.workspace_instance_id),
                leaf_instance_id=leaf.workspace_instance_id,
                plan_sha256=leaf.plan_sha256,
            )
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
    leaf_state = plan_workspace_document(
        workspace_instance_id=leaf_id,
        plan_sha256=plan_sha256,
    )
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
    updated = replaceable_workspace_document(
        workspace_instance_id=str(state.get("workspace_instance_id")),
        leaf_instance_id=recovered.workspace_instance_id,
        plan_sha256=recovered.plan_sha256,
        maintenance=_retired_leaf_maintenance(current, root),
    )
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
    index = versioned_workspace_document(
        workspace_instance_id=_new_uuid(excluding=upgraded.workspace_instance_id),
        iterations=[{"ordinal": 1, **_leaf_pointer(upgraded, root)}],
        maintenance=_retired_leaf_maintenance(current, root),
    )
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
    state = versioned_workspace_document(
        workspace_instance_id=root_id,
        iterations=[*iterations, {"ordinal": next_iteration, **_leaf_pointer(leaf, root)}],
        maintenance=state.get("maintenance") if isinstance(state.get("maintenance"), Mapping) else None,
    )
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
    iterations = state.get("iterations")
    if not isinstance(iterations, list):
        raise _integrity("Versioned workspace index is malformed.")
    updated = versioned_workspace_document(
        workspace_instance_id=str(state.get("workspace_instance_id")),
        iterations=[*iterations, {"ordinal": next_iteration, **_leaf_pointer(leaf, root)}],
        maintenance=state.get("maintenance") if isinstance(state.get("maintenance"), Mapping) else None,
    )
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
