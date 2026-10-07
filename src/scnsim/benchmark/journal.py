"""Immutable benchmark journal files and small append heads.

This module owns only the file-level mechanics for the benchmark journal.
Task/global mutation semantics and the public record projection remain in
``storage.py``. The workspace lock is always acquired by that domain owner.
"""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from ..errors import EvidenceIntegrityError
from ..workspace.primitives import _inside
from ..workspace.storage import (
    _WorkspacePublishIndeterminate,
    _atomic_write,
    _fsync_directory,
    _publish_workspace_state,
)
from .prepared import record_bytes, record_document


_TASK_HEAD = "scnsim.benchmark_task_head"
_GLOBAL_HEAD = "scnsim.benchmark_global_head"
_TASK_COMMIT = "scnsim.benchmark_task_commit"
_GLOBAL_COMMIT = "scnsim.benchmark_global_commit"
_JOURNAL_VERSION = 2
_HEAD_PUBLICATION_RECONCILIATION = "_scnsim_head_publication_reconciliation"


class _DurabilityWitness:
    """Durability facts scoped to one exclusive journal transaction."""

    def __init__(self) -> None:
        self.synced_directories: set[str] = set()
        # A committed child name is narrower evidence than a synced parent:
        # it must not certify unrelated or orphan names in the same directory.
        self.committed_directory_edges: set[tuple[str, str]] = set()
        self.durable_artifacts: dict[str, tuple[str, int, tuple[int, int, int, int, int]]] = {}

    @staticmethod
    def _directory_key(root: Path, path: Path) -> str:
        relative = path.relative_to(root).as_posix()
        return relative or "."

    def directory_is_synced(self, root: Path, path: Path) -> bool:
        return self._directory_key(root, path) in self.synced_directories

    def invalidate_directory(self, root: Path, path: Path) -> None:
        self.synced_directories.discard(self._directory_key(root, path))

    def mark_directory_synced(self, root: Path, path: Path) -> None:
        self.synced_directories.add(self._directory_key(root, path))

    def directory_edge_is_committed(self, root: Path, parent: Path, child: str) -> bool:
        return (self._directory_key(root, parent), child) in self.committed_directory_edges

    def remember_verified_file_path(self, relative_file_path: str) -> None:
        """Remember directory-name edges from a file just verified under lock."""
        current = Path()
        for child in Path(relative_file_path).parts[:-1]:
            parent = current.as_posix() if current.parts else "."
            self.committed_directory_edges.add((parent, child))
            current /= child

    @staticmethod
    def _artifact_stat(path: Path) -> tuple[int, int, int, int, int]:
        value = path.stat()
        return (
            value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns,
        )

    def artifact_is_durable(
        self,
        relative: str,
        path: Path,
        *,
        digest: str,
        byte_length: int,
    ) -> bool:
        known = self.durable_artifacts.get(relative)
        if known is None:
            return False
        if (
            known[:2] != (digest, byte_length)
            or self._artifact_stat(path) != known[2]
        ):
            self.durable_artifacts.pop(relative, None)
            return False
        return True

    def mark_artifact(
        self,
        relative: str,
        path: Path,
        *,
        digest: str,
        byte_length: int,
    ) -> None:
        self.durable_artifacts[relative] = (
            digest, byte_length, self._artifact_stat(path),
        )


def _integrity(message: str, **evidence: object) -> EvidenceIntegrityError:
    return EvidenceIntegrityError(message, stage="benchmark_record", evidence=evidence)


def _relative(path: str | os.PathLike[str]) -> str:
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise _integrity("Benchmark journal path must be text.", path=repr(raw))
    value = Path(raw)
    if (
        not raw
        or value.is_absolute()
        or raw.startswith("/")
        or "\\" in raw
        or "//" in raw
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        raise _integrity("Benchmark journal path must be workspace-relative.", path=str(value))
    return value.as_posix()


def ensure_directories(
    root: Path,
    relative_directory: str | os.PathLike[str],
    *,
    durability_witness: _DurabilityWitness | None = None,
) -> Path:
    """Create a workspace-relative directory and durably publish each name."""
    relative = _relative(relative_directory)
    _inside(root, relative)
    current = root
    for part in Path(relative).parts:
        next_path = current / part
        if next_path.is_symlink():
            raise _integrity("Benchmark journal path traverses a symlink.", path=str(next_path))
        if next_path.exists():
            if not next_path.is_dir():
                raise _integrity("Benchmark journal parent is not a directory.", path=str(next_path))
            if (
                durability_witness is None
                or (
                    not durability_witness.directory_is_synced(root, current)
                    and not durability_witness.directory_edge_is_committed(
                        root, current, next_path.name,
                    )
                )
            ):
                _fsync_directory(current)
                if durability_witness is not None:
                    durability_witness.mark_directory_synced(root, current)
        else:
            if durability_witness is not None:
                durability_witness.invalidate_directory(root, current)
            try:
                next_path.mkdir()
            except FileExistsError:
                if not next_path.is_dir() or next_path.is_symlink():
                    raise _integrity("Benchmark journal parent is not a directory.", path=str(next_path))
            _fsync_directory(current)
            if durability_witness is not None:
                durability_witness.mark_directory_synced(root, current)
        current = next_path
    return current


def artifact_reference(
    relative_path: str,
    payload: bytes,
    *,
    role: str,
    digest: str | None = None,
) -> dict[str, object]:
    return {
        "path": _relative(relative_path),
        "role": role,
        "byte_length": len(payload),
        "sha256": sha256(payload).hexdigest() if digest is None else digest,
    }


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_immutable(
    root: Path,
    relative_path: str | os.PathLike[str],
    payload: bytes,
    *,
    role: str,
    durability_witness: _DurabilityWitness | None = None,
) -> dict[str, object]:
    """Publish exact immutable bytes and return their existing ArtifactRef shape."""
    relative = _relative(relative_path)
    parent = Path(relative).parent
    target = _inside(root, relative)
    if parent.parts:
        ensure_directories(root, parent, durability_witness=durability_witness)
    if target.is_symlink():
        raise _integrity("Benchmark journal artifact must not be a symlink.", path=str(target))
    if target.exists():
        if not target.is_file():
            raise _integrity("Benchmark journal artifact path already holds different bytes.", path=str(target))
        digest = (
            sha256(payload).hexdigest()
            if durability_witness is not None
            else None
        )
        if (
            durability_witness is not None
            and durability_witness.artifact_is_durable(
                relative, target,
                digest=str(digest), byte_length=len(payload),
            )
        ):
            return artifact_reference(relative, payload, role=role, digest=str(digest))
        if target.read_bytes() != payload:
            raise _integrity("Benchmark journal artifact path already holds different bytes.", path=str(target))
        if durability_witness is None:
            _fsync_file(target)
            _fsync_directory(target.parent)
        elif relative not in durability_witness.durable_artifacts:
            _fsync_file(target)
            if not durability_witness.directory_is_synced(root, target.parent):
                _fsync_directory(target.parent)
                durability_witness.mark_directory_synced(root, target.parent)
            durability_witness.mark_artifact(
                relative, target,
                digest=str(digest), byte_length=len(payload),
            )
    else:
        if durability_witness is not None:
            durability_witness.invalidate_directory(root, target.parent)
        _atomic_write(target, payload)
        reference = artifact_reference(relative, payload, role=role)
        if durability_witness is not None:
            durability_witness.mark_directory_synced(root, target.parent)
            durability_witness.mark_artifact(
                relative, target,
                digest=str(reference["sha256"]), byte_length=len(payload),
            )
        return reference
    return artifact_reference(relative, payload, role=role, digest=digest)


def read_immutable(
    root: Path,
    reference: Mapping[str, object],
    *,
    role: str | None = None,
) -> bytes:
    """Read a referenced artifact after path, role, size, and digest checks."""
    required = {"path", "role", "byte_length", "sha256"}
    if set(reference) != required:
        raise _integrity("Benchmark journal artifact reference has an unexpected field set.")
    path_value = reference["path"]
    byte_length = reference["byte_length"]
    if (
        not isinstance(path_value, str)
        or not isinstance(reference["role"], str)
        or not isinstance(byte_length, int)
        or isinstance(byte_length, bool)
        or byte_length < 0
    ):
        raise _integrity("Benchmark journal artifact reference is malformed.")
    relative = _relative(path_value)
    if role is not None and reference["role"] != role:
        raise _integrity("Benchmark journal artifact has the wrong role.", expected=role)
    path = _inside(root, relative)
    if path.is_symlink() or not path.is_file():
        raise _integrity("Benchmark journal artifact is missing or symlinked.", path=str(path))
    payload = path.read_bytes()
    if (
        len(payload) != byte_length
        or sha256(payload).hexdigest() != reference["sha256"]
    ):
        raise _integrity("Benchmark journal artifact does not match its recorded identity.", path=str(path))
    return payload


def write_document(
    root: Path,
    relative_path: str | os.PathLike[str],
    document: Mapping[str, object],
    *,
    role: str,
    durability_witness: _DurabilityWitness | None = None,
) -> dict[str, object]:
    payload = record_bytes(dict(document))
    return write_immutable(
        root, relative_path, payload, role=role,
        durability_witness=durability_witness,
    )


def read_document(
    root: Path,
    reference: Mapping[str, object],
    *,
    schema: str,
    role: str,
    bind: Mapping[str, object],
) -> dict[str, Any]:
    payload = read_immutable(root, reference, role=role)
    document = record_document(payload)
    if not isinstance(document, dict):
        raise _integrity("Benchmark journal document is not an object.", path=reference["path"])
    if (
        record_bytes(document) != payload
        or document.get("schema") != schema
        or document.get("schema_version") != _JOURNAL_VERSION
        or any(document.get(key) != value for key, value in bind.items())
    ):
        raise _integrity("Benchmark journal document does not match its binding.", path=reference["path"])
    return document


def publish_head(
    root: Path,
    relative_path: str | os.PathLike[str],
    document: Mapping[str, object],
    *,
    durability_witness: _DurabilityWitness | None = None,
) -> None:
    """Atomically publish a tiny mutable pointer after its commit is durable."""
    relative = _relative(relative_path)
    parent = Path(relative).parent
    target = _inside(root, relative)
    if parent.parts:
        ensure_directories(root, parent, durability_witness=durability_witness)
    if target.is_symlink():
        raise _integrity("Benchmark journal head must not be a symlink.", path=str(target))
    payload = record_bytes(dict(document))
    try:
        _publish_workspace_state(target, payload)
    except _WorkspacePublishIndeterminate as error:
        primary = error.__cause__ if isinstance(error.__cause__, BaseException) else error
        setattr(primary, _HEAD_PUBLICATION_RECONCILIATION, {
            "path": str(target),
            "new_sha256": sha256(payload).hexdigest(),
            "visible_state": "unreconciled_after_replace",
        })
        raise primary.with_traceback(primary.__traceback__) from None


def read_head(
    root: Path,
    relative_path: str | os.PathLike[str],
    *,
    schema: str,
    bind: Mapping[str, object],
) -> dict[str, Any]:
    relative = _relative(relative_path)
    path = _inside(root, relative)
    if path.is_symlink() or not path.is_file():
        raise _integrity("Benchmark journal head is missing or symlinked.", path=str(path))
    payload = path.read_bytes()
    document = record_document(payload)
    if not isinstance(document, dict):
        raise _integrity("Benchmark journal head is not an object.", path=str(path))
    if (
        record_bytes(document) != payload
        or document.get("schema") != schema
        or document.get("schema_version") != _JOURNAL_VERSION
        or any(document.get(key) != value for key, value in bind.items())
    ):
        raise _integrity("Benchmark journal head does not match its binding.", path=str(path))
    return document


def read_chain(
    root: Path,
    head: Mapping[str, object],
    *,
    schema: str,
    role: str,
    bind: Mapping[str, object],
) -> list[dict[str, Any]]:
    """Return a verified commit chain oldest-first from its pinned head."""
    reference = head.get("commit")
    expected_sequence = head.get("commit_sequence")
    if not isinstance(expected_sequence, int) or expected_sequence < 0:
        raise _integrity("Benchmark journal head has an invalid commit sequence.")
    if reference is None:
        if expected_sequence != 0:
            raise _integrity("Empty benchmark journal head has a nonzero sequence.")
        return []
    if not isinstance(reference, dict):
        raise _integrity("Benchmark journal head commit reference is malformed.")
    commits: list[dict[str, Any]] = []
    seen: set[str] = set()
    while reference is not None:
        if not isinstance(reference, dict):
            raise _integrity("Benchmark journal predecessor reference is malformed.")
        digest = str(reference.get("sha256"))
        if digest in seen:
            raise _integrity("Benchmark journal commit chain contains a cycle.")
        seen.add(digest)
        commit = read_document(root, reference, schema=schema, role=role, bind=bind)
        if commit.get("commit_sequence") != expected_sequence:
            raise _integrity("Benchmark journal commit sequence is not contiguous.", expected=expected_sequence)
        commits.append(commit)
        reference = commit.get("previous")
        expected_sequence -= 1
    if expected_sequence != 0:
        raise _integrity("Benchmark journal chain terminates before its first commit.")
    commits.reverse()
    return commits


__all__ = [
    "_GLOBAL_COMMIT", "_GLOBAL_HEAD", "_JOURNAL_VERSION", "_TASK_COMMIT", "_TASK_HEAD",
    "artifact_reference", "ensure_directories", "publish_head", "read_chain", "read_document",
    "read_head", "read_immutable", "write_document", "write_immutable",
]
