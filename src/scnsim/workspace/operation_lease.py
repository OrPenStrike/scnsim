"""Stable OS leases that keep one bound Workspace leaf alive for an operation.

The root activity inode is distinct from SQLite and Plan locks. An operation
holds a shared lease from before runtime preparation through result decoding;
replaceable-leaf mutation takes the exclusive side. Native supervisors may
inherit a duplicate descriptor so a surviving kernel continues to pin the leaf
after its parent exits. A second, stable request-hash lease serializes exact
cache selection and attempt publication without holding a Workspace lock over
the numerical call.
"""
from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path
from uuid import UUID

if sys.platform in {"linux", "darwin"}:
    import fcntl

from ..errors import EvidenceIntegrityError, UnsupportedRuntimePlatformError

_LOCK_NAME = ".scnsim-operation-lease.lock"
_OPERATION_LOCKS = ".scnsim-operation-leases"
_REQUEST_LOCKS = ".scnsim-request-execution-leases"
_REQUEST_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _close_descriptors(
    descriptors: tuple[int, ...],
    *,
    primary_error: BaseException | None,
    context: str,
) -> None:
    """Close every owned descriptor without replacing an active failure."""
    first_close_error: BaseException | None = None
    for descriptor in descriptors:
        if descriptor < 0:
            continue
        try:
            os.close(descriptor)
        except BaseException as error:
            if primary_error is not None:
                primary_error.add_note(f"{context} descriptor close also failed: {error!r}")
            elif first_close_error is None:
                first_close_error = error
            else:
                first_close_error.add_note(f"{context} descriptor close also failed: {error!r}")
    if primary_error is None and first_close_error is not None:
        raise first_close_error


def _operation_id_text(operation_id: str) -> str:
    """Return the one path-safe, stable spelling of an operation UUID."""
    if not isinstance(operation_id, str):
        raise TypeError("operation_id must be a string")
    try:
        return str(UUID(operation_id))
    except (ValueError, AttributeError) as error:
        raise ValueError("operation_id must be a UUID") from error


def _open(root: Path) -> int:
    if sys.platform not in {"linux", "darwin"}:
        raise UnsupportedRuntimePlatformError(
            "Workspace operation leases are supported only on Linux and macOS.",
            stage="operation_lease",
            evidence={"platform": sys.platform},
        )
    if root.is_symlink() or not root.is_dir():
        raise EvidenceIntegrityError(
            "Workspace operation lease root is missing or symlinked.",
            stage="operation_lease", evidence={"root": str(root)},
        )
    path = root / _LOCK_NAME
    if path.is_symlink():
        raise EvidenceIntegrityError(
            "Workspace operation lease path must not be a symlink.",
            stage="operation_lease", evidence={"path": str(path)},
        )
    return os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)


def _open_operation(root: Path, operation_id: str) -> int:
    """Open the never-unlinked inode reserved for one operation identity."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise EvidenceIntegrityError(
            "Workspace operation lease root is missing or symlinked.",
            stage="operation_lease", evidence={"root": str(root)},
        )
    operation_id = _operation_id_text(operation_id)
    directory = root / _OPERATION_LOCKS
    if directory.is_symlink():
        raise EvidenceIntegrityError(
            "Workspace operation-lock directory must not be a symlink.",
            stage="operation_lease", evidence={"path": str(directory)},
        )
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise EvidenceIntegrityError(
            "Workspace operation-lock directory is unsafe.",
            stage="operation_lease", evidence={"path": str(directory)},
        )
    path = directory / f"{operation_id}.lock"
    if path.is_symlink():
        raise EvidenceIntegrityError(
            "Workspace per-operation lock path must not be a symlink.",
            stage="operation_lease", evidence={"path": str(path)},
        )
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise EvidenceIntegrityError(
                "Workspace per-operation lock is not a regular file.",
                stage="operation_lease", evidence={"path": str(path)},
            )
    except BaseException as original:
        _close_descriptors(
            (descriptor,), primary_error=original,
            context="Operation lease descriptor validation",
        )
        raise
    return descriptor


class OperationLease:
    """One locked descriptor; closing it releases only this descriptor's lease."""

    __slots__ = ("operation_id", "root", "_root_descriptor", "_operation_descriptor", "_entered")

    def __init__(self, root: Path, operation_id: str):
        self.operation_id = _operation_id_text(operation_id)
        self.root = Path(root)
        self._root_descriptor = -1
        self._operation_descriptor = -1
        self._entered = False

    @property
    def descriptor(self) -> int:
        """The root activity descriptor, retained for existing callers."""
        if self._root_descriptor < 0:
            raise RuntimeError("operation lease is not active")
        return self._root_descriptor

    @property
    def operation_descriptor(self) -> int:
        if self._operation_descriptor < 0:
            raise RuntimeError("operation lease is not active")
        return self._operation_descriptor

    @property
    def descriptors(self) -> tuple[int, int]:
        """Root activity and operation-specific descriptors, in that order."""
        return self.descriptor, self.operation_descriptor

    def duplicate_descriptor(self) -> int:
        """Return a duplicate of the root activity gate descriptor."""
        return os.dup(self.descriptor)

    def duplicate_descriptors(self) -> tuple[int, int]:
        """Duplicate both descriptors, closing a partial duplicate on error."""
        root_descriptor = os.dup(self.descriptor)
        try:
            operation_descriptor = os.dup(self.operation_descriptor)
        except BaseException as original:
            _close_descriptors(
                (root_descriptor,), primary_error=original,
                context="Operation lease duplicate",
            )
            raise
        return root_descriptor, operation_descriptor

    def __enter__(self) -> OperationLease:
        if self._entered:
            raise RuntimeError("operation lease cannot be entered twice")
        root_descriptor = _open(self.root)
        operation_descriptor = -1
        try:
            fcntl.flock(root_descriptor, fcntl.LOCK_SH)
            operation_descriptor = _open_operation(self.root, self.operation_id)
            fcntl.flock(operation_descriptor, fcntl.LOCK_EX)
        except BaseException as original:
            _close_descriptors(
                (operation_descriptor, root_descriptor),
                primary_error=original,
                context="Operation lease acquisition",
            )
            raise
        self._root_descriptor = root_descriptor
        self._operation_descriptor = operation_descriptor
        self._entered = True
        return self

    def __exit__(self, _kind, error, _traceback) -> None:
        operation_descriptor = self._operation_descriptor
        root_descriptor = self._root_descriptor
        self._operation_descriptor = -1
        self._root_descriptor = -1
        # Do not issue LOCK_UN. A supervisor duplicate must retain each lease
        # until its owned native process tree has drained.
        _close_descriptors(
            (operation_descriptor, root_descriptor),
            primary_error=error,
            context="Operation lease release",
        )


def operation_lease(binding, operation_id: str) -> OperationLease:
    """Return the operation's root-scoped lease context for a bound leaf."""
    return OperationLease(Path(binding.root), operation_id)


class RequestExecutionLease:
    """Serialize execution and result selection for one exact request in a leaf."""

    __slots__ = ("root", "leaf", "request_sha256", "_descriptor", "_binding")

    def __init__(self, binding, request_sha256: str):
        if not isinstance(request_sha256, str) or _REQUEST_SHA256.fullmatch(request_sha256) is None:
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest")
        self.root = Path(binding.root)
        self.leaf = Path(binding.leaf)
        self.request_sha256 = request_sha256
        self._descriptor = -1
        self._binding = binding

    @property
    def descriptor(self) -> int:
        if self._descriptor < 0:
            raise RuntimeError("request execution lease is not active")
        return self._descriptor

    def __enter__(self) -> RequestExecutionLease:
        if self._descriptor >= 0:
            raise RuntimeError("request execution lease cannot be entered twice")
        if (
            self.root.is_symlink() or not self.root.is_dir()
            or self.leaf.is_symlink() or not self.leaf.is_dir()
        ):
            raise EvidenceIntegrityError(
                "Request execution lease requires a present bound Workspace leaf.",
                stage="request_lease", evidence={"leaf": str(self.leaf)},
            )
        self._binding.assert_current()
        self.leaf = Path(self._binding.leaf)
        directory = self.leaf / _REQUEST_LOCKS
        if directory.is_symlink():
            raise EvidenceIntegrityError(
                "Request execution lease directory must not be a symlink.",
                stage="request_lease", evidence={"path": str(directory)},
            )
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise EvidenceIntegrityError(
                "Request execution lease directory is unsafe.",
                stage="request_lease", evidence={"path": str(directory)},
            )
        path = directory / f"{self.request_sha256}.lock"
        if path.is_symlink():
            raise EvidenceIntegrityError(
                "Request execution lease path must not be a symlink.",
                stage="request_lease", evidence={"path": str(path)},
            )
        descriptor = os.open(
            path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise EvidenceIntegrityError(
                    "Request execution lease is not a regular file.",
                    stage="request_lease", evidence={"path": str(path)},
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._binding.assert_current()
            if Path(self._binding.leaf) != self.leaf:
                raise EvidenceIntegrityError(
                    "Bound Workspace leaf changed while acquiring its request lease.",
                    stage="request_lease", evidence={"leaf": str(self.leaf)},
                )
        except BaseException as original:
            _close_descriptors(
                (descriptor,), primary_error=original,
                context="Request execution lease acquisition",
            )
            raise
        self._descriptor = descriptor
        return self

    def __exit__(self, _kind, error, _traceback) -> None:
        descriptor = self._descriptor
        self._descriptor = -1
        # A native supervisor may hold a duplicate until the process tree drains.
        # Closing this descriptor releases only the caller's reference.
        _close_descriptors(
            (descriptor,), primary_error=error,
            context="Request execution lease release",
        )


def request_execution_lease(binding, request_sha256: str) -> RequestExecutionLease:
    """Return the stable per-request execution/selection lease for one leaf."""
    return RequestExecutionLease(binding, request_sha256)


class _IdleOperationLease:
    """Nonblocking exclusive operation lock used while pruning its scratch."""

    def __init__(self, root: Path, operation_id: str):
        self.root = Path(root)
        self.operation_id = _operation_id_text(operation_id)
        self.descriptor = -1
        self.acquired = False

    def __enter__(self):
        descriptor = _open_operation(self.root, self.operation_id)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return self
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor = descriptor
        self.acquired = True
        return self

    def __exit__(self, _kind, _error, _traceback):
        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor >= 0:
            os.close(descriptor)


def _idle_operation_lease(root: Path, operation_id: str) -> _IdleOperationLease:
    """Acquire one operation inode if idle; caller must hold the root SH gate."""
    return _IdleOperationLease(root, operation_id)


class _ExclusiveActivityLease:
    """Private gate used only while binding can replace or remove a leaf."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.descriptor = -1

    def __enter__(self):
        descriptor = _open(self.root)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor = descriptor
        return self

    def __exit__(self, _kind, _error, _traceback):
        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor >= 0:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _exclusive_activity_lease(root: Path) -> _ExclusiveActivityLease:
    """Acquire the root gate before any replaceable-leaf transition."""
    return _ExclusiveActivityLease(root)
