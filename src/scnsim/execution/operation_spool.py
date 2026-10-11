"""Operation-local scratch for large optimization records.

The spool is temporary working storage, never a checkpoint or evidence
authority. Durable identity and verification remain with the bound SQLite
transaction. Workspace owns lease-guarded orphan pruning, shared by Run
recovery and spool allocation.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import shutil
from typing import Any

from ..errors import EvidenceIntegrityError
from ..numeric_encoding import record_bytes, record_document
from ..workspace.operation_lease import _operation_id_text
from ..workspace.operation_scratch import (
    _OPERATION_SCRATCH,
    cleanup_inactive_operation_scratch,
)


@dataclass(frozen=True, slots=True)
class SpoolReference:
    """A byte range owned by one live OperationSpool."""

    owner: str
    offset: int
    byte_length: int
    sha256: str


class OperationSpool:
    """Private append-only temporary bytes beneath one bound operations root."""

    def __init__(self, operations_root: Path, operation_id: str, *, lease_root: Path):
        root = Path(operations_root)
        self._lease_root = Path(lease_root)
        self._owner = _operation_id_text(operation_id)
        self._scratch_root = root / _OPERATION_SCRATCH
        if root.is_symlink() or not root.is_dir():
            raise EvidenceIntegrityError(
                "Operation scratch root is missing or unsafe", stage="operation_spool"
            )
        if self._scratch_root.is_symlink():
            raise EvidenceIntegrityError(
                "Operation scratch directory is symlinked", stage="operation_spool"
            )
        self._scratch_root.mkdir(mode=0o700, exist_ok=True)
        if self._scratch_root.is_symlink() or not self._scratch_root.is_dir():
            raise EvidenceIntegrityError(
                "Operation scratch directory is unsafe", stage="operation_spool"
            )
        cleanup_inactive_operation_scratch(
            root, self._lease_root, exclude_operation_id=self._owner
        )
        self._directory = self._scratch_root / self._owner
        if self._directory.is_symlink():
            raise EvidenceIntegrityError(
                "Operation scratch path is symlinked", stage="operation_spool"
            )
        if self._directory.exists():
            raise EvidenceIntegrityError(
                "Operation scratch identity already exists", stage="operation_spool",
                evidence={"operation_id": self._owner},
            )
        self._directory.mkdir(mode=0o700)
        self._path = self._directory / "records.bin"
        self._file = self._path.open("w+b")
        self._by_content: dict[tuple[str, int], SpoolReference] = {}
        self._closed = False

    def __enter__(self) -> OperationSpool:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(
                "Operation scratch cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        return False

    def activate(self) -> _SpoolBinding:
        return _SpoolBinding(self)

    def put_record(self, value: Any) -> SpoolReference:
        return self.put_bytes(record_bytes(value))

    def get_record(self, reference: SpoolReference) -> dict[str, Any]:
        raw = self.get_bytes(reference)
        value = record_document(raw)
        if record_bytes(value) != raw:
            raise EvidenceIntegrityError(
                "Operation scratch record is not canonical", stage="operation_spool"
            )
        return value

    def put_bytes(self, payload: bytes) -> SpoolReference:
        self._require_open()
        raw = bytes(payload)
        digest = sha256(raw).hexdigest()
        key = (digest, len(raw))
        existing = self._by_content.get(key)
        if existing is not None:
            if self.get_bytes(existing) != raw:
                raise EvidenceIntegrityError(
                    "Operation scratch digest identifies different bytes", stage="operation_spool"
                )
            return existing
        self._file.seek(0, 2)
        offset = self._file.tell()
        self._file.write(raw)
        reference = SpoolReference(self._owner, offset, len(raw), digest)
        self._by_content[key] = reference
        return reference

    def get_bytes(self, reference: SpoolReference) -> bytes:
        self._require_open()
        if reference.owner != self._owner:
            raise EvidenceIntegrityError(
                "Operation scratch reference belongs to another operation", stage="operation_spool"
            )
        self._file.seek(reference.offset)
        raw = self._file.read(reference.byte_length)
        if (len(raw) != reference.byte_length
                or sha256(raw).hexdigest() != reference.sha256):
            raise EvidenceIntegrityError(
                "Operation scratch bytes differ from their reference", stage="operation_spool"
            )
        return raw

    def close(self) -> None:
        if self._closed:
            return
        self._file.close()
        if self._directory.is_symlink():
            raise EvidenceIntegrityError(
                "Operation scratch path became symlinked", stage="operation_spool",
                evidence={"operation_id": self._owner},
            )
        shutil.rmtree(self._directory)
        self._by_content.clear()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise EvidenceIntegrityError("Operation scratch is already closed", stage="operation_spool")


_ACTIVE: ContextVar[OperationSpool | None] = ContextVar("scnsim_operation_spool", default=None)


class _SpoolBinding:
    def __init__(self, spool: OperationSpool):
        self._spool = spool
        self._token: Token[OperationSpool | None] | None = None

    def __enter__(self) -> OperationSpool:
        self._token = _ACTIVE.set(self._spool)
        return self._spool

    def __exit__(self, exc_type, exc, traceback) -> bool:
        assert self._token is not None
        _ACTIVE.reset(self._token)
        return False


def current_operation_spool() -> OperationSpool | None:
    """Return this context's operation scratch, if a JAX optimization owns one."""
    return _ACTIVE.get()


def is_spool_reference(value: object) -> bool:
    return isinstance(value, SpoolReference)


def prepare_resume_checkpoint(checkpoint: dict[str, Any], spool: OperationSpool) -> dict[str, Any]:
    """Keep only coordinator-required resume state and spill immutable bodies.

    The verified checkpoint hydration retains the public numerical ledger in
    SQLite. CMA needs the baseline dependency anchors, completed-candidate
    duplicate keys, best score/ordinal, and committed RNG state only.
    """
    checkpoint.pop("generations", None)
    cache = checkpoint["cache"]
    for key in tuple(cache):
        if not is_spool_reference(cache[key]):
            cache[key] = spool.put_record(cache[key])

    anchor_ids = set(checkpoint.get("anchors", {}).values())
    baseline = checkpoint["baseline"]
    dependencies = baseline.get("dependencies", {})
    baseline["dependencies"] = {key: dependencies[key] for key in anchor_ids}

    best = checkpoint["best"]
    checkpoint["best"] = {
        "evaluation_ordinal": best["evaluation_ordinal"],
        "cost_f64": best["cost_f64"],
    }
    return checkpoint
