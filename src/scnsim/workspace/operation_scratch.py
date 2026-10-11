"""Lease-guarded cleanup for temporary operation-owned spool payloads.

Workspace recovery owns orphan removal. JAX spool creation calls the same
mechanism before allocating its own operation directory. Durable task,
checkpoint, and result evidence is stored elsewhere and is never traversed.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from ..errors import EvidenceIntegrityError
from .operation_lease import _idle_operation_lease, _operation_id_text


_OPERATION_SCRATCH = ".scnsim-operation-scratch"


def cleanup_inactive_operation_scratch(
    operations_root: Path,
    lease_root: Path,
    *,
    exclude_operation_id: str | None = None,
) -> None:
    """Remove only UUID-owned spool directories whose stable lease is idle.

    The operation lock is acquired nonblocking for each candidate, so an
    active coordinator or surviving native-process guardian retains its
    scratch. Persistent lease files are not removed or recreated.
    """

    root = Path(operations_root)
    if root.is_symlink():
        raise EvidenceIntegrityError(
            "Operation scratch root is symlinked", stage="operation_spool"
        )
    if not root.exists():
        return
    if not root.is_dir():
        raise EvidenceIntegrityError(
            "Operation scratch root is not a directory", stage="operation_spool"
        )

    scratch_root = root / _OPERATION_SCRATCH
    if scratch_root.is_symlink():
        raise EvidenceIntegrityError(
            "Operation scratch directory is symlinked", stage="operation_spool"
        )
    if not scratch_root.exists():
        return
    if not scratch_root.is_dir():
        raise EvidenceIntegrityError(
            "Operation scratch path is not a directory", stage="operation_spool"
        )

    excluded = (
        None if exclude_operation_id is None
        else _operation_id_text(exclude_operation_id)
    )
    for entry in tuple(scratch_root.iterdir()):
        if entry.name == excluded or entry.is_symlink() or not entry.is_dir():
            continue
        try:
            owner = _operation_id_text(entry.name)
        except (TypeError, ValueError):
            continue
        with _idle_operation_lease(Path(lease_root), owner) as lease:
            if lease.acquired and entry.exists() and not entry.is_symlink() and entry.is_dir():
                shutil.rmtree(entry)
