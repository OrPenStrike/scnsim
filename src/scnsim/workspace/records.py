"""Immutable records exchanged by workspace transaction owners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import EvidenceIntegrityError, _freeze

@dataclass(frozen=True)
class AttemptAllocation:
    """Reserved, unsealed operation-owned staging for one immutable attempt."""

    request_sha256: str
    ordinal: int
    ordinal_text: str
    staging_directory: Path
    final_directory: Path
    operation_id: str
    operation_lease: object = field(repr=False, compare=False)

    @property
    def attempt_directory_text(self) -> str:
        return f"requests/{self.request_sha256}/attempts/{self.ordinal_text}"

    @property
    def staging_directory_text(self) -> str:
        leaf_directory = self.final_directory.parents[3]
        return self.staging_directory.relative_to(leaf_directory).as_posix()

@dataclass(frozen=True)
class VerifiedSuccess:
    """One verified success chain, ready for result reconstruction."""

    request: Mapping[str, object]
    attempt: Mapping[str, object]
    receipt: Mapping[str, object]
    result: Mapping[str, object]
    directory: Path
    native_index_ref: Mapping[str, object] | None = None
    native_index_metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.native_index_ref is not None:
            object.__setattr__(self, "native_index_ref", _freeze(self.native_index_ref))
        if self.native_index_metadata is not None:
            object.__setattr__(self, "native_index_metadata", _freeze(self.native_index_metadata))

@dataclass(frozen=True)
class BaselineCheckpoint:
    """Verified request-owned optimization baseline recovery evidence."""

    checkpoint_sha256: str
    seal_sha256: str
    checkpoint: Mapping[str, object]
    source_attempt: Mapping[str, object]
    directory: Path

@dataclass(frozen=True)
class PointCheckpoint:
    record: Mapping[str, object]
    seal_sha256: str
    directory: Path

class _IncomingCheckpointEvidenceError(EvidenceIntegrityError):
    """Untrusted child checkpoint bytes failed before publication ownership."""
