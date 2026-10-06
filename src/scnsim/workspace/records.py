"""Immutable records exchanged by workspace transaction owners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..errors import EvidenceIntegrityError

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
