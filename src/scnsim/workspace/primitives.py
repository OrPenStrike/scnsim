"""Filesystem-boundary helpers for workspace-owned paths."""

from __future__ import annotations

from pathlib import Path

from .validation.common import _CONTROL, _integrity

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
