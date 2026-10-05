"""Exact workspace file publication and decoding primitives."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from ..canonical import canonical_json_bytes as _canonical_bytes
from .validation.common import _integrity

_ATTEMPT = re.compile(r"^(?!000000$)(?:[0-9]{6}|[1-9][0-9]{6,})$")
_STAGING = re.compile(
    r"^\.staging-((?!000000-)(?:[0-9]{6}|[1-9][0-9]{6,}))"
    r"-([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$"
)
_LEAF_STAGING = re.compile(
    r"^\.staging-leaf-([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$"
)
_CHECKPOINT_STAGING = re.compile(
    r"^\.staging-baseline-checkpoint-"
    r"([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$"
)

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

def _decode_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _integrity(f"{label.capitalize()} bytes are not JSON.", error=str(error)) from error
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise _integrity(f"{label.capitalize()} bytes are not canonical JSON.")
    return value

def _path_entry_exists(path: Path) -> bool:
    """Return lexical directory-entry presence, including dangling symlinks."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True

def _remove_leaf(leaf: Path) -> None:
    if leaf.parent.name != "leaves" or leaf.parent.is_symlink() or leaf.is_symlink():
        raise _integrity("Refusing to remove a non-leaf workspace path.", leaf=str(leaf))
    shutil.rmtree(leaf)
    _fsync_directory(leaf.parent)
