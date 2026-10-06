"""Canonical metadata and integrity manifests for stored Zarr artifacts.

Manifest construction validates actual artifact bytes and their dataset layout;
it neither reconstructs typed Results nor authorizes workspace publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
import json
import math
import re
from ..canonical import (
    _identifier,
    _integrity,
    _validation,
    canonical_json_bytes,
    canonical_value,
    relative_path,
    sha256_hex,
)


_CHUNK = re.compile(r"^[0-9]+(?:\.[0-9]+){0,4}$")


def zarr_group_metadata_bytes() -> bytes:
    """The one exact V2 root-group metadata byte sequence accepted by V1."""

    return b'{"zarr_format":2}'


def zarr_array_metadata_bytes(*, shape: Sequence[int], chunks: Sequence[int]) -> bytes:
    """Return exact compact V2 Float64 C-order dataset metadata bytes."""

    if (
        not shape
        or len(shape) != len(chunks)
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < 0
            or (item == 0 and index != 0)
            for index, item in enumerate(shape)
        )
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in chunks)
    ):
        raise _validation("invalid Zarr shape/chunks")
    return canonical_json_bytes({
        "chunks": list(chunks), "compressor": None, "dimension_separator": ".", "dtype": "<f8",
        "fill_value": None, "filters": None, "order": "C", "shape": list(shape), "zarr_format": 2,
    })


def zarr_artifact_manifest(
    *, artifact_directory: Path, artifact_id: str, artifact_path: str
) -> dict[str, object]:
    """Validate one Julia-written V2 artifact tree and return its canonical manifest.

    This deliberately accepts only the root group plus `values` or paired
    `real`/`imag` Float64 arrays; callers compare the returned metadata with
    their typed result catalog.
    """

    if artifact_directory.is_symlink():
        raise _integrity("Zarr artifact directory is symlinked", path=str(artifact_directory))
    root = artifact_directory.resolve(strict=True)
    if not root.is_dir():
        raise _integrity("Zarr artifact directory is missing", path=str(artifact_directory))
    entries: list[tuple[str, Path]] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise _integrity("Zarr artifact contains a symlink", path=str(candidate))
        if candidate.is_file():
            relative = candidate.relative_to(root).as_posix()
            entries.append((relative_path(relative), candidate))
        elif candidate.is_dir():
            continue
        else:
            raise _integrity("Zarr artifact contains a non-regular filesystem entry", path=str(candidate))
    entries.sort(key=lambda item: item[0])
    files = {path: candidate for path, candidate in entries}
    if files.get(".zgroup") is None or files[".zgroup"].read_bytes() != zarr_group_metadata_bytes():
        raise _integrity("Zarr root metadata bytes differ from V1 contract")
    datasets = _zarr_datasets(files)
    allowed = {".zgroup"}
    for dataset in datasets:
        allowed.add(f"{dataset}/.zarray")
        allowed.update(path for path in files if path.startswith(f"{dataset}/") and not path.endswith("/.zarray"))
    if set(files) != allowed:
        raise _integrity("Zarr artifact contains unsupported metadata or paths", paths=sorted(set(files) - allowed))
    manifest_files = [
        {"path": path, "mode": "regular", "byte_length": file.stat().st_size, "sha256": sha256_hex(file.read_bytes())}
        for path, file in entries
    ]
    return canonical_value({
        "schema": "scnsim.artifact_manifest", "schema_version": 1,
        "artifact_id": _identifier(artifact_id, field="artifact_id"),
        "artifact_path": relative_path(artifact_path), "zarr_format": 2,
        "group_metadata_path": ".zgroup", "datasets": [
            {"path": dataset, "metadata_path": f"{dataset}/.zarray", "chunk_paths": sorted(
                path for path in files if path.startswith(f"{dataset}/") and path != f"{dataset}/.zarray"
            )}
            for dataset in datasets
        ],
        "files": manifest_files,
    })  # type: ignore[return-value]


def _zarr_datasets(files: Mapping[str, Path]) -> list[str]:
    present = [name for name in ("values", "real", "imag") if f"{name}/.zarray" in files]
    if present not in (["values"], ["real", "imag"]):
        raise _integrity("Zarr artifact must contain values or paired real/imag datasets")
    for dataset in present:
        metadata_path = f"{dataset}/.zarray"
        try:
            metadata = json.loads(files[metadata_path].read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _integrity("Zarr array metadata is invalid JSON", path=metadata_path) from error
        shape = metadata.get("shape") if isinstance(metadata, Mapping) else None
        chunks = metadata.get("chunks") if isinstance(metadata, Mapping) else None
        if not isinstance(shape, list) or not isinstance(chunks, list):
            raise _integrity("Zarr array metadata lacks shape/chunks", path=metadata_path)
        if files[metadata_path].read_bytes() != zarr_array_metadata_bytes(shape=shape, chunks=chunks):
            raise _integrity("Zarr array metadata bytes differ from V1 contract", path=metadata_path)
        counts = [math.ceil(size / chunk) for size, chunk in zip(shape, chunks)]
        expected = {
            f"{dataset}/" + ".".join(str(index) for index in indices)
            for indices in product(*(range(count) for count in counts))
        }
        actual = {
            path for path in files
            if path.startswith(f"{dataset}/") and path != metadata_path
        }
        if actual != expected:
            raise _integrity(
                "Zarr chunk grid is incomplete or has extra chunks",
                missing=sorted(expected - actual),
                extra=sorted(actual - expected),
            )
        for path in actual:
            chunk = path.removeprefix(f"{dataset}/")
            if not _CHUNK.fullmatch(chunk):
                raise _integrity("Zarr chunk name is invalid", path=path)
            elements = math.prod(chunks)
            if files[path].stat().st_size != 8 * elements:
                raise _integrity("Zarr chunk byte length disagrees with declared chunk shape", path=path)
    return present
