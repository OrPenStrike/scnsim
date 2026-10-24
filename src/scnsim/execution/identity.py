"""Static semantic runtime identity from installed SCNSim package bytes.

Package-relative manifests cover shipped Python, compensated CPU kernel source,
schemas, and static metadata. The historical python_source_sha256 key
also binds packaged native kernel source, independent of platform binary bytes.
Host runtime discovery and process launch do not participate."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from ..canonical import sha256_hex
from ..authoring.provenance import _excluded_source_path


def _runtime_identity_base() -> dict[str, object]:
    package = Path(__file__).resolve().parents[1]

    def manifest(paths: Sequence[Path]) -> str:
        rows = [
            {"path": path.relative_to(package).as_posix(), "mode": "100644", "sha256": sha256(path.read_bytes()).hexdigest()}
            for path in sorted(paths)
        ]
        return sha256_hex({"schema": "scnsim.source_manifest", "schema_version": 1, "files": rows})

    python_files = [
        *(
            path for path in package.rglob("*.py")
            if not _excluded_source_path(path.relative_to(package).as_posix())
        ),
        *package.glob("_schemas/*.json"),
        package / "_native" / "compensated_cpu.cpp",
    ]
    if not all(path.is_file() for path in python_files):
        raise RuntimeError("SCNSim packaged runtime resources are incomplete")
    return {"python_source_sha256": manifest(python_files)}


def _jax_runtime_identity(base: dict[str, object], *, precision: str) -> dict[str, object]:
    """Static JAX request identity; no import, device discovery or native probe."""
    from importlib.metadata import version
    from .config import runtime_resource_identity

    return {
        "backend": "jax", "precision": precision,
        "python_source_sha256": base["python_source_sha256"],
        "scipy_version": version("scipy"),
        "jax_version": version("jax"), "jaxlib_version": version("jaxlib"),
        "cmaes_version": version("cmaes"),
        "resources": runtime_resource_identity(),
    }
