"""Package-owned Julia resources and exact runtime preparation.

Discovery and instantiation finish before attempt allocation. This owner never
touches a Run workspace, selects a solver, or creates terminal evidence."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from shutil import copyfileobj
from typing import Any
from ..errors import RuntimePreparationError, UnsupportedRuntimePlatformError


_EXPECTED_JULIA_VERSION = "1.12.6"


_THREAD_ENVIRONMENT = {
    # JosephsonCircuits uses FFTW in the HB slice.  Keep the environment
    # deterministic before Julia starts; the HB adapter independently sets
    # and verifies the runtime count before it accepts a case.
    "FFTW_NUM_THREADS": "1",
    "JULIA_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True)
class PreparedRuntime:
    """An exact Julia executable selected before a workspace attempt exists."""

    executable: Path
    julia_version: str
    runtime_metadata: Mapping[str, object]


def _runtime_resources() -> Any:
    return resources.files("scnsim").joinpath("_julia")


def _copy_resource_tree(source: Any, destination: Path) -> None:
    """Materialize a package resource when a zip-style importer provides it."""

    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            _copy_resource_tree(child, target)
        else:
            with child.open("rb") as input_file, target.open("wb") as output_file:
                copyfileobj(input_file, output_file)


@contextmanager
def packaged_julia_resources() -> Iterator[tuple[Path, Path, Mapping[str, object]]]:
    """Yield absolute project/entrypoint paths sourced only from this package."""

    source = _runtime_resources()
    # Wheels are normally extracted to a real directory.  Keep the fallback for
    # zip-style importers without consulting the caller's cwd.
    try:
        source_path = Path(source)  # type: ignore[arg-type]
    except TypeError:
        source_path = None
    if source_path is not None and source_path.is_dir():
        with _yield_packaged_paths(source_path) as paths:
            yield paths
        return
    with tempfile.TemporaryDirectory(prefix="scnsim-julia-") as temporary:
        materialized = Path(temporary) / "_julia"
        _copy_resource_tree(source, materialized)
        with _yield_packaged_paths(materialized) as paths:
            yield paths


@contextmanager
def _yield_packaged_paths(
    root: Path,
) -> Iterator[tuple[Path, Path, Mapping[str, object]]]:
    project = root
    entrypoint = root / "bin" / "scnsim_request.jl"
    runtime_file = root / "runtime.json"
    if not (project / "Project.toml").is_file() or not (project / "Manifest.toml").is_file():
        raise RuntimePreparationError(
            "packaged SCNSim Julia project is incomplete",
            stage="package_resources",
            evidence={"project": str(project)},
        )
    if not entrypoint.is_file() or not runtime_file.is_file():
        raise RuntimePreparationError(
            "packaged SCNSim Julia entrypoint or runtime metadata is missing",
            stage="package_resources",
            evidence={"root": str(root)},
        )
    try:
        runtime = json.loads(runtime_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimePreparationError(
            "packaged runtime metadata is unreadable",
            stage="runtime_metadata",
            evidence={"path": str(runtime_file), "error": str(error)},
        ) from error
    if not isinstance(runtime, dict):
        raise RuntimePreparationError(
            "packaged runtime metadata must be an object",
            stage="runtime_metadata",
            evidence={"path": str(runtime_file)},
        )
    yield project, entrypoint, runtime


def _require_supported_platform() -> None:
    system = platform.system()
    if system not in {"Linux", "Darwin"} or sys.maxsize <= 2**32:
        raise UnsupportedRuntimePlatformError(
            "SCNSim V1 backend requires a 64-bit Linux or macOS runtime",
            stage="runtime_platform",
            evidence={"system": system, "machine": platform.machine()},
        )


def _runtime_version(runtime: Mapping[str, object]) -> str:
    value = runtime.get("julia_version")
    if value != _EXPECTED_JULIA_VERSION:
        raise RuntimePreparationError(
            "packaged runtime metadata does not declare Julia 1.12.6",
            stage="runtime_metadata",
            evidence={"declared_julia_version": value},
        )
    return _EXPECTED_JULIA_VERSION


def _discover_julia(version: str) -> tuple[Path, str]:
    """Use JuliaPkg only as the documented executable finder/installer."""

    try:
        from juliapkg.compat import Compat
        from juliapkg.find_julia import find_julia
        from juliapkg.state import STATE
    except ImportError as error:
        raise RuntimePreparationError(
            "JuliaPkg is required to prepare the SCNSim backend",
            stage="runtime_discovery",
            evidence={"error": str(error)},
        ) from error
    try:
        executable, discovered_version = find_julia(
            compat=Compat.parse(f"={version}"),
            prefix=STATE["install"],
            install=True,
            upgrade=False,
        )
    except Exception as error:  # JuliaPkg intentionally owns its acquisition details.
        raise RuntimePreparationError(
            "JuliaPkg could not find or install the required Julia runtime",
            stage="runtime_discovery",
            evidence={"required_julia_version": version, "error": str(error)},
        ) from error
    path = Path(executable).resolve()
    if not path.is_file() or str(discovered_version) != version:
        raise RuntimePreparationError(
            "JuliaPkg returned a runtime other than the required exact patch",
            stage="runtime_discovery",
            evidence={"executable": str(path), "reported_version": str(discovered_version)},
        )
    return path, str(discovered_version)


def _verify_julia_version(executable: Path, expected: str) -> None:
    try:
        completed = subprocess.run(
            [str(executable), "--startup-file=no", "--history-file=no", "--version"],
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env=_child_environment(),
        )
    except OSError as error:
        raise RuntimePreparationError(
            "the Julia executable selected by JuliaPkg cannot be started",
            stage="runtime_verification",
            evidence={"executable": str(executable), "error": str(error)},
        ) from error
    observed = completed.stdout.strip() or completed.stderr.strip()
    if completed.returncode != 0 or observed != f"julia version {expected}":
        raise RuntimePreparationError(
            "the Julia executable did not report the exact required patch",
            stage="runtime_verification",
            evidence={
                "executable": str(executable),
                "returncode": completed.returncode,
                "observed": observed,
                "expected": expected,
            },
        )


def _instantiate_packaged_project(executable: Path, project: Path) -> None:
    """Instantiate exactly the committed environment and reject Manifest drift."""

    manifest = project / "Manifest.toml"
    try:
        before = manifest.read_bytes()
        completed = subprocess.run(
            [
                str(executable),
                "--startup-file=no",
                "--history-file=no",
                "--threads=1",
                f"--project={project}",
                "-e",
                "using Pkg; Pkg.instantiate(); using SCNSimBackend",
            ],
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env=_child_environment(),
            cwd=str(project),
        )
        after = manifest.read_bytes()
    except (OSError, UnicodeError) as error:
        raise RuntimePreparationError(
            "the packaged SCNSim Julia project could not be instantiated",
            stage="runtime_preparation",
            evidence={"project": str(project), "error": str(error)},
        ) from error
    if before != after:
        raise RuntimePreparationError(
            "Julia preparation modified the committed SCNSim Manifest",
            stage="runtime_preparation",
            evidence={"project": str(project)},
        )
    if completed.returncode != 0:
        raise RuntimePreparationError(
            "the packaged SCNSim Julia project failed to instantiate or import",
            stage="runtime_preparation",
            evidence={
                "project": str(project),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )


def prepare_runtime() -> PreparedRuntime:
    """Prepare and exact-version-check Julia without touching a workspace."""

    _require_supported_platform()
    with packaged_julia_resources() as (project, _, runtime):
        version = _runtime_version(runtime)
        executable, discovered_version = _discover_julia(version)
        _verify_julia_version(executable, discovered_version)
        _instantiate_packaged_project(executable, project)
    return PreparedRuntime(executable, discovered_version, runtime)


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(_THREAD_ENVIRONMENT)
    return environment
