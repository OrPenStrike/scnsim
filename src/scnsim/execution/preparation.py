"""Package-owned Julia resources and exact runtime preparation.

Every discovery probe and native launch belongs to an explicit operation supervisor.
Borrowed supervisors remain owned by the caller; standalone preparation owns its
full managed scope. Discovery and instantiation finish before attempt allocation.
This owner never
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


@contextmanager
def _native_supervisor_scope(native_supervisor=None):
    """Borrow the operation owner or own one standalone native lifecycle.

    The import is deferred until a native feature is actually invoked; Python
    compilation and JAX authoring do not initialize native discovery helpers.
    """
    if native_supervisor is not None:
        yield native_supervisor
        return
    from uuid import uuid4
    from .native_supervisor import NativeSupervisor

    with NativeSupervisor(operation_id=str(uuid4())) as supervisor:
        yield supervisor


def _discover_julia(version: str, *, feature: str = "Julia runtime preparation", native_supervisor=None) -> tuple[Path, str]:
    """Find an installed exact-patch runtime; never acquire Julia implicitly."""

    instructions = (
        f'Install Python support with `uv pip install "scnsim[julia]"`, then install Julia {version} '
        "manually and expose its executable on PATH or configure PYTHON_JULIAPKG_EXE."
    )

    try:
        with _native_supervisor_scope(native_supervisor) as supervisor:
            executable, discovered_version = supervisor.discover_julia(version)
    except Exception as error:  # Keep native causes and the actual helper phase.
        if isinstance(error, ImportError) and getattr(error, '_scnsim_discovery_phase', None) == 'import':
            message = f"{feature} requires Julia support; JuliaPkg import failed: {error}. {instructions}"
        else:
            message = f"Julia runtime discovery failed for {feature}: {error}. Required Julia {version}. {instructions}"
        raise RuntimePreparationError(
            message, stage="runtime_discovery",
            evidence={"requested_feature": feature, "required_julia_version": version,
                      "error": str(error), "discovery_phase": getattr(error, '_scnsim_discovery_phase', None)},
        ) from error
    path = Path(executable).resolve()
    if not path.is_file() or str(discovered_version) != version:
        raise RuntimePreparationError(
            f"JuliaPkg returned a runtime other than the required exact patch for {feature}",
            stage="runtime_discovery",
            evidence={"requested_feature": feature, "required_julia_version": version,
                      "executable": str(path), "reported_version": str(discovered_version)},
        )
    return path, str(discovered_version)


def _verify_julia_version(executable: Path, expected: str, *, native_supervisor=None) -> None:
    with _native_supervisor_scope(native_supervisor) as supervisor:
        try:
            completed = supervisor.run(
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


def _instantiate_packaged_project(executable: Path, project: Path, *, native_supervisor=None) -> None:
    """Instantiate exactly the committed environment and reject Manifest drift."""
    with _native_supervisor_scope(native_supervisor) as supervisor:

        manifest = project / "Manifest.toml"
        try:
            before = manifest.read_bytes()
            completed = supervisor.run(
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


def prepare_runtime(*, feature: str = "Julia runtime preparation", native_supervisor=None) -> PreparedRuntime:
    """Prepare installed Julia for the requested feature without acquiring a runtime."""
    _require_supported_platform()
    with _native_supervisor_scope(native_supervisor) as supervisor:
        with packaged_julia_resources() as (project, _, runtime):
            version = _runtime_version(runtime)
            executable, discovered_version = _discover_julia(version, feature=feature, native_supervisor=supervisor)
            _verify_julia_version(executable, discovered_version, native_supervisor=supervisor)
            _instantiate_packaged_project(executable, project, native_supervisor=supervisor)
        return PreparedRuntime(executable, discovered_version, runtime)


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(_THREAD_ENVIRONMENT)
    return environment
