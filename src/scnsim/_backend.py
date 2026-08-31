"""The deliberately small Python side of SCNSim's one-process Julia boundary.

This module owns runtime selection and transport only.  Request construction,
attempt sealing, and receipt/artifact validation stay with their respective
owners so a child process never becomes a second authority for those records.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from shutil import copyfileobj
from typing import Any

from .errors import (
    BackendProtocolError,
    RuntimePreparationError,
    UnsupportedRuntimePlatformError,
)


_EXPECTED_JULIA_VERSION = "1.12.6"
_SHA256_LENGTH = 64
_THREAD_ENVIRONMENT = {
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


@dataclass(frozen=True)
class BootstrapReady:
    """Validated child bootstrap evidence, before the attempt is sealed."""

    request_sha256: str
    attempt_ordinal: int
    julia_version: str
    julia_threads: int
    blas_threads: int
    blas_vendor: str


@dataclass(frozen=True)
class TerminalOutcome:
    """Transport facts returned only after a successful child exit/outcome pair."""

    outcome: Mapping[str, object]
    progress: tuple[Mapping[str, object], ...]
    stdout_log: tuple[str, ...]
    stderr_log: tuple[str, ...]


def _canonical_json_line(value: Mapping[str, object]) -> str:
    """Return the protocol's one permitted canonical JSONL representation."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise BackendProtocolError(
            "protocol frame cannot be represented as canonical JSON",
            stage="protocol_frame",
            evidence={"error": str(error)},
        ) from error


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_absolute_file(value: str | os.PathLike[str], *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise BackendProtocolError(
            f"{label} must be an existing absolute file",
            stage="launch_arguments",
            evidence={"label": label, "path": str(path)},
        )
    return path


def _require_absolute_directory(value: str | os.PathLike[str], *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise BackendProtocolError(
            f"{label} must be an existing absolute directory",
            stage="launch_arguments",
            evidence={"label": label, "path": str(path)},
        )
    return path


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


def _terminal_argv(
    prepared: PreparedRuntime,
    project: Path,
    entrypoint: Path,
    request_path: Path,
    staging_directory: Path,
) -> list[str]:
    return [
        str(prepared.executable),
        "--startup-file=no",
        "--history-file=no",
        "--threads=1",
        f"--project={project}",
        str(entrypoint),
        "--request",
        str(request_path),
        "--staging",
        str(staging_directory),
    ]


def _read_lines(
    stream: Any,
    sink: queue.Queue[str | None] | list[str],
    errors: list[BaseException],
) -> None:
    try:
        for line in iter(stream.readline, ""):
            if isinstance(sink, queue.Queue):
                sink.put(line)
            else:
                sink.append(line)
    except BaseException as error:  # Reader failures must not become invisible logs.
        errors.append(error)
    finally:
        if isinstance(sink, queue.Queue):
            sink.put(None)


def _protocol_error(
    message: str,
    *,
    stage: str,
    stdout_log: list[str],
    stderr_log: list[str],
    extra: Mapping[str, object] | None = None,
) -> BackendProtocolError:
    evidence: dict[str, object] = {
        "stdout_log": tuple(stdout_log),
        "stderr_log": tuple(stderr_log),
    }
    evidence.update(extra or {})
    return BackendProtocolError(message, stage=stage, evidence=evidence)


def _decode_canonical_line(raw: str, *, stage: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BackendProtocolError(
            "child emitted malformed JSONL protocol framing",
            stage=stage,
            evidence={"line": raw.rstrip("\n"), "error": str(error)},
        ) from error
    if not isinstance(decoded, dict) or raw != _canonical_json_line(decoded):
        raise BackendProtocolError(
            "child protocol JSONL is not the required canonical frame",
            stage=stage,
            evidence={"line": raw.rstrip("\n")},
        )
    return decoded


def _validate_bootstrap(
    raw: str,
    *,
    request_sha256: str,
    attempt_ordinal: int,
    expected_version: str,
) -> BootstrapReady:
    frame = _decode_canonical_line(raw, stage="bootstrap")
    required = {
        "schema": "scnsim.bootstrap_ready",
        "schema_version": 1,
        "request_sha256": request_sha256,
        "attempt_ordinal": attempt_ordinal,
        "julia_version": expected_version,
        "julia_threads": 1,
        "blas_threads": 1,
    }
    if set(frame) != {*required, "blas_vendor"} or any(
        frame.get(key) != value for key, value in required.items()
    ) or not isinstance(frame.get("blas_vendor"), str) or not frame["blas_vendor"]:
        raise BackendProtocolError(
            "child bootstrap evidence does not match the sealed launch",
            stage="bootstrap",
            evidence={"frame": frame},
        )
    return BootstrapReady(
        request_sha256=request_sha256,
        attempt_ordinal=attempt_ordinal,
        julia_version=expected_version,
        julia_threads=1,
        blas_threads=1,
        blas_vendor=str(frame["blas_vendor"]),
    )


def _validate_progress(
    raw: str,
    *,
    request_sha256: str,
    attempt_sha256: str,
) -> Mapping[str, object] | None:
    try:
        frame = _decode_canonical_line(raw, stage="progress")
    except BackendProtocolError:
        return None
    if frame.get("schema") != "scnsim.progress":
        return None
    required = {
        "schema": "scnsim.progress",
        "schema_version": 1,
        "request_sha256": request_sha256,
        "attempt_sha256": attempt_sha256,
        "event": "optimization_generation_complete",
    }
    if any(frame.get(key) != value for key, value in required.items()) or set(frame) != {
        *required,
        "completed_generation",
        "completed_evaluations",
        "max_evaluations",
    } or any(
        not isinstance(frame.get(key), int) or frame[key] < 1
        for key in ("completed_generation", "completed_evaluations", "max_evaluations")
    ):
        raise BackendProtocolError(
            "child progress frame does not bind the authorized request and attempt",
            stage="progress",
            evidence={"frame": frame},
        )
    return frame


def _terminate_process_group(process: subprocess.Popen[str]) -> str:
    if process.poll() is not None:
        return "terminated"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "terminated"
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        return "killed_after_grace"
    return "terminated"


def _read_outcome(
    staging_directory: Path,
    *,
    request_sha256: str,
    attempt_sha256: str,
) -> Mapping[str, object]:
    path = staging_directory / "outcome.json"
    try:
        if (
            staging_directory.parent.is_symlink()
            or staging_directory.is_symlink()
            or not staging_directory.is_dir()
            or path.is_symlink()
            or not path.is_file()
        ):
            raise OSError("outcome.json is not a regular file")
        raw = path.read_text(encoding="utf-8")
        outcome = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise BackendProtocolError(
            "a completed child did not produce a readable outcome envelope",
            stage="outcome",
            evidence={"path": str(path), "error": str(error)},
        ) from error
    if not isinstance(outcome, dict) or outcome.get("schema") != "scnsim.outcome" or outcome.get(
        "schema_version"
    ) != 1 or outcome.get("request_sha256") != request_sha256 or outcome.get(
        "attempt_sha256"
    ) != attempt_sha256:
        raise BackendProtocolError(
            "child outcome envelope does not bind the authorized request and attempt",
            stage="outcome",
            evidence={"path": str(path)},
        )
    return outcome


def run_terminal(
    prepared: PreparedRuntime,
    *,
    request_path: str | os.PathLike[str],
    staging_directory: str | os.PathLike[str],
    request_sha256: str,
    attempt_ordinal: int,
    authorize: Callable[[BootstrapReady], str],
) -> TerminalOutcome:
    """Run exactly one authorized Julia request and return transport evidence.

    ``authorize`` is deliberately called only after a matching bootstrap frame:
    the workspace owner uses it to seal ``attempt.json`` and returns its hash.
    """

    if not _is_sha256(request_sha256) or not isinstance(attempt_ordinal, int) or attempt_ordinal < 1:
        raise BackendProtocolError(
            "terminal launch received invalid request identity or ordinal",
            stage="launch_arguments",
            evidence={"request_sha256": request_sha256, "attempt_ordinal": attempt_ordinal},
        )
    request = _require_absolute_file(request_path, label="request")
    staging = _require_absolute_directory(staging_directory, label="staging")
    stdout_lines: queue.Queue[str | None] = queue.Queue()
    stdout_log: list[str] = []
    stderr_log: list[str] = []
    reader_errors: list[BaseException] = []
    with packaged_julia_resources() as (project, entrypoint, runtime):
        expected_version = _runtime_version(runtime)
        if prepared.julia_version != expected_version:
            raise BackendProtocolError(
                "prepared Julia runtime changed after preflight",
                stage="runtime_identity_after_allocation",
                evidence={"prepared": prepared.julia_version, "expected": expected_version},
            )
        argv = _terminal_argv(prepared, project, entrypoint, request, staging)
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
                shell=False,
                cwd=str(project),
                env=_child_environment(),
                start_new_session=True,
            )
        except OSError as error:
            raise BackendProtocolError(
                "Julia child process could not be created after attempt allocation",
                stage="process_start",
                evidence={"argv": tuple(argv), "error": str(error)},
            ) from error
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        stdout_reader = threading.Thread(
            target=_read_lines,
            args=(process.stdout, stdout_lines, reader_errors),
            daemon=True,
        )
        stderr_reader = threading.Thread(
            target=_read_lines,
            args=(process.stderr, stderr_log, reader_errors),
            daemon=True,
        )
        stdout_reader.start()
        stderr_reader.start()
        try:
            first = stdout_lines.get()
            if first is None:
                raise _protocol_error(
                    "Julia child ended before its required bootstrap frame",
                    stage="bootstrap",
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    extra={
                        "returncode": process.poll(),
                        "reader_errors": tuple(str(error) for error in reader_errors),
                    },
                )
            bootstrap = _validate_bootstrap(
                first,
                request_sha256=request_sha256,
                attempt_ordinal=attempt_ordinal,
                expected_version=expected_version,
            )
            attempt_sha256 = authorize(bootstrap)
            if not _is_sha256(attempt_sha256):
                raise BackendProtocolError(
                    "workspace authorization did not return a canonical attempt hash",
                    stage="launch_authorization",
                    evidence={"attempt_sha256": attempt_sha256},
                )
            authorization = {
                "schema": "scnsim.launch_authorization",
                "schema_version": 1,
                "request_sha256": request_sha256,
                "attempt_sha256": attempt_sha256,
            }
            process.stdin.write(_canonical_json_line(authorization))
            process.stdin.close()
            progress: list[Mapping[str, object]] = []
            while True:
                line = stdout_lines.get()
                if line is None:
                    break
                if '"schema":"scnsim.bootstrap_ready"' in line:
                    raise _protocol_error(
                        "Julia child emitted a second bootstrap frame",
                        stage="protocol_stdout",
                        stdout_log=stdout_log,
                        stderr_log=stderr_log,
                    )
                event = _validate_progress(
                    line,
                    request_sha256=request_sha256,
                    attempt_sha256=attempt_sha256,
                )
                if event is None:
                    stdout_log.append(line)
                else:
                    progress.append(event)
            returncode = process.wait()
            stderr_reader.join()
            if reader_errors:
                raise _protocol_error(
                    "Julia child stream reader failed",
                    stage="process_transport",
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    extra={"reader_errors": tuple(str(error) for error in reader_errors)},
                )
            if returncode != 0:
                raise _protocol_error(
                    "Julia child exited unsuccessfully",
                    stage="process_exit",
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    extra={"returncode": returncode},
                )
            outcome = _read_outcome(
                staging,
                request_sha256=request_sha256,
                attempt_sha256=attempt_sha256,
            )
            return TerminalOutcome(
                outcome=outcome,
                progress=tuple(progress),
                stdout_log=tuple(stdout_log),
                stderr_log=tuple(stderr_log),
            )
        except KeyboardInterrupt as error:
            error.termination = _terminate_process_group(process)  # type: ignore[attr-defined]
            raise
        except BackendProtocolError:
            _terminate_process_group(process)
            raise
        except (BrokenPipeError, OSError, ValueError) as error:
            _terminate_process_group(process)
            raise _protocol_error(
                "Julia child transport failed",
                stage="process_transport",
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                extra={"error": str(error)},
            ) from error
        finally:
            if process.poll() is None:
                _terminate_process_group(process)
            stdout_reader.join()
            stderr_reader.join()


def run_preflight(
    prepared: PreparedRuntime,
    *,
    plan_path: str | os.PathLike[str],
    request_path: str | os.PathLike[str],
) -> Mapping[str, object]:
    """Compile one temp-backed bound request without attempt evidence."""

    plan = _require_absolute_file(plan_path, label="preflight plan")
    request = _require_absolute_file(request_path, label="preflight request")
    with packaged_julia_resources() as (project, entrypoint, runtime):
        expected_version = _runtime_version(runtime)
        if prepared.julia_version != expected_version:
            raise RuntimePreparationError(
                "prepared Julia runtime does not match packaged runtime metadata",
                stage="runtime_identity",
                evidence={"prepared": prepared.julia_version, "expected": expected_version},
            )
        argv = [
            str(prepared.executable),
            "--startup-file=no",
            "--history-file=no",
            "--threads=1",
            f"--project={project}",
            str(entrypoint),
            "--preflight",
            str(plan),
            "--request",
            str(request),
        ]
        try:
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                shell=False,
                cwd=str(project),
                env=_child_environment(),
            )
        except OSError as error:
            raise BackendProtocolError(
                "Julia preflight process could not be created",
                stage="preflight_start",
                evidence={"argv": tuple(argv), "error": str(error)},
            ) from error
    lines = completed.stdout.splitlines(keepends=True)
    if completed.returncode != 0 or len(lines) != 1:
        raise BackendProtocolError(
            "Julia preflight did not return exactly one successful protocol frame",
            stage="preflight",
            evidence={
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
    frame = _decode_canonical_line(lines[0], stage="preflight")
    if frame.get("schema") == "scnsim.preflight_failure" and frame.get("schema_version") == 1:
        if set(frame) != {"schema", "schema_version", "failure"} or not isinstance(frame.get("failure"), Mapping):
            raise BackendProtocolError(
                "Julia preflight returned a malformed typed failure frame",
                stage="preflight",
                evidence={"frame": frame},
            )
        return frame
    if frame.get("schema") != "scnsim.preflight" or frame.get("schema_version") != 1:
        raise BackendProtocolError(
            "Julia preflight returned an unexpected protocol frame",
            stage="preflight",
            evidence={"frame": frame},
        )
    return frame


__all__ = [
    "BootstrapReady",
    "PreparedRuntime",
    "TerminalOutcome",
    "packaged_julia_resources",
    "prepare_runtime",
    "run_preflight",
    "run_terminal",
]
