"""One-process Julia transport, cancellation, and compiler-only launches.

Terminal authorization and checkpoint publication are supplied by the workspace
coordinator; this owner returns transport facts, not trusted typed Results."""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from ..errors import (
    BackendProtocolError,
    OptimizationProgressCallbackError,
    RuntimePreparationError,
)
from .preparation import (
    PreparedRuntime,
    _child_environment,
    _runtime_version,
    packaged_julia_resources,
)
from .protocol import (
    BootstrapReady,
    TerminalOutcome,
    _canonical_json_line,
    _decode_canonical_line,
    _is_sha256,
    _read_outcome,
    _reserved_optimization_frame,
    _validate_bootstrap,
    _validate_checkpoint_ready,
    _validate_point_ready,
    _validate_progress,
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


def run_terminal(
    prepared: PreparedRuntime,
    *,
    request_path: str | os.PathLike[str],
    staging_directory: str | os.PathLike[str],
    request_sha256: str,
    attempt_ordinal: int,
    authorize: Callable[[BootstrapReady], str],
    checkpoint: Mapping[str, object] | None = None,
    publish_checkpoint: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    on_progress: Callable[[Mapping[str, object]], object] | None = None,
    point_recovery: tuple[Mapping[str, object], ...] | None = None,
    publish_point: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
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
    try:
        request_document = json.loads(request.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BackendProtocolError(
            "terminal launch could not read the sealed request operation",
            stage="launch_arguments",
            evidence={"request": str(request), "error": str(error)},
        ) from error
    if not isinstance(request_document, Mapping):
        raise BackendProtocolError(
            "terminal launch request is not an object",
            stage="launch_arguments",
            evidence={"request": str(request)},
        )
    hb_operation = request_document.get("operation") == "solve_hb"
    optimization_operation = request_document.get("operation") == "optimize_direct"
    sweep_operation = request_document.get("parameter_source", {}).get("kind") in {"grid", "points"}
    if sweep_operation != (point_recovery is not None and publish_point is not None):
        raise BackendProtocolError("sweep launch lacks exact point recovery controls", stage="launch_arguments")
    if (checkpoint is not None or publish_checkpoint is not None) and not optimization_operation:
        raise BackendProtocolError(
            "baseline checkpoint controls are valid only for optimization",
            stage="launch_arguments",
        )
    if optimization_operation and publish_checkpoint is None:
        raise BackendProtocolError(
            "optimization launch lacks its checkpoint publisher",
            stage="launch_arguments",
        )
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
                hb_operation=hb_operation,
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
            process.stdin.flush()
            checkpoint_committed = False
            if optimization_operation and checkpoint is not None:
                if (
                    set(checkpoint) != {"checkpoint_sha256", "seal_sha256"}
                    or not _is_sha256(checkpoint.get("checkpoint_sha256"))
                    or not _is_sha256(checkpoint.get("seal_sha256"))
                ):
                    raise BackendProtocolError(
                        "reused checkpoint identity is malformed",
                        stage="optimization_checkpoint",
                    )
                committed = {
                    "schema": "scnsim.optimization_checkpoint_committed",
                    "schema_version": 1,
                    "event": "baseline_checkpoint_committed",
                    "request_sha256": request_sha256,
                    "attempt_sha256": attempt_sha256,
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "seal_sha256": checkpoint["seal_sha256"],
                    "disposition": "reused",
                }
                process.stdin.write(_canonical_json_line(committed))
                process.stdin.flush()
                process.stdin.close()
                checkpoint_committed = True
            elif sweep_operation:
                recovery = {"schema": "scnsim.point_recovery", "schema_version": 1,
                    "request_sha256": request_sha256, "attempt_sha256": attempt_sha256,
                    "entries": list(point_recovery or ())}
                process.stdin.write(_canonical_json_line(recovery))
                process.stdin.flush()
            elif not optimization_operation:
                process.stdin.close()
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
                if re.search(r'"schema"\s*:\s*"scnsim\.point_checkpoint[^\"]*"', line):
                    if not sweep_operation:
                        raise BackendProtocolError("unexpected point checkpoint frame", stage="point_checkpoint")
                    ready = _validate_point_ready(line, request_sha256=request_sha256,
                        attempt_sha256=attempt_sha256)
                    assert publish_point is not None
                    published = publish_point(ready)
                    if set(published) != {"record_sha256", "seal_sha256"} or published["record_sha256"] != ready["record_sha256"] or not _is_sha256(published["seal_sha256"]):
                        raise BackendProtocolError("point publisher returned inconsistent identity", stage="point_checkpoint")
                    process.stdin.write(_canonical_json_line({
                        "schema": "scnsim.point_checkpoint_committed", "schema_version": 1,
                        "request_sha256": request_sha256, "attempt_sha256": attempt_sha256,
                        "ordinal": ready["ordinal"], "record_sha256": ready["record_sha256"],
                        "seal_sha256": published["seal_sha256"]}))
                    process.stdin.flush()
                    continue
                if _reserved_optimization_frame(line):
                    if not optimization_operation or checkpoint_committed:
                        raise _protocol_error(
                            "child emitted an unexpected optimization checkpoint frame",
                            stage="optimization_checkpoint",
                            stdout_log=stdout_log,
                            stderr_log=stderr_log,
                        )
                    ready = _validate_checkpoint_ready(
                        line,
                        request_sha256=request_sha256,
                        attempt_sha256=attempt_sha256,
                    )
                    assert publish_checkpoint is not None
                    published = publish_checkpoint(ready)
                    if (
                        set(published) != {"checkpoint_sha256", "seal_sha256"}
                        or published.get("checkpoint_sha256") != ready["checkpoint_sha256"]
                        or not _is_sha256(published.get("seal_sha256"))
                    ):
                        raise BackendProtocolError(
                            "published checkpoint identity disagrees with ready frame",
                            stage="optimization_checkpoint",
                        )
                    committed = {
                        "schema": "scnsim.optimization_checkpoint_committed",
                        "schema_version": 1,
                        "event": "baseline_checkpoint_committed",
                        "request_sha256": request_sha256,
                        "attempt_sha256": attempt_sha256,
                        "checkpoint_sha256": published["checkpoint_sha256"],
                        "seal_sha256": published["seal_sha256"],
                        "disposition": "published",
                    }
                    process.stdin.write(_canonical_json_line(committed))
                    process.stdin.flush()
                    process.stdin.close()
                    checkpoint_committed = True
                    continue
                event = _validate_progress(
                    line,
                    request_sha256=request_sha256,
                    attempt_sha256=attempt_sha256,
                )
                if event is None:
                    stdout_log.append(line)
                else:
                    if optimization_operation and not checkpoint_committed:
                        raise _protocol_error(
                            "optimization progress preceded baseline checkpoint commit",
                            stage="optimization_checkpoint",
                            stdout_log=stdout_log,
                            stderr_log=stderr_log,
                        )
                    if optimization_operation:
                        controls = request_document["spec"]["optimizer"]
                        if (event["total_generations"] != controls["complete_generations"] or
                            event["requested_budget"] != controls["max_evaluations"] or
                            event["achievable_evaluations"] != 1 + controls["complete_generations"] * controls["resolved_population_size"] or
                            event["evaluated_count"] != 1 + event["completed_generations"] * controls["resolved_population_size"]):
                            raise BackendProtocolError("optimization progress disagrees with request budget", stage="progress")
                    if on_progress is not None:
                        try:
                            on_progress(event)
                        except KeyboardInterrupt:
                            raise
                        except Exception as error:
                            raise OptimizationProgressCallbackError(
                                "optimization progress callback failed",
                                stage="progress_callback", evidence={"error_type": type(error).__name__},
                            ) from error
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
            if optimization_operation and not checkpoint_committed and outcome.get("status") == "success":
                raise _protocol_error(
                    "optimization succeeded without a committed baseline checkpoint",
                    stage="optimization_checkpoint",
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                )
            return TerminalOutcome(
                outcome=outcome,
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
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()


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
    if frame.get("schema") != "scnsim.preflight" or frame.get("schema_version") != 2:
        raise BackendProtocolError(
            "Julia preflight returned an unexpected protocol frame",
            stage="preflight",
            evidence={"frame": frame},
        )
    return frame


def run_compiler_audit(
    prepared: PreparedRuntime,
    *,
    plan_path: str | os.PathLike[str],
    point_path: str | os.PathLike[str],
) -> Mapping[str, object]:
    """Compile one resolved point and accept exactly one canonical audit frame."""

    plan = _require_absolute_file(plan_path, label="compiler-audit plan")
    point = _require_absolute_file(point_path, label="compiler-audit point")
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
            "--compiler-audit",
            str(plan),
            "--point",
            str(point),
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
        except (OSError, UnicodeError) as error:
            raise BackendProtocolError(
                "compiler-audit process could not be executed",
                stage="compiler_audit",
                evidence={"error": str(error)},
            ) from error
    if completed.returncode != 0:
        raise BackendProtocolError(
            "compiler-audit process exited unsuccessfully",
            stage="compiler_audit",
            evidence={"returncode": completed.returncode, "stderr": completed.stderr[-4096:]},
        )
    try:
        lines = completed.stdout.splitlines()
        if len(lines) != 1:
            raise ValueError("expected one stdout frame")
        frame = json.loads(lines[0])
        if not isinstance(frame, dict) or completed.stdout != _canonical_json_line(frame):
            raise ValueError("stdout frame is not canonical")
    except (ValueError, json.JSONDecodeError) as error:
        raise BackendProtocolError(
            "compiler-audit output is malformed",
            stage="compiler_audit",
            evidence={"error": str(error)},
        ) from error
    return frame
