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
import sys
import threading
from time import perf_counter_ns
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
    _native_supervisor_scope,
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
    *,
    julia_threads: int = 1,
    entrypoint_arguments: tuple[str, ...] | None = None,
) -> list[str]:
    argv = [
        str(prepared.executable),
        "--startup-file=no",
        "--history-file=no",
        f"--threads={julia_threads}",
        f"--project={project}",
        str(entrypoint),
    ]
    if entrypoint_arguments is None:
        argv.extend(("--request", str(request_path), "--staging", str(staging_directory)))
    else:
        argv.extend(entrypoint_arguments)
    return argv


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


def _validate_bootstrap_for_threads(
    raw: str,
    *,
    request_sha256: str,
    attempt_ordinal: int,
    expected_version: str,
    hb_operation: bool,
    expected_julia_threads: int,
    expected_blas_threads: int,
) -> BootstrapReady:
    """Validate the opt-in benchmark thread profile; ordinary calls remain 1-thread."""
    if expected_julia_threads == 1 and expected_blas_threads == 1:
        return _validate_bootstrap(
            raw,
            request_sha256=request_sha256,
            attempt_ordinal=attempt_ordinal,
            expected_version=expected_version,
            hb_operation=hb_operation,
        )

    frame = _decode_canonical_line(raw, stage="bootstrap")
    required = {
        "schema": "scnsim.bootstrap_ready",
        "schema_version": 1,
        "request_sha256": request_sha256,
        "attempt_ordinal": attempt_ordinal,
        "julia_version": expected_version,
        "julia_threads": expected_julia_threads,
        "blas_threads": expected_blas_threads,
    }
    expected_fields = {*required, "blas_vendor"}
    if hb_operation:
        expected_fields.add("fftw_threads")
    if (
        set(frame) != expected_fields
        or any(frame.get(key) != value for key, value in required.items())
        or not isinstance(frame.get("blas_vendor"), str)
        or not frame["blas_vendor"]
        or (hb_operation and frame.get("fftw_threads") != 1)
    ):
        raise BackendProtocolError(
            "child bootstrap evidence does not match the sealed benchmark thread profile",
            stage="bootstrap",
            evidence={
                "frame": frame,
                "expected_julia_threads": expected_julia_threads,
                "expected_blas_threads": expected_blas_threads,
            },
        )
    return BootstrapReady(
        request_sha256=request_sha256,
        attempt_ordinal=attempt_ordinal,
        julia_version=expected_version,
        julia_threads=expected_julia_threads,
        blas_threads=expected_blas_threads,
        blas_vendor=str(frame["blas_vendor"]),
        fftw_threads=1 if hb_operation else None,
    )


def _observe_timing(
    observer: Callable[[str, int, int, Mapping[str, object]], object] | None,
    stage: str,
    start_ns: int,
    end_ns: int,
    **details: object,
) -> None:
    """Send one actual process interval through optional benchmark plumbing."""
    if observer is not None:
        observer(stage, start_ns, end_ns, details)


def _cpu_limited_argv(argv: list[str], profile: Mapping[str, object] | None) -> list[str]:
    """Exec the target only after the benchmark worker receives its CPU set."""
    if profile is None:
        return argv
    cpus = profile.get("cpus")
    topology = profile.get("topology")
    if cpus is None:
        return argv
    bootstrap = (
        "import json,os,sys; p=json.loads(sys.argv[1]); "
        "os.sched_setaffinity(0,set(p['cpus'])); "
        "a=sorted(os.sched_getaffinity(0)); "
        "os.environ['SCNSIM_BENCHMARK_CPU_AFFINITY']=json.dumps(" 
        "{'requested_cpus':p['cpus'],'actual_cpus':a,'topology':p['topology']},"
        "sort_keys=True,separators=(',',':')); "
        "v=sys.argv[2:]; os.execvpe(v[0],v,os.environ)"
    )
    return [
        sys.executable,
        "-S",
        "-c",
        bootstrap,
        json.dumps({"cpus": list(cpus), "topology": topology}, sort_keys=True, separators=(",", ":")),
        *argv,
    ]


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


def _next_stdout_line(lines, process, supervisor):
    """Observe root exit even when a descendant still holds stdout open."""
    while True:
        try:
            return lines.get(timeout=.05)
        except queue.Empty:
            if process.poll() is not None:
                supervisor.drain()


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
    _timing_observer: Callable[[str, int, int, Mapping[str, object]], object] | None = None,
    _entrypoint: str | os.PathLike[str] | None = None,
    _entrypoint_arguments: tuple[str, ...] | None = None,
    _julia_threads: int = 1,
    _julia_blas_threads: int = 1,
    _cpu_affinity: Mapping[str, object] | None = None,
    _preserve_progress_callback_exception: bool = False,
    native_supervisor=None,
) -> TerminalOutcome:
    """Run exactly one authorized Julia request and return transport evidence.

    ``authorize`` is deliberately called only after a matching bootstrap frame:
    the workspace owner uses it to seal ``attempt.json`` and returns its hash.
    """
    with _native_supervisor_scope(native_supervisor) as supervisor:

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
        if (_entrypoint is None) != (_entrypoint_arguments is None):
            raise BackendProtocolError(
                "alternate Julia entrypoint and argument vector must be supplied together",
                stage="launch_arguments",
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
            selected_entrypoint = entrypoint if _entrypoint is None else _require_absolute_file(
                _entrypoint, label="alternate Julia entrypoint"
            )
            argv = _terminal_argv(
                prepared,
                project,
                selected_entrypoint,
                request,
                staging,
                julia_threads=_julia_threads,
                entrypoint_arguments=_entrypoint_arguments,
            )
            child_environment = _child_environment()
            child_environment["JULIA_NUM_THREADS"] = str(_julia_threads)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                child_environment[name] = str(_julia_blas_threads)
            if _cpu_affinity is not None:
                child_environment["SCNSIM_BENCHMARK_CPU_AFFINITY"] = json.dumps(
                    {"requested_cpus": _cpu_affinity.get("cpus"), "actual_cpus": None,
                     "topology": _cpu_affinity.get("topology")}, sort_keys=True, separators=(",", ":")
                )
            launch_argv = _cpu_limited_argv(argv, _cpu_affinity)
            terminal_started_ns = perf_counter_ns()
            launch_started_ns = terminal_started_ns
            try:
                process = supervisor.popen(
                    launch_argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="strict",
                    bufsize=1,
                    shell=False,
                    cwd=str(project),
                    env=child_environment,
                    start_new_session=True,
                )
            except OSError as error:
                _observe_timing(
                    _timing_observer, "julia_process_start", launch_started_ns,
                    perf_counter_ns(), status="launch_error", error_type=type(error).__name__,
                )
                raise BackendProtocolError(
                    "Julia child process could not be created after attempt allocation",
                    stage="process_start",
                    evidence={"argv": tuple(argv), "error": str(error)},
                ) from error
            try:
                actual_affinity = tuple(sorted(os.sched_getaffinity(process.pid))) if hasattr(os, "sched_getaffinity") else None
            except OSError:
                actual_affinity = None
            _observe_timing(
                _timing_observer, "julia_process_start", launch_started_ns,
                perf_counter_ns(), status="started", pid=process.pid,
                cpu_affinity=actual_affinity,
                observation_boundary="after_popen_before_validated_bootstrap_ready",
            )
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
                ready_wait_started_ns = perf_counter_ns()
                first = _next_stdout_line(stdout_lines, process, supervisor)
                _observe_timing(
                    _timing_observer, "julia_ready_wait", ready_wait_started_ns,
                    perf_counter_ns(), status="received" if first is not None else "eof",
                )
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
                bootstrap = _validate_bootstrap_for_threads(
                    first,
                    request_sha256=request_sha256,
                    attempt_ordinal=attempt_ordinal,
                    expected_version=expected_version,
                    hb_operation=hb_operation,
                    expected_julia_threads=_julia_threads,
                    expected_blas_threads=_julia_blas_threads,
                )
                if _timing_observer is not None:
                    affinity_started_ns = perf_counter_ns()
                    bootstrap_affinity = None
                    affinity_status = "unavailable"
                    if hasattr(os, "sched_getaffinity"):
                        try:
                            bootstrap_affinity = tuple(
                                sorted(os.sched_getaffinity(process.pid))
                            )
                            affinity_status = "observed"
                        except OSError:
                            pass
                    _observe_timing(
                        _timing_observer,
                        "julia_bootstrap_affinity",
                        affinity_started_ns,
                        perf_counter_ns(),
                        status=affinity_status,
                        pid=process.pid,
                        cpu_affinity=bootstrap_affinity,
                        observation_boundary="validated_bootstrap_ready",
                        process_role="julia_child",
                    )
                authorization_started_ns = perf_counter_ns()
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
                _observe_timing(
                    _timing_observer, "julia_authorization", authorization_started_ns,
                    perf_counter_ns(), status="authorized", attempt_sha256=attempt_sha256,
                )
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
                    checkpoint_ack_started_ns = perf_counter_ns()
                    process.stdin.write(_canonical_json_line(committed))
                    process.stdin.flush()
                    process.stdin.close()
                    checkpoint_committed = True
                    _observe_timing(
                        _timing_observer, "julia_checkpoint_ack", checkpoint_ack_started_ns,
                        perf_counter_ns(), disposition="reused",
                    )
                elif sweep_operation:
                    recovery = {"schema": "scnsim.point_recovery", "schema_version": 1,
                        "request_sha256": request_sha256, "attempt_sha256": attempt_sha256,
                        "entries": list(point_recovery or ())}
                    process.stdin.write(_canonical_json_line(recovery))
                    process.stdin.flush()
                elif not optimization_operation:
                    process.stdin.close()
                while True:
                    line = _next_stdout_line(stdout_lines, process, supervisor)
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
                        checkpoint_started_ns = perf_counter_ns()
                        published = publish_point(ready)
                        if set(published) != {"record_sha256", "seal_sha256"} or published["record_sha256"] != ready["record_sha256"] or not _is_sha256(published["seal_sha256"]):
                            raise BackendProtocolError("point publisher returned inconsistent identity", stage="point_checkpoint")
                        process.stdin.write(_canonical_json_line({
                            "schema": "scnsim.point_checkpoint_committed", "schema_version": 1,
                            "request_sha256": request_sha256, "attempt_sha256": attempt_sha256,
                            "ordinal": ready["ordinal"], "record_sha256": ready["record_sha256"],
                            "seal_sha256": published["seal_sha256"]}))
                        process.stdin.flush()
                        _observe_timing(
                            _timing_observer, "julia_point_checkpoint", checkpoint_started_ns,
                            perf_counter_ns(), status="committed", ordinal=ready["ordinal"],
                        )
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
                        checkpoint_started_ns = perf_counter_ns()
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
                        _observe_timing(
                            _timing_observer, "julia_checkpoint_commit", checkpoint_started_ns,
                            perf_counter_ns(), status="committed",
                        )
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
                                if _preserve_progress_callback_exception:
                                    raise
                                raise OptimizationProgressCallbackError(
                                    "optimization progress callback failed",
                                    stage="progress_callback", evidence={"error_type": type(error).__name__},
                                ) from error
                process_wait_started_ns = perf_counter_ns()
                returncode = process.wait()
                _observe_timing(
                    _timing_observer, "julia_process_wait", process_wait_started_ns,
                    perf_counter_ns(), returncode=returncode,
                )
                supervisor.drain()
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
                outcome_decode_started_ns = perf_counter_ns()
                outcome = _read_outcome(
                    staging,
                    request_sha256=request_sha256,
                    attempt_sha256=attempt_sha256,
                )
                _observe_timing(
                    _timing_observer, "julia_outcome_read", outcome_decode_started_ns,
                    perf_counter_ns(), status=outcome.get("status", "unknown"),
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
                raise
            except BackendProtocolError:
                raise
            except (BrokenPipeError, OSError, ValueError) as error:
                raise _protocol_error(
                    "Julia child transport failed",
                    stage="process_transport",
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    extra={"error": str(error)},
                ) from error
            finally:
                original = sys.exception()
                cleanup_error = None
                def cleanup(action, label):
                    nonlocal cleanup_error
                    try:
                        return action()
                    except BaseException as secondary:
                        target = original if original is not None else cleanup_error
                        if target is None:
                            cleanup_error = secondary
                        else:
                            target.add_note(f"{label}: {secondary!r}")
                        return None

                # Drain root and detached descendants before reader joins, on
                # normal exit and every callback/transport/interruption exit.
                termination = cleanup(supervisor.drain, "Julia terminal native drain")
                if isinstance(original, KeyboardInterrupt) and termination is not None:
                    original.termination = termination
                cleanup(lambda: _observe_timing(
                    _timing_observer, "julia_terminal_transport", terminal_started_ns,
                    perf_counter_ns(), returncode=process.returncode,
                ), "Julia transport timing cleanup")
                for reader in (stdout_reader, stderr_reader):
                    while reader.is_alive():
                        cleanup(reader.join, "Julia stream reader join")
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None and not stream.closed:
                        cleanup(stream.close, "Julia stream close")
                if original is None and cleanup_error is not None:
                    raise cleanup_error


def run_preflight(
    prepared: PreparedRuntime,
    *,
    plan_path: str | os.PathLike[str],
    request_path: str | os.PathLike[str],
    native_supervisor=None,
) -> Mapping[str, object]:
    """Compile one temp-backed bound request without attempt evidence."""
    with _native_supervisor_scope(native_supervisor) as supervisor:

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
                completed = supervisor.run(
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
    native_supervisor=None,
) -> Mapping[str, object]:
    """Compile one resolved point and accept exactly one canonical audit frame."""
    with _native_supervisor_scope(native_supervisor) as supervisor:

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
                completed = supervisor.run(
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
