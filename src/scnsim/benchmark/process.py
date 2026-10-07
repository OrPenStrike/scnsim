"""Authenticated Python task child transport for isolated benchmark arms."""

from __future__ import annotations

import builtins
import json
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from ..canonical import canonical_json_bytes
from ..errors import BackendProtocolError, SCNSimError
from ..execution.preparation import _child_environment
from ..execution.process import _read_lines, _terminate_process_group
from ..execution.protocol import _canonical_json_line, _decode_canonical_line
from .identity import (
    _native_thread_spec,
    environment_identity,
    environment_snapshot,
    task_identifier,
    task_request_identity,
)
from .prepared import PreparedBenchmark, record_data, record_document


_REQUEST_SCHEMA = "scnsim.benchmark_python_task_request"
_EVENT_SCHEMA = "scnsim.benchmark_task_event"
_AUTH_SCHEMA = "scnsim.benchmark_task_authorization"
_CHECKPOINT_ACK_SCHEMA = "scnsim.benchmark_checkpoint_committed"
_EVENT_KINDS = frozenset({
    "ready", "authorized", "started", "timing", "baseline_ready",
    "checkpoint_committed", "population_observed", "generation_ready",
    "evaluation", "completed", "interrupted", "failed",
})

_PYTHON_CHILD_BOOTSTRAP = """
import json, os, runpy, site, sys
profile = json.loads(sys.argv[1])
cpus = profile.get('cpus')
if cpus is not None and hasattr(os, 'sched_setaffinity'):
    os.sched_setaffinity(0, set(cpus))
actual = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
os.environ['SCNSIM_BENCHMARK_CPU_AFFINITY'] = json.dumps({
    'requested_cpus': cpus, 'actual_cpus': actual,
    'topology': profile.get('topology'),
}, sort_keys=True, separators=(',', ':'))
request_path = sys.argv[2]
site.main()
sys.argv = ['scnsim.benchmark.process', '--child', request_path]
runpy.run_module('scnsim.benchmark.process', run_name='__main__')
""".strip()


def cpu_affinity_profile(cpu_threads: int) -> dict[str, object]:
    """Select one allowed logical CPU per physical core where topology exists."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return {"cpus": None, "topology": "affinity_unavailable", "selected_count": None}
    allowed = sorted(os.sched_getaffinity(0))
    by_core: dict[tuple[int, int], list[int]] = {}
    topology_available = bool(allowed)
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((topology / "physical_package_id").read_text(encoding="ascii"))
            core = int((topology / "core_id").read_text(encoding="ascii"))
        except (OSError, ValueError):
            topology_available = False
            break
        by_core.setdefault((package, core), []).append(cpu)
    if topology_available:
        candidates = [min(values) for _, values in sorted(by_core.items())]
        topology_name = "physical_core_representatives"
    else:
        candidates = allowed
        topology_name = "logical_cpus_topology_unavailable"
    selected = candidates[:cpu_threads]
    return {"cpus": selected, "topology": topology_name, "selected_count": len(selected)}


def _protocol_error(message: str, *, stage: str, evidence: Mapping[str, object] | None = None) -> BackendProtocolError:
    return BackendProtocolError(message, stage=stage, evidence=evidence or {})


def _json_line(value: Mapping[str, object]) -> str:
    def plain(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(child) for child in item]
        return item
    return _canonical_json_line(record_data(plain(dict(value))))


def _canonical_object(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise _protocol_error("Python task request must be an absolute regular file", stage="launch_arguments",
                              evidence={"path": str(path)})
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _protocol_error("Python task request cannot be read", stage="launch_arguments",
                              evidence={"path": str(path), "error": str(error)}) from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise _protocol_error("Python task request is not canonical JSON", stage="launch_arguments",
                              evidence={"path": str(path)})
    return record_document(raw)


def _validate_event(
    raw: str,
    *,
    expected_task_id: str | None,
    expected_sequence: int,
) -> dict[str, Any]:
    frame = _decode_canonical_line(raw, stage="benchmark_child_protocol")
    if (
        set(frame) != {"schema", "schema_version", "task_id", "sequence", "kind", "payload"}
        or frame.get("schema") != _EVENT_SCHEMA
        or frame.get("schema_version") != 1
        or not isinstance(frame.get("task_id"), str)
        or not isinstance(frame.get("sequence"), int)
        or frame.get("sequence") != expected_sequence
        or frame.get("kind") not in _EVENT_KINDS
        or not isinstance(frame.get("payload"), dict)
    ):
        raise _protocol_error("Python task child emitted a malformed event", stage="benchmark_child_protocol",
                              evidence={"sequence": expected_sequence})
    if expected_task_id is not None and frame["task_id"] != expected_task_id:
        raise _protocol_error("Python task child changed task identity", stage="benchmark_child_protocol",
                              evidence={"expected_task_id": expected_task_id, "observed_task_id": frame["task_id"]})
    return record_document(canonical_json_bytes(dict(frame)))


def _ready_context(
    payload: Mapping[str, object],
    *,
    benchmark_sha256: str,
    expected_arm: str,
    expected_sample: int,
    expected_attempt_id: str,
    expected_device: str,
    expected_cpu_threads: int,
    expected_julia_threads: int | None = None,
    expected_julia_blas_threads: int | None = None,
    mesh_identity: Mapping[str, object],
    expected_task_id: str | None = None,
) -> dict[str, object]:
    context = payload["context"]
    environment = payload["environment"]
    backend = payload["backend"]
    if not isinstance(context, dict) or not isinstance(environment, dict) or not isinstance(backend, dict):
        raise _protocol_error("Python task ready event lacks identities", stage="benchmark_child_protocol")
    if (
        context.get("arm") != expected_arm
        or context.get("sample") != expected_sample
        or context.get("attempt_id") != expected_attempt_id
        or context.get("device") != str(backend.get("device", expected_device))
        or context.get("cpu_threads") != expected_cpu_threads
        or environment.get("arm") != expected_arm
        or environment.get("device_requested") != expected_device
        or environment.get("cpu_threads_requested") != expected_cpu_threads
        or (
            expected_julia_threads is not None
            and environment.get("julia_threads_requested") != expected_julia_threads
        )
        or (
            expected_julia_blas_threads is not None
            and environment.get("julia_blas_threads_requested") != expected_julia_blas_threads
        )
        or (
            expected_julia_threads is not None
            and backend.get("requested_julia_threads") != expected_julia_threads
        )
        or (
            expected_julia_blas_threads is not None
            and backend.get("requested_julia_blas_threads") != expected_julia_blas_threads
        )
    ):
        raise _protocol_error("Python task child ready identity differs from its launch", stage="benchmark_child_protocol")
    observed_environment_sha = environment_identity(environment)
    request_sha = task_request_identity(
        benchmark_sha256=benchmark_sha256,
        arm=expected_arm,
        environment_sha256=observed_environment_sha,
        device=str(context["device"]),
        cpu_threads=expected_cpu_threads,
        backend_identity=backend,
        mesh_identity=mesh_identity,
    )
    task_id = task_identifier(request_sha256=request_sha, sample=expected_sample)
    if (
        context.get("environment_sha256") != observed_environment_sha
        or context.get("request_sha256") != request_sha
        or context.get("task_id") != task_id
        or (expected_task_id is not None and task_id != expected_task_id)
    ):
        raise _protocol_error("Python task child task hash does not match its evidence", stage="benchmark_child_protocol")
    return dict(context)


def _send_parent_frame(process: subprocess.Popen[str], frame: Mapping[str, object]) -> None:
    if process.stdin is None:
        raise _protocol_error("Python task child stdin is unavailable", stage="benchmark_child_protocol")
    process.stdin.write(_json_line(frame))
    process.stdin.flush()


def _read_parent_frame() -> Mapping[str, object]:
    line = sys.stdin.readline()
    if not line:
        raise _protocol_error("Python task child authorization stream ended", stage="benchmark_child_protocol")
    return _decode_canonical_line(line, stage="benchmark_child_protocol")


def _emit_child(
    *,
    task_id: str,
    process_stdout: Any,
    next_ack: Callable[[int], Mapping[str, object]],
    sequence_ref: list[int],
) -> Callable[[str, dict[str, object]], None]:
    def emit(kind: str, payload: dict[str, object]) -> None:
        if kind not in _EVENT_KINDS - {"ready", "authorized", "started", "checkpoint_committed", "completed", "interrupted", "failed"}:
            raise ValueError(f"unsupported Python task event: {kind!r}")
        frame = {
            "schema": _EVENT_SCHEMA,
            "schema_version": 1,
            "task_id": task_id,
            "sequence": sequence_ref[0],
            "kind": kind,
            "payload": payload,
        }
        process_stdout.write(_json_line(frame))
        process_stdout.flush()
        current = sequence_ref[0]
        sequence_ref[0] += 1
        if kind in {"baseline_ready", "generation_ready"}:
            ack = next_ack(current)
            reference = ack["reference"]
            if not isinstance(reference, dict) or not isinstance(reference.get("evidence"), dict):
                raise _protocol_error("Python task evidence acknowledgement is malformed", stage="benchmark_checkpoint")
            checkpoint = reference.get("checkpoint")
            if checkpoint is not None:
                committed = {
                    "schema": _EVENT_SCHEMA,
                    "schema_version": 1,
                    "task_id": task_id,
                    "sequence": sequence_ref[0],
                    "kind": "checkpoint_committed",
                    "payload": {"evidence": reference["evidence"], "checkpoint": checkpoint},
                }
                process_stdout.write(_json_line(committed))
                process_stdout.flush()
                sequence_ref[0] += 1

    return emit


def _remote_failure(error: BaseException) -> dict[str, object]:
    if isinstance(error, SCNSimError):
        return {
            "type": "scnsim",
            "kind": error.kind,
            "category": error.category,
            "stage": error.stage,
            "message": str(error),
            "evidence": dict(error.evidence),
        }
    record: dict[str, object] = {
        "type": "python",
        "module": type(error).__module__,
        "exception": type(error).__name__,
        "message": str(error),
    }
    if isinstance(error, ImportError):
        record["name"] = error.name
    return record


def _child_main(request_path: str) -> int:
    request = _canonical_object(Path(request_path))
    if request.get("schema") != _REQUEST_SCHEMA or request.get("schema_version") != 1:
        raise _protocol_error("Python task request has an unsupported schema", stage="launch_arguments")
    prepared = PreparedBenchmark.from_wire(request["prepared"])
    if prepared.request_sha256 != request["benchmark_sha256"]:
        raise _protocol_error("Python task request changed benchmark identity", stage="launch_arguments")
    arm = request["arm"]
    sample = request["sample"]
    attempt_id = request["attempt_id"]
    device = request["device"]
    cpu_threads = request["cpu_threads"]
    native_threads = (
        _native_thread_spec(
            cpu_threads,
            julia_threads=request.get("julia_threads"),
            julia_blas_threads=request.get("julia_blas_threads"),
        )
        if arm in {"python_julia_reuse", "python_julia_lu"}
        else None
    )
    from .task import create_python_backend

    backend = None
    try:
        try:
            backend = create_python_backend(
                arm=arm, device=device, cpu_threads=cpu_threads,
                julia_threads=None if native_threads is None else native_threads.julia_threads,
                julia_blas_threads=None if native_threads is None else native_threads.blas_threads,
            )
            backend_identity = backend.identity()
            actual_device = str(backend_identity.get("device", device))
            environment = environment_snapshot(
                arm=arm, device=device, cpu_threads=cpu_threads,
                backend=backend_identity, native_threads=native_threads,
            )
            environment_sha = environment_identity(environment)
            mesh_identity = prepared.declaration()["benchmark"]["mesh"]
            request_sha = task_request_identity(
                benchmark_sha256=prepared.request_sha256,
                arm=arm,
                environment_sha256=environment_sha,
                device=actual_device,
                cpu_threads=cpu_threads,
                backend_identity=backend_identity,
                mesh_identity=mesh_identity,
            )
            task_id = task_identifier(request_sha256=request_sha, sample=sample)
        except BaseException as error:
            sys.stdout.write(_json_line({
                "schema": _EVENT_SCHEMA,
                "schema_version": 1,
                "task_id": "",
                "sequence": 0,
                "kind": "failed",
                "payload": {
                    "arm": arm,
                    "sample": sample,
                    "attempt_id": attempt_id,
                    "failure": _remote_failure(error),
                },
            }))
            sys.stdout.flush()
            return 1
        context = {
            "task_id": task_id,
            "request_sha256": request_sha,
            "arm": arm,
            "sample": sample,
            "attempt_id": attempt_id,
            "environment_sha256": environment_sha,
            "device": actual_device,
            "cpu_threads": cpu_threads,
        }
        sys.stdout.write(_json_line({
            "schema": _EVENT_SCHEMA,
            "schema_version": 1,
            "task_id": task_id,
            "sequence": 0,
            "kind": "ready",
            "payload": {"context": context, "environment": environment, "backend": backend_identity},
        }))
        sys.stdout.flush()
        authorization = _read_parent_frame()
        if (
            authorization.get("schema") != _AUTH_SCHEMA
            or authorization.get("schema_version") != 1
            or authorization.get("task_id") != task_id
            or authorization.get("request_sha256") != request_sha
            or authorization.get("attempt_id") != attempt_id
        ):
            raise _protocol_error("Python task authorization does not match ready identity", stage="launch_authorization")
        def next_ack(event_sequence: int) -> Mapping[str, object]:
            frame = _read_parent_frame()
            if (
                set(frame) != {"schema", "schema_version", "task_id", "event_sequence", "reference"}
                or frame.get("schema") != _CHECKPOINT_ACK_SCHEMA
                or frame.get("schema_version") != 1
                or frame.get("task_id") != task_id
                or frame.get("event_sequence") != event_sequence
                or not isinstance(frame.get("reference"), dict)
            ):
                raise _protocol_error("Python task checkpoint acknowledgement is malformed", stage="benchmark_checkpoint")
            return frame

        sequence_ref = [1]
        started = {"schema": _EVENT_SCHEMA, "schema_version": 1, "task_id": task_id,
                   "sequence": sequence_ref[0], "kind": "started", "payload": {"attempt_id": attempt_id}}
        sys.stdout.write(_json_line(started))
        sys.stdout.flush()
        sequence_ref[0] += 1
        emit = _emit_child(task_id=task_id, process_stdout=sys.stdout, next_ack=next_ack,
                           sequence_ref=sequence_ref)
        checkpoint = request.get("checkpoint")
        from .api import execute_python_task

        try:
            result = execute_python_task(prepared, backend, emit=emit, checkpoint=checkpoint)
        except KeyboardInterrupt:
            frame = {"schema": _EVENT_SCHEMA, "schema_version": 1, "task_id": task_id,
                     "sequence": sequence_ref[0], "kind": "interrupted",
                     "payload": {"kind": "keyboard_interrupt", "attempt_id": attempt_id}}
            sys.stdout.write(_json_line(frame))
            sys.stdout.flush()
            sequence_ref[0] += 1
            return 130
        except BaseException as error:
            frame = {"schema": _EVENT_SCHEMA, "schema_version": 1, "task_id": task_id,
                     "sequence": sequence_ref[0], "kind": "failed",
                     "payload": {"attempt_id": attempt_id, "failure": _remote_failure(error)}}
            sys.stdout.write(_json_line(frame))
            sys.stdout.flush()
            sequence_ref[0] += 1
            return 1
        frame = {"schema": _EVENT_SCHEMA, "schema_version": 1, "task_id": task_id,
                 "sequence": sequence_ref[0], "kind": "completed", "payload": {"attempt_id": attempt_id, "result": result}}
        sys.stdout.write(_json_line(frame))
        sys.stdout.flush()
        return 0
    finally:
        if backend is not None:
            backend.close()


def _restore_remote_failure(record: Mapping[str, object]) -> BaseException:
    if record.get("type") == "scnsim":
        from .. import errors
        for candidate in vars(errors).values():
            if isinstance(candidate, type) and issubclass(candidate, SCNSimError) and candidate.kind == record.get("kind"):
                return candidate(
                    str(record["message"]), stage=str(record["stage"]),
                    evidence=record.get("evidence", {}),
                )
        return SCNSimError(str(record["message"]), stage=str(record["stage"]), evidence=record.get("evidence", {}))
    name = record.get("exception")
    candidate = getattr(builtins, str(name), RuntimeError)
    if not isinstance(candidate, type) or not issubclass(candidate, BaseException):
        candidate = RuntimeError
    message = str(record.get("message", name or "Python benchmark task failed"))
    if candidate is ModuleNotFoundError:
        return ModuleNotFoundError(message, name=record.get("name"))
    if candidate is ImportError:
        return ImportError(message, name=record.get("name"))
    return candidate(message)


def run_python_child(
    request_path: str | os.PathLike[str],
    *,
    arm: str,
    sample: int,
    attempt_id: str,
    benchmark_sha256: str,
    device: str,
    cpu_threads: int,
    expected_julia_threads: int | None = None,
    expected_julia_blas_threads: int | None = None,
    mesh_identity: Mapping[str, object],
    on_event: Callable[[Mapping[str, object]], Mapping[str, object] | None],
    timing: object | None = None,
    expected_task_id: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Launch one exact Python task and synchronously acknowledge checkpoints."""
    request = Path(request_path)
    if not request.is_absolute() or request.is_symlink() or not request.is_file():
        raise _protocol_error("Python child request path must be an absolute regular file", stage="launch_arguments",
                              evidence={"path": str(request)})
    environment = _child_environment()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(cpu_threads)
    affinity = cpu_affinity_profile(cpu_threads)
    argv = [
        sys.executable,
        "-S",
        "-c",
        _PYTHON_CHILD_BOOTSTRAP,
        json.dumps(affinity, sort_keys=True, separators=(",", ":")),
        str(request),
    ]
    stdout_queue: queue.Queue[str | None] = queue.Queue()
    stderr_lines: list[str] = []
    reader_errors: list[BaseException] = []
    task_id: str | None = None
    pending_intervals: list[tuple[str, int, int, dict[str, object]]] = []

    def observe(stage: str, start_ns: int, end_ns: int, details: Mapping[str, object]) -> None:
        if timing is None:
            return
        if task_id is None:
            pending_intervals.append((stage, start_ns, end_ns, dict(details)))
            return
        timing.add_interval(stage, task_id=task_id, start_tick_ns=start_ns,
                            end_tick_ns=end_ns, details=details)

    def flush_pending(target_task_id: str) -> None:
        if timing is None:
            return
        for stage, start_ns, end_ns, details in pending_intervals:
            timing.add_interval(stage, task_id=target_task_id, start_tick_ns=start_ns,
                                end_tick_ns=end_ns, details=details)
        pending_intervals.clear()

    launch_start = perf_counter_ns()
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
            env=environment,
            start_new_session=True,
        )
    except OSError as error:
        if timing is not None:
            timing.add_interval("python_process_start", task_id="benchmark", start_tick_ns=launch_start,
                                end_tick_ns=perf_counter_ns(), details={"status": "launch_error", "error_type": type(error).__name__})
        raise BackendProtocolError("Python task child could not be created", stage="process_start",
                                   evidence={"argv": tuple(argv), "error": str(error)}) from error
    observe("python_process_start", launch_start, perf_counter_ns(),
            {"status": "started", "pid": process.pid})
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    stdout_reader = threading.Thread(target=_read_lines, args=(process.stdout, stdout_queue, reader_errors), daemon=True)
    stderr_reader = threading.Thread(target=_read_lines, args=(process.stderr, stderr_lines, reader_errors), daemon=True)
    stdout_reader.start()
    stderr_reader.start()
    context: dict[str, object] | None = None
    completion: dict[str, object] | None = None
    remote_failure: Mapping[str, object] | None = None
    interrupted = False
    expected_sequence = 0
    terminal_start = perf_counter_ns()
    try:
        ready_start = perf_counter_ns()
        line = stdout_queue.get()
        observe("python_child_ready", ready_start, perf_counter_ns(),
                {"status": "received" if line is not None else "eof"})
        if line is None:
            raise _protocol_error("Python task child exited before ready", stage="benchmark_child_ready",
                                  evidence={"returncode": process.poll(), "stderr": tuple(stderr_lines)})
        frame = _validate_event(line, expected_task_id=None, expected_sequence=expected_sequence)
        if frame["kind"] == "failed" and frame["task_id"] == "":
            flush_pending("benchmark")
            on_event(frame)
            remote_failure = frame["payload"]["failure"]
            expected_sequence += 1
            process.wait()
            raise _restore_remote_failure(remote_failure)
        if frame["kind"] != "ready":
            raise _protocol_error("Python task child did not begin with ready", stage="benchmark_child_ready")
        task_id = frame["task_id"]
        payload = frame["payload"]
        context = _ready_context(
            payload,
            benchmark_sha256=benchmark_sha256,
            expected_arm=arm,
            expected_sample=sample,
            expected_attempt_id=attempt_id,
            expected_device=device,
            expected_cpu_threads=cpu_threads,
            expected_julia_threads=expected_julia_threads,
            expected_julia_blas_threads=expected_julia_blas_threads,
            mesh_identity=mesh_identity,
            expected_task_id=expected_task_id,
        )
        flush_pending(task_id)
        expected_sequence += 1
        authorization_start = perf_counter_ns()
        ready_ack = on_event(frame)
        if ready_ack is None:
            raise _protocol_error("Python task ready event was not durably authorized", stage="launch_authorization")
        try:
            _send_parent_frame(process, {
                "schema": _AUTH_SCHEMA,
                "schema_version": 1,
                "task_id": task_id,
                "request_sha256": context["request_sha256"],
                "attempt_id": attempt_id,
            })
        except BrokenPipeError as error:
            # The child may have exited while the ready event was being committed.
            # Collect its terminal streams before reporting the parent write failure.
            returncode = process.wait()
            stdout_reader.join()
            stderr_reader.join()
            trailing_stdout: list[str] = []
            while True:
                try:
                    line = stdout_queue.get_nowait()
                except queue.Empty:
                    break
                if line is not None:
                    trailing_stdout.append(line)
            raise _protocol_error(
                "Python task child exited before authorization was accepted",
                stage="launch_authorization",
                evidence={
                    "returncode": returncode,
                    "stderr": tuple(stderr_lines),
                    "stdout": tuple(trailing_stdout),
                    "reader_errors": tuple(str(item) for item in reader_errors),
                    "authorization_error": str(error),
                },
            ) from error
        observe("python_child_authorization", authorization_start, perf_counter_ns(),
                {"attempt_id": attempt_id})
        terminal_seen = False
        while True:
            line = stdout_queue.get()
            if line is None:
                break
            frame = _validate_event(line, expected_task_id=task_id, expected_sequence=expected_sequence)
            expected_sequence += 1
            kind = frame["kind"]
            if terminal_seen:
                raise _protocol_error("Python task child emitted after a terminal event", stage="benchmark_child_protocol")
            if kind in {"completed", "failed", "interrupted"}:
                terminal_seen = True
            ack = on_event(frame)
            if kind in {"baseline_ready", "generation_ready"}:
                if ack is None:
                    raise _protocol_error("Python task checkpoint was not committed", stage="benchmark_checkpoint")
                _send_parent_frame(process, {
                    "schema": _CHECKPOINT_ACK_SCHEMA,
                    "schema_version": 1,
                    "task_id": task_id,
                    "event_sequence": frame["sequence"],
                    "reference": dict(ack),
                })
            elif kind == "completed":
                completion = frame["payload"]
            elif kind == "failed":
                remote_failure = frame["payload"]["failure"]
            elif kind == "interrupted":
                interrupted = True
        returncode = process.wait()
        stderr_reader.join()
        if reader_errors:
            raise _protocol_error("Python task process stream reader failed", stage="process_transport",
                                  evidence={"errors": tuple(str(error) for error in reader_errors)})
        if interrupted:
            raise KeyboardInterrupt
        if remote_failure is not None:
            raise _restore_remote_failure(remote_failure)
        if returncode != 0:
            raise _protocol_error("Python task child exited unsuccessfully", stage="process_exit",
                                  evidence={"returncode": returncode, "stderr": tuple(stderr_lines)})
        if completion is None or context is None:
            raise _protocol_error("Python task child exited without completed result", stage="benchmark_child_result",
                                  evidence={"returncode": returncode, "stderr": tuple(stderr_lines)})
        if completion.get("attempt_id") != attempt_id or not isinstance(completion.get("result"), dict):
            raise _protocol_error("Python task completed event has a malformed result", stage="benchmark_child_result")
        return context, completion["result"]
    except KeyboardInterrupt as error:
        error.termination = _terminate_process_group(process)  # type: ignore[attr-defined]
        raise
    except BaseException:
        _terminate_process_group(process)
        raise
    finally:
        if process.poll() is None:
            _terminate_process_group(process)
        observe("python_child_process", terminal_start, perf_counter_ns(),
                {"returncode": process.poll()})
        if task_id is None:
            flush_pending("benchmark")
        stdout_reader.join()
        stderr_reader.join()
        active_error = sys.exc_info()[0] is not None
        for stream in (process.stdin, process.stdout, process.stderr):
            if not stream.closed:
                try:
                    stream.close()
                except OSError:
                    if not active_error:
                        raise


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        return _child_main(sys.argv[2])
    raise SystemExit("usage: python -m scnsim.benchmark.process --child REQUEST.json")


if __name__ == "__main__":
    raise SystemExit(main())
