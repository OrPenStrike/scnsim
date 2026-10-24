"""Generate and verify source-bound Chapter 8 two-kernel evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from engineer_publication_binding import check_publication_binding, installed_course_runtime

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "examples" / "engineer" / "chapter-08"
FIGURES = ROOT / "examples" / "engineer" / "figures"
MANIFEST = FIGURES / "chapter-08-artifacts.json"
SOURCES = tuple(CHAPTER / name for name in ("_01-persist.qmd", "_02-resolve.qmd"))
WRAPPERS = tuple(CHAPTER / name for name in ("01-persist.qmd", "02-resolve.qmd"))
EXPECTED_CELL_IDS = (
    "ch8-build-persisted-plan", "ch8-prepare-persisted-requests",
    "ch8-execute-persisted-results", "ch8-build-result-report",
    "ch8-rebuild-plan-after-restart", "ch8-rebuild-request-after-restart",
    "ch8-resolve-only-after-restart",
)
FIRST_KERNEL_IDS = EXPECTED_CELL_IDS[:4]
SECOND_KERNEL_IDS = EXPECTED_CELL_IDS[4:]
ARTIFACT_NAMES = (
    "08-direct-s11.svg", "08-persisted-results.json", "08-persisted-results.md",
    "08-report.html", "08-resolve-only.md",
)
FAILED_GENERATOR_SHA256 = "033f58db9acd7631226682e5188b8f23f48762afe1117ee3537063f5ff2b3426"
FAILED_RECEIPT_SCHEMA = "scnsim.private_chapter8_failed_run_capture"
ORIGINAL_FAILED_RUN_SCHEMA = "scnsim.engineer_chapter8_original_failed_run.v1"
NO_EXECUTION_GUARD_CODE = '''
from scnsim.execution.jax_backend import JaxBackend as _CourseBackend
_course_original_native = _CourseBackend.execute_native
_course_native_calls = {"attempted": 0, "completed": 0}
def _course_native(self, *args, **kwargs):
    _course_native_calls["attempted"] += 1
    if _course_resolve_only:
        raise RuntimeError("resolve-only course kernel forbids JaxBackend.execute_native")
    outcome = _course_original_native(self, *args, **kwargs)
    _course_native_calls["completed"] += 1
    return outcome
_CourseBackend.execute_native = _course_native
'''


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _inventory(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("08-*") if path.is_file()}


def _parse_cells() -> tuple[tuple[str, str], ...]:
    cells = []
    for source in SOURCES:
        lines = source.read_text(encoding="utf-8").splitlines(keepends=True); index = 0
        while index < len(lines):
            if lines[index].strip() != "```{python}": index += 1; continue
            index += 1; body = []
            while index < len(lines) and lines[index].strip() != "```": body.append(lines[index]); index += 1
            if index == len(lines): raise ValueError(f"unterminated Python cell in {source.relative_to(ROOT)}")
            cell_id = next((line.split(":", 1)[1].strip() for line in body if line.startswith("#| id:")), None)
            if cell_id is None: raise ValueError(f"missing Python cell id in {source.relative_to(ROOT)}")
            cells.append((cell_id, "".join(line for line in body if not line.startswith("#|")))); index += 1
    if tuple(cell_id for cell_id, _ in cells) != EXPECTED_CELL_IDS: raise ValueError("Chapter 8 cell order changed")
    return tuple(cells)


def _source_tree() -> dict[str, object]:
    paused = ROOT / "src" / "scnsim" / "_agent_knowledge"
    rows = {str(path.relative_to(ROOT)): _hash(path) for path in sorted((ROOT / "src" / "scnsim").rglob("*")) if path.is_file() and not path.is_relative_to(paused) and path.suffix in {".py", ".jl", ".json", ".toml", ".cpp"}}
    return {"sha256": _hash_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()), "files": rows}


def _binding() -> dict[str, object]:
    cells = _parse_cells(); sources = (CHAPTER / "chapter.qmd", *SOURCES, *WRAPPERS)
    return {"generator_sha256": _hash(Path(__file__)), "sources": {str(path.relative_to(ROOT)): _hash(path) for path in (*sources, ROOT / "scripts" / "engineer_publication_binding.py")}, "source_tree": _source_tree(), "cell_ids": list(EXPECTED_CELL_IDS), "cells": {cell_id: _hash_bytes(code.encode()) for cell_id, code in cells}}


def _environment() -> dict[str, object]:
    runtime_binding = installed_course_runtime(ROOT)
    import scnsim
    scnsim.configure_runtime(cpu_threads=10)
    values = {"python": sys.version.split()[0], "scnsim": runtime_binding["scnsim"]}
    for name in ("ipykernel", "jupyter_client", "kaleido", "matplotlib", "nbformat", "numpy", "pint", "plotly", "schemdraw"): values[name] = importlib.metadata.version(name)
    return values | {"runtime_binding": runtime_binding}


def _workspace(requested: Path | None) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="scnsim-engineer-chapter8-")) if requested is None else requested.resolve()
    if ROOT == workspace or ROOT in workspace.parents: raise ValueError("execution workspace must be outside the repository")
    if workspace.exists() and any(workspace.iterdir()): raise RuntimeError("execution workspace must be new or empty")
    workspace.mkdir(parents=True, exist_ok=True); return workspace


def _tree_hashes(directory: Path) -> dict[str, str]:
    if not directory.exists(): return {}
    return {str(path.relative_to(directory)): _hash(path) for path in sorted(directory.rglob("*")) if path.is_file()}


# Historical receipt verification below preserves earlier publication lineage only.
# Current generation never resumes or reinterprets a Julia receipt.
def _json_object_bytes(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required receipt evidence is not a regular file: {path}")
    try:
        data = path.read_bytes()
        value = json.loads(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"required receipt evidence is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"required receipt evidence is not an object: {path}")
    return value, data


def _json_object(path: Path) -> dict[str, object]:
    return _json_object_bytes(path)[0]


def _canonical_sha256(value: object) -> str:
    return _hash_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _preparation_without_explain(code: str) -> str:
    tree = ast.parse(code)
    expected = ast.parse(
        "run.explain(quantity_view, root_spec, parameters=baseline_parameters).show()\n"
    ).body[0]
    if not tree.body or ast.dump(tree.body[-1], include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise RuntimeError("Chapter 8 preparation cell no longer ends with the exact explain display")
    final = tree.body[-1]
    if final.end_lineno is None or any(
        line.strip() for line in code.splitlines(keepends=True)[final.end_lineno:]
    ):
        raise RuntimeError("Chapter 8 preparation cell has content after the explain display")
    return "".join(code.splitlines(keepends=True)[: final.lineno - 1])


def _unbound_plan_sha256(build_code: str) -> str:
    from scnsim.canonical import canonical_json_bytes, sha256_hex
    from scnsim.authoring.identity import canonical_plan_snapshot

    namespace: dict[str, object] = {}
    exec(build_code, namespace)
    plan = namespace.get("plan")
    if plan is None:
        raise RuntimeError("Chapter 8 build cell did not declare plan")
    with plan._run_seal_preparation():  # type: ignore[attr-defined]
        snapshot = plan._capture_authoring_snapshot()  # type: ignore[attr-defined]
        return sha256_hex(canonical_json_bytes(canonical_plan_snapshot(snapshot)))


def _resume_workspace(requested: Path) -> Path:
    if requested.is_symlink():
        raise RuntimeError("resume workspace must not be a symbolic link")
    workspace = requested.resolve()
    if ROOT == workspace or ROOT in workspace.parents:
        raise ValueError("resume workspace must be outside the repository")
    if workspace.is_symlink() or not workspace.is_dir():
        raise RuntimeError("resume workspace must be an existing regular directory")
    return workspace


def _resume_input(
    receipt_path: Path,
    expected_capture_sha256: str,
    workspace: Path,
) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    if not _is_lower_hex(expected_capture_sha256, 64):
        raise ValueError("expected Chapter 8 capture SHA-256 must be 64 lowercase hexadecimal characters")
    receipt, receipt_bytes = _json_object_bytes(receipt_path)
    if _hash_bytes(receipt_bytes) != expected_capture_sha256:
        raise RuntimeError("Chapter 8 failed-run capture differs from the caller-owned expected SHA-256")
    expected_keys = {
        "capture_classification", "cell_ids", "cells_sha256", "environment", "failure",
        "generator_sha256", "plan", "requests", "runtime_identity", "schema",
        "schema_version", "sources_sha256", "source_tree_sha256", "source_tree_git_tree",
    }
    if set(receipt) != expected_keys or receipt.get("schema") != FAILED_RECEIPT_SCHEMA or receipt.get("schema_version") != 1:
        raise RuntimeError("Chapter 8 failed-run receipt shape is unsupported")
    if receipt.get("capture_classification") != "post_failure_read_only_capture":
        raise RuntimeError("Chapter 8 resume requires the reviewed post-failure capture")
    if receipt.get("generator_sha256") != FAILED_GENERATOR_SHA256:
        raise RuntimeError("Chapter 8 failed-run generator identity is unsupported")

    binding = _binding()
    if receipt.get("source_tree_sha256") != binding["source_tree"]["sha256"]:
        raise RuntimeError("Chapter 8 execution source tree differs from the failed run")
    if receipt.get("cell_ids") != binding["cell_ids"] or receipt.get("cells_sha256") != binding["cells"]:
        raise RuntimeError("Chapter 8 executable cells differ from the failed run")
    if receipt.get("sources_sha256") != binding["sources"]:
        raise RuntimeError("Chapter 8 source files differ from the failed run")

    from scnsim.runtime import _runtime_identity_base

    if receipt.get("runtime_identity") != _runtime_identity_base():
        raise RuntimeError("Chapter 8 runtime semantic identity differs from the failed run")
    plan = receipt.get("plan")
    requests = receipt.get("requests")
    failure = receipt.get("failure")
    if not isinstance(plan, dict) or not isinstance(requests, list) or not isinstance(failure, dict):
        raise RuntimeError("Chapter 8 failed-run receipt evidence is malformed")
    if failure.get("cell_id") != "ch8-execute-persisted-results" or failure.get("process_telemetry") != "UNAVAILABLE":
        raise RuntimeError("Chapter 8 failed-run boundary is unsupported")
    failure_receipt = receipt_path.parent / "failure-receipt.md"
    if (
        failure_receipt.is_symlink()
        or not failure_receipt.is_file()
        or _hash_bytes(failure_receipt.read_bytes()) != failure.get("failure_receipt_sha256")
    ):
        raise RuntimeError("Chapter 8 failed-run narrative receipt differs from the capture")
    cells = dict(_parse_cells())
    plan_sha256 = _unbound_plan_sha256(cells[FIRST_KERNEL_IDS[0]])
    if plan_sha256 != plan.get("plan_sha256"):
        raise RuntimeError("Chapter 8 unbound Plan identity differs from the preserved workspace")

    execution = workspace / "execution"
    shared = execution / "workspaces" / "engineer-chapter-08"
    if execution.is_symlink() or shared.is_symlink() or not shared.is_dir():
        raise RuntimeError("Chapter 8 preserved workspace is missing or redirected")
    if any(path.is_symlink() for path in shared.rglob("*")):
        raise RuntimeError("Chapter 8 preserved workspace contains a symbolic link")
    root_document = _json_object(shared / "workspace.json")
    if _hash(shared / "workspace.json") != plan.get("workspace_sha256"):
        raise RuntimeError("Chapter 8 preserved root identity differs from the receipt")
    active = root_document.get("active_leaf")
    expected_active = {
        "directory": plan.get("active_leaf"),
        "plan_sha256": plan_sha256,
        "workspace_instance_id": plan.get("workspace_instance_id"),
    }
    if active != expected_active or not isinstance(active, dict):
        raise RuntimeError("Chapter 8 active leaf differs from the preserved receipt")
    leaf = shared / str(active["directory"])
    if leaf.is_symlink() or not leaf.is_dir() or _hash(leaf / "plan.json") != plan_sha256:
        raise RuntimeError("Chapter 8 active Plan evidence differs from the preserved receipt")

    request_rows: dict[str, dict[str, object]] = {}
    for row in requests:
        if not isinstance(row, dict) or not isinstance(row.get("request_sha256"), str):
            raise RuntimeError("Chapter 8 request receipt row is malformed")
        request_rows[str(row["request_sha256"])] = row
    request_root = leaf / "requests"
    if set(path.name for path in request_root.iterdir() if path.is_dir()) != set(request_rows):
        raise RuntimeError("Chapter 8 preserved request inventory differs from the receipt")
    for request_sha256, row in request_rows.items():
        request = request_root / request_sha256
        attempt = request / "attempts" / "000001"
        if _hash(request / "request.json") != request_sha256:
            raise RuntimeError("Chapter 8 preserved request bytes differ from their identity")
        if _hash(attempt / "result.json") != row.get("result_sha256"):
            raise RuntimeError("Chapter 8 preserved Result bytes differ from the receipt")
        if _hash(attempt / "receipt.json") != row.get("receipt_file_sha256"):
            raise RuntimeError("Chapter 8 preserved sealed receipt bytes differ from the capture")
        outcome = _json_object(attempt / "outcome.json")
        if (
            row.get("status") != "success"
            or outcome.get("status") != "success"
            or outcome.get("attempt_sha256") != row.get("attempt_sha256")
            or outcome.get("result_sha256") != row.get("result_sha256")
        ):
            raise RuntimeError("Chapter 8 preserved outcome differs from the successful receipt")
    output = workspace / "artifacts"
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
        raise RuntimeError("Chapter 8 resume output directory must remain present and empty")
    return receipt, binding, execution, shared


def _original_failed_run_payload(
    receipt: dict[str, object], expected_capture_sha256: str
) -> dict[str, object]:
    failure = receipt["failure"]
    if not isinstance(failure, dict):
        raise RuntimeError("Chapter 8 failed-run failure evidence is malformed")
    return {
        "schema": ORIGINAL_FAILED_RUN_SCHEMA,
        "capture_sha256": expected_capture_sha256,
        "failure_receipt_sha256": failure["failure_receipt_sha256"],
        "generator_sha256": receipt["generator_sha256"],
        "source_tree_sha256": receipt["source_tree_sha256"],
        "source_tree_git_tree": receipt["source_tree_git_tree"],
        "cell_ids": receipt["cell_ids"],
        "cells": receipt["cells_sha256"],
        "sources": receipt["sources_sha256"],
        "runtime_identity": receipt["runtime_identity"],
        "plan": receipt["plan"],
        "failed_cell_id": failure["cell_id"],
        "failure": failure["error"],
        "kernel_identity": "UNAVAILABLE",
        "process_inventory": "UNAVAILABLE",
        "successful_results": receipt["requests"],
    }


def _verify_original_failed_run(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"payload", "payload_sha256"}:
        raise RuntimeError("Chapter 8 original failed-run record is malformed")
    payload = value.get("payload")
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "capture_sha256", "failure_receipt_sha256", "generator_sha256",
        "source_tree_sha256", "source_tree_git_tree", "cell_ids", "cells", "sources",
        "runtime_identity", "plan", "failed_cell_id", "failure", "kernel_identity",
        "process_inventory", "successful_results",
    }:
        raise RuntimeError("Chapter 8 original failed-run payload shape is malformed")
    digest_fields = {
        "capture_sha256", "failure_receipt_sha256", "generator_sha256", "source_tree_sha256",
    }
    if any(not _is_lower_hex(payload.get(field), 64) for field in digest_fields):
        raise RuntimeError("Chapter 8 original failed-run digest evidence is malformed")
    if (
        not _is_lower_hex(value.get("payload_sha256"), 64)
        or value.get("payload_sha256") != _canonical_sha256(payload)
        or payload.get("schema") != ORIGINAL_FAILED_RUN_SCHEMA
        or payload.get("generator_sha256") != FAILED_GENERATOR_SHA256
        or payload.get("cell_ids") != list(EXPECTED_CELL_IDS)
        or not isinstance(payload.get("failure"), str)
        or payload.get("failed_cell_id") != "ch8-execute-persisted-results"
        or payload.get("kernel_identity") != "UNAVAILABLE"
        or payload.get("process_inventory") != "UNAVAILABLE"
    ):
        raise RuntimeError("Chapter 8 original failed-run payload seal is invalid")
    cells = payload.get("cells")
    sources = payload.get("sources")
    runtime_identity = payload.get("runtime_identity")
    plan = payload.get("plan")
    results = payload.get("successful_results")
    if (
        not isinstance(cells, dict)
        or set(cells) != set(EXPECTED_CELL_IDS)
        or any(not _is_lower_hex(value, 64) for value in cells.values())
        or not isinstance(sources, dict)
        or set(sources) != {str(path.relative_to(ROOT)) for path in (*SOURCES, *WRAPPERS, CHAPTER / "chapter.qmd")}
        or any(not _is_lower_hex(value, 64) for value in sources.values())
        or not isinstance(runtime_identity, dict)
        or set(runtime_identity) != {
            "python_source_sha256", "julia_source_sha256", "project_sha256",
            "manifest_sha256", "julia_version",
        }
        or not isinstance(plan, dict)
        or set(plan) != {
            "active_leaf", "catalog_dirty_overlay_sha256", "catalog_git_commit",
            "catalog_source_sha256", "plan_sha256", "workspace_instance_id", "workspace_sha256",
        }
        or not isinstance(results, list)
        or len(results) != 2
    ):
        raise RuntimeError("Chapter 8 original failed-run provenance is malformed")
    if (
        any(
            not _is_lower_hex(runtime_identity.get(field), 64)
            for field in (
                "python_source_sha256", "julia_source_sha256", "project_sha256", "manifest_sha256",
            )
        )
        or not isinstance(runtime_identity.get("julia_version"), str)
        or not _is_lower_hex(payload.get("source_tree_git_tree"), 40)
        or not isinstance(plan.get("active_leaf"), str)
        or not isinstance(plan.get("workspace_instance_id"), str)
        or not _is_lower_hex(plan.get("catalog_git_commit"), 40)
        or any(
            not _is_lower_hex(plan.get(field), 64)
            for field in (
                "catalog_dirty_overlay_sha256", "catalog_source_sha256", "plan_sha256", "workspace_sha256",
            )
        )
    ):
        raise RuntimeError("Chapter 8 original failed-run identity fields are malformed")
    expected_result_keys = {
        "attempt_sha256", "operation", "receipt_file_sha256", "request_sha256",
        "result_sha256", "status",
    }
    if (
        any(not isinstance(row, dict) or set(row) != expected_result_keys for row in results)
        or {row["operation"] for row in results} != {"solve_direct", "evaluate_direct"}
        or any(row.get("status") != "success" for row in results)
        or any(
            not _is_lower_hex(row.get(field), 64)
            for row in results
            for field in (
                "attempt_sha256", "receipt_file_sha256", "request_sha256", "result_sha256",
            )
        )
    ):
        raise RuntimeError("Chapter 8 original successful-Result inventory is malformed")
    return payload


def _execute_message(client: Any, code: str, label: str) -> str:
    message_id = client.execute(code, store_history=True, stop_on_error=True)
    streams = []
    while True:
        message = client.get_iopub_msg(timeout=900)
        if message.get("parent_header", {}).get("msg_id") != message_id: continue
        message_type = message["header"]["msg_type"]
        content = message["content"]
        if message_type == "stream": streams.append(content.get("text", ""))
        elif message_type == "error":
            error = RuntimeError(f"{label}: {content.get('ename')}: {content.get('evalue')}")
            error.add_note("\n".join(content.get("traceback", ())))
            raise error
        elif message_type == "status" and content.get("execution_state") == "idle": break
    return "".join(streams)


def _kernel_identity(manager: Any) -> dict[str, object]:
    process = manager.provisioner.process
    if process is None:
        raise RuntimeError("Jupyter did not expose its owned kernel process")
    identity = {"kernel_id": manager.kernel_id, "pid": process.pid,
                "connection_file": Path(manager.connection_file).name}
    if Path("/proc").is_dir():
        fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
        identity.update(pgid=int(fields[2]), startticks=fields[19])
    return identity


def _payload(stream: str, marker: str) -> dict[str, object]:
    lines = [line for line in stream.splitlines() if line.startswith(marker)]
    if len(lines) != 1: raise RuntimeError(f"missing unique {marker} payload")
    value = json.loads(lines[0][len(marker):])
    if not isinstance(value, dict): raise RuntimeError(f"{marker} payload is malformed")
    return value


def _kernel_temporary_directory(environment):
    """Keep per-kernel browser sockets under the inherited platform temp root."""
    return tempfile.TemporaryDirectory(prefix="scnsim-c8-", dir=environment.get("TMPDIR"))


def _run_kernel(cells: tuple[tuple[str, str], ...], cwd: Path, final_code: str,
                marker: str, *, resolve_only: bool) -> tuple[dict[str, object], dict[str, object]]:
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    expected = installed_course_runtime(ROOT)
    spec_root = cwd.parent / "kernel-specs"
    spec_directory = spec_root / "scnsim-engineer-candidate"
    spec_directory.mkdir(parents=True, exist_ok=True)
    spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "SCNSim installed candidate", "language": "python"}
    spec_bytes = (json.dumps(spec, sort_keys=True) + "\n").encode()
    spec_path = spec_directory / "kernel.json"
    if spec_path.exists() and spec_path.read_bytes() != spec_bytes:
        raise RuntimeError("private course kernelspec differs from candidate interpreter")
    spec_path.write_bytes(spec_bytes)
    manager = KernelManager(kernel_name="scnsim-engineer-candidate",
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec_root)]))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("SCNSIM_RUN_JULIA_TESTS", None)
    for name in ("IPYTHONDIR", "JUPYTER_RUNTIME_DIR"):
        path = cwd.parent / "kernel-runtime" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    client = None
    process = None
    identity = {}
    private_facts = {}
    primary = None
    cleanup_error = None
    kernel_tmp = None
    try:
        kernel_tmp = _kernel_temporary_directory(environment)
        environment["TMPDIR"] = kernel_tmp.name
        manager.start_kernel(cwd=str(cwd), env=environment)
        process = manager.provisioner.process
        identity = _kernel_identity(manager)
        client = manager.client()
        client.start_channels()
        client.wait_for_ready(timeout=120)
        bootstrap = f'''
import json as _course_json, os as _course_os, sys as _course_sys, runpy as _course_runpy
from pathlib import Path as _CoursePath
_course_helper = _course_runpy.run_path({str(ROOT / 'scripts/engineer_publication_binding.py')!r})
_course_binding = _course_helper['installed_course_runtime'](_CoursePath({str(ROOT)!r}))
if _course_binding != {expected!r} or _course_sys.executable != {sys.executable!r} or _course_sys.prefix != {sys.prefix!r}:
    raise RuntimeError('kernel differs from exact installed parent wheel/interpreter')
import scnsim as _course_scnsim
_course_scnsim.configure_runtime(cpu_threads=10)
_course_resolve_only = {resolve_only!r}
print('__SCNSIM_KERNEL_BINDING__' + _course_json.dumps(dict(
    executable=_course_sys.executable, prefix=_course_sys.prefix,
    module=_course_scnsim.__file__, binding=_course_binding), sort_keys=True))
'''
        private_facts = _payload(_execute_message(client, bootstrap, "installed kernel binding"),
                                 "__SCNSIM_KERNEL_BINDING__")
        _execute_message(client, NO_EXECUTION_GUARD_CODE, "JAX native execution guard")
        for cell_id, code in cells:
            _execute_message(client, code, cell_id)
        payload = _payload(_execute_message(client, final_code, marker), marker)
        counts = _payload(_execute_message(client,
            "print('__SCNSIM_NATIVE__' + _course_json.dumps(_course_native_calls, sort_keys=True))",
            "same-call native dispatch counts"), "__SCNSIM_NATIVE__")
        identity["native_dispatch"] = counts
        identity["runtime_binding"] = expected
        if resolve_only and (counts["attempted"] != 0 or counts["completed"] != 0):
            raise RuntimeError("resolve-only kernel attempted native execution")
        return identity, payload
    except BaseException as error:
        primary = error
        raise
    finally:
        def cleanup(label, action):
            nonlocal cleanup_error
            try:
                action()
            except BaseException as error:
                if primary is None and cleanup_error is None:
                    cleanup_error = error
                # Diagnostic formatting must not replace the owned failure.
                try:
                    message = label + ": " + repr(error)
                    private_facts.setdefault("cleanup_errors", []).append(
                        {"phase": label, "message": repr(error)})
                    owner = primary if primary is not None else cleanup_error
                    if owner is not error:
                        owner.add_note(message)
                except BaseException:
                    pass
        if client is not None and process is not None:
            close_code = """
_course_closed = []
for _course_name in ('run', 'restart_run'):
    if _course_name in globals():
        globals()[_course_name].close()
        _course_closed.append(_course_name)
if '_course_original_native' in globals():
    _CourseBackend.execute_native = _course_original_native
print('__SCNSIM_CLOSED__' + _course_json.dumps({'closed_runs': _course_closed}))
"""
            def close_runs():
                if process.poll() is not None:
                    return
                private_facts["closed_runs"] = _payload(
                    _execute_message(client, close_code, "close owned Runs"), "__SCNSIM_CLOSED__")
            cleanup("CircuitRun.close", close_runs)
        if client is not None:
            cleanup("stop kernel channels", client.stop_channels)
        # Preserve the existing Jupyter shutdown policy; add explicit wait evidence.
        cleanup("shutdown kernel", lambda: manager.shutdown_kernel(now=True))
        if process is not None:
            def wait():
                private_facts["returncode"] = process.wait()
                private_facts["waited"] = True
            cleanup("wait/reap kernel", wait)
            if hasattr(os, "getpgid") and identity.get("pgid") is not None:
                def group_members():
                    rows = []
                    for path in Path("/proc").iterdir():
                        if not path.name.isdecimal():
                            continue
                        try:
                            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
                        except (FileNotFoundError, ProcessLookupError):
                            continue
                        if int(fields[2]) == identity["pgid"]:
                            rows.append(int(path.name))
                    private_facts["owned_pgid_members_after_wait"] = rows
                    if rows:
                        raise RuntimeError("owned kernel group remains after wait")
                cleanup("owned kernel group observation", group_members)
        if kernel_tmp is not None:
            private_facts["kernel_tmpdir"] = kernel_tmp.name
            cleanup("kernel temporary directory", kernel_tmp.cleanup)
        def persist():
            private_facts["process"] = dict(identity)
            private_facts["primary_error"] = None if primary is None else repr(primary)
            path = cwd.parent / ("kernel-" + (manager.kernel_id or "not-started") + "-runtime.json")
            with path.open("w") as output:
                json.dump(private_facts, output, indent=2, sort_keys=True)
                output.write("\n"); output.flush(); os.fsync(output.fileno())
            identity["private_process_facts_sha256"] = _hash(path)
        cleanup("persist private kernel facts", persist)
        if primary is None and cleanup_error is not None:
            raise cleanup_error


def _identity_code(name: str) -> str:
    return "{field: getattr(" + name + ".identity, field) for field in ('plan_sha256','request_sha256','attempt_sha256','result_sha256')}"


def generate(workspace: Path) -> None:
    environment = _environment()
    cells = dict(_parse_cells()); execution = workspace / "execution"; execution.mkdir(); output = workspace / "artifacts"; output.mkdir()
    report_path = output / "08-report.html"; figure_path = output / "08-direct-s11.svg"
    first_code = f'''\nimport json as _receipt_json, os as _receipt_os\nfrom pathlib import Path as _ReceiptPath\nreport.save(_ReceiptPath({str(report_path)!r}))\ndirect.s.plot(magnitude="db", theme=Theme.AUTO).write_image(_ReceiptPath({str(figure_path)!r}), format="svg")\n_receipt_payload = {{\n "pid": _receipt_os.getpid(),\n "direct_identity": {_identity_code("direct")},\n "root_identity": {_identity_code("root")},\n "root_frequency_GHz": float(root.frequency.to("gigahertz").magnitude),\n "root_linewidth_MHz": float(root.linewidth.to("megahertz").magnitude),\n "matrix_coordinates": list(direct.s.view.coordinates),\n}}\nprint("__SCNSIM_FIRST__" + _receipt_json.dumps(_receipt_payload, sort_keys=True))\n'''
    first_cells = tuple((cell_id, cells[cell_id]) for cell_id in FIRST_KERNEL_IDS)
    first_kernel, first = _run_kernel(first_cells, execution, first_code, "__SCNSIM_FIRST__", resolve_only=False)
    shared = execution / "workspaces" / "engineer-chapter-08"; before = _tree_hashes(shared)
    second_code = f'''\nimport json as _receipt_json, os as _receipt_os\n_receipt_payload = {{\n "pid": _receipt_os.getpid(),\n "resolved_identity": {_identity_code("resolved_root")},\n "root_frequency_GHz": float(resolved_root.frequency.to("gigahertz").magnitude),\n "root_linewidth_MHz": float(resolved_root.linewidth.to("megahertz").magnitude),\n}}\nprint("__SCNSIM_SECOND__" + _receipt_json.dumps(_receipt_payload, sort_keys=True))\n'''
    second_cells = tuple((cell_id, cells[cell_id]) for cell_id in SECOND_KERNEL_IDS)
    second_kernel, second = _run_kernel(second_cells, execution, second_code, "__SCNSIM_SECOND__", resolve_only=True); after = _tree_hashes(shared)
    if first_kernel["kernel_id"] == second_kernel["kernel_id"]: raise RuntimeError("Chapter 8 did not use two independent Jupyter kernels")
    if first["root_identity"] != second["resolved_identity"]: raise RuntimeError("resolved root identity differs from persisted root")
    if before != after: raise RuntimeError("resolve-only kernel changed the shared workspace")
    public_results = {"direct_identity": first["direct_identity"], "root_identity": first["root_identity"], "root_frequency_GHz": first["root_frequency_GHz"], "root_linewidth_MHz": first["root_linewidth_MHz"], "matrix_coordinates": first["matrix_coordinates"]}
    (output / "08-persisted-results.json").write_text(json.dumps(public_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "08-persisted-results.md").write_text(f"### Persisted Result summary\n\n| Result | value |\n|---|---:|\n| loaded root frequency | {first['root_frequency_GHz']:.12g} GHz |\n| loaded root linewidth | {first['root_linewidth_MHz']:.12g} MHz |\n\n![The declared Direct S11 presentation.](../figures/08-direct-s11.svg)\n", encoding="utf-8")
    (output / "08-resolve-only.md").write_text(f"### Resolve-only restart outcome\n\nA second independent Jupyter kernel resolved the exact stored root: **{second['root_frequency_GHz']:.12g} GHz**, linewidth **{second['root_linewidth_MHz']:.12g} MHz**. Its Result identity matches Lesson 1, and the shared workspace bytes were unchanged.\n", encoding="utf-8")
    artifacts = {name: _hash(output / name) for name in ARTIFACT_NAMES}; binding = _binding()
    if _inventory(output) != set(ARTIFACT_NAMES): raise RuntimeError("Chapter 8 generated artifact inventory is not exact")
    manifest = {"schema": "scnsim.engineer_chapter8_artifacts.v1", "binding": binding, "execution": {"generator_sha256": binding["generator_sha256"], "source_tree_sha256": binding["source_tree"]["sha256"], "cells": binding["cells"], "kernels": {"first": first_kernel, "second": second_kernel}, "first_kernel_cell_ids": list(FIRST_KERNEL_IDS), "second_kernel_cell_ids": list(SECOND_KERNEL_IDS), "second_kernel_operations": ["resolve"], "shared_workspace_unchanged_during_resolve": True, "native_execution_guard": "JaxBackend.execute_native.v1"}, "environment": environment, "results": public_results | {"resolved_identity": second["resolved_identity"]}, "artifacts": artifacts}
    FIGURES.mkdir(parents=True, exist_ok=True)
    unknown = _inventory(FIGURES) - set(ARTIFACT_NAMES)
    if unknown: raise RuntimeError(f"unknown Chapter 8 artifacts require review: {sorted(unknown)!r}")
    for name in ARTIFACT_NAMES: shutil.copy2(output / name, FIGURES / name)
    if _inventory(FIGURES) != set(ARTIFACT_NAMES): raise RuntimeError("Chapter 8 published artifact inventory is not exact")
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"; MANIFEST.write_text(text, encoding="utf-8"); receipt = workspace / "generation-receipt.json"; receipt.write_text(text, encoding="utf-8"); print(receipt)




def validate_published_manifest(
    manifest: object,
    *,
    artifact_sha256: dict[str, str],
) -> None:
    """Validate Chapter 8's sealed execution and continuation without recursion."""

    if not isinstance(manifest, dict):
        raise RuntimeError("Chapter 8 artifact manifest is malformed")
    schema = manifest.get("schema")
    if schema not in {
        "scnsim.engineer_chapter8_artifacts.v1",
        "scnsim.engineer_chapter8_artifacts.v2",
    }:
        raise RuntimeError("Chapter 8 artifact manifest schema is unsupported")
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        raise RuntimeError("Chapter 8 execution record is malformed")
    binding = manifest.get("binding")
    if not isinstance(binding, dict):
        raise RuntimeError("Chapter 8 execution binding is malformed")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != set(ARTIFACT_NAMES)
        or artifacts != artifact_sha256
        or any(not _is_lower_hex(value, 64) for value in artifacts.values())
    ):
        raise RuntimeError("Chapter 8 artifact inventory or bytes are stale")
    if schema == "scnsim.engineer_chapter8_artifacts.v2":
        _verify_original_failed_run(execution.get("original_failed_run"))
    if schema == "scnsim.engineer_chapter8_artifacts.v2":
        if execution.get("mode") != "receipt_bound_resume":
            raise RuntimeError("Chapter 8 continuation mode is malformed")
        original = _verify_original_failed_run(execution.get("original_failed_run"))
        continuation = execution.get("continuation", {})
        restart = execution.get("restart", {})
        if (
            original.get("generator_sha256") != FAILED_GENERATOR_SHA256
            or original.get("kernel_identity") != "UNAVAILABLE"
            or original.get("process_inventory") != "UNAVAILABLE"
            or original.get("failed_cell_id") != "ch8-execute-persisted-results"
        ):
            raise RuntimeError("Chapter 8 original failed-run evidence is malformed")
        if continuation.get("kernel", {}).get("kernel_id") == restart.get("kernel", {}).get("kernel_id"):
            raise RuntimeError("Chapter 8 continuation and restart kernel identities are not distinct")
        if (
            continuation.get("cell_ids") != [
                "ch8-build-persisted-plan", "ch8-prepare-without-explain",
                "ch8-forbid-continuation-execution", "ch8-resolve-existing-results",
                "ch8-build-result-report",
            ]
            or continuation.get("operations") != ["resolve_direct", "resolve_root", "display", "build_report", "save_report", "export_svg"]
            or continuation.get("numerical_operations") != []
            or continuation.get("forbidden_execution_calls") != []
            or restart.get("cell_ids") != [
                *SECOND_KERNEL_IDS[:2], "ch8-forbid-restart-execution", SECOND_KERNEL_IDS[2]
            ]
            or restart.get("teaching_cell_ids") != list(SECOND_KERNEL_IDS)
            or restart.get("operations") != ["resolve_root"]
            or restart.get("numerical_operations") != []
            or restart.get("forbidden_execution_calls") != []
            or execution.get("shared_workspace_unchanged_during_resume") is not True
        ):
            raise RuntimeError("Chapter 8 receipt-bound continuation evidence is malformed")
        if (
            continuation.get("generator_sha256") != binding.get("generator_sha256")
            or restart.get("generator_sha256") != binding.get("generator_sha256")
        ):
            raise RuntimeError("Chapter 8 continuation generator identity is malformed")
    else:
        kernels = execution.get("kernels", {})
        if kernels.get("first", {}).get("kernel_id") == kernels.get("second", {}).get("kernel_id"): raise RuntimeError("Chapter 8 kernel identities are not distinct")
        if execution.get("second_kernel_cell_ids") != list(SECOND_KERNEL_IDS) or execution.get("second_kernel_operations") != ["resolve"] or execution.get("shared_workspace_unchanged_during_resolve") is not True: raise RuntimeError("Chapter 8 restart evidence is malformed")
    if binding.get("generator_sha256") == _hash(Path(__file__)):
        kernels = execution.get("kernels", {})
        first = kernels.get("first", {})
        second = kernels.get("second", {})
        if (execution.get("native_execution_guard") != "JaxBackend.execute_native.v1"
                or second.get("native_dispatch") != {"attempted": 0, "completed": 0}
                or first.get("runtime_binding") != second.get("runtime_binding")
                or not second.get("runtime_binding")):
            raise RuntimeError("Chapter 8 current installed JAX guard evidence is malformed")
    results = manifest.get("results")
    if not isinstance(results, dict) or set(results) != {
        "direct_identity", "matrix_coordinates", "resolved_identity",
        "root_frequency_GHz", "root_identity", "root_linewidth_MHz",
    }:
        raise RuntimeError("Chapter 8 Result summary is malformed")
    if results.get("root_identity") != results.get("resolved_identity"):
        raise RuntimeError("Chapter 8 persisted and resolved root identities differ")
    if schema == "scnsim.engineer_chapter8_artifacts.v2":
        original = _verify_original_failed_run(execution.get("original_failed_run"))
        original_plan = original["plan"]
        original_results = {
            row["operation"]: row for row in original["successful_results"]
        }
        for name, operation in (
            ("direct_identity", "solve_direct"),
            ("root_identity", "evaluate_direct"),
            ("resolved_identity", "evaluate_direct"),
        ):
            identity = results.get(name)
            row = original_results[operation]
            expected = {
                "plan_sha256": original_plan["plan_sha256"],
                "request_sha256": row["request_sha256"],
                "attempt_sha256": row["attempt_sha256"],
                "result_sha256": row["result_sha256"],
            }
            if identity != expected:
                raise RuntimeError(f"Chapter 8 {name} differs from the original verified Result")


def check() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if _inventory(FIGURES) != set(ARTIFACT_NAMES):
        raise RuntimeError("Chapter 8 published artifact inventory is not exact")
    validate_published_manifest(
        manifest,
        artifact_sha256={name: _hash(FIGURES / name) for name in ARTIFACT_NAMES},
    )
    check_publication_binding(manifest, _binding())
    print("engineer Chapter 8 artifacts are current")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    if args.check:
        if args.workspace is not None:
            parser.error("--workspace cannot be used with --check")
        check()
    else:
        generate(_workspace(args.workspace))


if __name__ == "__main__": main()
