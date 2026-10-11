"""Current operation success reads over one verified SQLite snapshot.

Ordinary JAX results still use their existing operation projection. Optimization
uses the sealed v6 completion index and fixed reader, never task-history replay.
"""
from collections.abc import Mapping

from ..numeric_encoding import record_bytes, record_document


def read_success(root, snapshot, descriptor, *, attempt_id=None, selection=None,
                 projection_consumer=None, binding_identity=None):
    from . import sqlite_storage as storage
    from .sqlite_storage import _materialize_task

    task_id = descriptor["task_id"]
    request_ref = next(
        (item for item in descriptor["artifacts"] if item.get("role") == "operation_request"),
        None,
    )
    if request_ref is None or request_ref.get("sha256") != descriptor["request_sha256"]:
        raise storage._error("Operation task lacks its canonical request.", task_id=task_id)
    request_raw = storage.read_object(root, request_ref, role="operation_request")
    request = record_document(request_raw)
    if not isinstance(request, dict) or record_bytes(request) != request_raw:
        raise storage._error("Operation request bytes are not canonical.", task_id=task_id)

    if request.get("operation") == "optimize_direct":
        from ..workspace.results import read_optimization_success
        from ..workspace.validation.common import _integrity

        if selection is not None and attempt_id is not None and selection.get("attempt_id") != attempt_id:
            selection = None
        if selection is not None:
            selected_attempt = selection.get("attempt_id")
            completion_ref = selection.get("workspace_completion")
            result_ref = selection.get("result_ref")
            if not isinstance(selected_attempt, str) or not isinstance(completion_ref, Mapping):
                raise _integrity("Selected Optimization success lacks its sealed completion.",
                                 task_id=task_id)
            attempt_id = selected_attempt
        else:
            if not isinstance(attempt_id, str):
                return None
            completion_pointer = snapshot.read_pointer(f"completion/{task_id}/{attempt_id}")
            if (completion_pointer is None
                    or completion_pointer["reference"].get("role") != "workspace_completion"):
                return None
            completion_ref = completion_pointer["reference"]
            result_ref = None

        if not isinstance(binding_identity, Mapping):
            raise _integrity("Fixed Optimization reading requires its bound Workspace identity.",
                             task_id=task_id, attempt_id=attempt_id)
        success = read_optimization_success(
            snapshot, binding_identity, descriptor=descriptor, attempt_id=attempt_id,
            completion_ref=completion_ref, result_ref=result_ref, selection=selection,
        )
    else:
        task = _materialize_task(root, snapshot, descriptor)
        from .task_history import operation_success_from_task
        success = operation_success_from_task(task, attempt_id=attempt_id)
        if success is None:
            return None

    return projection_consumer(success) if projection_consumer is not None else success
