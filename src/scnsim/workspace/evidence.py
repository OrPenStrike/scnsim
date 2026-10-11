"""Workspace-owned operation-evidence entry points."""
from __future__ import annotations

from collections.abc import Mapping


def _store():
    from . import sqlite_storage

    return sqlite_storage


def start_operation(binding, row, *, clock_binding=None):
    return _store().start_operation(binding, row, clock_binding=clock_binding)


def register_operation_execution(binding, *, row, task, request, attempt_id, resume_from=None):
    return _store().register_operation_execution(
        binding, row=row, task=task, request=request, attempt_id=attempt_id,
        resume_from=resume_from,
    )


def bind_operation(binding, *, row):
    return _store().bind_operation(binding, row=row)


def bind_operation_attempt(binding, *, row):
    return _store().bind_operation_attempt(binding, row=row)


def finish_operation(binding, row, *, failure=None):
    return _store().finish_operation(binding, row, failure=failure)


def publish_timing_batch(binding, batch):
    return _store().publish_timing_batch(binding, batch)


def query_operation_rows(binding, *, operation_ids=None, method=None, backend=None,
                         precision=None, status=None):
    return _store().query_operation_rows(
        binding, operation_ids=operation_ids, method=method, backend=backend,
        precision=precision, status=status,
    )


def open_record(workspace):
    return _store().open_record(workspace)


def read_native_result_artifact(*, binding_identity, directory, index_ref, artifact_ref):
    from .store import read_native_result_artifact as read_artifact

    return read_artifact(
        binding_identity=binding_identity,
        directory=directory,
        index_ref=index_ref,
        artifact_ref=artifact_ref,
    )


def operation_workspace(binding):
    return _store().operation_workspace(binding)


def error_document(error):
    return _store().error_document(error)


def is_diagnostic_event(kind):
    return _store().is_diagnostic_event(kind)


def recover_operation_workspace(binding):
    return _store().recover_operation_workspace(binding)


def __getattr__(name: str):
    """Resolve the current Workspace task/checkpoint transaction surface."""
    allowed = {
        "TaskWriter", "append_event", "begin_task_writer", "commit_barrier",
        "complete_operation", "find_operation_success", "flush_completed",
        "operation_task_record", "read_checkpoint", "read_operation_success",
        "task_record", "update_attempt",
    }
    if name not in allowed:
        raise AttributeError(name)
    return getattr(_store(), name)

