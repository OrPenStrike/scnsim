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


def complete_sensitivity_operation(binding, *, operation_id, task_id, attempt_id,
                                    manifest, result_summary, blocks, outcome,
                                    equation_evidence):
    return _store().complete_sensitivity_operation(
        binding, operation_id=operation_id, task_id=task_id, attempt_id=attempt_id,
        manifest=manifest, result_summary=result_summary, blocks=blocks, outcome=outcome,
        equation_evidence=equation_evidence,
    )


def complete_direct_operation(binding, *, operation_id, task_id, attempt_id,
                               metadata, chunks, summary=None):
    return _store().complete_direct_operation(
        binding, operation_id=operation_id, task_id=task_id, attempt_id=attempt_id,
        metadata=metadata, chunks=chunks, summary=summary,
    )


def complete_sweep_operation(binding, *, operation_id, task_id, attempt_id,
                             metadata, chunks, summary=None):
    return _store().complete_sweep_operation(
        binding, operation_id=operation_id, task_id=task_id, attempt_id=attempt_id,
        metadata=metadata, chunks=chunks, summary=summary,
    )


def commit_physical_stage_batch(binding, *, operation_id, task_id, attempt_id,
                                stage_id, model_identity, rows, stage_state, checkpoint,
                                transitions=()):
    return _store().commit_physical_stage_batch(
        binding, operation_id=operation_id, task_id=task_id, attempt_id=attempt_id,
        stage_id=stage_id, model_identity=model_identity, rows=rows,
        stage_state=stage_state, checkpoint=checkpoint, transitions=transitions,
    )


def read_physical_optimization_checkpoint(binding, *, task_id, attempt_id,
                                           expected_request_sha256, reference=None,
                                           expected_stage_id=None):
    return _store().read_physical_optimization_checkpoint(
        binding, task_id=task_id, attempt_id=attempt_id,
        expected_request_sha256=expected_request_sha256, reference=reference,
        expected_stage_id=expected_stage_id,
    )


def read_physical_optimization_candidate(binding, *, task_id, attempt_id, locator,
                                          expected_request_sha256=None, resume_from=None):
    return _store().read_physical_optimization_candidate(
        binding, task_id=task_id, attempt_id=attempt_id, locator=locator,
        expected_request_sha256=expected_request_sha256, resume_from=resume_from,
    )


def read_physical_accepted_iterate(binding, *, attempt_id, stage_id,
                                   accepted_iterate_id):
    return _store().read_physical_accepted_iterate(
        binding, attempt_id=attempt_id, stage_id=stage_id,
        accepted_iterate_id=accepted_iterate_id,
    )


def complete_optimization_operation(binding, *, operation_id, task_id, attempt_id,
                                    result, manifest, stages, transitions, outcome):
    return _store().complete_optimization_operation(
        binding, operation_id=operation_id, task_id=task_id, attempt_id=attempt_id,
        result=result, manifest=manifest, stages=stages, transitions=transitions,
        outcome=outcome,
    )


def read_sensitivity_result(binding, task_id, *, attempt_id=None, projection_consumer=None):
    return _store().read_sensitivity_result(
        binding, task_id, attempt_id=attempt_id, projection_consumer=projection_consumer,
    )


def read_operation_attempt(binding, attempt_id, *, projection_consumer=None):
    return _store().read_operation_attempt(
        binding, attempt_id, projection_consumer=projection_consumer,
    )


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
        "complete_operation", "find_operation_success",
        "operation_task_record", "read_operation_success",
        "task_record", "update_attempt",
    }
    if name not in allowed:
        raise AttributeError(name)
    return getattr(_store(), name)
