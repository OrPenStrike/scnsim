"""Current operation-domain writer and verified SQLite projections.

The core owns native transactions and byte objects; this adapter owns task,
occurrence, checkpoint and selected-result meanings. Ordinary analysis task
events and any root-anchor baseline remain bound to their terminal transaction.
Physical Optimization stage batches use their dedicated publisher below.
Older file-journal Workspaces remain untouched and require recomputation.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..errors import EvidenceIntegrityError, UnsupportedEvidenceVersionError
from .operation_store import OperationStore, Snapshot, _VERSION as _OPERATION_STORE_VERSION, _reference
from .binary_blocks import record_bytes, record_document
from ..numeric_encoding import record_bytes as numerical_record_bytes
from .operation_scratch import cleanup_inactive_operation_scratch

_ACTIVE: ContextVar[tuple[Path, Snapshot] | None] = ContextVar('operation_sql_snapshot', default=None)
_MARKER = '$scnsim_benchmark_journal'
_OCCURRENCE = frozenset({'attempt_id','candidate_key','cache_hit','evaluation_ordinal','generation',
                        'population_column','latent_coordinates','source_index','origin',
                        'continuation_t_f64','numerical_source_id'})
_DIAGNOSTIC = frozenset({'evaluation','progress'})


def _error(message, **evidence):
    return EvidenceIntegrityError(message, stage='operation_store', evidence=evidence)


def error_document(error: BaseException) -> dict[str, object]:
    """Encode an operation failure without changing its owning exception."""
    if isinstance(error, Exception) and hasattr(error, "kind") and hasattr(error, "stage"):
        value: dict[str, object] = {
            "type": type(error).__name__,
            "kind": error.kind,
            "category": error.category,
            "stage": error.stage,
            "message": str(error),
        }
        if hasattr(error, "evidence"):
            value["evidence"] = _error_evidence(dict(error.evidence))
        return value
    return {"type": type(error).__name__, "module": type(error).__module__, "message": str(error)}


def is_diagnostic_event(kind: str) -> bool:
    return kind in _DIAGNOSTIC


def _error_evidence(value):
    """Encode path-like failure evidence without importing benchmark storage."""
    import os
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _error_evidence(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_error_evidence(item) for item in value]
    return value


def _new_task_descriptor(task):
    value = record_document(record_bytes(dict(task)))
    required = {
        'task_id', 'request_sha256', 'arm', 'sample', 'attempts',
        'events', 'measurements', 'environment', 'artifacts',
    }
    if set(value) != required:
        raise _error('Benchmark task record has an unexpected field set.', fields=sorted(value))
    if value['attempts'] or value['events'] or value['measurements']:
        raise _error('New benchmark task identity cannot include prior task history.',
                     task_id=value['task_id'])
    return {key: value[key] for key in ('task_id', 'request_sha256', 'arm', 'sample',
                                       'environment', 'artifacts')}


def _unsupported_old_record(path):
    return UnsupportedEvidenceVersionError(
        "This Workspace uses an unsupported evidence format; use a new Workspace and recompute.",
        stage="operation_store",
        evidence={"path": str(path), "action": "use a new Workspace and recompute"},
    )


def _root(workspace):
    return Path(workspace).expanduser().resolve(strict=False)


def operation_workspace(binding):
    return Path(binding.leaf) / 'operations'


def _bound(binding, *, phase_scope=None):
    return OperationStore(operation_workspace(binding), plan_sha256=binding.plan_sha256,
                          workspace_instance_id=binding.workspace_instance_id, phase_scope=phase_scope)


def _writer_store(workspace, binding, *, phase_scope=None):
    store = _bound(binding, phase_scope=phase_scope)
    if _root(workspace) != store.root:
        raise _error('Operation write root differs from its bound Plan leaf.')
    return store


def _existing(root, *, phase_scope=None):
    store = OperationStore.open_readonly(root)
    if store is None:
        raise _error('Current operation database is absent.', path=str(root))
    if phase_scope is not None:
        # Preserve the archived binding while passing the caller's optional
        # trace dependency through the normal constructor.
        store = OperationStore(root, plan_sha256=store.plan_sha256,
                               workspace_instance_id=store.workspace_instance_id, phase_scope=phase_scope)
    return store


@contextmanager
def _snapshot(store):
    with store.reader() as snapshot:
        if snapshot is None:
            raise _error('Operation database disappeared before its read.')
        token = _ACTIVE.set((store.root, snapshot))
        try:
            yield snapshot
        finally:
            _ACTIVE.reset(token)


def read_object(root, reference, *, role):
    if reference.get('role') != role:
        raise _error('SQLite object has the wrong domain role.', expected=role, actual=reference.get('role'))
    active = _ACTIVE.get()
    if active is not None and active[0] == Path(root):
        return active[1].get_object(reference)
    with _snapshot(_existing(Path(root))) as snapshot:
        return snapshot.get_object(reference)


def read_document(root, reference, *, schema, role, bind):
    raw = read_object(root, reference, role=role)
    value = record_document(raw)
    if not isinstance(value, dict) or record_bytes(value) != raw or value.get('schema') != schema or value.get('schema_version') != 3:
        raise _error('SQLite domain object is malformed or noncanonical.', reference=dict(reference))
    if any(value.get(key) != expected for key, expected in bind.items()):
        raise _error('SQLite domain object belongs to another task.', reference=dict(reference))
    return value


def read_value(root, reference):
    """Read one canonical numerical body; task and occurrence identity live elsewhere."""
    raw = read_object(root, reference, role='benchmark_value')
    value = record_document(raw)
    if not isinstance(value, dict) or record_bytes(value) != raw:
        raise _error('SQLite numerical body is not a canonical record.', reference=dict(reference))
    return value


def _decode(snapshot, reference):
    raw = snapshot.get_object(reference)
    value = record_document(raw)
    if record_bytes(value) != raw:
        raise _error('SQLite operation object is not canonical.', reference=dict(reference))
    return value


def _decode_timing_reference(snapshot, operation_id, entry):
    if entry['reference'].get('role') != 'operation_timing_batch_ref':
        raise _error('Operation timing stream reference has the wrong role.', operation_id=operation_id)
    reference_row = _decode(snapshot, entry['reference'])
    if (not isinstance(reference_row, dict)
            or reference_row.get('schema') != 'scnsim.operation_timing_reference'
            or reference_row.get('schema_version') != 1
            or reference_row.get('operation_id') != operation_id):
        raise _error('Operation timing stream reference is malformed.', operation_id=operation_id)
    batch_ref = reference_row.get('reference')
    if not isinstance(batch_ref, Mapping) or batch_ref.get('role') != 'operation_timing_batch':
        raise _error('Operation timing batch reference has the wrong role.', operation_id=operation_id)
    batch_raw = snapshot.get_object(batch_ref)
    batch = record_document(batch_raw)
    if (record_bytes(batch) != batch_raw
            or batch.get('schema') != 'scnsim.operation_timing_batch'
            or batch.get('schema_version') != 1
            or batch.get('operation_id') != operation_id
            or batch.get('batch_id') != reference_row.get('batch_id')):
        raise _error('Operation timing batch does not match its stream reference.', operation_id=operation_id)
    return batch


def _pointer(snapshot, name):
    pointer = snapshot.read_pointer(name)
    return None if pointer is None else _decode(snapshot, pointer['reference'])


def _operation_manifest(binding, clock_binding):
    declaration = {'schema':'scnsim.operation_trace','schema_version':1,
                   'plan_sha256':binding.plan_sha256,'workspace_instance_id':binding.workspace_instance_id}
    return {'schema':'scnsim.benchmark_record','schema_version':_OPERATION_STORE_VERSION,
                'benchmark_sha256':sha256(record_bytes(declaration)).hexdigest(),
                'declaration':declaration,'plan_sha256':binding.plan_sha256,
                'workspace_instance_id':binding.workspace_instance_id,'clock':dict(clock_binding)}


def initialize_operation_record(binding, *, clock_binding):
    """Create the SQLite schema before the atomic initial manifest/root write."""
    del clock_binding
    store = _bound(binding)
    store.initialize()
    return store.root


def recover_operation_workspace(binding):
    """Recover this bound leaf and prune only lease-proven inactive scratch.

    Caller holds binding.writer(); no complete evidence audit or reader-side
    recovery is implied. Missing SQL storage stays absent, never migrated.
    Temporary spool payload cleanup is confined to this verified leaf's
    operation scratch root and protected by each owner's stable operation
    lease; committed evidence and lease files are not removed.
    """
    store = _bound(binding)
    if store.path.exists() or store.path.is_symlink():
        store.recover()
    cleanup_inactive_operation_scratch(operation_workspace(binding), binding.root)


def _index_operation(tx, row, reference):
    tx.index_operation(row['operation_id'], reference, method=row.get('method'), backend=row.get('backend'),
                       precision=row.get('precision'), status=row.get('status'), start_ns=row.get('start_ns'),
                       end_ns=row.get('end_ns'), clock_id=row.get('clock',{}).get('id'))


def _publish_operation(binding, row):
    store = _bound(binding)
    row=dict(row)
    row_bytes = None if row is None else record_bytes(row)
    txid = str(uuid4())
    with store.transaction(txid, expected_revisions={}) as tx:
        _index_operation(tx, row, tx.put_object(row_bytes, role='operation'))
        details = row.get('details')
        if isinstance(details, Mapping) and details.get('cache_hit') is True:
            request_sha = row.get('request_sha256')
            selected_pointer = tx.read_pointer('success/'+str(request_sha))
            if selected_pointer is None or selected_pointer['reference'].get('role') != 'success_selection':
                selected = None
            else:
                selected = _decode(tx, selected_pointer['reference'])
            if selected is None:
                raise _error('A cache-hit operation lacks a selected numerical result.',
                             operation_id=row.get('operation_id'))
            association = _operation_association(
                row['operation_id'], selected['task_id'], kind='cache_hit',
                selected_attempt_id=selected['attempt_id'],
                selected_result_ref=selected['result_ref'],
            )
            association_ref = tx.put_object(record_bytes(association), role='operation_task_association')
            tx.set_pointer(_association_pointer(row['operation_id']), association_ref)
    return {'committed':True, 'transaction_id':txid}


def start_operation(binding, row, *, clock_binding=None):
    """Publish a fresh manifest and its initial operation root together."""
    store = _bound(binding)
    store.initialize()
    row = dict(row)
    clock = row.get('clock', {}) if clock_binding is None else clock_binding
    manifest_raw = record_bytes(_operation_manifest(binding, clock))
    operation_raw = record_bytes(row)
    transaction_id = str(uuid4())
    with store.transaction(transaction_id, expected_revisions={}) as tx:
        existing = tx.read_pointer('manifest')
        if existing is None:
            tx.select_first('manifest', tx.put_object(manifest_raw, role='operation_manifest'))
        else:
            manifest = _decode(tx, existing['reference'])
            if (manifest.get('schema') != 'scnsim.benchmark_record'
                    or manifest.get('schema_version') != _OPERATION_STORE_VERSION
                    or manifest.get('plan_sha256') != binding.plan_sha256
                    or manifest.get('workspace_instance_id') != binding.workspace_instance_id):
                raise _error('Operation manifest differs from its bound Plan leaf.')
        reference = tx.put_object(operation_raw, role='operation')
        _index_operation(tx, row, reference)
    return {'committed': True, 'transaction_id': transaction_id}


@contextmanager
def _registration_transaction(store, transaction_id):
    transaction = None
    try:
        with store.transaction(transaction_id, expected_revisions={}) as current:
            transaction = current
            yield current
    except BaseException as error:
        outcome = getattr(error, 'operation_transaction_outcome', None)
        if outcome is None and transaction is not None:
            outcome = transaction.outcome
        if outcome is not None:
            error.operation_registration_outcome = dict(outcome)
            registration = getattr(transaction, 'operation_registration_result', None)
            if outcome.get('status') == 'committed' and registration is not None:
                error.operation_registration_result = dict(registration)
        raise


def register_operation_execution(binding, *, row, task, request, attempt_id, resume_from=None):
    """Atomically register the request, task, running attempt, and operation binding.

    The caller holds the bound Plan writer lock. Request bytes and the exact
    in-memory operation/task identities are committed in one OperationStore
    transaction, so a task cannot exist without its owning operation attempt.
    """
    request_sha256 = request['request_sha256']
    request_bytes = request['request_bytes']
    if sha256(request_bytes).hexdigest() != request_sha256:
        raise _error('Prepared operation request does not match its identity.')
    operation_row = dict(row)
    operation_id = operation_row['operation_id']
    task_row = dict(task)
    if task_row.get('request_sha256') != request_sha256:
        raise _error('Operation task request differs from its prepared request.', task_id=task_row.get('task_id'))
    store = _bound(binding)
    task_id = task_row['task_id']
    transaction_id = str(uuid4())
    task_change = {
        'kind': 'attempt_begin',
        'attempt': {
            'attempt_id': attempt_id,
            'status': 'running',
            'resume_from': resume_from,
            'artifacts': [],
            'failure': None,
            'interruption': None,
        },
    }
    with _registration_transaction(store, transaction_id) as tx:
        operation = next((item for item in tx.query_operations(operation_ids=(operation_id,))), None)
        if operation is None or operation['reference'].get('role') != 'operation':
            raise _error('Operation root must be committed before execution registration.',
                         operation_id=operation_id)
        request_ref = tx.put_object(request_bytes, role='operation_request')
        task_row['artifacts'] = [request_ref]
        descriptor = _new_task_descriptor(task_row)
        if descriptor['task_id'] != task_id or descriptor['request_sha256'] != request_sha256:
            raise _error('Prepared task descriptor differs from its operation request.', task_id=task_id)
        existing_descriptor = _pointer(tx, 'task/' + task_id)
        if existing_descriptor is None:
            descriptor_raw = record_bytes(descriptor)
            descriptor_ref = tx.put_object(descriptor_raw, role='task_descriptor')
            tx.set_pointer('task/' + task_id, descriptor_ref)
            tx.append('tasks', 'task_descriptor', descriptor_raw)
        else:
            existing_fields = ('task_id', 'request_sha256', 'arm', 'sample')
            if (any(existing_descriptor.get(key) != descriptor.get(key) for key in existing_fields)
                    or existing_descriptor.get('environment', {}).get('environment_sha256')
                    != descriptor.get('environment', {}).get('environment_sha256')
                    or request_ref not in existing_descriptor.get('artifacts', ())):
                raise _error('Benchmark task identity was rebound.', task_id=task_id)
        if tx.read_pointer('attempt/' + task_id + '/' + attempt_id) is not None:
            raise _error('Attempt identifier is already recorded.', attempt_id=attempt_id)
        attempt = task_change['attempt']
        attempt_ref = tx.put_object(record_bytes(attempt), role='attempt')
        tx.set_pointer('attempt/' + task_id + '/' + attempt_id, attempt_ref)
        tx.set_pointer('current_attempt/' + task_id, attempt_ref)
        _, change_ref = tx.append_with_ref(_stream(task_id), 'task_change', record_bytes(task_change))
        tx.set_pointer(_attempt_state_pointer(task_id, attempt_id), change_ref)
        association = _operation_association(operation_id, task_id, attempt_id=attempt_id)
        tx.set_pointer(_association_pointer(operation_id),
                        tx.put_object(record_bytes(association), role='operation_task_association'))
        details = dict(operation_row.get('details', {}))
        details['request'] = request_ref
        operation_row['details'] = details
        operation_ref = tx.put_object(record_bytes(operation_row), role='operation')
        _index_operation(tx, operation_row, operation_ref)
        tx.operation_registration_result = {
            'request_ref': request_ref,
            'attempt_binding': {
                'operation_id': operation_id,
                'task_id': task_id,
                'attempt_id': attempt_id,
                'request_sha256': request_sha256,
                'environment_sha256': descriptor['environment']['environment_sha256'],
            },
        }
    return {
        'ack': {'committed': True, 'transaction_id': transaction_id},
        'request_ref': request_ref,
        'attempt_binding': {
            'operation_id': operation_id,
            'task_id': task_id,
            'attempt_id': attempt_id,
            'request_sha256': request_sha256,
            'environment_sha256': descriptor['environment']['environment_sha256'],
        },
    }


def bind_operation(binding, *, row):
    return _publish_operation(binding, row)


def bind_operation_attempt(binding, *, row):
    return _publish_operation(binding, row)


def finish_operation(binding, row, *, failure=None):
    value = dict(row)
    if failure is not None:
        value['details'] = {**value.get('details',{}), 'failure':_error_evidence(failure)}
    return _publish_operation(binding, value)


def _association_pointer(operation_id):
    return 'operation_task/'+operation_id


def _attempt_state_pointer(task_id, attempt_id):
    return 'attempt_state/'+task_id+'/'+attempt_id


def _task_state_from_snapshot(snapshot, operation, store):
    operation_id = operation.get('operation_id')
    pointer = snapshot.read_pointer(_association_pointer(operation_id))
    if pointer is None:
        return {'association':None, 'task_status':None, 'latest_state_reference':None,
                'latest_state':None, 'checkpoint_reference':None,
                'checkpoint':None, 'checkpoint_attempt_id':None}
    association = _decode(snapshot, pointer['reference'])
    if (pointer['reference'].get('role') != 'operation_task_association'
            or association.get('schema') != 'scnsim.operation_task_association'
            or association.get('schema_version') != 1
            or association.get('operation_id') != operation_id):
        raise _error('Operation task association is malformed.', operation_id=operation_id)
    task_id = association.get('task_id')
    attempt_id = association.get('attempt_id')
    if association.get('kind') == 'cache_hit':
        attempt_id = association.get('selected_attempt_id')
    latest_reference = None
    latest_state = None
    task_status = None
    checkpoint_reference = None
    checkpoint = None
    checkpoint_attempt_id = None
    if isinstance(task_id, str) and isinstance(attempt_id, str):
        state = snapshot.read_pointer(_attempt_state_pointer(task_id, attempt_id))
        if state is not None:
            latest_reference = state['reference']
            if latest_reference.get('role') != 'task_change':
                raise _error('Latest task-state pointer has the wrong role.', attempt_id=attempt_id)
            latest_state = _decode(snapshot, latest_reference)
            if latest_state.get('kind') == 'attempt_begin':
                task_status = latest_state.get('attempt', {}).get('status')
            elif latest_state.get('kind') == 'attempt_update':
                task_status = latest_state.get('status')
            if latest_state.get('attempt_id') not in (None, attempt_id):
                raise _error('Latest task-state pointer belongs to another attempt.', attempt_id=attempt_id)

        request_sha256 = operation.get('request_sha256')
        if isinstance(request_sha256, str):
            physical = _physical_checkpoint_from_snapshot(
                snapshot, store, task_id=task_id, attempt_id=attempt_id,
                expected_request_sha256=request_sha256,
            )
            if physical is not None:
                checkpoint_reference = physical['checkpoint_ref']
                checkpoint = physical['resume_from']
                checkpoint_attempt_id = attempt_id

        # A resumed attempt may not have committed a newer checkpoint yet.
        # Its allocation names the exact source; expose it only after verifying
        # that source's immutable checkpoint boundary in this same snapshot.
        if checkpoint is None:
            attempt_pointer = snapshot.read_pointer(f'attempt/{task_id}/{attempt_id}')
            if attempt_pointer is not None:
                if attempt_pointer.get('reference', {}).get('role') != 'attempt':
                    raise _error('Attempt pointer has the wrong role.', attempt_id=attempt_id)
                attempt_record = _decode(snapshot, attempt_pointer['reference'])
                if attempt_record.get('attempt_id') != attempt_id:
                    raise _error('Attempt pointer belongs to another attempt.', attempt_id=attempt_id)
                resume_from = attempt_record.get('resume_from')
                if resume_from is not None:
                    if (not isinstance(resume_from, Mapping)
                            or set(resume_from) != {'task_id', 'attempt_id', 'checkpoint_ref'}
                            or resume_from.get('task_id') != task_id
                            or not isinstance(resume_from.get('attempt_id'), str)
                            or not isinstance(resume_from.get('checkpoint_ref'), Mapping)):
                        raise _error('Attempt resume source is malformed.', attempt_id=attempt_id)
                    source_attempt_id = resume_from['attempt_id']
                    source = _physical_checkpoint_from_snapshot(
                        snapshot, store, task_id=task_id, attempt_id=source_attempt_id,
                        expected_request_sha256=request_sha256,
                        reference=resume_from['checkpoint_ref'],
                    )
                    if source is None:
                        raise _error('Attempt resume source has no committed checkpoint.',
                                     task_id=task_id, attempt_id=source_attempt_id)
                    checkpoint_reference = source['checkpoint_ref']
                    checkpoint = source['resume_from']
                    checkpoint_attempt_id = source_attempt_id

        legacy_checkpoint = snapshot.read_pointer('checkpoint/'+task_id+'/'+attempt_id)
        if checkpoint_reference is None and legacy_checkpoint is not None:
            checkpoint_reference = legacy_checkpoint['reference']
            if checkpoint_reference.get('role') != 'checkpoint_selection':
                raise _error('Current checkpoint selection has the wrong role.', attempt_id=attempt_id)
            selection = _decode(snapshot, checkpoint_reference)
            if selection.get('attempt_id') != attempt_id or selection.get('task_id') != task_id:
                raise _error('Current checkpoint selection belongs to another attempt.', attempt_id=attempt_id)
            checkpoint = dict(checkpoint_reference)
            checkpoint_attempt_id = attempt_id
    return {'association':association, 'task_status':task_status,
            'latest_state_reference':latest_reference, 'latest_state':latest_state,
            'checkpoint_reference':checkpoint_reference, 'checkpoint':checkpoint,
            'checkpoint_attempt_id':checkpoint_attempt_id}


def _attempt_after_changes(attempt, changes):
    """Apply committed task changes to the small mutable attempt projection."""
    result = dict(attempt)
    result['artifacts'] = list(attempt.get('artifacts', ()))

    def add_artifacts(values):
        for artifact in values or ():
            if artifact not in result['artifacts']:
                result['artifacts'].append(artifact)

    for change in changes:
        kind = change.get('kind')
        if kind == 'attempt_update' and change.get('attempt_id') == result.get('attempt_id'):
            status = change.get('status')
            if status is not None:
                result['status'] = status
                result['failure'] = change.get('failure')
                result['interruption'] = change.get('interruption')
            if change.get('checkpoint') is not None:
                result['checkpoint'] = change['checkpoint']
            add_artifacts(change.get('artifacts'))
        elif kind == 'event':
            event = change.get('event', {})
            payload = event.get('payload', {}) if isinstance(event, Mapping) else {}
            if not isinstance(payload, Mapping) or payload.get('attempt_id') != result.get('attempt_id'):
                continue
            event_kind = event.get('kind')
            if event_kind in {'completed', 'failed', 'interrupted'}:
                result['status'] = {'completed':'success', 'failed':'failure',
                                    'interrupted':'interrupted'}[event_kind]
                result['failure'] = payload.get('failure') if event_kind == 'failed' else None
                result['interruption'] = payload.get('interruption') if event_kind == 'interrupted' else None
            if change.get('checkpoint') is not None:
                result['checkpoint'] = change['checkpoint']
            add_artifacts(change.get('artifacts'))
    return result


def _query_operation_snapshot(snapshot, *, store, operation_ids=None, method=None, backend=None,
                              precision=None, status=None):
    rows = snapshot.query_operations(operation_ids=operation_ids, method=method, backend=backend,
                                     precision=precision, status=None)
    operations = []
    for row in rows:
        if row['reference'].get('role') != 'operation':
            raise _error('Indexed operation root has the wrong role.', operation_id=None)
        operation = record_document(row['payload'])
        if record_bytes(operation) != row['payload']:
            raise _error('Indexed operation row is not canonical.', reference=row['reference'])
        operations.append(operation)
    task_states = {
        row['operation_id']:_task_state_from_snapshot(snapshot, row, store)
        for row in operations
    }
    if status is not None:
        matching_ids = {
            operation['operation_id']
            for operation in operations
            if operation.get('status') == status
            or task_states[operation['operation_id']].get('task_status') == status
        }
        operations = [operation for operation in operations
                      if operation['operation_id'] in matching_ids]
        task_states = {operation_id: state for operation_id, state in task_states.items()
                       if operation_id in matching_ids}
    timing_batches = []
    for operation in operations:
        operation_id = operation['operation_id']
        stream = snapshot.read_stream('operation_timing/'+operation_id)
        for entry in stream['entries']:
            if entry['kind'] != 'operation_timing_batch_ref':
                raise _error('Operation timing stream contains an unknown entry.', operation_id=operation_id)
            timing_batches.append(_decode_timing_reference(snapshot, operation_id, entry))
    return {'operations':operations, 'timing_batches':timing_batches, 'task_states':task_states}


def query_operation_rows(binding, *, operation_ids=None, method=None, backend=None, precision=None, status=None):
    store = _bound(binding)
    if not store.path.exists() and not store.path.is_symlink():
        return None
    with _snapshot(store) as snapshot:
        return _query_operation_snapshot(snapshot, store=store, operation_ids=operation_ids, method=method,
                                         backend=backend, precision=precision, status=status)


def publish_timing_batch(binding, batch):
    """Commit one canonical timing batch independently of numerical state."""
    value = dict(batch)
    raw = record_bytes(value)
    if (value.get('schema') != 'scnsim.operation_timing_batch'
            or value.get('schema_version') != 1
            or not isinstance(value.get('operation_id'), str)
            or not isinstance(value.get('batch_id'), str)):
        raise _error('Operation timing batch identity is malformed.')
    if record_bytes(record_document(raw)) != raw:
        raise _error('Operation timing batch is not canonical.', batch_id=value['batch_id'])
    operation_id = value['operation_id']
    batch_id = value['batch_id']
    transaction_id = str(uuid4())
    batch_reference = None
    transaction = None
    try:
        with binding.writer():
            store = _bound(binding)
            with store.transaction(transaction_id, expected_revisions={}) as tx:
                batch_reference = tx.put_object(raw, role='operation_timing_batch')
                reference_row = {
                    'schema': 'scnsim.operation_timing_reference',
                    'schema_version': 1,
                    'operation_id': operation_id,
                    'batch_id': batch_id,
                    'reference': batch_reference,
                }
                tx.append('operation_timing/'+operation_id, 'operation_timing_batch_ref',
                          record_bytes(reference_row))
                transaction = tx
    except Exception as error:
        outcome = getattr(error, 'operation_transaction_outcome', None)
        if outcome is None and transaction is not None:
            outcome = transaction.outcome
        if outcome is not None and outcome.get('status') == 'committed' and batch_reference is not None:
            return {'state':'committed', 'reconciled':True, 'transaction_id':transaction_id,
                    'batch_id':batch_id, 'reference':batch_reference}
        raise
    return {'state':'committed', 'reconciled':False, 'transaction_id':transaction_id,
            'batch_id':batch_id, 'reference':batch_reference}


def open_record(workspace):
    root = _root(workspace)
    store = OperationStore.open_readonly(root)
    if store is None:
        legacy_path = root/'benchmark.json'
        if legacy_path.exists():
            raise _unsupported_old_record(legacy_path)
        raise _error('Operation database is absent.', path=str(root))
    from ..diagnostics.operations import project_indexed_operation_document
    with _snapshot(store) as snapshot:
        indexed = _query_operation_snapshot(snapshot, store=store)
    return project_indexed_operation_document(
        indexed, workspace=root, plan_sha256=store.plan_sha256,
        workspace_instance_id=store.workspace_instance_id
    )


def _manifest(snapshot):
    value = _pointer(snapshot, 'manifest')
    if value is None:
        raise _error('SQLite operation store lacks its bound manifest.')
    return value


def _descriptor(snapshot, task_id):
    descriptor = _pointer(snapshot, 'task/'+task_id)
    if descriptor is None:
        raise _error('Operation task identity is not recorded.',task_id=task_id)
    return descriptor


def _operation_association(operation_id, task_id, *, attempt_id=None, kind='execution',
                           selected_attempt_id=None, selected_result_ref=None):
    association = {
        'schema':'scnsim.operation_task_association', 'schema_version':1,
        'operation_id':operation_id, 'kind':kind, 'task_id':task_id, 'attempt_id':attempt_id,
    }
    if kind == 'cache_hit':
        association['selected_attempt_id'] = selected_attempt_id
        association['selected_result_ref'] = selected_result_ref
    return association


def ensure_task(workspace, task, *, binding, operation_id=None):
    """Register or verify identity without reading the task's saved history."""
    root = _root(workspace)
    descriptor = _new_task_descriptor(task)
    raw = record_bytes(descriptor)
    store = _writer_store(root, binding)
    with store.transaction(str(uuid4()),expected_revisions={}) as tx:
        prior = _pointer(tx,'task/'+descriptor['task_id'])
        if prior is not None:
            fields=('task_id','request_sha256','arm','sample')
            if any(prior[key]!=descriptor[key] for key in fields) or prior['environment']['environment_sha256']!=descriptor['environment']['environment_sha256']:
                raise _error('Benchmark task identity was rebound.',task_id=descriptor['task_id'])
        else:
            ref=tx.put_object(raw,role='task_descriptor')
            tx.set_pointer('task/'+descriptor['task_id'],ref)
            tx.append('tasks','task_descriptor',raw)
        if operation_id is not None:
            association = _operation_association(operation_id, descriptor['task_id'])
            current = _pointer(tx, _association_pointer(operation_id))
            if current is None:
                reference = tx.put_object(record_bytes(association), role='operation_task_association')
                tx.set_pointer(_association_pointer(operation_id), reference)
            elif current != association:
                raise _error('Operation task association changed during registration.',
                             operation_id=operation_id)


def ensure_operation_task(binding, task, *, operation_id):
    ensure_task(operation_workspace(binding), task, binding=binding, operation_id=operation_id)


def _stream(task_id):
    return 'task/'+task_id+'/history'


def _append_change(store, task_id, change, *, operation_id):
    raw=record_bytes(change)
    with store.transaction(str(uuid4()),expected_revisions={}) as tx:
        _descriptor(tx,task_id)
        association = _pointer(tx, _association_pointer(operation_id))
        attempt_id = change.get('attempt_id')
        if (association is None or association.get('task_id') != task_id
                or association.get('attempt_id') != attempt_id):
            raise _error('Attempt update differs from its operation association.',
                         operation_id=operation_id, task_id=task_id, attempt_id=attempt_id)
        _, reference = tx.append_with_ref(_stream(task_id),'task_change',raw)
        if change.get('kind') == 'attempt_update':
            tx.set_pointer(_attempt_state_pointer(task_id, change['attempt_id']), reference)
        elif change.get('kind') == 'attempt_begin':
            tx.set_pointer(_attempt_state_pointer(task_id, change['attempt']['attempt_id']), reference)


def begin_attempt(workspace, *, binding, operation_id, task_id, attempt_id, resume_from=None):
    root=_root(workspace)
    store=_writer_store(root,binding)
    attempt={'attempt_id':attempt_id,'status':'allocated','resume_from':resume_from,
             'artifacts':[],'failure':None,'interruption':None}
    raw=record_bytes({'kind':'attempt_begin','attempt':attempt})
    with store.transaction(str(uuid4()),expected_revisions={}) as tx:
        _descriptor(tx,task_id)
        if tx.read_pointer('attempt/'+task_id+'/'+attempt_id) is not None:
            raise _error('Attempt identifier is already recorded.',attempt_id=attempt_id)
        ref=tx.put_object(record_bytes(attempt),role='attempt')
        tx.set_pointer('attempt/'+task_id+'/'+attempt_id,ref)
        tx.set_pointer('current_attempt/'+task_id,ref)
        _, change_reference = tx.append_with_ref(_stream(task_id),'task_change',raw)
        tx.set_pointer(_attempt_state_pointer(task_id, attempt_id), change_reference)
        association = _pointer(tx, _association_pointer(operation_id))
        if (association is None or association.get('operation_id') != operation_id
                or association.get('task_id') != task_id or association.get('kind') != 'execution'):
            raise _error('Operation task association is missing or belongs to another task.',
                         operation_id=operation_id, task_id=task_id)
        if association.get('attempt_id') not in (None, attempt_id):
            raise _error('Operation task association already names another attempt.', operation_id=operation_id)
        association['attempt_id'] = attempt_id
        tx.set_pointer(_association_pointer(operation_id),
                       tx.put_object(record_bytes(association), role='operation_task_association'))


def update_attempt(workspace, *, binding, operation_id, task_id, attempt_id, status, failure=None, interruption=None, artifacts=(),checkpoint=None):
    _append_change(_writer_store(workspace,binding),task_id,{'kind':'attempt_update','attempt_id':attempt_id,'status':status,
          'failure':failure,'interruption':interruption,'checkpoint':checkpoint,'artifacts':list(artifacts)},
          operation_id=operation_id)


def task_record(workspace, task_id):
    root=_root(workspace)
    store=OperationStore.open_readonly(root)
    if store is None:
        if (root/'benchmark.json').exists():
            raise _unsupported_old_record(root/'benchmark.json')
        raise _error('Operation database is absent.', path=str(root))
    with _snapshot(store) as snapshot:
        descriptor=_pointer(snapshot,'task/'+task_id)
        if descriptor is None:
            raise _error('Operation task identity is not recorded.', task_id=task_id)
        return _materialize_task(root,snapshot,descriptor)


def _materialize_task(root,snapshot,descriptor):
    stream=snapshot.read_stream(_stream(descriptor['task_id']))
    changes=[_decode(snapshot,row['reference']) for row in stream['entries']]
    from .task_history import task_document_from_changes
    task,_=task_document_from_changes(root, descriptor, changes, _manifest(snapshot))
    return task


def operation_task_record(binding,task_id):
    store=_bound(binding)
    if store.path.exists() or store.path.is_symlink():
        with _snapshot(store) as snapshot:
            descriptor=_pointer(snapshot,'task/'+task_id)
            task=None if descriptor is None else _materialize_task(store.root,snapshot,descriptor)
    else:
        task=None
    if task is None:
        raise _error('Operation task identity is not recorded.', task_id=task_id)
    request=next((item for item in task['artifacts'] if item.get('role')=='operation_request'),None)
    if request is None or request['sha256']!=task['request_sha256']:
        raise _error('Operation task lacks its canonical request.',task_id=task_id)
    raw=read_object(operation_workspace(binding), request, role='operation_request')
    if sha256(raw).hexdigest()!=task['request_sha256']:
        raise _error('Operation request differs from task identity.',task_id=task_id)
    return task


def write_artifact(workspace, relative_path, payload, *, binding, role):
    # Historical callers supplied a suggested filename. Current content refs
    # deliberately identify an object, never a pretend row-as-file path.
    with _writer_store(workspace,binding).transaction(str(uuid4()),expected_revisions={}) as tx:
        return tx.put_object(payload,role=role)


def store_operation_request(binding, *, request_sha256,request_bytes):
    if sha256(request_bytes).hexdigest()!=request_sha256:
        raise _error('Prepared operation request does not match its identity.')
    return write_artifact(operation_workspace(binding),None,request_bytes,binding=binding,role='operation_request')


class _Objects:
    """Collect the canonical immutable objects for one task transaction."""
    def __init__(self):
        self.objects={}
    def put(self,value,role):
        return self.put_bytes(record_bytes(value),role)
    def put_bytes(self,raw,role):
        digest=sha256(raw).hexdigest()
        ref=_reference(digest, role, len(raw))
        self.objects[(digest,role)]=(bytes(raw),role)
        return ref
    def publish(self,tx):
        for raw,role in self.objects.values():
            tx.put_object(raw,role=role)


class TaskWriter:
    """Publish ordinary task events and a deferred root-anchor baseline."""
    def __init__(self,root,task_id,attempt_id,*,binding,operation_id,diagnostics):
        self.root=Path(root); self.task_id=task_id; self.attempt_id=attempt_id
        self.operation_id=operation_id
        self.store=_writer_store(self.root,binding)
        self.diagnostics=diagnostics
        self.pending=[]
        # Root-analysis baseline evidence joins the terminal transaction; it
        # is not an independent recovery boundary.
        self.baseline_pending=None
        self.publication_uncertain=None; self.last_ack=self._empty_ack()
        with _snapshot(self.store) as snapshot:
            self.descriptor=_descriptor(snapshot,task_id)
            association=_pointer(snapshot,_association_pointer(operation_id))
            if (association is None or association.get('task_id')!=task_id
                    or association.get('attempt_id')!=attempt_id):
                raise _error('Task writer differs from its operation association.',
                             operation_id=operation_id, task_id=task_id, attempt_id=attempt_id)
            self.revision=snapshot.stream_revision(_stream(task_id))
            pointer=snapshot.read_pointer('event_sequence/'+task_id)
            self.sequence_hint=0 if pointer is None else _decode(snapshot,pointer['reference'])['next_sequence']
            attempt_pointer=snapshot.read_pointer('attempt/'+task_id+'/'+attempt_id)
            if attempt_pointer is None or attempt_pointer['reference'].get('role')!='attempt':
                raise _error('Task writer has no canonical attempt allocation.',
                             task_id=task_id,attempt_id=attempt_id)
            self.attempt_document=_decode(snapshot,attempt_pointer['reference'])
            if self.attempt_document.get('attempt_id')!=attempt_id:
                raise _error('Task writer attempt allocation differs from its identity.',
                             task_id=task_id,attempt_id=attempt_id)

    @staticmethod
    def _empty_ack():
        return {'committed':False,'checkpoint':None}

    def _ensure_publishable(self):
        if self.publication_uncertain is not None:
            raise _error('Task writer cannot replay an unacknowledged publication.',
                         task_id=self.task_id,publication=self.publication_uncertain)

    def _stage(self, value):
        return record_document(record_bytes(value))

    def append_event(self,*,kind,payload,force=False):
        self._ensure_publishable(); self.last_ack=self._empty_ack()
        if kind in {"timing", "operation_span"}:
            raise _error("Timing diagnostics cannot be appended to numerical task history.", kind=kind)
        value=self._stage(payload)
        self.pending.append((kind,value))
        event={'task_id':self.task_id,'sequence':self.sequence_hint,'kind':kind,
               'payload':value}
        self.sequence_hint+=1
        if kind not in _DIAGNOSTIC or force:
            self._publish([],extra_events=list(self.pending))
            self.pending.clear()
        return event

    def commit_barrier(self,kind,payload):
        self._ensure_publishable(); self.last_ack=self._empty_ack()
        if kind != 'baseline_ready':
            raise _error('Task writer supports only deferred root-anchor baselines.', kind=kind)
        value=dict(payload)
        value.pop('resume_state',None)
        segment={'kind':kind,'payload':self._stage(value),'events':list(self.pending)}
        self.pending.clear()
        event={'task_id':self.task_id,'sequence':self.sequence_hint,'kind':kind,
               'payload':value}
        self.sequence_hint+=1
        if self.baseline_pending is not None:
            raise _error('Task baseline was already staged for this attempt.',
                         task_id=self.task_id, attempt_id=self.attempt_id)
        self.baseline_pending=segment
        self.last_ack={'committed':False,'state':'baseline_sealed','checkpoint':None}
        return event,self.last_ack

    def _value(self,objects,value):
        occurrence={key:value[key] for key in _OCCURRENCE if key in value}
        body={key:item for key,item in value.items() if key not in _OCCURRENCE}
        reference=objects.put(body,'benchmark_value')
        return {'reference':reference,'occurrence':occurrence}

    def _publish(self,segments,*,extra_events=(),terminal=None):
        self._ensure_publishable()
        objects=_Objects(); changes=[]
        baseline=None
        with _snapshot(self.store) as snapshot:
            pointer=snapshot.read_pointer('event_sequence/'+self.task_id)
            sequence=0 if pointer is None else _decode(snapshot,pointer['reference'])['next_sequence']
            revision=snapshot.stream_revision(_stream(self.task_id))

        def event(kind,payload,**additional):
            nonlocal sequence
            value=dict(payload)
            compact=value
            if kind=='evaluation':
                compact={_MARKER:{'kind':'value',**self._value(objects,value)}}
                if 'attempt_id' in value: compact['attempt_id']=value['attempt_id']
            item={'task_id':self.task_id,'sequence':sequence,'kind':kind,'payload':compact}
            changes.append({'kind':'event','event':item,**additional}); sequence+=1

        for segment in segments:
            for kind,payload in segment['events']:
                event(kind,payload)
            if segment['kind'] != 'baseline_ready':
                raise _error('Task writer contains an unsupported barrier.', kind=segment['kind'])
            payload=segment['payload']
            marker=self._value(objects,payload['baseline'])
            anchors=None
            if isinstance(payload.get('anchors'),Mapping):
                anchors=objects.put({'schema':'scnsim.benchmark_anchor_values','schema_version':3,
                    'task_id':self.task_id,'anchors':payload['anchors']},'benchmark_anchor_values')
            attributes={k:v for k,v in payload.items() if k not in {'attempt_id','baseline','anchors'}}
            block={'schema':'scnsim.benchmark_baseline_evidence','schema_version':3,'task_id':self.task_id,
                'attempt_id':self.attempt_id,'baseline':marker['reference'],
                'baseline_occurrence':marker['occurrence'],'anchors':anchors,'attributes':attributes}
            baseline=objects.put(block,'benchmark_baseline_evidence')
            event(segment['kind'],{'attempt_id':self.attempt_id,
                 _MARKER:{'kind':'barrier','evidence':baseline,'checkpoint':None}},
                 evidence=baseline,checkpoint=None,artifacts=[])
        for kind,payload in extra_events:
            event(kind,payload)
        result_ref=None; terminal_attempt=None; attempt_reference=None
        if terminal is not None:
            result_ref=objects.put(terminal,'operation_result')
            completion={'attempt_id':self.attempt_id,'result':result_ref}
            terminal_attempt = _attempt_after_changes(self.attempt_document, changes)
            terminal_attempt['status']='success'
            terminal_attempt['failure']=None
            terminal_attempt['interruption']=None
            if result_ref not in terminal_attempt['artifacts']:
                terminal_attempt['artifacts'].append(result_ref)
            attempt_reference=objects.put(terminal_attempt,'attempt')
            event('completed',completion)
            changes.append({'kind':'attempt_update','attempt_id':self.attempt_id,'status':'success',
                  'failure':None,'interruption':None,'checkpoint':None,'artifacts':[result_ref]})
        success_selection=None
        if result_ref is not None:
            selected={'event':'request_success_selected','request_sha256':self.descriptor['request_sha256'],
                      'task_id':self.task_id,'attempt_id':self.attempt_id,'result_ref':result_ref}
            success_selection=objects.put(selected,'success_selection')
        nextref=objects.put({'next_sequence':sequence},'event_sequence')
        txid=str(uuid4()); store=self.store
        committed_attempt = terminal_attempt if terminal_attempt is not None else _attempt_after_changes(
            self.attempt_document, changes
        )
        if attempt_reference is None and committed_attempt != self.attempt_document:
            attempt_reference=objects.put(committed_attempt,'attempt')
        transaction=None
        try:
            with store.transaction(txid,expected_revisions={_stream(self.task_id):revision}) as tx:
                transaction=tx
                current=_pointer(tx,'current_attempt/'+self.task_id)
                if current is None or current['attempt_id']!=self.attempt_id:
                    raise _error('Operation event belongs to another current attempt.',attempt_id=self.attempt_id)
                association=_pointer(tx,_association_pointer(self.operation_id))
                if (association is None or association.get('task_id')!=self.task_id
                        or association.get('attempt_id')!=self.attempt_id):
                    raise _error('Task publication differs from its operation association.',
                                 operation_id=self.operation_id, task_id=self.task_id,
                                 attempt_id=self.attempt_id)
                objects.publish(tx)
                if attempt_reference is not None:
                    tx.set_pointer('attempt/'+self.task_id+'/'+self.attempt_id,attempt_reference)
                    tx.set_pointer('current_attempt/'+self.task_id,attempt_reference)
                for change in changes:
                    raw=record_bytes(change)
                    _, change_reference = tx.append_with_ref(_stream(self.task_id),'task_change',raw)
                    if change.get('kind') == 'attempt_update':
                        tx.set_pointer(_attempt_state_pointer(self.task_id, change['attempt_id']), change_reference)
                    elif change.get('kind') == 'attempt_begin':
                        tx.set_pointer(_attempt_state_pointer(self.task_id, change['attempt']['attempt_id']), change_reference)
                tx.set_pointer('event_sequence/'+self.task_id,nextref)
                if result_ref is not None:
                    tx.select_first('success/'+self.descriptor['request_sha256'],success_selection)
        except BaseException as error:
            self.publication_uncertain=getattr(error,'operation_transaction_outcome',None) or (
                transaction.outcome if transaction is not None else {'status':'unknown','transaction_id':txid})
            raise
        self.revision=revision+len(changes)
        self.attempt_document=committed_attempt
        self.sequence_hint=sequence+max(0,len(self.pending)-len(extra_events))
        self.last_ack={'committed':True,'checkpoint':None,'evidence':baseline,
                      'transaction_id':txid}
        return result_ref,self.last_ack


def begin_task_writer(workspace,*,binding,operation_id,task_id,attempt_id,diagnostics):
    return TaskWriter(
        _root(workspace), task_id, attempt_id, binding=binding,
        operation_id=operation_id, diagnostics=diagnostics,
    )


def append_event(workspace,*,task_id,kind,payload,writer,force=False):
    return writer.append_event(kind=kind,payload=payload,force=force)


def commit_barrier(workspace,*,writer,kind,payload):
    return writer.commit_barrier(kind,payload)


def complete_operation(binding,*,writer,terminal_bytes):
    if (operation_workspace(binding)!=writer.root or binding.plan_sha256!=writer.store.plan_sha256
            or binding.workspace_instance_id!=writer.store.workspace_instance_id):
        raise _error('Operation completion belongs to another bound Plan leaf.')
    terminal=record_document(terminal_bytes)
    if record_bytes(terminal)!=terminal_bytes:
        raise _error('Operation terminal bytes are not canonical.')
    # Root-anchor evidence and terminal task state share one required commit.
    segments=([writer.baseline_pending] if writer.baseline_pending is not None else [])
    result,ack=writer._publish(segments,extra_events=list(writer.pending),terminal=terminal)
    writer.pending.clear()
    writer.baseline_pending=None
    return result,ack


def complete_sensitivity_operation(binding, *, operation_id, task_id, attempt_id,
                                    manifest, result_summary, blocks, outcome,
                                    equation_evidence):
    """Publish one complete or explicitly partial Sensitivity terminal atomically.

    Block bodies are serialized and inserted one at a time inside the same
    SQLite transaction. Only their immutable references and compact summaries
    remain in the in-memory index. A partial terminal is addressable by its
    exact attempt, but never becomes the request's complete-success selection.
    """
    if outcome not in {"success", "partial"}:
        raise _error("Sensitivity terminal outcome is unsupported.", outcome=outcome)
    store = _bound(binding)
    with _snapshot(store) as snapshot:
        descriptor = _descriptor(snapshot, task_id)
        association = _pointer(snapshot, _association_pointer(operation_id))
        descriptor_pointer = snapshot.read_pointer(f"task/{task_id}")
        attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
        current_pointer = snapshot.read_pointer(f"current_attempt/{task_id}")
        if (descriptor.get("request_sha256") is None
                or association is None
                or association.get("operation_id") != operation_id
                or association.get("task_id") != task_id
                or association.get("attempt_id") != attempt_id
                or attempt_pointer is None
                or current_pointer is None
                or current_pointer.get("reference") != attempt_pointer.get("reference")):
            raise _error("Sensitivity terminal is not bound to the current operation attempt.",
                         operation_id=operation_id, task_id=task_id, attempt_id=attempt_id)
        attempt = _decode(snapshot, attempt_pointer["reference"])
        if (attempt.get("attempt_id") != attempt_id
                or attempt.get("status") not in {"allocated", "running"}):
            raise _error("Sensitivity attempt is not in a publishable state.",
                         task_id=task_id, attempt_id=attempt_id)
        request_ref = next((item for item in descriptor.get("artifacts", ())
                            if item.get("role") == "operation_request"), None)
        if (not isinstance(request_ref, Mapping)
                or request_ref.get("sha256") != descriptor.get("request_sha256")):
            raise _error("Sensitivity task lacks its canonical request reference.", task_id=task_id)
        request_raw = snapshot.get_object(request_ref)
        request = record_document(request_raw)
        if (not isinstance(request, dict) or record_bytes(request) != request_raw
                or request.get("operation") != "sensitivity"):
            raise _error("Sensitivity terminal is bound to another request kind.", task_id=task_id)
        stream = _stream(task_id)
        revision = snapshot.stream_revision(stream)
        sequence_pointer = snapshot.read_pointer("event_sequence/" + task_id)
        sequence = (0 if sequence_pointer is None else
                    _decode(snapshot, sequence_pointer["reference"]).get("next_sequence"))
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise _error("Sensitivity task event sequence is malformed.", task_id=task_id)

    workspace_instance_id = store.workspace_instance_id
    plan_sha256 = store.plan_sha256
    request_sha256 = descriptor["request_sha256"]
    if not isinstance(manifest, Mapping) or set(manifest) != {
            "program_sha256", "quantity_layout", "derivative_layout"}:
        raise _error("Sensitivity manifest metadata has an unsupported field set.")
    if not isinstance(result_summary, Mapping):
        raise _error("Sensitivity terminal summary is malformed.")
    from .validation.common import _valid_sha
    _valid_sha(manifest["program_sha256"])
    evidence_fields = {
        "program_sha256", "derivative_kind", "wrt", "output_layout", "certificates",
        "derivative_available", "derivative_reasons", "observables",
    }
    def requests_static_output(value):
        if isinstance(value, Mapping):
            return (value.get("type") in {"static_capacitance_projection", "static_capacitance_element_projection"}
                    or any(requests_static_output(item) for item in value.values()))
        if isinstance(value, (tuple, list)):
            return any(requests_static_output(item) for item in value)
        return False

    if requests_static_output(request["spec"]["quantities"]):
        evidence_fields.add("static_capacitance")
    if isinstance(equation_evidence, Mapping) and "static_capacitance" in equation_evidence:
        static_facts = equation_evidence["static_capacitance"]
        if not isinstance(static_facts, (tuple, list)) or not static_facts:
            raise _error("Static Sensitivity equation evidence must contain its observable facts.")
        static_fields = {
            "observable_index", "row_coordinates", "column_coordinates", "zero_charge",
            "coordinate_order", "original_coordinate_order", "finite_mesh", "coordinate_transform",
        }
        seen_static = set()
        from ..numeric_encoding import array_from_record
        for row in static_facts:
            if not isinstance(row, Mapping) or set(row) != static_fields:
                raise _error("Static Sensitivity observable evidence has an unsupported field set.")
            index = row["observable_index"]
            if type(index) is not int or index < 0 or index in seen_static:
                raise _error("Static Sensitivity observable ordinal is malformed or duplicated.")
            seen_static.add(index)
            for key in ("row_coordinates", "column_coordinates", "zero_charge",
                        "coordinate_order", "original_coordinate_order"):
                coordinates = row[key]
                if (not isinstance(coordinates, (tuple, list))
                        or any(not isinstance(value, str) or not value for value in coordinates)
                        or len(set(coordinates)) != len(coordinates)):
                    raise _error("Static Sensitivity coordinate axes are malformed.", field=key)
            mesh = row["finite_mesh"]
            # _compile_mesh retains explicit FixedMeshSpec sections as their
            # declaration; plan_mesh emits its ordered per-line derivation list.
            # These are current producer forms, not migration/fallback records.
            if isinstance(mesh, Mapping):
                if (set(mesh) != {"type", "resolution", "sections"}
                        or mesh["type"] != "fixed_mesh" or mesh["resolution"] is not None
                        or not isinstance(mesh["sections"], (tuple, list))):
                    raise _error("Static Sensitivity fixed-section provenance is malformed.")
                section_rows = mesh["sections"]
                section_fields = {"path", "count"}
            elif isinstance(mesh, (tuple, list)):
                section_rows = mesh
                section_fields = {"path", "baseline_sections", "planned_sections",
                                  "maximum_length_m", "modal_delay_bound_s_per_m", "physical_policy"}
            else:
                raise _error("Static Sensitivity finite-mesh provenance is malformed.")
            seen_paths = set()
            for section in section_rows:
                if not isinstance(section, Mapping) or set(section) != section_fields:
                    raise _error("Static Sensitivity mesh section has an unsupported field set.")
                path = section["path"]
                if (not isinstance(path, (tuple, list)) or not path
                        or any(not isinstance(part, str) or not part for part in path)
                        or tuple(path) in seen_paths):
                    raise _error("Static Sensitivity mesh occurrence path is malformed or duplicated.")
                seen_paths.add(tuple(path))
                count_fields = ("count",) if isinstance(mesh, Mapping) else ("baseline_sections", "planned_sections")
                if any(type(section[key]) is not int or section[key] < 1 for key in count_fields):
                    raise _error("Static Sensitivity mesh section count is malformed.")
                if not isinstance(mesh, Mapping):
                    if not isinstance(section["physical_policy"], Mapping):
                        raise _error("Static Sensitivity mesh physical-policy provenance is malformed.")
                    for key in ("maximum_length_m", "modal_delay_bound_s_per_m"):
                        value = section[key]
                        if value is not None and type(value) is not float and value not in ("nan", "+inf", "-inf"):
                            raise _error("Static Sensitivity mesh derivation scalar is malformed.", field=key)
            transforms = row["coordinate_transform"]
            if not isinstance(transforms, (tuple, list)):
                raise _error("Static Sensitivity transform facts are malformed.")
            for pair in transforms:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise _error("Static Sensitivity transform must retain its alpha/beta pair.")
                for value in pair:
                    if (not isinstance(value, Mapping) or set(value) != {"dtype", "shape", "data_hex"}
                            or value["dtype"] not in {"float32", "float64"}
                            or value["shape"] != [] or not isinstance(value["data_hex"], str)):
                        raise _error("Static Sensitivity transform scalar encoding is malformed.")
                    # Existing exact-dtype codec owns scalar bit/shape decoding.
                    array_from_record(value)
    if not isinstance(equation_evidence, Mapping) or set(equation_evidence) != evidence_fields:
        raise _error("Sensitivity equation evidence has an unsupported field set.")
    derivative_layout = manifest["derivative_layout"]
    if (equation_evidence.get("program_sha256") != manifest["program_sha256"]
            or not isinstance(derivative_layout, Mapping)
            or equation_evidence.get("derivative_kind") != derivative_layout.get("kind")
            or equation_evidence.get("wrt") != derivative_layout.get("parameters")
            or not isinstance(equation_evidence.get("output_layout"), Mapping)
            or not isinstance(equation_evidence.get("certificates"), (tuple, list))
            or type(equation_evidence.get("derivative_available")) is not bool
            or not isinstance(equation_evidence.get("derivative_reasons"), (tuple, list))
            or not isinstance(equation_evidence.get("observables"), (tuple, list))):
        raise _error("Sensitivity equation evidence differs from its derivative declaration.")
    _valid_sha(equation_evidence["program_sha256"])
    manifest_body = {
        "schema": "scnsim.sensitivity_manifest", "schema_version": 1, "evidence_version": 3,
        "workspace_instance_id": workspace_instance_id, "plan_sha256": plan_sha256,
        "request_sha256": request_sha256, "task_id": task_id, "attempt_id": attempt_id,
        "outcome": outcome, "request_ref": dict(request_ref),
        "program_sha256": manifest["program_sha256"],
        "quantity_layout": manifest["quantity_layout"],
        "derivative_layout": manifest["derivative_layout"],
    }
    manifest_raw = record_bytes(manifest_body)
    equation_evidence_body = {
        "schema": "scnsim.sensitivity_equation_evidence", "schema_version": 1,
        "workspace_instance_id": workspace_instance_id, "plan_sha256": plan_sha256,
        "request_sha256": request_sha256, "task_id": task_id, "attempt_id": attempt_id,
        **dict(equation_evidence),
    }
    equation_evidence_raw = record_bytes(equation_evidence_body)
    transaction_id = str(uuid4())
    blocks_index = []
    primal_refs = {}
    result_ref = manifest_ref = index_root_ref = completion_ref = attempt_ref = None
    equation_evidence_ref = None
    with store.transaction(transaction_id, expected_revisions={_stream(task_id): revision}) as tx:
        current_association = _pointer(tx, _association_pointer(operation_id))
        current_attempt = tx.read_pointer(f"attempt/{task_id}/{attempt_id}")
        current_task = tx.read_pointer(f"task/{task_id}")
        if (descriptor_pointer is None or current_task is None
                or current_task.get("reference") != descriptor_pointer.get("reference")
                or current_association != association
                or current_attempt is None
                or current_attempt.get("reference") != attempt_pointer.get("reference")):
            raise _error("Sensitivity attempt binding changed before terminal publication.",
                         operation_id=operation_id, task_id=task_id, attempt_id=attempt_id)
        manifest_ref = tx.put_object(manifest_raw, role="sensitivity_manifest")
        equation_evidence_ref = tx.put_object(
            equation_evidence_raw, role="sensitivity_equation_evidence",
        )
        for ordinal, source in enumerate(blocks):
            if not isinstance(source, Mapping):
                raise _error("Sensitivity block descriptor is malformed.", ordinal=ordinal)
            kind = source.get("kind")
            block_id = source.get("block_id")
            quantity_id = source.get("quantity_id")
            output_slice = source.get("output_slice")
            parameter_slice = source.get("parameter_slice")
            summary = source.get("summary_json")
            primal_ids = source.get("primal_blocks")
            payload = source.get("payload")
            if (kind not in {"primal", "derivative"}
                    or not isinstance(block_id, str) or not block_id
                    or (quantity_id is not None and (not isinstance(quantity_id, str) or not quantity_id))
                    or (output_slice is not None and not isinstance(output_slice, Mapping))
                    or (parameter_slice is not None and not isinstance(parameter_slice, Mapping))
                    or not isinstance(summary, Mapping)
                    or not isinstance(payload, Mapping)
                    or not isinstance(primal_ids, list)
                    or any(not isinstance(value, str) or not value for value in primal_ids)
                    or len(set(primal_ids)) != len(primal_ids)):
                raise _error("Sensitivity block descriptor is incomplete.", ordinal=ordinal)
            primal_blocks = []
            if kind == "primal":
                if (primal_ids or quantity_id is None or output_slice is None
                        or parameter_slice is not None or block_id in primal_refs):
                    raise _error("Primal Sensitivity block identity or axes are malformed.",
                                 block_id=block_id)
            else:
                if not primal_ids or any(primal_id not in primal_refs for primal_id in primal_ids):
                    raise _error("Derivative Sensitivity block lacks its prior primal blocks.",
                                 block_id=block_id, primal_blocks=primal_ids)
                if quantity_id is None and output_slice is not None:
                    raise _error("Aggregate Sensitivity derivative cannot select one output slice.",
                                 block_id=block_id)
                primal_blocks = [
                    {"block_id": primal_id, "block_ref": primal_refs[primal_id]}
                    for primal_id in primal_ids
                ]
            block_body = {
                "schema": "scnsim.sensitivity_block", "schema_version": 1,
                "workspace_instance_id": workspace_instance_id, "plan_sha256": plan_sha256,
                "request_sha256": request_sha256, "task_id": task_id, "attempt_id": attempt_id,
                "block_id": block_id, "kind": kind, "quantity_id": quantity_id,
                "output_slice": None if output_slice is None else dict(output_slice),
                "parameter_slice": None if parameter_slice is None else dict(parameter_slice),
                "primal_blocks": primal_blocks,
                "primal_status": source.get("primal_status"),
                "derivative_status": source.get("derivative_status"), "payload": dict(payload),
            }
            block_ref = tx.put_object(record_bytes(block_body), role="sensitivity_block")
            if kind == "primal":
                primal_refs[block_id] = block_ref
            blocks_index.append({
                "ordinal": ordinal, "block_id": block_id, "kind": kind,
                "block_ref": block_ref, "quantity_id": quantity_id,
                "output_slice": None if output_slice is None else dict(output_slice),
                "parameter_slice": None if parameter_slice is None else dict(parameter_slice),
                "primal_blocks": primal_blocks,
                "primal_status": source.get("primal_status"),
                "derivative_status": source.get("derivative_status"),
                "summary_json": dict(summary),
            })

        result_summary_body = dict(result_summary)
        if "equation_evidence_ref" in result_summary_body:
            raise _error("Sensitivity result summary cannot replace its equation evidence reference.")
        result_summary_body["equation_evidence_ref"] = equation_evidence_ref
        result_body = {
            "schema": "scnsim.sensitivity_result", "schema_version": 1,
            "outcome": outcome, "workspace_instance_id": workspace_instance_id,
            "plan_sha256": plan_sha256, "request_sha256": request_sha256,
            "task_id": task_id, "attempt_id": attempt_id,
            "manifest_ref": manifest_ref, "block_count": len(blocks_index),
            "summary": result_summary_body,
        }
        result_ref = tx.put_object(record_bytes(result_body), role="operation_result")
        index_root = {
            "schema": "scnsim.sensitivity_index_root", "schema_version": 1,
            "workspace_instance_id": workspace_instance_id, "plan_sha256": plan_sha256,
            "request_sha256": request_sha256, "task_id": task_id, "attempt_id": attempt_id,
            "outcome": outcome, "result_ref": result_ref, "manifest_ref": manifest_ref,
            "request_ref": dict(request_ref), "block_count": len(blocks_index),
            "blocks": blocks_index,
        }
        index_root_ref = tx.put_object(record_bytes(index_root), role="sensitivity_index_root")
        terminal_attempt = dict(attempt)
        terminal_artifacts = list(attempt.get("artifacts", ()))
        for artifact in (result_ref, manifest_ref, index_root_ref, equation_evidence_ref):
            if artifact not in terminal_artifacts:
                terminal_artifacts.append(artifact)
        terminal_attempt.update(status=outcome, failure=None, interruption=None,
                                artifacts=terminal_artifacts)
        attempt_ref = tx.put_object(record_bytes(terminal_attempt), role="attempt")
        completion = {
            "schema": "scnsim.sensitivity_completion", "schema_version": 1,
            "workspace_instance_id": workspace_instance_id, "plan_sha256": plan_sha256,
            "request_sha256": request_sha256, "task_id": task_id, "attempt_id": attempt_id,
            "outcome": outcome, "result_ref": result_ref, "manifest_ref": manifest_ref,
            "index_root_ref": index_root_ref, "terminal_attempt_ref": attempt_ref,
            "block_count": len(blocks_index),
        }
        completion_ref = tx.put_object(record_bytes(completion), role="sensitivity_completion")
        event_payload = {
            "attempt_id": attempt_id, "outcome": outcome, "result": result_ref,
            "sensitivity_completion": completion_ref,
        }
        task_event = {"task_id": task_id, "sequence": sequence,
                      "kind": "sensitivity_terminal", "payload": event_payload}
        tx.append_with_ref(_stream(task_id), "task_change",
                           record_bytes({"kind": "event", "event": task_event,
                                         "artifacts": [result_ref, completion_ref,
                                                       equation_evidence_ref]}))
        attempt_change = {"kind": "attempt_update", "attempt_id": attempt_id,
                          "status": outcome, "failure": None, "interruption": None,
                          "checkpoint": None, "artifacts": [result_ref, manifest_ref,
                                                                 index_root_ref, completion_ref,
                                                                 equation_evidence_ref]}
        _, state_ref = tx.append_with_ref(_stream(task_id), "task_change", record_bytes(attempt_change))
        tx.set_pointer(f"attempt/{task_id}/{attempt_id}", attempt_ref)
        tx.set_pointer(f"current_attempt/{task_id}", attempt_ref)
        tx.set_pointer(_attempt_state_pointer(task_id, attempt_id), state_ref)
        tx.set_pointer(f"sensitivity_completion/{task_id}/{attempt_id}", completion_ref)
        next_ref = tx.put_object(record_bytes({"next_sequence": sequence + 1}), role="event_sequence")
        tx.set_pointer("event_sequence/" + task_id, next_ref)
        if outcome == "success":
            selection = {
                "event": "request_success_selected", "request_sha256": request_sha256,
                "task_id": task_id, "attempt_id": attempt_id, "result_ref": result_ref,
                "sensitivity_completion": completion_ref,
            }
            selection_ref = tx.put_object(record_bytes(selection), role="success_selection")
            tx.select_first("success/" + request_sha256, selection_ref)
        tx.index_sensitivity_blocks(completion_ref["sha256"], blocks_index)
    return {
        "result_ref": result_ref, "manifest_ref": manifest_ref,
        "index_root_ref": index_root_ref, "completion_ref": completion_ref,
        "attempt_ref": attempt_ref, "equation_evidence_ref": equation_evidence_ref,
        "ack": {"committed": True, "transaction_id": transaction_id,
                "outcome": outcome, "block_count": len(blocks_index)},
    }


def _complete_chunked_operation(binding, *, domain, operation_id, task_id, attempt_id,
                                metadata, chunks, summary):
    """Atomically seal one streamed Direct or parameter-sweep result."""
    if domain not in {"direct", "sweep"}:
        raise _error("Chunked result domain is unsupported.", domain=domain)
    if not isinstance(metadata, Mapping) or not isinstance(summary, Mapping):
        raise _error("Chunked result terminal metadata is malformed.", domain=domain)
    store = _bound(binding)
    with _snapshot(store) as snapshot:
        descriptor = _descriptor(snapshot, task_id)
        association = _pointer(snapshot, _association_pointer(operation_id))
        descriptor_pointer = snapshot.read_pointer(f"task/{task_id}")
        attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
        current_pointer = snapshot.read_pointer(f"current_attempt/{task_id}")
        if (association is None or association.get("operation_id") != operation_id
                or association.get("task_id") != task_id
                or association.get("attempt_id") != attempt_id
                or attempt_pointer is None or current_pointer is None
                or current_pointer.get("reference") != attempt_pointer.get("reference")):
            raise _error("Chunked result is not bound to the current operation attempt.",
                         operation_id=operation_id, task_id=task_id, attempt_id=attempt_id)
        attempt = _decode(snapshot, attempt_pointer["reference"])
        if attempt.get("attempt_id") != attempt_id or attempt.get("status") not in {"allocated", "running"}:
            raise _error("Chunked result attempt is not in a publishable state.",
                         task_id=task_id, attempt_id=attempt_id)
        request_ref = next((item for item in descriptor.get("artifacts", ())
                            if item.get("role") == "operation_request"), None)
        if (not isinstance(request_ref, Mapping)
                or request_ref.get("sha256") != descriptor.get("request_sha256")):
            raise _error("Chunked result task lacks its canonical request reference.",
                         task_id=task_id)
        request_raw = snapshot.get_object(request_ref)
        request = record_document(request_raw)
        source = request.get("parameter_source") if isinstance(request, Mapping) else None
        spec = request.get("spec") if isinstance(request, Mapping) else None
        is_sweep = isinstance(source, Mapping) and source.get("kind") in {"grid", "points"}
        is_direct_points = (
            isinstance(source, Mapping) and source.get("kind") in {"grid", "points"}
            and isinstance(spec, Mapping) and spec.get("type") == "direct_solve"
        )
        if (not isinstance(request, dict) or record_bytes(request) != request_raw
                or request.get("runtime_semantic", {}).get("backend") != "jax"
                or (domain == "direct" and (
                    request.get("operation") != "solve_direct"
                    or not isinstance(source, Mapping) or source.get("kind") != "point"
                ))
                or (domain == "sweep" and (
                    not (
                        request.get("operation") == "evaluate_direct" and is_sweep
                        or request.get("operation") == "solve_direct" and is_direct_points
                    )
                ))):
            raise _error("Chunked result domain differs from its canonical request.",
                         domain=domain, task_id=task_id)
        expected_sweep_points = None
        if domain == "sweep":
            from .validation.common import _parameter_source_points

            expected_sweep_points = iter(_parameter_source_points(source))
        stream = _stream(task_id)
        revision = snapshot.stream_revision(stream)
        sequence_pointer = snapshot.read_pointer("event_sequence/" + task_id)
        sequence = (0 if sequence_pointer is None else
                    _decode(snapshot, sequence_pointer["reference"]).get("next_sequence"))
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise _error("Chunked result task event sequence is malformed.", task_id=task_id)

    identity = {
        "workspace_instance_id": store.workspace_instance_id,
        "plan_sha256": store.plan_sha256,
        "request_sha256": descriptor["request_sha256"],
        "task_id": task_id,
        "attempt_id": attempt_id,
        "outcome": "success",
    }
    from .validation.common import _valid_sha
    if domain == "direct":
        if not isinstance(metadata.get("program_sha256"), str):
            raise _error("Direct result metadata lacks its physical program identity.",
                         domain=domain)
        _valid_sha(metadata["program_sha256"])
    elif (metadata.get("program_identity_scope") != "per_point"
          or metadata.get("program_sha256") is not None):
        raise _error("Sweep metadata must identify its per-point physical programs.",
                     domain=domain)
    manifest_ref = result_ref = index_root_ref = completion_ref = attempt_ref = None
    chunk_rows = []
    point_count = 0
    status_counts = {}
    transaction_id = str(uuid4())
    with store.transaction(transaction_id, expected_revisions={_stream(task_id): revision}) as tx:
        current_association = _pointer(tx, _association_pointer(operation_id))
        current_attempt = tx.read_pointer(f"attempt/{task_id}/{attempt_id}")
        current_task = tx.read_pointer(f"task/{task_id}")
        if (descriptor_pointer is None or current_task is None
                or current_task.get("reference") != descriptor_pointer.get("reference")
                or current_association != association
                or current_attempt is None
                or current_attempt.get("reference") != attempt_pointer.get("reference")):
            raise _error("Chunked result binding changed before terminal publication.",
                         operation_id=operation_id, task_id=task_id, attempt_id=attempt_id)
        manifest_body = {
            "schema": f"scnsim.{domain}_manifest", "schema_version": 1, "evidence_version": 3,
            **identity, "request_ref": dict(request_ref), "metadata": dict(metadata),
        }
        manifest_ref = tx.put_object(record_bytes(manifest_body), role=f"{domain}_manifest")
        for ordinal, source_chunk in enumerate(chunks(tx) if callable(chunks) else chunks):
            if not isinstance(source_chunk, Mapping):
                raise _error("Chunk source is malformed.", domain=domain, ordinal=ordinal)
            count = source_chunk.get("point_count")
            payload = source_chunk.get("payload")
            chunk_summary = source_chunk.get("summary_json")
            if (not isinstance(count, int) or isinstance(count, bool) or count <= 0
                    or not isinstance(payload, Mapping) or not isinstance(chunk_summary, Mapping)):
                raise _error("Chunk source has incomplete rows or payload.",
                             domain=domain, ordinal=ordinal)
            first_point = point_count
            if domain == "direct":
                if (not isinstance(payload.get("frequencies_hz"), Mapping)
                        or not isinstance(payload.get("families"), Mapping)
                        or set(payload["families"]) != {"S", "Y", "Z"}):
                    raise _error("Direct chunk payload lacks its three actual matrix families.",
                                 ordinal=ordinal)
            else:
                points = payload.get("points")
                if not isinstance(points, list) or len(points) != count:
                    raise _error("Sweep chunk point count differs from its payload.",
                                 ordinal=ordinal)
                from ..authoring.identity import canonical_parameters_sha256
                occurrence_summaries = []
                stored_points = []
                for offset, point in enumerate(points):
                    if not isinstance(point, Mapping) or point.get("ordinal") != first_point + offset:
                        raise _error("Sweep point ordinal differs from its sealed chunk position.",
                                     ordinal=ordinal, offset=offset)
                    try:
                        expected_source_index, expected_parameters = next(expected_sweep_points)
                    except StopIteration as error:
                        raise _error("Sweep chunks contain more points than their request.",
                                     ordinal=ordinal, offset=offset) from error
                    if (point.get("source_index") != expected_source_index
                            or point.get("parameters") != expected_parameters
                            or point.get("parameters_sha256")
                            != canonical_parameters_sha256(expected_parameters)):
                        raise _error("Sweep point differs from its exact request position.",
                                     ordinal=ordinal, offset=offset)
                    status = point.get("status")
                    if status not in {"success", "failure"}:
                        raise _error("Sweep point status is malformed.",
                                     ordinal=ordinal, offset=offset)
                    program_sha256 = point.get("program_sha256")
                    discretization = point.get("discretization")
                    if not isinstance(program_sha256, str) or not isinstance(discretization, list):
                        raise _error("Sweep point lacks its actual program and mesh identity.",
                                     ordinal=ordinal, offset=offset)
                    _valid_sha(program_sha256)
                    status_counts[status] = status_counts.get(status, 0) + 1
                    occurrence = {
                        "ordinal": point["ordinal"],
                        "source_index": expected_source_index,
                        "parameters_sha256": canonical_parameters_sha256(expected_parameters),
                        "status": status,
                        "program_sha256": program_sha256,
                        "discretization": discretization,
                    }
                    if status == "failure":
                        failure = point.get("failure")
                        if (not isinstance(failure, Mapping)
                                or any(not isinstance(failure.get(key), str) or not failure[key]
                                       for key in ("kind", "stage", "detail"))):
                            raise _error("Sweep point failure summary is malformed.",
                                         ordinal=ordinal, offset=offset)
                        occurrence["failure"] = {
                            "kind": failure["kind"], "stage": failure["stage"],
                            "message": failure["detail"],
                        }
                    occurrence_summaries.append(occurrence)
                    stored_points.append(_externalize_sweep_point(tx, point))
                chunk_summary = {"occurrences": occurrence_summaries}
                payload = {**payload, "points": stored_points}
            reserved = {"schema", "schema_version", *identity.keys(), "ordinal", "first_point", "point_count"}
            if reserved.intersection(payload):
                raise _error("Chunk payload repeats its immutable envelope fields.",
                             domain=domain, ordinal=ordinal)
            body = {
                "schema": f"scnsim.{domain}_chunk", "schema_version": 1,
                **identity, "ordinal": ordinal, "first_point": first_point,
                "point_count": count, **dict(payload),
            }
            if domain == "direct":
                body["array_context"] = {key: metadata[key] for key in (
                    "terminal_ids", "input_channels", "output_channels", "family_units")
                    if key in metadata}
            reference = tx.put_object(record_bytes(body), role=f"{domain}_chunk")
            if domain == "direct":
                if not isinstance(chunk_summary.get("evidence"), Mapping):
                    raise _error("Direct chunk lacks its same-call equation evidence.", ordinal=ordinal)
                summary_body = {
                    "schema": "scnsim.direct_chunk_summary", "schema_version": 1,
                    **identity, "ordinal": ordinal, "evidence": dict(chunk_summary["evidence"]),
                }
                summary_reference = tx.put_object(record_bytes(summary_body), role="direct_chunk_summary")
                chunk_summary = {"summary_ref": summary_reference}
            chunk_rows.append({
                "ordinal": ordinal, "first_point": first_point, "point_count": count,
                "chunk_ref": reference, "summary_json": dict(chunk_summary),
            })
            point_count += count

        if domain == "direct":
            expected = request["spec"].get("frequencies", ())
            if point_count != len(expected):
                raise _error("Direct chunk rows differ from the canonical frequency count.",
                             expected=len(expected), actual=point_count)
        elif source.get("kind") == "points":
            expected = len(source["points"])
            if point_count != expected:
                raise _error("Sweep chunk rows differ from the canonical point count.",
                             expected=expected, actual=point_count)
        else:
            expected = 1
            for dimension in source["shape"]:
                expected *= dimension
            if point_count != expected:
                raise _error("Sweep chunk rows differ from the canonical grid size.",
                             expected=expected, actual=point_count)
        if domain == "sweep":
            try:
                next(expected_sweep_points)
            except StopIteration:
                pass
            else:
                raise _error("Sweep chunks omit request-declared points.",
                             actual=point_count)

        result_body = {
            "schema": f"scnsim.{domain}_result", "schema_version": 1,
            **identity, "type": domain, "manifest_ref": manifest_ref,
            "point_count": point_count, "chunk_count": len(chunk_rows),
            "summary": _externalize_sweep_summary(tx, summary)
            if domain == "sweep" else dict(summary),
        }
        result_ref = tx.put_object(record_bytes(result_body), role="operation_result")
        root_body = {
            "schema": f"scnsim.{domain}_index_root", "schema_version": 1,
            **identity, "result_ref": result_ref, "manifest_ref": manifest_ref,
            "request_ref": dict(request_ref), "point_count": point_count,
            "chunk_count": len(chunk_rows), "chunks": chunk_rows,
        }
        if domain == "sweep":
            root_body["status_summary"] = status_counts
        index_root_ref = tx.put_object(record_bytes(root_body), role=f"{domain}_index_root")
        terminal_attempt = dict(attempt)
        artifacts = list(terminal_attempt.get("artifacts", ()))
        for reference in (result_ref, manifest_ref, index_root_ref):
            if reference not in artifacts:
                artifacts.append(reference)
        terminal_attempt.update(status="success", failure=None, interruption=None,
                                artifacts=artifacts)
        attempt_ref = tx.put_object(record_bytes(terminal_attempt), role="attempt")
        completion_body = {
            "schema": f"scnsim.{domain}_completion", "schema_version": 1,
            **identity, "result_ref": result_ref, "manifest_ref": manifest_ref,
            "index_root_ref": index_root_ref, "terminal_attempt_ref": attempt_ref,
            "point_count": point_count, "chunk_count": len(chunk_rows),
        }
        completion_ref = tx.put_object(record_bytes(completion_body), role=f"{domain}_completion")
        event = {"task_id": task_id, "sequence": sequence,
                 "kind": f"{domain}_terminal",
                 "payload": {"attempt_id": attempt_id, "outcome": "success",
                             "result": result_ref, f"{domain}_completion": completion_ref}}
        tx.append_with_ref(
            _stream(task_id), "task_change",
            record_bytes({"kind": "event", "event": event,
                          "artifacts": [result_ref, completion_ref]}),
        )
        attempt_change = {"kind": "attempt_update", "attempt_id": attempt_id,
                          "status": "success", "failure": None, "interruption": None,
                          "checkpoint": None,
                          "artifacts": [result_ref, manifest_ref, index_root_ref, completion_ref]}
        _, state_ref = tx.append_with_ref(_stream(task_id), "task_change",
                                          record_bytes(attempt_change))
        tx.set_pointer(f"attempt/{task_id}/{attempt_id}", attempt_ref)
        tx.set_pointer(f"current_attempt/{task_id}", attempt_ref)
        tx.set_pointer(_attempt_state_pointer(task_id, attempt_id), state_ref)
        tx.set_pointer(f"completion/{task_id}/{attempt_id}", completion_ref)
        next_ref = tx.put_object(record_bytes({"next_sequence": sequence + 1}), role="event_sequence")
        tx.set_pointer("event_sequence/" + task_id, next_ref)
        selector = {
            "event": "request_success_selected", "request_sha256": descriptor["request_sha256"],
            "task_id": task_id, "attempt_id": attempt_id, "result_ref": result_ref,
            f"{domain}_completion": completion_ref,
        }
        selector_ref = tx.put_object(record_bytes(selector), role="success_selection")
        tx.select_first("success/" + descriptor["request_sha256"], selector_ref)
    return {
        "result_ref": result_ref, "manifest_ref": manifest_ref,
        "index_root_ref": index_root_ref, "completion_ref": completion_ref,
        "attempt_ref": attempt_ref,
        "ack": {"committed": True, "transaction_id": transaction_id,
                "outcome": "success", "point_count": point_count,
                "chunk_count": len(chunk_rows)},
    }


def _externalize_sweep_point(tx, point):
    """Store immutable dependency/evidence bodies once, leaving point locators.

    The public fixed reader hydrates these exact references when a single point
    is selected. Compact sweep indexes never read or duplicate numerical bodies.
    """
    if not isinstance(point, Mapping):
        raise _error("Sweep point is malformed while externalizing its bodies.")
    stored = dict(point)
    _externalize_sweep_record(tx, stored)
    continuations = stored.get("continuations")
    if continuations is not None:
        if not isinstance(continuations, list):
            raise _error("Sweep continuation records are malformed.")
        stored["continuations"] = [
            _externalize_sweep_record(tx, continuation)
            for continuation in continuations
        ]
    return stored


def _externalize_sweep_summary(tx, summary):
    stored = dict(summary)
    anchor = stored.get("root_anchor")
    if anchor is None:
        return stored
    if not isinstance(anchor, Mapping):
        raise _error("Sweep root-anchor summary is malformed.")
    anchor_record = dict(anchor)
    baseline = anchor_record.get("baseline")
    if not isinstance(baseline, Mapping):
        raise _error("Sweep root-anchor baseline is malformed.")
    anchor_record["baseline"] = _externalize_sweep_record(tx, baseline)
    stored["root_anchor"] = anchor_record
    return stored


def _externalize_sweep_record(tx, record):
    if not isinstance(record, Mapping):
        raise _error("Sweep evidence record is malformed.")
    stored = dict(record)
    if "evidence" in stored:
        if "evidence_ref" in stored:
            raise _error("Sweep evidence record repeats inline and referenced evidence.")
        evidence = stored.pop("evidence")
        if not isinstance(evidence, Mapping):
            raise _error("Sweep evidence body is malformed.")
        stored["evidence_ref"] = tx.put_object(record_bytes(evidence), role="benchmark_value")
    if "dependencies" in stored:
        if "dependency_refs" in stored:
            raise _error("Sweep point repeats inline and referenced dependencies.")
        dependencies = stored.pop("dependencies")
        stored["dependency_refs"] = _physical_dependency_references(tx, dependencies)
    failure = stored.get("failure")
    if isinstance(failure, Mapping) and "evidence" in failure:
        if "evidence_ref" in failure:
            raise _error("Sweep failure repeats inline and referenced evidence.")
        failure_record = dict(failure)
        evidence = failure_record.pop("evidence")
        if not isinstance(evidence, Mapping):
            raise _error("Sweep failure evidence body is malformed.")
        failure_record["evidence_ref"] = tx.put_object(
            record_bytes(evidence), role="benchmark_value"
        )
        stored["failure"] = failure_record
    return stored


def complete_direct_operation(binding, *, operation_id, task_id, attempt_id,
                               metadata, chunks, summary=None):
    return _complete_chunked_operation(
        binding, domain="direct", operation_id=operation_id, task_id=task_id,
        attempt_id=attempt_id, metadata=metadata, chunks=chunks,
        summary={} if summary is None else summary,
    )


def complete_sweep_operation(binding, *, operation_id, task_id, attempt_id,
                             metadata, chunks, summary=None):
    return _complete_chunked_operation(
        binding, domain="sweep", operation_id=operation_id, task_id=task_id,
        attempt_id=attempt_id, metadata=metadata, chunks=chunks,
        summary={} if summary is None else summary,
    )


def _physical_stage_stream(task_id, attempt_id):
    return f"physical-optimization/{task_id}/{attempt_id}"


def _physical_checkpoint_pointer(task_id, attempt_id):
    return f"physical_optimization_checkpoint/{task_id}/{attempt_id}"


def _physical_checkpoint_boundary_pointer(task_id, attempt_id):
    return f"physical_optimization_checkpoint_boundary/{task_id}/{attempt_id}"


def _physical_stage_key(stage_id):
    return sha256(stage_id.encode("utf-8")).hexdigest()


def _physical_stage_state_pointer(task_id, attempt_id, stage_id):
    return (f"physical_optimization_stage_state/{task_id}/{attempt_id}/"
            f"{_physical_stage_key(stage_id)}")


def _physical_candidate_pointer(task_id, attempt_id, stage_id, evaluation_ordinal):
    return (f"physical_optimization_candidate/{task_id}/{attempt_id}/"
            f"{_physical_stage_key(stage_id)}/{evaluation_ordinal}")


def _physical_accepted_pointer(task_id, attempt_id, stage_id, candidate_sha256):
    return (f"physical_optimization_accepted/{task_id}/{attempt_id}/"
            f"{_physical_stage_key(stage_id)}/{candidate_sha256}")


def _physical_baseline_pointer(task_id, attempt_id):
    return f"physical_optimization_baseline/{task_id}/{attempt_id}"


def _physical_identity(store, request_sha256, task_id, attempt_id):
    return {
        "workspace_instance_id": store.workspace_instance_id,
        "plan_sha256": store.plan_sha256,
        "request_sha256": request_sha256,
        "task_id": task_id,
        "attempt_id": attempt_id,
    }


def _physical_locator_identity(locator, *, default_task_id, default_attempt_id):
    if not isinstance(locator, Mapping) or not isinstance(locator.get("candidate_ref"), Mapping):
        return None
    kind = locator.get("kind", "evaluation")
    if not set(locator).issubset({"kind", "stage_id", "evaluation_ordinal", "candidate_ref",
                                  "cost_f64", "source_task_id", "source_attempt_id"}):
        return None
    if ("source_task_id" in locator) != ("source_attempt_id" in locator):
        return None
    if kind == "baseline":
        stage_id = locator.get("stage_id")
        evaluation_ordinal = locator.get("evaluation_ordinal")
    elif kind == "evaluation":
        stage_id = locator.get("stage_id")
        evaluation_ordinal = locator.get("evaluation_ordinal")
        if (not isinstance(stage_id, str) or not stage_id
                or not isinstance(evaluation_ordinal, int)
                or isinstance(evaluation_ordinal, bool) or evaluation_ordinal < 0):
            return None
    else:
        return None
    source_task_id = locator.get("source_task_id", default_task_id)
    source_attempt_id = locator.get("source_attempt_id", default_attempt_id)
    if (not isinstance(source_task_id, str) or not source_task_id
            or not isinstance(source_attempt_id, str) or not source_attempt_id):
        return None
    return (source_task_id, source_attempt_id, kind, stage_id, evaluation_ordinal,
            record_bytes(dict(locator["candidate_ref"])))


def _physical_candidate_locators(value, *, default_task_id, default_attempt_id):
    if isinstance(value, Mapping):
        identity = _physical_locator_identity(
            value, default_task_id=default_task_id, default_attempt_id=default_attempt_id,
        )
        if identity is not None:
            yield identity
        for item in value.values():
            yield from _physical_candidate_locators(
                item, default_task_id=default_task_id,
                default_attempt_id=default_attempt_id,
            )
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _physical_candidate_locators(
                item, default_task_id=default_task_id,
                default_attempt_id=default_attempt_id,
            )


def _physical_dependency_references(tx, dependencies):
    if not isinstance(dependencies, Mapping):
        raise _error("Physical Optimization candidate dependencies are malformed.")
    references = {}
    for body_id, body in dependencies.items():
        if not isinstance(body_id, str) or not isinstance(body, Mapping):
            raise _error("Physical Optimization dependency body is malformed.")
        raw = record_bytes(dict(body))
        # The producer's logical body ID and the storage object address are
        # separate authorities; both bind the same decoded exact array bits.
        digest = sha256(numerical_record_bytes(dict(body))).hexdigest()
        if digest != body_id:
            raise _error("Physical Optimization dependency body identity changed.",
                         body_id=body_id, actual_sha256=digest)
        references[body_id] = tx.put_object(raw, role="benchmark_value")
    return references


def _physical_hydrate_dependencies(snapshot, candidate):
    dependencies = candidate.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise _error("Physical Optimization candidate dependencies are malformed.",
                     task_id=candidate.get("task_id"), attempt_id=candidate.get("attempt_id"))
    hydrated = {}
    for body_id, reference in dependencies.items():
        if (not isinstance(body_id, str) or not isinstance(reference, Mapping)
                or reference.get("role") != "benchmark_value"):
            raise _error("Physical Optimization dependency reference is malformed.",
                         body_id=body_id)
        raw = snapshot.get_object(reference)
        body = record_document(raw)
        if (not isinstance(body, dict) or record_bytes(body) != raw
                or sha256(numerical_record_bytes(body)).hexdigest() != body_id):
            raise _error("Physical Optimization dependency body differs from its content identity.",
                         body_id=body_id)
        hydrated[body_id] = body
    candidate["dependencies"] = hydrated
    return candidate


def _physical_resume_authority(snapshot, store, resume_from, *, expected_request_sha256):
    """Return the exact checkpoint-bound candidate locators for one resume.

    This follows only the checkpoint explicitly named by the attempt. It does
    not search task history or infer ancestry from a latest pointer.
    """
    if not isinstance(resume_from, Mapping):
        return None
    task_id = resume_from.get("task_id")
    attempt_id = resume_from.get("attempt_id")
    selected_ref = resume_from.get("checkpoint_ref")
    if (set(resume_from) != {"task_id", "attempt_id", "checkpoint_ref"}
            or not isinstance(task_id, str) or not task_id
            or not isinstance(attempt_id, str) or not attempt_id
            or not isinstance(selected_ref, Mapping)
            or selected_ref.get("role") != "optimization_checkpoint"):
        raise _error("Physical Optimization resume ancestry is malformed.",
                     task_id=task_id, attempt_id=attempt_id)
    descriptor_pointer = snapshot.read_pointer(f"task/{task_id}")
    attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
    checkpoint_pointer = snapshot.read_pointer(_physical_checkpoint_pointer(task_id, attempt_id))
    # A later failure/interruption update may replace the latest attempt-state
    # artifacts without changing the checkpoint the caller explicitly named.
    # Resume authority therefore follows the checkpoint's immutable publishing
    # state change, not the mutable latest-state selector.
    state_pointer = snapshot.read_pointer(
        _physical_checkpoint_boundary_pointer(task_id, attempt_id)
    )
    if (descriptor_pointer is None or attempt_pointer is None
            or checkpoint_pointer is None or state_pointer is None
            or descriptor_pointer.get("reference", {}).get("role") != "task_descriptor"
            or attempt_pointer.get("reference", {}).get("role") != "attempt"
            or checkpoint_pointer.get("reference") != dict(selected_ref)
            or state_pointer.get("reference", {}).get("role") != "task_change"):
        raise _error("Physical Optimization resume checkpoint is no longer the selected source.",
                     task_id=task_id, attempt_id=attempt_id)
    descriptor = _decode(snapshot, descriptor_pointer["reference"])
    request_sha256, _, _ = _physical_request(
        snapshot, store, descriptor, task_id=task_id, attempt_id=attempt_id,
    )
    if request_sha256 != expected_request_sha256:
        raise _error("Physical Optimization resume source belongs to another request.",
                     task_id=task_id, attempt_id=attempt_id)
    attempt = _decode(snapshot, attempt_pointer["reference"])
    state_change = _decode(snapshot, state_pointer["reference"])
    checkpoint_state = state_change.get("checkpoint")
    if (attempt.get("attempt_id") != attempt_id
            or state_change.get("kind") != "attempt_update"
            or state_change.get("attempt_id") != attempt_id
            or not isinstance(checkpoint_state, Mapping)
            or checkpoint_state.get("reference") != dict(selected_ref)):
        raise _error("Physical Optimization resume source state differs from its checkpoint.",
                     task_id=task_id, attempt_id=attempt_id)
    raw = snapshot.get_object(selected_ref)
    checkpoint = record_document(raw)
    if (not isinstance(checkpoint, dict) or record_bytes(checkpoint) != raw
            or checkpoint.get("schema") != "scnsim.optimization_checkpoint"
            or checkpoint.get("schema_version") != 1
            or checkpoint.get("workspace_instance_id") != store.workspace_instance_id
            or checkpoint.get("plan_sha256") != store.plan_sha256
            or checkpoint.get("request_sha256") != expected_request_sha256
            or checkpoint.get("task_id") != task_id
            or checkpoint.get("attempt_id") != attempt_id
            or not isinstance(checkpoint.get("state"), Mapping)):
        raise _error("Physical Optimization resume checkpoint body is misbound.",
                     task_id=task_id, attempt_id=attempt_id)
    allowed = frozenset(_physical_candidate_locators(
        checkpoint["state"], default_task_id=task_id, default_attempt_id=attempt_id,
    ))
    return {"task_id": task_id, "attempt_id": attempt_id,
            "checkpoint_ref": dict(selected_ref), "allowed_locators": allowed}


def _physical_checkpoint_prefix(snapshot, store, resume_from, *, expected_request_sha256,
                                seen=()):
    """Read the exact committed stage-batch prefix ending at one checkpoint.

    The source attempt's task-state artifacts name the batch that published
    the selected checkpoint. Following that reference in the attempt's own
    stage stream gives a bounded, immutable prefix; later attempts or a latest
    task selector never contribute rows.
    """
    authority = _physical_resume_authority(
        snapshot, store, resume_from, expected_request_sha256=expected_request_sha256,
    )
    if authority is None:
        return {"sources": [], "rows": []}
    task_id = authority["task_id"]
    attempt_id = authority["attempt_id"]
    checkpoint_ref = authority["checkpoint_ref"]
    identity = (task_id, attempt_id, checkpoint_ref["sha256"])
    if identity in seen:
        raise _error("Physical Optimization checkpoint ancestry contains a cycle.",
                     task_id=task_id, attempt_id=attempt_id)

    checkpoint_raw = snapshot.get_object(checkpoint_ref)
    checkpoint = record_document(checkpoint_raw)
    attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
    state_pointer = snapshot.read_pointer(
        _physical_checkpoint_boundary_pointer(task_id, attempt_id)
    )
    if (not isinstance(checkpoint, dict) or record_bytes(checkpoint) != checkpoint_raw
            or checkpoint.get("schema") != "scnsim.optimization_checkpoint"
            or checkpoint.get("schema_version") != 1
            or checkpoint.get("task_id") != task_id
            or checkpoint.get("attempt_id") != attempt_id
            or not isinstance(attempt_pointer, Mapping)
            or attempt_pointer.get("reference", {}).get("role") != "attempt"
            or not isinstance(state_pointer, Mapping)
            or state_pointer.get("reference", {}).get("role") != "task_change"):
        raise _error("Physical Optimization checkpoint prefix binding is malformed.",
                     task_id=task_id, attempt_id=attempt_id)
    attempt = _decode(snapshot, attempt_pointer["reference"])
    state_change = _decode(snapshot, state_pointer["reference"])
    artifacts = state_change.get("artifacts")
    checkpoint_state = state_change.get("checkpoint")
    if (state_change.get("kind") != "attempt_update"
            or state_change.get("attempt_id") != attempt_id
            or not isinstance(checkpoint_state, Mapping)
            or checkpoint_state.get("reference") != dict(checkpoint_ref)
            or not isinstance(artifacts, list)):
        raise _error("Physical Optimization checkpoint prefix has no terminal batch boundary.",
                     task_id=task_id, attempt_id=attempt_id)
    boundary_refs = [item for item in artifacts
                     if isinstance(item, Mapping)
                     and item.get("role") == "optimization_stage_batch"]
    if len(boundary_refs) != 1:
        raise _error("Physical Optimization checkpoint state does not name one batch boundary.",
                     task_id=task_id, attempt_id=attempt_id)
    boundary_ref = dict(boundary_refs[0])
    stream = snapshot.read_stream(_physical_stage_stream(task_id, attempt_id))
    prefix_batch_refs = []
    prefix_rows = []
    found_boundary = False
    for entry in stream["entries"]:
        batch_ref = entry.get("reference")
        if (entry.get("kind") != "optimization_stage_batch"
                or not isinstance(batch_ref, Mapping)
                or batch_ref.get("role") != "optimization_stage_batch"):
            raise _error("Physical Optimization checkpoint stream has an invalid entry.",
                         task_id=task_id, attempt_id=attempt_id)
        batch_raw = snapshot.get_object(batch_ref)
        batch = record_document(batch_raw)
        if (not isinstance(batch, dict) or record_bytes(batch) != batch_raw
                or batch.get("schema") != "scnsim.optimization_stage_batch"
                or batch.get("schema_version") != 1
                or any(batch.get(key) != value for key, value in _physical_identity(
                    store, expected_request_sha256, task_id, attempt_id,
                ).items())):
            raise _error("Physical Optimization checkpoint prefix batch is malformed.",
                         task_id=task_id, attempt_id=attempt_id)
        prefix_batch_refs.append(dict(batch_ref))
        stage_id = batch.get("stage_id")
        model_identity = batch.get("model_identity")
        rows = batch.get("rows")
        if (not isinstance(stage_id, str) or not isinstance(model_identity, Mapping)
                or not isinstance(rows, list)):
            raise _error("Physical Optimization checkpoint prefix batch rows are malformed.",
                         task_id=task_id, attempt_id=attempt_id)
        for row in rows:
            candidate_ref = row.get("candidate_ref") if isinstance(row, Mapping) else None
            if (not isinstance(row, Mapping) or not isinstance(candidate_ref, Mapping)
                    or candidate_ref.get("role") != "optimization_candidate"):
                raise _error("Physical Optimization checkpoint prefix candidate is malformed.",
                             task_id=task_id, attempt_id=attempt_id)
            kind = row.get("kind")
            evaluation_ordinal = row.get("evaluation_ordinal")
            if (kind not in {"baseline", "evaluation"}
                    or not isinstance(evaluation_ordinal, int)
                    or isinstance(evaluation_ordinal, bool) or evaluation_ordinal < 0):
                raise _error("Physical Optimization checkpoint prefix candidate identity is malformed.",
                             task_id=task_id, attempt_id=attempt_id)
            history_ordinal = row.get("history_ordinal")
            if kind == "evaluation" and (
                    not isinstance(history_ordinal, int) or isinstance(history_ordinal, bool)
                    or history_ordinal < 0):
                raise _error("Physical Optimization checkpoint prefix history ordinal is malformed.",
                             task_id=task_id, attempt_id=attempt_id)
            row_summary = row.get("summary_json", {})
            if not isinstance(row_summary, Mapping):
                raise _error("Physical Optimization checkpoint prefix summary is malformed.",
                             task_id=task_id, attempt_id=attempt_id)
            candidate_stage = stage_id
            candidate_model = model_identity
            source_locator = None
            if kind == "baseline":
                # A resumed operation can carry its original baseline into a
                # stage with a different model. Read that single baseline body
                # to retain its scientific identity and provenance; evaluation
                # rows use the sealed stage-batch summaries below.
                candidate_raw = snapshot.get_object(candidate_ref)
                candidate = record_document(candidate_raw)
                if (not isinstance(candidate, dict) or record_bytes(candidate) != candidate_raw
                        or candidate.get("schema") != "scnsim.optimization_candidate"
                        or candidate.get("schema_version") != 1
                        or any(candidate.get(key) != value for key, value in _physical_identity(
                            store, expected_request_sha256, task_id, attempt_id,
                        ).items())
                        or candidate.get("kind") != kind
                        or candidate.get("evaluation_ordinal") != evaluation_ordinal):
                    raise _error("Physical Optimization checkpoint prefix baseline is misbound.",
                                 task_id=task_id, attempt_id=attempt_id)
                candidate_stage = candidate.get("stage_id")
                candidate_model = candidate.get("model_identity")
                source_locator = candidate.get("source_locator")
            if (not isinstance(candidate_stage, str) or not isinstance(candidate_model, Mapping)
                    or (kind == "evaluation"
                        and (candidate_stage != stage_id
                             or dict(candidate_model) != dict(model_identity)))
                    or (source_locator is not None and not isinstance(source_locator, Mapping))):
                raise _error("Physical Optimization checkpoint prefix candidate model is malformed.",
                             task_id=task_id, attempt_id=attempt_id)
            prefix_rows.append({
                "stage_id": candidate_stage, "model_identity": dict(candidate_model),
                "kind": kind, "evaluation_ordinal": evaluation_ordinal,
                "history_ordinal": history_ordinal, "candidate_ref": dict(candidate_ref),
                "summary_json": dict(row_summary),
                "source_task_id": task_id, "source_attempt_id": attempt_id,
                "source_batch_ref": dict(batch_ref),
                **({"source_locator": dict(source_locator)}
                   if isinstance(source_locator, Mapping) else {}),
            })
        if dict(batch_ref) == boundary_ref:
            if batch.get("checkpoint_ref") != dict(checkpoint_ref):
                raise _error("Physical Optimization batch boundary differs from its checkpoint.",
                             task_id=task_id, attempt_id=attempt_id)
            found_boundary = True
            break
    if not found_boundary:
        raise _error("Physical Optimization checkpoint batch is outside its committed stream prefix.",
                     task_id=task_id, attempt_id=attempt_id)

    parent_resume = attempt.get("resume_from")
    if parent_resume is None:
        parent = {"sources": [], "rows": []}
    elif isinstance(parent_resume, Mapping):
        parent = _physical_checkpoint_prefix(
            snapshot, store, parent_resume,
            expected_request_sha256=expected_request_sha256,
            seen=(*seen, identity),
        )
    else:
        raise _error("Physical Optimization source attempt has malformed resume ancestry.",
                     task_id=task_id, attempt_id=attempt_id)
    for row in prefix_rows:
        source_locator = row.get("source_locator")
        if source_locator is None:
            continue
        source_kind = source_locator.get("kind", "evaluation")
        source_task_id = source_locator.get("source_task_id")
        source_attempt_id = source_locator.get("source_attempt_id")
        source_ref = source_locator.get("candidate_ref")
        if (source_kind != "baseline" or not isinstance(source_task_id, str)
                or not isinstance(source_attempt_id, str) or not isinstance(source_ref, Mapping)):
            raise _error("Physical Optimization inherited baseline ancestry is malformed.",
                         task_id=row["source_task_id"], attempt_id=row["source_attempt_id"])
        matches = [parent_row for parent_row in parent["rows"]
                   if parent_row["source_task_id"] == source_task_id
                   and parent_row["source_attempt_id"] == source_attempt_id
                   and parent_row["kind"] == source_kind
                   and parent_row["stage_id"] == source_locator.get("stage_id")
                   and parent_row["evaluation_ordinal"] == source_locator.get("evaluation_ordinal")
                   and parent_row["candidate_ref"] == dict(source_ref)
                   and (source_locator.get("source_batch_ref") is None
                        or parent_row["source_batch_ref"] == source_locator["source_batch_ref"])]
        if (len(matches) != 1 or matches[0]["stage_id"] != row["stage_id"]
                or matches[0]["model_identity"] != row["model_identity"]):
            raise _error("Physical Optimization inherited baseline changed its source identity.",
                         task_id=row["source_task_id"], attempt_id=row["source_attempt_id"])
    source = {"task_id": task_id, "attempt_id": attempt_id,
              "checkpoint_ref": dict(checkpoint_ref),
              "batch_refs": prefix_batch_refs}
    return {"sources": [*parent["sources"], source],
            "rows": [*parent["rows"], *prefix_rows]}


def _resolve_physical_candidate_locator(tx, locator, *, task_id, attempt_id,
                                       store, current_candidates, current_baseline,
                                       resume_authority=None):
    if not isinstance(locator, Mapping):
        raise _error("Physical Optimization candidate locator is malformed.")
    if (locator.get("kind", "evaluation") not in {"baseline", "evaluation"}
            or not set(locator).issubset({"kind", "stage_id", "evaluation_ordinal",
                                          "candidate_ref", "cost_f64",
                                          "source_task_id", "source_attempt_id"})):
        raise _error("Physical Optimization candidate locator fields are malformed.",
                     task_id=task_id, attempt_id=attempt_id)
    source_task_id = locator.get("source_task_id")
    source_attempt_id = locator.get("source_attempt_id")
    if source_task_id is not None or source_attempt_id is not None:
        if (not isinstance(source_task_id, str) or not source_task_id
                or not isinstance(source_attempt_id, str) or not source_attempt_id
                or resume_authority is None):
            raise _error("Physical Optimization source locator lacks a verified resume ancestry.",
                         task_id=task_id, attempt_id=attempt_id)
        locator_identity = _physical_locator_identity(
            locator, default_task_id=source_task_id,
            default_attempt_id=source_attempt_id,
        )
        if locator_identity not in resume_authority["allowed_locators"]:
            raise _error("Physical Optimization source locator is outside the selected checkpoint.",
                         source_task_id=source_task_id,
                         source_attempt_id=source_attempt_id)
        kind = locator.get("kind", "evaluation")
        stage_id = locator.get("stage_id")
        evaluation_ordinal = locator.get("evaluation_ordinal")
        selected_ref, _ = _physical_read_candidate(
            tx, store, task_id=source_task_id,
            attempt_id=source_attempt_id, stage_id=stage_id,
            evaluation_ordinal=evaluation_ordinal,
            reference=locator.get("candidate_ref"), kind=kind,
        )
        return {**dict(locator), "candidate_ref": selected_ref}
    supplied_ref = locator.get("candidate_ref")
    if locator.get("kind") == "baseline":
        selected_ref = current_baseline
        if selected_ref is None:
            pointer = tx.read_pointer(_physical_baseline_pointer(task_id, attempt_id))
            selected_ref = None if pointer is None else pointer["reference"]
    else:
        stage_id = locator.get("stage_id")
        evaluation_ordinal = locator.get("evaluation_ordinal")
        if (not isinstance(stage_id, str) or not stage_id
                or not isinstance(evaluation_ordinal, int)
                or isinstance(evaluation_ordinal, bool) or evaluation_ordinal < 0):
            raise _error("Physical Optimization candidate locator lacks an exact occurrence.")
        selected_ref = current_candidates.get((stage_id, evaluation_ordinal))
        if selected_ref is None:
            pointer = tx.read_pointer(_physical_candidate_pointer(
                task_id, attempt_id, stage_id, evaluation_ordinal,
            ))
            selected_ref = None if pointer is None else pointer["reference"]
    if not isinstance(selected_ref, Mapping):
        raise _error("Physical Optimization candidate locator names no committed occurrence.",
                     task_id=task_id, attempt_id=attempt_id)
    if supplied_ref is not None and dict(supplied_ref) != dict(selected_ref):
        raise _error("Physical Optimization candidate locator reference changed.",
                     task_id=task_id, attempt_id=attempt_id)
    return {**dict(locator), "candidate_ref": dict(selected_ref)}


def _resolve_physical_locators(value, *, tx, task_id, attempt_id,
                               store, current_candidates, current_baseline,
                               resume_authority=None):
    if isinstance(value, Mapping):
        locator_keys = {"stage_id", "evaluation_ordinal", "candidate_ref", "cost_f64", "kind",
                        "source_task_id", "source_attempt_id"}
        if ("stage_id" in value and "evaluation_ordinal" in value
                and set(value).issubset(locator_keys)):
            return _resolve_physical_candidate_locator(
                tx, value, task_id=task_id, attempt_id=attempt_id,
                store=store,
                current_candidates=current_candidates, current_baseline=current_baseline,
                resume_authority=resume_authority,
            )
        if (value.get("kind") == "baseline"
                and set(value).issubset({"kind", "candidate_ref", "source_task_id",
                                         "source_attempt_id"})):
            return _resolve_physical_candidate_locator(
                tx, value, task_id=task_id, attempt_id=attempt_id,
                store=store,
                current_candidates=current_candidates, current_baseline=current_baseline,
                resume_authority=resume_authority,
            )
        return {
            key: _resolve_physical_locators(
                item, tx=tx, task_id=task_id, attempt_id=attempt_id,
                store=store,
                current_candidates=current_candidates, current_baseline=current_baseline,
                resume_authority=resume_authority,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_physical_locators(
                item, tx=tx, task_id=task_id, attempt_id=attempt_id,
                store=store,
                current_candidates=current_candidates, current_baseline=current_baseline,
                resume_authority=resume_authority,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _resolve_physical_locators(
                item, tx=tx, task_id=task_id, attempt_id=attempt_id,
                store=store,
                current_candidates=current_candidates, current_baseline=current_baseline,
                resume_authority=resume_authority,
            )
            for item in value
        )
    return value


def commit_physical_stage_batch(binding, *, operation_id, task_id, attempt_id,
                                stage_id, model_identity, rows, stage_state, checkpoint,
                                transitions=()):
    """Atomically publish one completed physical Optimization stage batch.

    Candidate bodies are immutable objects. The batch stream retains only their
    references and compact summaries; its controller checkpoint is the exact
    committed post-iteration/post-generation state supplied by the coordinator.
    """
    if not isinstance(stage_id, str) or not stage_id or not isinstance(model_identity, Mapping):
        raise _error("Physical Optimization stage identity is malformed.", stage_id=stage_id)
    if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes, bytearray)):
        raise _error("Physical Optimization stage rows are malformed.", stage_id=stage_id)
    if not isinstance(stage_state, Mapping) or (checkpoint is not None and not isinstance(checkpoint, Mapping)):
        raise _error("Physical Optimization committed state is malformed.", stage_id=stage_id)
    if not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes)):
        raise _error("Physical Optimization transition rows are malformed.", stage_id=stage_id)

    store = _bound(binding)
    batch_stream = _physical_stage_stream(task_id, attempt_id)
    task_stream = _stream(task_id)
    with _snapshot(store) as snapshot:
        descriptor = _descriptor(snapshot, task_id)
        association = _pointer(snapshot, _association_pointer(operation_id))
        descriptor_pointer = snapshot.read_pointer(f"task/{task_id}")
        attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
        current_attempt_pointer = snapshot.read_pointer(f"current_attempt/{task_id}")
        if (descriptor.get("request_sha256") is None
                or association is None
                or association.get("operation_id") != operation_id
                or association.get("task_id") != task_id
                or association.get("attempt_id") != attempt_id
                or attempt_pointer is None
                or current_attempt_pointer is None
                or current_attempt_pointer.get("reference") != attempt_pointer.get("reference")):
            raise _error("Physical Optimization batch is not bound to the current attempt.",
                         operation_id=operation_id, task_id=task_id, attempt_id=attempt_id)
        attempt = _decode(snapshot, attempt_pointer["reference"])
        if attempt.get("status") not in {"allocated", "running"}:
            raise _error("Physical Optimization attempt is not publishable.",
                         task_id=task_id, attempt_id=attempt_id)
        request_ref = next((item for item in descriptor.get("artifacts", ())
                            if item.get("role") == "operation_request"), None)
        if not isinstance(request_ref, Mapping) or request_ref.get("sha256") != descriptor.get("request_sha256"):
            raise _error("Physical Optimization task lacks its bound request.", task_id=task_id)
        request_raw = snapshot.get_object(request_ref)
        request = record_document(request_raw)
        if (not isinstance(request, dict) or record_bytes(request) != request_raw
                or request.get("operation") != "optimize_direct"
                or request.get("runtime_semantic", {}).get("backend") != "jax"):
            raise _error("Physical Optimization batch is bound to another request authority.",
                         task_id=task_id, attempt_id=attempt_id)
        task_revision = snapshot.stream_revision(task_stream)
        batch_revision = snapshot.stream_revision(batch_stream)
        sequence_pointer = snapshot.read_pointer("event_sequence/" + task_id)
        sequence = (0 if sequence_pointer is None else
                    _decode(snapshot, sequence_pointer["reference"]).get("next_sequence"))
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise _error("Physical Optimization task event sequence is malformed.", task_id=task_id)

    identity = _physical_identity(store, descriptor["request_sha256"], task_id, attempt_id)
    transaction_id = str(uuid4())
    batch_rows = []
    baseline_ref = None
    candidate_refs = {}
    accepted_ref = None
    with store.transaction(transaction_id, expected_revisions={
            task_stream: task_revision, batch_stream: batch_revision}) as tx:
        current_association = _pointer(tx, _association_pointer(operation_id))
        current_task = tx.read_pointer(f"task/{task_id}")
        current_attempt = tx.read_pointer(f"attempt/{task_id}/{attempt_id}")
        current_attempt_pointer = tx.read_pointer(f"current_attempt/{task_id}")
        if (descriptor_pointer is None or current_task is None
                or current_task.get("reference") != descriptor_pointer.get("reference")
                or current_association != association
                or current_attempt is None
                or current_attempt.get("reference") != attempt_pointer.get("reference")
                or current_attempt_pointer is None
                or current_attempt_pointer.get("reference") != attempt_pointer.get("reference")):
            raise _error("Physical Optimization attempt binding changed before batch publication.",
                         operation_id=operation_id, task_id=task_id, attempt_id=attempt_id)
        publishing_attempt = _decode(tx, current_attempt["reference"])
        if publishing_attempt.get("attempt_id") != attempt_id:
            raise _error("Physical Optimization attempt identity changed before publication.",
                         task_id=task_id, attempt_id=attempt_id)
        resume_authority = _physical_resume_authority(
            tx, store, publishing_attempt.get("resume_from"),
            expected_request_sha256=descriptor["request_sha256"],
        )
        seen_ordinals = set()
        for position, source in enumerate(rows):
            if not isinstance(source, Mapping):
                raise _error("Physical Optimization candidate row is malformed.",
                             stage_id=stage_id, position=position)
            candidate = source.get("candidate")
            summary = source.get("summary_json", {})
            kind = source.get("kind", "evaluation")
            history_ordinal = source.get("history_ordinal")
            if (not isinstance(candidate, Mapping) or not isinstance(summary, Mapping)
                    or kind not in {"baseline", "evaluation"}
                    or (kind == "evaluation" and (
                        not isinstance(history_ordinal, int) or isinstance(history_ordinal, bool)
                        or history_ordinal < 0 or history_ordinal in seen_ordinals))):
                raise _error("Physical Optimization candidate row is incomplete.",
                             stage_id=stage_id, position=position)
            evaluation_ordinal = candidate.get("evaluation_ordinal")
            parameters = candidate.get("parameters")
            if (not isinstance(evaluation_ordinal, int) or isinstance(evaluation_ordinal, bool)
                    or evaluation_ordinal < 0 or not isinstance(parameters, Mapping)
                    or not isinstance(candidate.get("cost_f64"), str)
                    or not isinstance(candidate.get("status"), str)):
                raise _error("Physical Optimization candidate identity is malformed.",
                             stage_id=stage_id, position=position)
            source_locator = source.get("source_locator")
            candidate_body = {
                **dict(candidate), "kind": kind,
                "schema": "scnsim.optimization_candidate", "schema_version": 1,
                **identity,
            }
            if source_locator is not None:
                if kind != "baseline" or not isinstance(source_locator, Mapping):
                    raise _error("Physical Optimization source provenance is malformed.",
                                 stage_id=stage_id, position=position)
                resolved_source = _resolve_physical_candidate_locator(
                    tx, source_locator, task_id=task_id, attempt_id=attempt_id,
                    store=store, current_candidates=candidate_refs,
                    current_baseline=baseline_ref,
                    resume_authority=resume_authority,
                )
                source_candidate_ref, source_candidate = _physical_read_candidate(
                    tx, store, task_id=resolved_source["source_task_id"],
                    attempt_id=resolved_source["source_attempt_id"],
                    stage_id=resolved_source.get("stage_id"),
                    evaluation_ordinal=resolved_source.get("evaluation_ordinal"),
                    reference=resolved_source["candidate_ref"],
                    kind=resolved_source.get("kind", "evaluation"),
                )
                if (candidate.get("stage_id") != source_candidate.get("stage_id")
                        or candidate.get("model_identity") != source_candidate.get("model_identity")):
                    raise _error("Physical Optimization inherited baseline changed its scientific identity.",
                                 stage_id=stage_id, position=position)
                candidate_body["stage_id"] = source_candidate["stage_id"]
                candidate_body["model_identity"] = source_candidate["model_identity"]
                candidate_body["source_locator"] = {
                    **resolved_source, "candidate_ref": source_candidate_ref,
                }
            else:
                candidate_body["stage_id"] = stage_id
                candidate_body["model_identity"] = dict(model_identity)
            candidate_body["dependencies"] = _physical_dependency_references(
                tx, candidate_body.get("dependencies"),
            )
            candidate_ref = tx.put_object(record_bytes(candidate_body), role="optimization_candidate")
            if kind == "baseline":
                if baseline_ref is not None:
                    raise _error("Physical Optimization batch contains more than one baseline.",
                                 stage_id=stage_id)
                baseline_ref = candidate_ref
                prior_baseline = tx.read_pointer(_physical_baseline_pointer(task_id, attempt_id))
                if prior_baseline is not None and prior_baseline.get("reference") != candidate_ref:
                    raise _error("Physical Optimization baseline occurrence changed.",
                                 task_id=task_id, attempt_id=attempt_id)
                tx.set_pointer(_physical_baseline_pointer(task_id, attempt_id), candidate_ref)
                batch_rows.append({"kind": "baseline", "evaluation_ordinal": evaluation_ordinal,
                                   "candidate_ref": candidate_ref})
            else:
                candidate_pointer = _physical_candidate_pointer(
                    task_id, attempt_id, stage_id, evaluation_ordinal,
                )
                if tx.read_pointer(candidate_pointer) is not None:
                    raise _error("Physical Optimization evaluation ordinal is already committed.",
                                 task_id=task_id, attempt_id=attempt_id, stage_id=stage_id,
                                 evaluation_ordinal=evaluation_ordinal)
                tx.set_pointer(candidate_pointer, candidate_ref)
                candidate_refs[(stage_id, evaluation_ordinal)] = candidate_ref
                seen_ordinals.add(history_ordinal)
                batch_rows.append({"kind": "evaluation", "history_ordinal": history_ordinal,
                                   "evaluation_ordinal": evaluation_ordinal,
                                   "candidate_ref": candidate_ref, "summary_json": dict(summary)})

        resolved_state = _resolve_physical_locators(
            dict(stage_state), tx=tx, store=store,
            task_id=task_id, attempt_id=attempt_id,
            current_candidates=candidate_refs, current_baseline=baseline_ref,
            resume_authority=resume_authority,
        )
        resolved_checkpoint = (None if checkpoint is None else _resolve_physical_locators(
            dict(checkpoint), tx=tx, store=store,
            task_id=task_id, attempt_id=attempt_id,
            current_candidates=candidate_refs, current_baseline=baseline_ref,
            resume_authority=resume_authority,
        ))
        stage_state_body = {
            "schema": "scnsim.optimization_stage_state", "schema_version": 1,
            **identity, "stage_id": stage_id, "model_identity": dict(model_identity),
            "state": resolved_state,
        }
        checkpoint_body = None if resolved_checkpoint is None else {
            "schema": "scnsim.optimization_checkpoint", "schema_version": 1,
            **identity, "stage_id": stage_id, "model_identity": dict(model_identity),
            "state": resolved_checkpoint,
        }
        state_ref = tx.put_object(record_bytes(stage_state_body), role="optimization_stage_state")
        checkpoint_ref = (None if checkpoint_body is None else
                          tx.put_object(record_bytes(checkpoint_body), role="optimization_checkpoint"))

        transition_refs = []
        for ordinal, source in enumerate(transitions):
            if not isinstance(source, Mapping):
                raise _error("Physical Optimization transition row is malformed.",
                             stage_id=stage_id, ordinal=ordinal)
            transition_id = source.get("transition_id")
            if not isinstance(transition_id, str) or not transition_id:
                raise _error("Physical Optimization transition identity is malformed.",
                             stage_id=stage_id, ordinal=ordinal)
            resolved_transition = _resolve_physical_locators(
                dict(source), tx=tx, store=store,
                task_id=task_id, attempt_id=attempt_id,
                current_candidates=candidate_refs, current_baseline=baseline_ref,
                resume_authority=resume_authority,
            )
            transition_body = {
                **resolved_transition, "schema": "scnsim.optimization_transition",
                "schema_version": 1, **identity, "stage_id": stage_id,
                "model_identity": dict(model_identity),
            }
            transition_ref = tx.put_object(
                record_bytes(transition_body), role="optimization_transition"
            )
            transition_refs.append({"transition_id": transition_id,
                                    "transition_ref": transition_ref})

        batch_body = {
            "schema": "scnsim.optimization_stage_batch", "schema_version": 1,
            **identity, "stage_id": stage_id, "model_identity": dict(model_identity),
            "state_ref": state_ref, "checkpoint_ref": checkpoint_ref,
            "baseline_ref": baseline_ref, "rows": batch_rows,
            "transition_refs": transition_refs,
        }
        _, batch_ref = tx.append_with_ref(
            batch_stream, "optimization_stage_batch", record_bytes(batch_body)
        )
        tx.set_pointer(_physical_stage_state_pointer(task_id, attempt_id, stage_id), state_ref)
        accepted = resolved_state.get("accepted_iterate") if isinstance(resolved_state, Mapping) else None
        if isinstance(accepted, Mapping) and isinstance(accepted.get("candidate_ref"), Mapping):
            accepted_stage = accepted.get("stage_id")
            accepted_ordinal = accepted.get("evaluation_ordinal")
            accepted_ref = accepted["candidate_ref"]
            if (accepted_stage != stage_id
                    or not isinstance(accepted_ordinal, int)
                    or isinstance(accepted_ordinal, bool)
                    or accepted_ref.get("role") != "optimization_candidate"):
                raise _error("Physical Optimization accepted iterate is malformed.",
                             task_id=task_id, attempt_id=attempt_id, stage_id=stage_id)
            expected_candidate = tx.read_pointer(_physical_candidate_pointer(
                task_id, attempt_id, accepted_stage, accepted_ordinal,
            ))
            if (expected_candidate is None
                    or expected_candidate.get("reference") != dict(accepted_ref)):
                raise _error("Physical Optimization accepted iterate was not committed in this attempt.",
                             task_id=task_id, attempt_id=attempt_id, stage_id=stage_id,
                             evaluation_ordinal=accepted_ordinal)
            accepted_body = record_document(tx.get_object(accepted_ref))
            if (not isinstance(accepted_body, dict)
                    or accepted_body.get("kind") != "evaluation"
                    or accepted_body.get("status") != "success"
                    or accepted_body.get("stage_id") != stage_id
                    or accepted_body.get("evaluation_ordinal") != accepted_ordinal
                    or accepted_body.get("model_identity") != dict(model_identity)):
                raise _error("Physical Optimization accepted iterate differs from its evaluation.",
                             task_id=task_id, attempt_id=attempt_id, stage_id=stage_id,
                             evaluation_ordinal=accepted_ordinal)
            tx.set_pointer(_physical_accepted_pointer(
                task_id, attempt_id, stage_id, accepted_ref["sha256"],
            ), batch_ref)
        change = {
            "kind": "attempt_update", "attempt_id": attempt_id, "status": "running",
            "failure": None, "interruption": None,
            "checkpoint": (None if checkpoint_ref is None else
                           {"reference": checkpoint_ref, "stage_id": stage_id}),
            "artifacts": [batch_ref, state_ref,
                          *([] if checkpoint_ref is None else [checkpoint_ref]),
                          *(row["candidate_ref"] for row in batch_rows)],
        }
        _, state_change_ref = tx.append_with_ref(task_stream, "task_change", record_bytes(change))
        tx.set_pointer(_attempt_state_pointer(task_id, attempt_id), state_change_ref)
        if checkpoint_ref is not None:
            tx.set_pointer(_physical_checkpoint_pointer(task_id, attempt_id), checkpoint_ref)
            tx.set_pointer(
                _physical_checkpoint_boundary_pointer(task_id, attempt_id), state_change_ref,
            )
    return {
        "ack": {"committed": True, "transaction_id": transaction_id,
                "stage_id": stage_id, "row_count": len(batch_rows)},
        "batch_ref": batch_ref, "checkpoint_ref": checkpoint_ref,
    }


def read_physical_optimization_checkpoint(binding, *, task_id, attempt_id,
                                           expected_request_sha256, reference=None,
                                           expected_stage_id=None):
    """Read the exact currently committed v7 physical Optimization checkpoint.

    The checkpoint pointer and its immutable publishing task-state pointer are
    read in one snapshot. Later failure/interruption state changes do not erase
    the committed checkpoint boundary. This does not search task history or
    infer a checkpoint from a result projection.
    """
    store = _bound(binding)
    with _snapshot(store) as snapshot:
        return _physical_checkpoint_from_snapshot(
            snapshot, store, task_id=task_id, attempt_id=attempt_id,
            expected_request_sha256=expected_request_sha256, reference=reference,
            expected_stage_id=expected_stage_id,
        )


def _physical_checkpoint_from_snapshot(snapshot, store, *, task_id, attempt_id,
                                       expected_request_sha256, reference=None,
                                       expected_stage_id=None):
    """Verify a v7 checkpoint using only its pointer and immutable batch boundary."""
    checkpoint_pointer = snapshot.read_pointer(_physical_checkpoint_pointer(task_id, attempt_id))
    boundary_pointer = snapshot.read_pointer(_physical_checkpoint_boundary_pointer(task_id, attempt_id))
    if checkpoint_pointer is None and boundary_pointer is None:
        return None
    if (checkpoint_pointer is None or boundary_pointer is None
            or boundary_pointer.get("reference", {}).get("role") != "task_change"):
        raise _error("Physical Optimization checkpoint pointer pair is incomplete.",
                     task_id=task_id, attempt_id=attempt_id)

    descriptor_pointer = snapshot.read_pointer(f"task/{task_id}")
    attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
    if (descriptor_pointer is None
            or descriptor_pointer.get("reference", {}).get("role") != "task_descriptor"
            or attempt_pointer is None
            or attempt_pointer.get("reference", {}).get("role") != "attempt"):
        raise _error("Physical Optimization checkpoint identities are absent.",
                     task_id=task_id, attempt_id=attempt_id)
    descriptor = _decode(snapshot, descriptor_pointer["reference"])
    if (descriptor.get("task_id") != task_id
            or descriptor.get("request_sha256") != expected_request_sha256):
        raise _error("Physical Optimization checkpoint belongs to another request.",
                     task_id=task_id, attempt_id=attempt_id)
    checkpoint_ref = checkpoint_pointer.get("reference")
    if (not isinstance(checkpoint_ref, Mapping)
            or checkpoint_ref.get("role") != "optimization_checkpoint"):
        raise _error("Physical Optimization checkpoint pointer has the wrong role.",
                     task_id=task_id, attempt_id=attempt_id)
    if reference is not None and dict(reference) != dict(checkpoint_ref):
        raise _error("Selected Physical Optimization checkpoint reference changed.",
                     task_id=task_id, attempt_id=attempt_id)

    state_change = _decode(snapshot, boundary_pointer["reference"])
    checkpoint_state = state_change.get("checkpoint")
    if (state_change.get("kind") != "attempt_update"
            or state_change.get("attempt_id") != attempt_id
            or not isinstance(checkpoint_state, Mapping)
            or checkpoint_state.get("reference") != dict(checkpoint_ref)):
        raise _error("Physical Optimization checkpoint differs from its publishing boundary.",
                     task_id=task_id, attempt_id=attempt_id)
    attempt = _decode(snapshot, attempt_pointer["reference"])
    if attempt.get("attempt_id") != attempt_id:
        raise _error("Physical Optimization checkpoint attempt identity changed.",
                     task_id=task_id, attempt_id=attempt_id)
    raw = snapshot.get_object(checkpoint_ref)
    checkpoint_body = record_document(raw)
    if (not isinstance(checkpoint_body, dict) or record_bytes(checkpoint_body) != raw
            or checkpoint_body.get("schema") != "scnsim.optimization_checkpoint"
            or checkpoint_body.get("schema_version") != 1
            or checkpoint_body.get("workspace_instance_id") != store.workspace_instance_id
            or checkpoint_body.get("plan_sha256") != store.plan_sha256
            or checkpoint_body.get("request_sha256") != expected_request_sha256
            or checkpoint_body.get("task_id") != task_id
            or checkpoint_body.get("attempt_id") != attempt_id
            or not isinstance(checkpoint_body.get("stage_id"), str)
            or not isinstance(checkpoint_body.get("model_identity"), Mapping)
            or not isinstance(checkpoint_body.get("state"), Mapping)):
        raise _error("Physical Optimization checkpoint body is malformed or misbound.",
                     task_id=task_id, attempt_id=attempt_id)
    if (checkpoint_state.get("stage_id") != checkpoint_body["stage_id"]
            or (expected_stage_id is not None
                and checkpoint_body.get("stage_id") != expected_stage_id)):
        raise _error("Physical Optimization checkpoint names another stage.",
                     task_id=task_id, attempt_id=attempt_id,
                     expected_stage_id=expected_stage_id,
                     actual_stage_id=checkpoint_body.get("stage_id"))
    return {
        "task_id": task_id,
        "attempt_id": attempt_id,
        "request_sha256": expected_request_sha256,
        "checkpoint_ref": dict(checkpoint_ref),
        "checkpoint": checkpoint_body,
        "resume_from": {"task_id": task_id, "attempt_id": attempt_id,
                        "checkpoint_ref": dict(checkpoint_ref)},
    }


def _physical_request(snapshot, store, descriptor, *, task_id, attempt_id):
    request_sha256 = descriptor.get("request_sha256")
    request_ref = next((item for item in descriptor.get("artifacts", ())
                        if isinstance(item, Mapping) and item.get("role") == "operation_request"), None)
    if (not isinstance(request_sha256, str) or not isinstance(request_ref, Mapping)
            or request_ref.get("sha256") != request_sha256):
        raise _error("Physical Optimization source lacks its bound request.",
                     task_id=task_id, attempt_id=attempt_id)
    raw = snapshot.get_object(request_ref)
    request = record_document(raw)
    if (not isinstance(request, dict) or record_bytes(request) != raw
            or request.get("operation") != "optimize_direct"
            or request.get("plan_sha256") != store.plan_sha256
            or request.get("runtime_semantic", {}).get("backend") != "jax"):
        raise _error("Physical Optimization source request is malformed or misbound.",
                     task_id=task_id, attempt_id=attempt_id)
    return request_sha256, dict(request_ref), request


def _physical_read_candidate(snapshot, store, *, task_id, attempt_id, stage_id,
                             evaluation_ordinal, reference=None, kind="evaluation"):
    if kind == "baseline":
        pointer = snapshot.read_pointer(_physical_baseline_pointer(task_id, attempt_id))
    else:
        pointer = snapshot.read_pointer(_physical_candidate_pointer(
            task_id, attempt_id, stage_id, evaluation_ordinal,
        ))
    if pointer is None:
        raise _error("Physical Optimization candidate occurrence is not committed.",
                     task_id=task_id, attempt_id=attempt_id, stage_id=stage_id,
                     evaluation_ordinal=evaluation_ordinal)
    candidate_ref = pointer.get("reference")
    if (not isinstance(candidate_ref, Mapping)
            or candidate_ref.get("role") != "optimization_candidate"
            or reference is not None and dict(candidate_ref) != dict(reference)):
        raise _error("Physical Optimization candidate index differs from its selected reference.",
                     task_id=task_id, attempt_id=attempt_id, stage_id=stage_id,
                     evaluation_ordinal=evaluation_ordinal)
    raw = snapshot.get_object(candidate_ref)
    candidate = record_document(raw)
    if (not isinstance(candidate, dict) or record_bytes(candidate) != raw
            or candidate.get("schema") != "scnsim.optimization_candidate"
            or candidate.get("schema_version") != 1
            or candidate.get("workspace_instance_id") != store.workspace_instance_id
            or candidate.get("plan_sha256") != store.plan_sha256
            or candidate.get("task_id") != task_id
            or candidate.get("attempt_id") != attempt_id
            or (stage_id is not None and candidate.get("stage_id") != stage_id)
            or candidate.get("kind") != kind
            or (kind != "baseline" and candidate.get("evaluation_ordinal") != evaluation_ordinal)
            or (kind == "baseline" and evaluation_ordinal is not None
                and candidate.get("evaluation_ordinal") != evaluation_ordinal)):
        raise _error("Physical Optimization candidate body differs from its exact index.",
                     task_id=task_id, attempt_id=attempt_id, stage_id=stage_id,
                     evaluation_ordinal=evaluation_ordinal)
    return dict(candidate_ref), candidate


def read_physical_optimization_candidate(binding, *, task_id, attempt_id, locator,
                                          expected_request_sha256=None, resume_from=None):
    """Read one checkpoint cache candidate by exact committed locator."""
    if not isinstance(locator, Mapping):
        raise TypeError("Physical Optimization candidate locator must be a mapping")
    kind = locator.get("kind", "evaluation")
    stage_id = locator.get("stage_id")
    evaluation_ordinal = locator.get("evaluation_ordinal")
    if (kind not in {"baseline", "evaluation"}
            or (stage_id is not None and (not isinstance(stage_id, str) or not stage_id))
            or (evaluation_ordinal is not None and (
                not isinstance(evaluation_ordinal, int) or isinstance(evaluation_ordinal, bool)
                or evaluation_ordinal < 0))
            or (kind == "evaluation" and (stage_id is None or evaluation_ordinal is None))):
        raise _error("Physical Optimization candidate locator is malformed.",
                     task_id=task_id, attempt_id=attempt_id)
    store = _bound(binding)
    with _snapshot(store) as snapshot:
        descriptor = _descriptor(snapshot, task_id)
        request_sha256, _, _ = _physical_request(
            snapshot, store, descriptor, task_id=task_id, attempt_id=attempt_id,
        )
        if expected_request_sha256 is not None and request_sha256 != expected_request_sha256:
            raise _error("Physical Optimization candidate belongs to another request.",
                         task_id=task_id, attempt_id=attempt_id)
        source_task_id = locator.get("source_task_id")
        source_attempt_id = locator.get("source_attempt_id")
        if source_task_id is not None or source_attempt_id is not None:
            if (source_task_id != task_id or source_attempt_id != attempt_id
                    or not isinstance(resume_from, Mapping)):
                raise _error("Physical Optimization source candidate lacks its exact resume binding.",
                             task_id=task_id, attempt_id=attempt_id)
            authority = _physical_resume_authority(
                snapshot, store, resume_from,
                expected_request_sha256=request_sha256,
            )
            locator_identity = _physical_locator_identity(
                locator, default_task_id=source_task_id,
                default_attempt_id=source_attempt_id,
            )
            if locator_identity not in authority["allowed_locators"]:
                raise _error("Physical Optimization candidate is outside its selected checkpoint.",
                             task_id=task_id, attempt_id=attempt_id)
        attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
        if attempt_pointer is None or attempt_pointer.get("reference", {}).get("role") != "attempt":
            raise _error("Physical Optimization candidate source attempt is absent.",
                         task_id=task_id, attempt_id=attempt_id)
        candidate_ref, candidate = _physical_read_candidate(
            snapshot, store, task_id=task_id, attempt_id=attempt_id,
            stage_id=stage_id, evaluation_ordinal=evaluation_ordinal,
            reference=locator.get("candidate_ref"), kind=kind,
        )
        _physical_hydrate_dependencies(snapshot, candidate)
        return {"candidate_ref": candidate_ref, "candidate": candidate}


def read_physical_accepted_iterate(binding, *, attempt_id, stage_id,
                                   accepted_iterate_id):
    """Read an explicitly named committed accepted iterate without history replay."""
    if (not isinstance(attempt_id, str) or not attempt_id
            or not isinstance(stage_id, str) or not stage_id
            or not isinstance(accepted_iterate_id, str) or not accepted_iterate_id):
        raise TypeError("accepted iterate selection requires exact attempt, stage, and content identities")
    store = _bound(binding)
    with _snapshot(store) as snapshot:
        located = snapshot.find_attempt_pointer(attempt_id)
        if located is None:
            return None
        task_id = located["task_id"]
        descriptor = _descriptor(snapshot, task_id)
        request_sha256, request_ref, request = _physical_request(
            snapshot, store, descriptor, task_id=task_id, attempt_id=attempt_id,
        )
        attempt_ref = located["reference"]
        attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
        if (attempt_pointer is None or attempt_pointer.get("reference") != attempt_ref
                or attempt_pointer.get("reference", {}).get("role") != "attempt"):
            raise _error("Physical Optimization source attempt pointer changed.",
                         task_id=task_id, attempt_id=attempt_id)
        attempt = _decode(snapshot, attempt_ref)
        if attempt.get("attempt_id") != attempt_id:
            raise _error("Physical Optimization source attempt identity changed.",
                         task_id=task_id, attempt_id=attempt_id)
        proof_pointer = snapshot.read_pointer(_physical_accepted_pointer(
            task_id, attempt_id, stage_id, accepted_iterate_id,
        ))
        if proof_pointer is None or proof_pointer.get("reference", {}).get("role") != "optimization_stage_batch":
            return None
        batch_ref = proof_pointer["reference"]
        batch_raw = snapshot.get_object(batch_ref)
        batch = record_document(batch_raw)
        if (not isinstance(batch, dict) or record_bytes(batch) != batch_raw
                or batch.get("schema") != "scnsim.optimization_stage_batch"
                or batch.get("schema_version") != 1
                or batch.get("workspace_instance_id") != store.workspace_instance_id
                or batch.get("plan_sha256") != store.plan_sha256
                or batch.get("request_sha256") != request_sha256
                or batch.get("task_id") != task_id
                or batch.get("attempt_id") != attempt_id
                or batch.get("stage_id") != stage_id
                or not isinstance(batch.get("state_ref"), Mapping)
                or batch["state_ref"].get("role") != "optimization_stage_state"):
            raise _error("Physical Optimization accepted-state proof is malformed.",
                         task_id=task_id, attempt_id=attempt_id, stage_id=stage_id)
        state_ref = batch["state_ref"]
        state_raw = snapshot.get_object(state_ref)
        state_body = record_document(state_raw)
        if (not isinstance(state_body, dict) or record_bytes(state_body) != state_raw
                or state_body.get("schema") != "scnsim.optimization_stage_state"
                or state_body.get("schema_version") != 1
                or state_body.get("workspace_instance_id") != store.workspace_instance_id
                or state_body.get("plan_sha256") != store.plan_sha256
                or state_body.get("request_sha256") != request_sha256
                or state_body.get("task_id") != task_id
                or state_body.get("attempt_id") != attempt_id
                or state_body.get("stage_id") != stage_id
                or state_body.get("model_identity") != batch.get("model_identity")
                or not isinstance(state_body.get("state"), Mapping)):
            raise _error("Physical Optimization accepted stage state is malformed.",
                         task_id=task_id, attempt_id=attempt_id, stage_id=stage_id)
        locator = state_body["state"].get("accepted_iterate")
        if (not isinstance(locator, Mapping) or locator.get("stage_id") != stage_id
                or not isinstance(locator.get("evaluation_ordinal"), int)
                or isinstance(locator.get("evaluation_ordinal"), bool)
                or not isinstance(locator.get("candidate_ref"), Mapping)
                or locator["candidate_ref"].get("sha256") != accepted_iterate_id):
            return None
        evaluation_ordinal = locator["evaluation_ordinal"]
        candidate_ref, candidate = _physical_read_candidate(
            snapshot, store, task_id=task_id, attempt_id=attempt_id,
            stage_id=stage_id, evaluation_ordinal=evaluation_ordinal,
            reference=locator["candidate_ref"], kind="evaluation",
        )
        if (candidate.get("status") != "success"
                or candidate.get("model_identity") != dict(batch["model_identity"])
                or candidate_ref["sha256"] != accepted_iterate_id):
            raise _error("Physical Optimization accepted candidate differs from its state proof.",
                         task_id=task_id, attempt_id=attempt_id, stage_id=stage_id,
                         evaluation_ordinal=evaluation_ordinal)
        return {
            "task_id": task_id, "attempt_id": attempt_id,
            "request_sha256": request_sha256, "request_ref": dict(request_ref),
            "stage_id": stage_id, "evaluation_ordinal": evaluation_ordinal,
            "accepted_iterate_id": accepted_iterate_id,
            "candidate_ref": candidate_ref, "candidate": candidate,
            "model_identity": dict(batch["model_identity"]),
        }


def _physical_candidate_locator(candidate_ref, candidate_body):
    return {
        "stage_id": candidate_body["stage_id"],
        "evaluation_ordinal": candidate_body["evaluation_ordinal"],
        "candidate_ref": dict(candidate_ref),
    }


def _physical_candidate_source_locator(selected):
    """Keep an actual candidate's source; terminal-only values have no batch."""
    return {
        "kind": selected["kind"], "stage_id": selected["stage_id"],
        "evaluation_ordinal": selected.get("evaluation_ordinal"),
        "candidate_ref": dict(selected["candidate_ref"]),
        "source_task_id": selected["source_task_id"],
        "source_attempt_id": selected["source_attempt_id"],
        **({"source_batch_ref": dict(selected["source_batch_ref"])}
           if "source_batch_ref" in selected else {}),
    }


def _physical_stage_history_baseline_locator(candidate_locators, baseline_ref,
                                              stage_id, model_identity):
    # Evaluation keys are (authored stage_id, ordinal), so an authored stage
    # named "baseline" is not a discriminator. The actual occurrence owns kind.
    saved_baselines = [row for row in candidate_locators.values()
                       if row.get("kind") == "baseline" and row["stage_id"] == stage_id]
    if any(row["model_identity"] != dict(model_identity) for row in saved_baselines):
        raise _error("Physical Optimization history baseline uses another model.",
                     stage_id=stage_id)
    current_baseline = (None if baseline_ref is None else
                        candidate_locators.get(("baseline", baseline_ref["sha256"])))
    if current_baseline is not None and current_baseline["stage_id"] == stage_id:
        saved_baseline = current_baseline
    elif len(saved_baselines) == 1:
        saved_baseline = saved_baselines[0]
    elif saved_baselines:
        raise _error("Physical Optimization history has ambiguous baseline sources.",
                     stage_id=stage_id)
    else:
        saved_baseline = None
    return (None if saved_baseline is None else
            _physical_candidate_source_locator(saved_baseline))


def complete_optimization_operation(binding, *, operation_id, task_id, attempt_id,
                                    result, manifest, stages, transitions, outcome):
    """Seal physical Optimization result/indexes and terminal task state atomically.

    Committed batch objects contain candidate references, never repeated
    candidate bodies. The completion transaction seals ordered stage/history
    manifests, the comparison selectors, the result body, and the terminal
    attempt pointer as one publication boundary.
    """
    if outcome not in {"success", "partial"}:
        raise _error("Physical Optimization terminal outcome is unsupported.", outcome=outcome)
    if (not isinstance(result, Mapping) or not isinstance(manifest, Mapping)
            or not isinstance(stages, Sequence) or isinstance(stages, (str, bytes))
            or not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes))):
        raise _error("Physical Optimization terminal descriptors are malformed.", task_id=task_id)

    store = _bound(binding)
    batch_stream = _physical_stage_stream(task_id, attempt_id)
    task_stream = _stream(task_id)
    with _snapshot(store) as snapshot:
        descriptor = _descriptor(snapshot, task_id)
        association = _pointer(snapshot, _association_pointer(operation_id))
        descriptor_pointer = snapshot.read_pointer(f"task/{task_id}")
        attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
        current_attempt_pointer = snapshot.read_pointer(f"current_attempt/{task_id}")
        if (descriptor.get("request_sha256") is None
                or association is None
                or association.get("operation_id") != operation_id
                or association.get("task_id") != task_id
                or association.get("attempt_id") != attempt_id
                or attempt_pointer is None
                or current_attempt_pointer is None
                or current_attempt_pointer.get("reference") != attempt_pointer.get("reference")):
            raise _error("Physical Optimization terminal is not bound to the current attempt.",
                         operation_id=operation_id, task_id=task_id, attempt_id=attempt_id)
        attempt = _decode(snapshot, attempt_pointer["reference"])
        if attempt.get("status") not in {"allocated", "running"}:
            raise _error("Physical Optimization attempt is not terminally publishable.",
                         task_id=task_id, attempt_id=attempt_id)
        request_ref = next((item for item in descriptor.get("artifacts", ())
                            if item.get("role") == "operation_request"), None)
        if not isinstance(request_ref, Mapping) or request_ref.get("sha256") != descriptor.get("request_sha256"):
            raise _error("Physical Optimization task lacks its canonical request.", task_id=task_id)
        request_raw = snapshot.get_object(request_ref)
        request = record_document(request_raw)
        if (not isinstance(request, dict) or record_bytes(request) != request_raw
                or request.get("operation") != "optimize_direct"
                or request.get("runtime_semantic", {}).get("backend") != "jax"):
            raise _error("Physical Optimization terminal is bound to another request authority.",
                         task_id=task_id, attempt_id=attempt_id)
        source_prefix = _physical_checkpoint_prefix(
            snapshot, store, attempt.get("resume_from"),
            expected_request_sha256=descriptor["request_sha256"],
        ) if attempt.get("resume_from") is not None else {"sources": [], "rows": []}
        batch_revision = snapshot.stream_revision(batch_stream)
        task_revision = snapshot.stream_revision(task_stream)
        sequence_pointer = snapshot.read_pointer("event_sequence/" + task_id)
        sequence = (0 if sequence_pointer is None else
                    _decode(snapshot, sequence_pointer["reference"]).get("next_sequence"))
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise _error("Physical Optimization task event sequence is malformed.", task_id=task_id)
        batch_stream_record = snapshot.read_stream(batch_stream)
        committed_batches = []
        for entry in batch_stream_record["entries"]:
            if entry.get("kind") != "optimization_stage_batch":
                raise _error("Physical Optimization batch stream contains an unknown entry.",
                             task_id=task_id)
            batch_raw = snapshot.get_object(entry["reference"])
            batch = record_document(batch_raw)
            if (not isinstance(batch, dict) or record_bytes(batch) != batch_raw
                    or batch.get("schema") != "scnsim.optimization_stage_batch"
                    or batch.get("schema_version") != 1):
                raise _error("Physical Optimization committed batch is malformed.",
                             task_id=task_id)
            baseline_metadata = {}
            for row in batch.get("rows", ()):
                if not isinstance(row, Mapping) or row.get("kind") != "baseline":
                    continue
                candidate_ref = row.get("candidate_ref")
                if (not isinstance(candidate_ref, Mapping)
                        or candidate_ref.get("role") != "optimization_candidate"):
                    raise _error("Physical Optimization baseline reference is malformed.",
                                 task_id=task_id, attempt_id=attempt_id)
                candidate_raw = snapshot.get_object(candidate_ref)
                candidate_body = record_document(candidate_raw)
                if (not isinstance(candidate_body, dict)
                        or record_bytes(candidate_body) != candidate_raw
                        or candidate_body.get("schema") != "scnsim.optimization_candidate"
                        or candidate_body.get("schema_version") != 1
                        or any(candidate_body.get(key) != value for key, value in _physical_identity(
                            store, descriptor["request_sha256"], task_id, attempt_id,
                        ).items())
                        or candidate_body.get("kind") != "baseline"
                        or candidate_body.get("evaluation_ordinal") != row.get("evaluation_ordinal")
                        or not isinstance(candidate_body.get("stage_id"), str)
                        or not isinstance(candidate_body.get("model_identity"), Mapping)
                        or (candidate_body.get("source_locator") is not None
                            and not isinstance(candidate_body.get("source_locator"), Mapping))):
                    raise _error("Physical Optimization baseline body is malformed.",
                                 task_id=task_id, attempt_id=attempt_id)
                baseline_metadata[candidate_ref["sha256"]] = {
                    "stage_id": candidate_body["stage_id"],
                    "model_identity": dict(candidate_body["model_identity"]),
                }
            committed_batches.append((dict(entry["reference"]), batch, baseline_metadata))

    identity = _physical_identity(store, descriptor["request_sha256"], task_id, attempt_id)
    transaction_id = str(uuid4())
    # The terminal stage list is the authority for stages that were actually
    # entered. A declared stage may have no complete population/iteration yet
    # (for example, a CMA budget that ends before its first full generation),
    # so its sealed history is legitimately empty. Batch streams alone cannot
    # represent that entered stage.
    stage_order = []
    for ordinal, source in enumerate(stages):
        if not isinstance(source, Mapping):
            raise _error("Physical Optimization stage summary is malformed.", ordinal=ordinal)
        stage_id = source.get("stage_id")
        if not isinstance(stage_id, str) or not stage_id or stage_id in stage_order:
            raise _error("Physical Optimization stage order is malformed.", ordinal=ordinal)
        stage_order.append(stage_id)
    last_batch_stage = -1
    stage_indexes = []
    candidate_locators = {}
    candidate_refs_by_stage = {}
    baseline_refs = []
    transition_refs = []
    for batch_ref, batch, baseline_metadata in committed_batches:
        if any(batch.get(key) != value for key, value in identity.items()):
            raise _error("Physical Optimization batch belongs to another task or attempt.",
                         task_id=task_id, attempt_id=attempt_id)
        stage_id, model_identity = batch.get("stage_id"), batch.get("model_identity")
        if not isinstance(stage_id, str) or not isinstance(model_identity, Mapping):
            raise _error("Physical Optimization batch stage identity is malformed.", task_id=task_id)
        if stage_id not in stage_order:
            raise _error("Physical Optimization batch names a stage that was not entered.",
                         stage_id=stage_id)
        batch_stage = stage_order.index(stage_id)
        if batch_stage < last_batch_stage:
            raise _error("Physical Optimization batch order differs from entered stages.",
                         stage_id=stage_id)
        last_batch_stage = batch_stage
        rows = batch.get("rows")
        if not isinstance(rows, list):
            raise _error("Physical Optimization batch rows are malformed.", stage_id=stage_id)
        stage_candidates = candidate_refs_by_stage.setdefault(stage_id, [])
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("candidate_ref"), Mapping):
                raise _error("Physical Optimization batch candidate locator is malformed.",
                             stage_id=stage_id)
            if row.get("kind") == "baseline":
                candidate_ref = dict(row["candidate_ref"])
                original_identity = baseline_metadata.get(candidate_ref.get("sha256"))
                if not isinstance(original_identity, Mapping):
                    raise _error("Physical Optimization baseline lacks its stored scientific identity.",
                                 stage_id=stage_id)
                baseline_refs.append(candidate_ref)
                baseline_row = {
                    "stage_id": original_identity["stage_id"],
                    "model_identity": dict(original_identity["model_identity"]),
                    "kind": "baseline", "evaluation_ordinal": row.get("evaluation_ordinal"),
                    "candidate_ref": candidate_ref,
                    "summary_json": dict(row.get("summary_json", {})),
                    "source_task_id": task_id, "source_attempt_id": attempt_id,
                    "source_batch_ref": dict(batch_ref),
                }
                candidate_locators[("baseline", candidate_ref["sha256"])] = baseline_row
                continue
            if row.get("kind") != "evaluation":
                raise _error("Physical Optimization batch row kind is unsupported.", stage_id=stage_id)
            history_ordinal = row.get("history_ordinal")
            evaluation_ordinal = row.get("evaluation_ordinal")
            summary = row.get("summary_json")
            if (not isinstance(history_ordinal, int) or isinstance(history_ordinal, bool)
                    or history_ordinal < 0 or not isinstance(evaluation_ordinal, int)
                    or isinstance(evaluation_ordinal, bool) or evaluation_ordinal < 0
                    or not isinstance(summary, Mapping)):
                raise _error("Physical Optimization batch row identity is malformed.", stage_id=stage_id)
            locator = {"kind": "evaluation", "stage_id": stage_id,
                       "evaluation_ordinal": evaluation_ordinal,
                       "candidate_ref": dict(row["candidate_ref"]),
                       "source_task_id": task_id, "source_attempt_id": attempt_id,
                       "source_batch_ref": dict(batch_ref)}
            stage_candidates.append({"ordinal": history_ordinal, "stage_id": stage_id,
                                     "candidate_ref": dict(row["candidate_ref"]),
                                     "summary_json": dict(summary), "model_identity": dict(model_identity),
                                     "evaluation_ordinal": evaluation_ordinal,
                                     "source_task_id": task_id, "source_attempt_id": attempt_id,
                                     "source_batch_ref": dict(batch_ref)})
            key = (stage_id, evaluation_ordinal)
            if key in candidate_locators:
                raise _error("Physical Optimization stage repeats an evaluation ordinal.",
                             stage_id=stage_id, evaluation_ordinal=evaluation_ordinal)
            candidate_locators[key] = locator
        for transition_ref in batch.get("transition_refs", ()):
            if not isinstance(transition_ref, Mapping):
                raise _error("Physical Optimization transition reference is malformed.", stage_id=stage_id)
            transition_refs.append(dict(transition_ref))

    ancestry_batches = {
        (source["task_id"], source["attempt_id"], record_bytes(dict(batch_ref)))
        for source in source_prefix["sources"] for batch_ref in source["batch_refs"]
    }
    source_rows_by_identity = {}
    source_rows_by_ref = {}
    for row in source_prefix["rows"]:
        identity_key = (row["source_task_id"], row["source_attempt_id"],
                        row["kind"], row["stage_id"], row["evaluation_ordinal"],
                        record_bytes(row["candidate_ref"]))
        source_rows_by_identity[identity_key] = row
        source_rows_by_ref[(row["source_task_id"], row["source_attempt_id"],
                            row["kind"], record_bytes(row["candidate_ref"]))] = row
        if row["kind"] == "baseline":
            candidate_locators[("baseline", row["candidate_ref"]["sha256"])] = row
            continue
        stage_candidates = candidate_refs_by_stage.setdefault(row["stage_id"], [])
        stage_candidates.append({
            "ordinal": row["history_ordinal"], "stage_id": row["stage_id"],
            "candidate_ref": dict(row["candidate_ref"]),
            "summary_json": dict(row["summary_json"]), "model_identity": dict(row["model_identity"]),
            "evaluation_ordinal": row["evaluation_ordinal"],
            "source_task_id": row["source_task_id"],
            "source_attempt_id": row["source_attempt_id"],
            "source_batch_ref": dict(row["source_batch_ref"]),
        })
        key = (row["stage_id"], row["evaluation_ordinal"])
        prior = candidate_locators.get(key)
        if prior is not None and dict(prior["candidate_ref"]) != dict(row["candidate_ref"]):
            raise _error("Physical Optimization source prefix repeats an evaluation ordinal.",
                         stage_id=row["stage_id"], evaluation_ordinal=row["evaluation_ordinal"])
        candidate_locators[key] = {
            "kind": "evaluation", "stage_id": row["stage_id"],
            "evaluation_ordinal": row["evaluation_ordinal"],
            "candidate_ref": dict(row["candidate_ref"]),
            "source_task_id": row["source_task_id"],
            "source_attempt_id": row["source_attempt_id"],
            "source_batch_ref": dict(row["source_batch_ref"]),
        }

    for stage_id, stage_rows in candidate_refs_by_stage.items():
        stage_rows.sort(key=lambda row: row["ordinal"])

    if len({reference["sha256"] for reference in baseline_refs}) > 1 or len(baseline_refs) > 1:
        raise _error("Physical Optimization has conflicting committed baseline occurrences.",
                     task_id=task_id)

    def candidate_reference(locator, *, label):
        if locator is None:
            return None
        if not isinstance(locator, Mapping):
            raise _error("Physical Optimization candidate locator is malformed.", field=label)
        kind = locator.get("kind", "evaluation")
        supplied_ref = locator.get("candidate_ref")
        source_task_id = locator.get("source_task_id")
        source_attempt_id = locator.get("source_attempt_id")
        if (source_task_id is None) != (source_attempt_id is None):
            raise _error("Physical Optimization candidate locator has incomplete source identity.",
                         field=label)
        if kind == "baseline":
            if source_task_id is not None:
                if not isinstance(supplied_ref, Mapping):
                    raise _error("Physical Optimization source baseline lacks its candidate reference.",
                                 field=label)
                selected = source_rows_by_ref.get((source_task_id, source_attempt_id,
                    "baseline", record_bytes(dict(supplied_ref))))
                if selected is None:
                    raise _error("Physical Optimization source baseline is outside the selected checkpoint prefix.",
                                 field=label, source_task_id=source_task_id,
                                 source_attempt_id=source_attempt_id)
            else:
                if baseline_ref is None:
                    raise _error("Physical Optimization baseline locator has no committed occurrence.",
                                 field=label)
                if supplied_ref is not None and dict(supplied_ref) != baseline_ref:
                    raise _error("Physical Optimization baseline locator reference changed.", field=label)
                selected = candidate_locators.get(("baseline", baseline_ref["sha256"]))
                if selected is None:
                    raise _error("Physical Optimization baseline has no committed batch source.",
                                 field=label)
        else:
            stage_id = locator.get("stage_id")
            evaluation_ordinal = locator.get("evaluation_ordinal")
            if (kind != "evaluation" or not isinstance(stage_id, str) or not stage_id
                    or not isinstance(evaluation_ordinal, int)
                    or isinstance(evaluation_ordinal, bool) or evaluation_ordinal < 0):
                raise _error("Physical Optimization candidate locator lacks stage identity.", field=label)
            if source_task_id is not None:
                if not isinstance(supplied_ref, Mapping):
                    raise _error("Physical Optimization source candidate lacks its candidate reference.",
                                 field=label)
                key = (source_task_id, source_attempt_id, kind, stage_id,
                       evaluation_ordinal, record_bytes(dict(supplied_ref)))
                selected = source_rows_by_identity.get(key)
                if selected is None:
                    raise _error("Physical Optimization source candidate is outside the selected checkpoint prefix.",
                                 field=label, stage_id=stage_id,
                                 evaluation_ordinal=evaluation_ordinal)
            else:
                selected = candidate_locators.get((stage_id, evaluation_ordinal))
                if selected is None:
                    raise _error("Physical Optimization selector does not name a committed occurrence.",
                                 field=label, stage_id=stage_id,
                                 evaluation_ordinal=evaluation_ordinal)
                if supplied_ref is not None and dict(supplied_ref) != dict(selected["candidate_ref"]):
                    raise _error("Physical Optimization candidate locator reference changed.", field=label)
        return _physical_candidate_source_locator(selected)

    manifest_body = {
        "schema": "scnsim.optimization_manifest", "schema_version": 1, "evidence_version": 3,
        **identity, "outcome": outcome,
        "request_ref": dict(request_ref), "declarations": dict(manifest),
        "stage_ids": list(stage_order),
    }
    manifest_ref = result_ref = comparison_ref = index_root_ref = completion_ref = attempt_ref = None
    stage_root_rows = []
    sql_stage_rows = []
    transition_index_rows = []
    baseline_ref = baseline_refs[0] if baseline_refs else None
    with store.transaction(transaction_id, expected_revisions={
            task_stream: task_revision, batch_stream: batch_revision}) as tx:
        current_association = _pointer(tx, _association_pointer(operation_id))
        current_task = tx.read_pointer(f"task/{task_id}")
        current_attempt = tx.read_pointer(f"attempt/{task_id}/{attempt_id}")
        current_attempt_pointer = tx.read_pointer(f"current_attempt/{task_id}")
        if (descriptor_pointer is None or current_task is None
                or current_task.get("reference") != descriptor_pointer.get("reference")
                or current_association != association
                or current_attempt is None
                or current_attempt.get("reference") != attempt_pointer.get("reference")
                or current_attempt_pointer is None
                or current_attempt_pointer.get("reference") != attempt_pointer.get("reference")):
            raise _error("Physical Optimization attempt binding changed before terminal publication.",
                         operation_id=operation_id, task_id=task_id, attempt_id=attempt_id)

        # A zero-evaluation or zero-complete-iteration result still owns the
        # actual baseline occurrence in its terminal transaction.
        baseline_body = result.get("baseline")
        if baseline_ref is None and baseline_body is not None:
            if not isinstance(baseline_body, Mapping):
                raise _error("Physical Optimization terminal baseline is malformed.", task_id=task_id)
            baseline_candidate = dict(baseline_body)
            stage_id = baseline_candidate.get("stage_id")
            model_identity = baseline_candidate.get("model_identity")
            if not isinstance(stage_id, str) or not isinstance(model_identity, Mapping):
                raise _error("Physical Optimization terminal baseline lacks its model identity.", task_id=task_id)
            baseline_candidate_body = {
                **baseline_candidate, "kind": "baseline",
                "schema": "scnsim.optimization_candidate", "schema_version": 1,
                **identity,
            }
            baseline_ref = tx.put_object(record_bytes(baseline_candidate_body), role="optimization_candidate")
            candidate_locators[("baseline", baseline_ref["sha256"])] = {
                "kind": "baseline", "stage_id": stage_id,
                "model_identity": dict(model_identity),
                "evaluation_ordinal": baseline_candidate["evaluation_ordinal"],
                "candidate_ref": baseline_ref,
                "source_task_id": task_id, "source_attempt_id": attempt_id,
            }

        manifest_ref = tx.put_object(record_bytes(manifest_body), role="optimization_manifest")
        completed_stage_ids = []
        for ordinal, source in enumerate(stages):
            if not isinstance(source, Mapping):
                raise _error("Physical Optimization stage summary is malformed.", ordinal=ordinal)
            stage_id = source.get("stage_id")
            model_identity = source.get("model_identity")
            strategy = source.get("strategy")
            objective_ids = source.get("objective_ids")
            variable_refs = source.get("variable_refs")
            status = source.get("status")
            evaluation_count = source.get("evaluation_count")
            iteration_count = source.get("iteration_count")
            if (not isinstance(stage_id, str) or stage_id in completed_stage_ids
                    or not isinstance(model_identity, Mapping) or not isinstance(strategy, str)
                    or not isinstance(objective_ids, Sequence) or isinstance(objective_ids, (str, bytes))
                    or not isinstance(variable_refs, Sequence) or isinstance(variable_refs, (str, bytes))
                    or not isinstance(status, str)
                    or not isinstance(evaluation_count, int) or isinstance(evaluation_count, bool)
                    or not isinstance(iteration_count, int) or isinstance(iteration_count, bool)):
                raise _error("Physical Optimization stage summary fields are malformed.", ordinal=ordinal)
            completed_stage_ids.append(stage_id)
            if stage_id not in stage_order and candidate_refs_by_stage.get(stage_id):
                raise _error("Physical Optimization stage summary has no committed batch.", stage_id=stage_id)
            stage_rows = sorted(candidate_refs_by_stage.get(stage_id, ()), key=lambda row: row["ordinal"])
            if len({row["ordinal"] for row in stage_rows}) != len(stage_rows):
                raise _error("Physical Optimization stage history repeats an ordinal.", stage_id=stage_id)
            if any(row["model_identity"] != dict(model_identity) for row in stage_rows):
                raise _error("Physical Optimization stage history uses another model.", stage_id=stage_id)
            summary = {
                key: source.get(key) for key in (
                    "strategy", "objective_ids", "variable_refs", "termination_reason", "status",
                    "evaluation_count", "iteration_count",
                )
            }
            observed_ref = candidate_reference(source.get("observed_best"), label="observed_best")
            accepted_ref = candidate_reference(source.get("accepted_iterate"), label="accepted_iterate")
            summary_body = {
                "schema": "scnsim.optimization_stage_summary", "schema_version": 1,
                **identity, "stage_id": stage_id, "model_identity": dict(model_identity),
                "summary": summary,
                "observed_best_ref": (None if observed_ref is None else observed_ref["candidate_ref"]),
                "observed_best_locator": observed_ref,
                "accepted_iterate_ref": (None if accepted_ref is None else accepted_ref["candidate_ref"]),
                "accepted_iterate_locator": accepted_ref,
            }
            history_baseline_locator = _physical_stage_history_baseline_locator(
                candidate_locators, baseline_ref, stage_id, model_identity,
            )
            history_body = {
                "schema": "scnsim.optimization_stage_history", "schema_version": 1,
                "baseline_locator": history_baseline_locator,
                **identity, "stage_id": stage_id, "model_identity": dict(model_identity),
                "row_count": len(stage_rows),
                "candidate_refs": [dict(row["candidate_ref"]) for row in stage_rows],
                "candidate_sources": [
                    {"task_id": row["source_task_id"],
                     "attempt_id": row["source_attempt_id"],
                     "batch_ref": dict(row["source_batch_ref"])}
                    for row in stage_rows
                ],
            }
            summary_ref = tx.put_object(record_bytes(summary_body), role="optimization_stage_summary")
            history_ref = tx.put_object(record_bytes(history_body), role="optimization_stage_history")
            stage_root_rows.append({"stage_id": stage_id, "model_identity": dict(model_identity),
                                    "summary_ref": summary_ref, "history_ref": history_ref})
            sql_stage_rows.append({"stage_id": stage_id, "model_identity": dict(model_identity),
                                   "summary_ref": summary_ref, "history_ref": history_ref,
                                   "rows": stage_rows})

        if completed_stage_ids != [source.get("stage_id") for source in stages]:
            raise _error("Physical Optimization stage order is malformed.", task_id=task_id)
        comparison = result.get("comparison")
        if not isinstance(comparison, Mapping):
            raise _error("Physical Optimization comparison declaration is malformed.", task_id=task_id)
        comparison_model_identity = comparison.get("comparison_model_identity")
        comparison_stage_id = comparison.get("comparison_stage_id")
        declarations = comparison.get("declarations")
        if (comparison_model_identity is not None and not isinstance(comparison_model_identity, Mapping)
                or comparison_stage_id is not None and not isinstance(comparison_stage_id, str)
                or not isinstance(declarations, Mapping)):
            raise _error("Physical Optimization comparison identity is malformed.", task_id=task_id)
        winner_locator = candidate_reference(comparison.get("winner"), label="comparison.winner")
        baseline_locator = candidate_reference(comparison.get("baseline"), label="comparison.baseline")
        winner_ref = None if winner_locator is None else winner_locator["candidate_ref"]
        selected_baseline_ref = None if baseline_locator is None else baseline_locator["candidate_ref"]
        if selected_baseline_ref is not None:
            baseline_ref = selected_baseline_ref
        comparison_body = {
            "schema": "scnsim.optimization_comparison", "schema_version": 1,
            **identity,
            "comparison_model_identity": (None if comparison_model_identity is None
                                          else dict(comparison_model_identity)),
            "comparison_stage_id": comparison_stage_id,
            "declarations": dict(declarations),
            "baseline_ref": baseline_ref,
            "baseline_locator": baseline_locator,
            "winner_ref": winner_ref,
            "winner_locator": winner_locator,
        }
        comparison_ref = tx.put_object(record_bytes(comparison_body), role="optimization_comparison")

        transition_ids = set()
        committed_transition_ids = {row["transition_id"] for row in transition_refs}
        for ordinal, transition in enumerate(transitions):
            if not isinstance(transition, Mapping):
                raise _error("Physical Optimization transition is malformed.", ordinal=ordinal)
            transition_id = transition.get("transition_id")
            if not isinstance(transition_id, str) or not transition_id or transition_id in transition_ids:
                raise _error("Physical Optimization transition identity is malformed.", ordinal=ordinal)
            transition_ids.add(transition_id)
            already_committed = next((row for row in transition_refs
                                      if row["transition_id"] == transition_id), None)
            if already_committed is not None:
                # The transition body was sealed at the stage boundary. The
                # terminal index reuses that exact reference.
                transition_index_rows.append(dict(already_committed))
                continue
            transition_body = {
                **dict(transition), "schema": "scnsim.optimization_transition",
                "schema_version": 1, **identity,
            }
            transition_ref = tx.put_object(record_bytes(transition_body), role="optimization_transition")
            transition_index_rows.append({"transition_id": transition_id,
                                          "transition_ref": transition_ref})

        if committed_transition_ids - transition_ids:
            raise _error("Physical Optimization terminal omits a committed stage transition.",
                         transition_ids=sorted(committed_transition_ids - transition_ids))

        result_body = {
            "schema": "scnsim.optimization_result", "schema_version": 1,
            **identity, "type": "optimization", "outcome": outcome,
            "manifest_ref": manifest_ref,
            "summary": dict(result.get("summary", {})),
        }
        result_ref = tx.put_object(record_bytes(result_body), role="operation_result")
        generation_ancestry_ref = None
        if source_prefix["sources"]:
            ancestry_body = {
                "schema": "scnsim.optimization_generation_ancestry",
                "schema_version": 1, **identity, "outcome": outcome,
                "sources": source_prefix["sources"],
            }
            generation_ancestry_ref = tx.put_object(
                record_bytes(ancestry_body), role="optimization_generation_ancestry",
            )
        index_root = {
            "schema": "scnsim.optimization_index_root", "schema_version": 1,
            **identity, "outcome": outcome, "result_ref": result_ref,
            "manifest_ref": manifest_ref, "comparison_ref": comparison_ref,
            "stages": stage_root_rows,
            "transitions": [{"transition_id": row["transition_id"],
                             "transition_ref": row["transition_ref"]}
                            for row in transition_index_rows],
            "baseline_ref": baseline_ref,
            "generation_ancestry_ref": generation_ancestry_ref,
        }
        index_root_ref = tx.put_object(record_bytes(index_root), role="optimization_index_root")
        completion = {
            "schema": "scnsim.workspace_completion", "schema_version": 1,
            **identity, "outcome": outcome, "result_ref": result_ref,
            "manifest_ref": manifest_ref, "index_root_ref": index_root_ref,
            "comparison_ref": comparison_ref, "baseline_ref": baseline_ref,
            "terminal_attempt_ref": None,
            "stage_count": len(stage_root_rows),
            "transition_count": len(transition_index_rows),
            "evaluation_count": sum(source["evaluation_count"] for source in stages),
            "generation_ancestry_ref": generation_ancestry_ref,
        }
        # The attempt document's digest is part of the selected terminal identity.
        # It contains artifact references only, never copies of candidate bodies.
        terminal_attempt = dict(attempt)
        artifacts = list(terminal_attempt.get("artifacts", ()))
        for reference in (result_ref, manifest_ref, comparison_ref, index_root_ref,
                          *([] if generation_ancestry_ref is None else [generation_ancestry_ref]),
                          *(row["summary_ref"] for row in stage_root_rows),
                          *(row["history_ref"] for row in stage_root_rows),
                          *(row["transition_ref"] for row in transition_index_rows)):
            if reference not in artifacts:
                artifacts.append(reference)
        terminal_status = "success" if outcome == "success" else "partial"
        terminal_attempt.update(status=terminal_status, failure=None, interruption=None,
                                artifacts=artifacts)
        attempt_ref = tx.put_object(record_bytes(terminal_attempt), role="attempt")
        completion["terminal_attempt_ref"] = attempt_ref
        completion_ref = tx.put_object(record_bytes(completion), role="workspace_completion")
        success_pointer = None
        if outcome == "success":
            selection = {
                "event": "request_success_selected", "request_sha256": descriptor["request_sha256"],
                "task_id": task_id, "attempt_id": attempt_id, "result_ref": result_ref,
                "workspace_completion": completion_ref,
            }
            selection_ref = tx.put_object(record_bytes(selection), role="success_selection")
            selected_ref = tx.select_first("success/" + descriptor["request_sha256"], selection_ref)
            # A named restart is a new exact attempt for the same immutable
            # request. Preserve the request's first-success selector while
            # sealing and returning this attempt independently by attempt ID.
            success_pointer = selected_ref
        task_event = {
            "task_id": task_id, "sequence": sequence, "kind": "optimization_terminal",
            "payload": {"attempt_id": attempt_id, "outcome": outcome,
                        "result": result_ref, "workspace_completion": completion_ref},
        }
        event_ref = tx.append_with_ref(
            task_stream, "task_change",
            record_bytes({"kind": "event", "event": task_event,
                          "artifacts": [result_ref, completion_ref]}),
        )[1]
        terminal_change = {
            "kind": "attempt_update", "attempt_id": attempt_id, "status": terminal_status,
            "failure": None, "interruption": None,
            "checkpoint": None,
            "artifacts": [result_ref, manifest_ref, comparison_ref, index_root_ref, completion_ref],
        }
        _, state_ref = tx.append_with_ref(task_stream, "task_change", record_bytes(terminal_change))
        tx.set_pointer(f"attempt/{task_id}/{attempt_id}", attempt_ref)
        tx.set_pointer(f"current_attempt/{task_id}", attempt_ref)
        tx.set_pointer(_attempt_state_pointer(task_id, attempt_id), state_ref)
        tx.set_pointer(f"completion/{task_id}/{attempt_id}", completion_ref)
        next_ref = tx.put_object(record_bytes({"next_sequence": sequence + 1}), role="event_sequence")
        tx.set_pointer("event_sequence/" + task_id, next_ref)
        tx.index_optimization(completion_ref["sha256"], sql_stage_rows, transition_index_rows)
    return {
        "result_ref": result_ref, "manifest_ref": manifest_ref,
        "comparison_ref": comparison_ref, "index_root_ref": index_root_ref,
        "completion_ref": completion_ref, "attempt_ref": attempt_ref,
        "success_selection_ref": success_pointer,
        "ack": {"committed": True, "transaction_id": transaction_id,
                "outcome": outcome, "stage_count": len(stage_root_rows),
                "evaluation_count": completion["evaluation_count"]},
    }


def read_operation_success(binding,task_id,*,attempt_id=None,projection_consumer=None):
    from .evidence_reader import read_success
    store = _bound(binding)
    if store.path.exists() or store.path.is_symlink():
        with _snapshot(store) as snapshot:
            descriptor = _pointer(snapshot, 'task/'+task_id)
            if descriptor is not None:
                selected = _pointer(snapshot, 'success/'+descriptor['request_sha256'])
                if selected is not None and attempt_id is not None and selected['attempt_id'] != attempt_id:
                    selected = None
                return read_success(
                    store.root, snapshot, descriptor, attempt_id=attempt_id, selection=selected,
                    projection_consumer=projection_consumer,
                    binding_identity={"root": str(binding.root), "leaf": str(binding.leaf),
                                      "plan_sha256": binding.plan_sha256,
                                      "workspace_instance_id": binding.workspace_instance_id},
                )
    raise _error('Operation task identity is not recorded.', task_id=task_id)


def read_operation_attempt(binding, attempt_id, *, projection_consumer=None):
    """Read one exact terminal attempt selected by its immutable attempt ID.

    The pointer lookup and all result verification share one SQLite snapshot;
    this path does not reconstruct a request from current Plan/runtime state or
    replay task history to discover the attempt.
    """
    store = _bound(binding)
    if not (store.path.exists() or store.path.is_symlink()):
        return None
    with _snapshot(store) as snapshot:
        located = snapshot.find_attempt_pointer(attempt_id)
        if located is None:
            return None
        task_id = located["task_id"]
        descriptor = _descriptor(snapshot, task_id)
        attempt = _decode(snapshot, located["reference"])
        if (attempt.get("attempt_id") != attempt_id
                or descriptor.get("task_id") != task_id
                or not isinstance(descriptor.get("request_sha256"), str)):
            raise _error("Exact attempt pointer differs from its task descriptor.",
                         task_id=task_id, attempt_id=attempt_id)
        from .evidence_reader import read_success
        return read_success(
            store.root, snapshot, descriptor, attempt_id=attempt_id, selection=None,
            projection_consumer=projection_consumer,
            binding_identity={"root": str(binding.root), "leaf": str(binding.leaf),
                              "plan_sha256": binding.plan_sha256,
                              "workspace_instance_id": binding.workspace_instance_id},
        )


def read_sensitivity_result(binding, task_id, *, attempt_id=None, projection_consumer=None):
    """Read a fixed successful result or an explicitly named Sensitivity attempt."""
    success = read_operation_success(
        binding, task_id, attempt_id=attempt_id,
    )
    if success is None:
        return None
    from .results import SensitivityResultSelection
    selection = success.get("selection")
    if not isinstance(selection, SensitivityResultSelection):
        raise _error("The selected operation is not a Sensitivity result.", task_id=task_id,
                     attempt_id=attempt_id)
    return projection_consumer(success) if projection_consumer is not None else success


def find_operation_success(binding,request_sha256,*,projection_consumer=None):
    store=_bound(binding)
    if store.path.exists() or store.path.is_symlink():
        with _snapshot(store) as snapshot:
            selected=_pointer(snapshot,'success/'+request_sha256)
            if selected is not None:
                descriptor=_descriptor(snapshot,selected['task_id'])
                from .evidence_reader import read_success
                consumer = projection_consumer
                if consumer is not None:
                    def consumer(success):
                        if (success is None or success['result_ref']!=selected['result_ref']
                                or descriptor['request_sha256']!=request_sha256
                                or selected.get('request_sha256')!=request_sha256
                                or selected.get('task_id')!=descriptor['task_id']):
                            raise _error('Selected operation success differs from committed task evidence.')
                        return projection_consumer(success)
                success=read_success(store.root,snapshot,descriptor,
                                     attempt_id=selected['attempt_id'],selection=selected,
                                     projection_consumer=consumer,
                                     binding_identity={"root": str(binding.root), "leaf": str(binding.leaf),
                                                       "plan_sha256": binding.plan_sha256,
                                                       "workspace_instance_id": binding.workspace_instance_id})
                if projection_consumer is not None:
                    return success
                if success is None or success['result_ref']!=selected['result_ref'] or descriptor['request_sha256']!=request_sha256:
                    raise _error('Selected operation success differs from committed task evidence.')
                return success
    legacy_path=operation_workspace(binding)/'benchmark.json'
    if legacy_path.exists():
        raise _unsupported_old_record(legacy_path)
    return None




class _ChunkReference:
    """One verified immutable staging object; it owns no execution lifetime."""
    __slots__ = ("reference", "byte_length")

    def __init__(self, reference):
        self.reference = dict(reference)
        self.byte_length = reference["byte_length"]


class ChunkStorage:
    """Same-call Direct/Sweep staging in checked SQLite objects, without spool.

    Chunk writes cannot publish completion. The terminal publisher consumes the
    references through its own transaction snapshot, so it never nests a reader
    lock inside the final writer lock.
    """
    def __init__(self, binding):
        self.binding = binding
        self.store = _bound(binding)

    def put_record(self, value):
        payload = record_bytes(value)
        with self.binding.writer():
            with self.store.transaction(str(uuid4()), expected_revisions={}) as tx:
                reference = tx.put_object(payload, role="chunk_staging")
        return _ChunkReference(reference)

    def get_record(self, reference, *, snapshot=None):
        if snapshot is not None:
            return _decode(snapshot, reference.reference)
        with _snapshot(self.store) as reader:
            return _decode(reader, reference.reference)
