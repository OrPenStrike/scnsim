"""Current operation-domain writer and verified SQLite projections.

The core owns native transactions and byte objects; this adapter owns task,
occurrence, checkpoint and selected-result meanings. Completed generation groups
publish atomically. Only their latest CMA/RNG state is retained; an unfinished
population can be archived diagnostically but never enters resumable evidence.
Older file-journal Workspaces remain untouched and require recomputation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..errors import EvidenceIntegrityError, UnsupportedEvidenceVersionError
from .operation_store import OperationStore, Snapshot, _VERSION as _OPERATION_STORE_VERSION, _reference
from ..diagnostics.identity import checkpoint_seal
from ..numeric_encoding import record_bytes, record_document
from .operation_scratch import cleanup_inactive_operation_scratch

_ACTIVE: ContextVar[tuple[Path, Snapshot] | None] = ContextVar('operation_sql_snapshot', default=None)
_MARKER = '$scnsim_benchmark_journal'
_OCCURRENCE = frozenset({'attempt_id','candidate_key','cache_hit','evaluation_ordinal','generation',
                        'population_column','latent_coordinates','source_index','origin',
                        'continuation_t_f64','numerical_source_id'})
_DIAGNOSTIC = frozenset({'population_observed','evaluation','progress'})


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
        "This Workspace uses an older benchmark evidence format; use a new Workspace and recompute.",
        stage="benchmark_record",
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
    binding._cleanup_staging()
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


def _task_state_from_snapshot(snapshot, operation):
    operation_id = operation.get('operation_id')
    pointer = snapshot.read_pointer(_association_pointer(operation_id))
    if pointer is None:
        return {'association':None, 'task_status':None, 'latest_state_reference':None,
                'latest_state':None, 'checkpoint_reference':None}
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
        checkpoint = snapshot.read_pointer('checkpoint/'+task_id+'/'+attempt_id)
        if checkpoint is not None:
            checkpoint_reference = checkpoint['reference']
            if checkpoint_reference.get('role') != 'checkpoint_selection':
                raise _error('Current checkpoint selection has the wrong role.', attempt_id=attempt_id)
            selection = _decode(snapshot, checkpoint_reference)
            if selection.get('attempt_id') != attempt_id or selection.get('task_id') != task_id:
                raise _error('Current checkpoint selection belongs to another attempt.', attempt_id=attempt_id)
    return {'association':association, 'task_status':task_status,
            'latest_state_reference':latest_reference, 'latest_state':latest_state,
            'checkpoint_reference':checkpoint_reference}


def _candidate_summary(value, generation, ordinal):
    """Build the frozen compact index row without copying scientific bodies."""
    components = []
    for objective in value.get('objectives', ()):
        terms = []
        for term in objective.get('terms', ()):
            failure = term.get('failure')
            if isinstance(failure, Mapping):
                failure = {key: failure.get(key) for key in ('kind', 'stage', 'detail')}
            terms.append({
                'term_ordinal': term.get('term_ordinal'),
                'status': term.get('status'),
                'value_f64': term.get('value_f64'),
                'ref_lineage': term.get('lineage'),
                'failure': failure,
            })
        components.append({
            'objective_id': objective.get('id'),
            'status': objective.get('status'),
            'value_f64': objective.get('value_f64'),
            'normalized_residual_f64': objective.get('normalized_residual_f64'),
            'weighted_cost_f64': objective.get('cost_f64'),
            'terms': terms,
        })
    return {
        'evaluation_ordinal': ordinal,
        'generation': generation,
        'status': 'failure' if value.get('failure') is not None else 'success',
        'cost_f64': value.get('cost_f64'),
        'parameters': value.get('parameters'),
        'components': components,
    }


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


def _completion_indexes(writer, tip, staged, current_generations, current_candidates):
    """Resolve the exact committed generation-index chain for one terminal."""
    if tip is None:
        return [], []
    reverse = []
    reference = tip
    seen = set()
    with _snapshot(writer.store) as snapshot:
        while reference is not None:
            digest = reference.get('sha256')
            if not isinstance(digest, str) or digest in seen:
                raise _error('Optimization generation evidence ancestry is malformed.', task_id=writer.task_id)
            seen.add(digest)
            local = staged.get(digest)
            if local is not None:
                generation_row, index_document, index_reference, previous_reference, block = local
            else:
                raw = snapshot.get_object(reference)
                block = record_document(raw)
                if record_bytes(block) != raw:
                    raise _error('Optimization generation evidence is not canonical.', task_id=writer.task_id)
                generation_number = block.get('attributes', {}).get('generation')
                index_pointer = snapshot.read_pointer(
                    f"generation_index/{writer.task_id}/{block.get('attempt_id')}/{generation_number}"
                )
                if index_pointer is None or index_pointer['reference'].get('role') != 'workspace_candidate_index':
                    raise _error('Committed generation lacks its candidate index.',
                                 task_id=writer.task_id, generation=generation_number)
                index_reference = index_pointer['reference']
                index_raw = snapshot.get_object(index_reference)
                index_document = record_document(index_raw)
                if record_bytes(index_document) != index_raw:
                    raise _error('Candidate index block is not canonical.', generation=generation_number)
                generation_row = {
                    'generation': generation_number,
                    'block_ref': dict(reference),
                    'first_ordinal': index_document.get('first_ordinal'),
                    'row_count': index_document.get('row_count'),
                    'summary_json': index_document.get('summary_json'),
                    'candidate_index_ref': dict(index_reference),
                }
            generation_number = block.get('attributes', {}).get('generation')
            if (index_document.get('schema') != 'scnsim.workspace_candidate_index'
                    or index_document.get('schema_version') != 1
                    or index_document.get('workspace_instance_id') != writer.store.workspace_instance_id
                    or index_document.get('plan_sha256') != writer.store.plan_sha256
                    or index_document.get('request_sha256') != writer.descriptor.get('request_sha256')
                    or index_document.get('task_id') != writer.task_id
                    or index_document.get('generation') != generation_number
                    or index_document.get('block_ref') != dict(reference)):
                raise _error('Candidate index block differs from its generation evidence.',
                             task_id=writer.task_id, generation=generation_number)
            candidate_rows = index_document.get('candidates')
            if not isinstance(candidate_rows, list) or len(candidate_rows) != generation_row['row_count']:
                raise _error('Candidate index row count differs from its generation.',
                             task_id=writer.task_id, generation=generation_number)
            evidence_rows = block.get('rows')
            if not isinstance(evidence_rows, list) or len(evidence_rows) != len(candidate_rows):
                raise _error('Generation evidence rows differ from its candidate index.',
                             task_id=writer.task_id, generation=generation_number)
            for offset, candidate in enumerate(candidate_rows):
                evidence_row = evidence_rows[offset]
                if (candidate.get('row_offset') != offset
                        or candidate.get('block_ref') != dict(reference)
                        or candidate.get('value_ref') != evidence_row.get('value')):
                    raise _error('Candidate index locator differs from generation evidence.',
                                 task_id=writer.task_id, generation=generation_number,
                                 row_offset=offset)
            reverse.append((generation_row, candidate_rows))
            reference = previous_reference if local is not None else block.get('previous')
    chronological = list(reversed(reverse))
    generations = []
    candidates = []
    next_ordinal = 1
    for expected_generation, (generation_row, candidate_rows) in enumerate(chronological, 1):
        if (generation_row.get('generation') != expected_generation
                or generation_row.get('first_ordinal') != next_ordinal
                or generation_row.get('row_count') != len(candidate_rows)):
            raise _error('Sealed generation index is not contiguous.', task_id=writer.task_id,
                         generation=generation_row.get('generation'))
        generations.append(generation_row)
        for offset, row in enumerate(candidate_rows):
            if row.get('ordinal') != next_ordinal or row.get('generation') != expected_generation:
                raise _error('Sealed candidate index order is not contiguous.', task_id=writer.task_id,
                             evaluation_ordinal=row.get('ordinal'))
            candidates.append(row)
            next_ordinal += 1
    return generations, candidates


def _query_operation_snapshot(snapshot, *, operation_ids=None, method=None, backend=None,
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
    task_states = {row['operation_id']:_task_state_from_snapshot(snapshot, row) for row in operations}
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
        return _query_operation_snapshot(snapshot, operation_ids=operation_ids, method=method,
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
        indexed = _query_operation_snapshot(snapshot)
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
    """Stage one publication's immutable objects without retaining their bytes."""
    def __init__(self, spool=None):
        self.spool=spool
        self.objects={}
    def put(self,value,role):
        return self.put_bytes(record_bytes(value),role)
    def put_bytes(self,raw,role):
        digest=sha256(raw).hexdigest()
        ref=_reference(digest, role, len(raw))
        staged=self.spool.put_bytes(raw) if self.spool is not None else bytes(raw)
        self.objects[(digest,role)]=(staged,role)
        return ref
    def publish(self,tx):
        for staged,role in self.objects.values():
            raw=self.spool.get_bytes(staged) if self.spool is not None else staged
            tx.put_object(raw,role=role)


class TaskWriter:
    """One attempt's staged completed prefix and latest exact CMA snapshot."""
    def __init__(self,root,task_id,attempt_id,*,binding,operation_id,diagnostics,checkpoint_document=None,
                 phase_scope=None,commit_every_generations=1,spool=None):
        self.root=Path(root); self.task_id=task_id; self.attempt_id=attempt_id
        self.operation_id=operation_id
        self.store=_writer_store(self.root,binding,phase_scope=phase_scope)
        self.diagnostics=diagnostics; self.phase_scope=phase_scope
        self.commit_every_generations=commit_every_generations
        self.spool=spool
        self.pending=[]; self.completed=[]; self.generation_rows=[]
        # The baseline is held in memory until it can join a complete
        # post-tell generation transaction. It is never a checkpoint by itself.
        self.baseline_pending=None
        self.latest_state=None; self.publication_uncertain=None; self.last_ack=self._empty_ack()
        links=checkpoint_document.get('_journal_links',{}) if checkpoint_document else {}
        self.baseline_block=links.get('baseline_evidence'); self.generation_block=links.get('generation_evidence')
        self.value_refs=dict(links.get('value_refs',{}))
        self.known_costs={}
        if checkpoint_document is not None:
            best=checkpoint_document['best']
            self.known_costs[best['evaluation_ordinal']]=best['cost_f64']
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
        return {'committed':False,'checkpoint':None,'latest_generation':None}

    def _ensure_publishable(self):
        if self.publication_uncertain is not None:
            raise _error('Task writer cannot replay an unacknowledged publication.',
                         task_id=self.task_id,publication=self.publication_uncertain)

    def _stage(self, value):
        if self.spool is None:
            return record_document(record_bytes(value))
        return self.spool.put_record(value)

    def _load(self, value):
        if self.spool is not None and hasattr(value, "owner"):
            return self.spool.get_record(value)
        return value

    def append_event(self,*,kind,payload,force=False):
        self._ensure_publishable(); self.last_ack=self._empty_ack()
        if kind in {"timing", "operation_span"}:
            raise _error("Timing diagnostics cannot be appended to numerical task history.", kind=kind)
        value=self._stage(payload)
        if kind=='evaluation' and isinstance(payload.get('generation'),int) and payload['generation']>0:
            self.generation_rows.append(value)
        self.pending.append((kind,value))
        event={'task_id':self.task_id,'sequence':self.sequence_hint,'kind':kind,
               'payload':payload if self.spool is not None else value}
        self.sequence_hint+=1
        if kind not in _DIAGNOSTIC or force:
            # A terminal/error archive may contain the unfinished generation's
            # diagnostics; it does not publish resumable generation evidence.
            self._publish([],extra_events=list(self.pending))
            self.pending.clear(); self.generation_rows.clear()
        return event

    def commit_barrier(self,kind,payload):
        self._ensure_publishable(); self.last_ack=self._empty_ack()
        value=dict(payload)
        state=value.pop('resume_state',None)
        segment={'kind':kind,'payload':self._stage(value),'events':list(self.pending),
                 'rows':list(self.generation_rows)}
        self.pending.clear(); self.generation_rows.clear()
        event={'task_id':self.task_id,'sequence':self.sequence_hint,'kind':kind,
               'payload':value if self.spool is None else payload}
        self.sequence_hint+=1
        if kind=='baseline_ready':
            if self.baseline_block is not None or self.baseline_pending is not None:
                raise _error('Task baseline was already staged for this attempt.',
                             task_id=self.task_id, attempt_id=self.attempt_id)
            self.baseline_pending=segment
            self.last_ack={'committed':False,'state':'baseline_sealed','checkpoint':None,
                           'latest_generation':None}
            return event,self.last_ack

        self.latest_state=state
        self.completed.append(segment)
        if len(self.completed)>=self.commit_every_generations:
            segments=([self.baseline_pending] if self.baseline_pending is not None else [])+self.completed
            self._publish(segments)
            self.completed.clear(); self.latest_state=None
            self.baseline_pending=None
        return event,self.last_ack

    def will_commit_barrier(self, kind):
        """Whether this barrier completes a durable group, without staging it."""
        if kind == 'baseline_ready':
            return False
        return len(self.completed) + 1 >= self.commit_every_generations

    def flush_completed(self,reason):
        self._ensure_publishable(); self.last_ack=self._empty_ack()
        if self.completed:
            segments=([self.baseline_pending] if self.baseline_pending is not None else [])+self.completed
            self._publish(segments)
            self.completed.clear(); self.latest_state=None
            self.baseline_pending=None
        return self.last_ack

    def flush(self):
        self.flush_completed('flush')
        if self.pending:
            self._publish([],extra_events=list(self.pending))
            self.pending.clear(); self.generation_rows.clear()

    def _value(self,objects,value,refs):
        occurrence={key:value[key] for key in _OCCURRENCE if key in value}
        body={key:item for key,item in value.items() if key not in _OCCURRENCE}
        key=value.get('candidate_key')
        if isinstance(key,str) and key in refs:
            reference=refs[key]
            if key in self.value_refs:
                with _snapshot(self.store):
                    read_value(self.root, reference)
        else:
            reference=objects.put(body,'benchmark_value')
            if isinstance(key,str): refs[key]=reference
        return {'reference':reference,'occurrence':occurrence}

    def _publish(self,segments,*,extra_events=(),terminal=None):
        self._ensure_publishable()
        objects=_Objects(self.spool); refs=dict(self.value_refs); changes=[]
        costs=dict(self.known_costs)
        baseline=self.baseline_block; generation=self.generation_block; checkpoint=None
        staged_generation_indexes = {}
        generation_index_updates = []
        completion_generation_rows = []
        completion_candidate_rows = []
        all_generation_rows = None
        all_candidate_rows = None
        # A completed prefix can be flushed while a new incomplete population
        # is buffered. Its diagnostic sequence hints must not shift publication.
        with _snapshot(self.store) as snapshot:
            pointer=snapshot.read_pointer('event_sequence/'+self.task_id)
            sequence=0 if pointer is None else _decode(snapshot,pointer['reference'])['next_sequence']
            revision=snapshot.stream_revision(_stream(self.task_id))
        latest=None

        def event(kind,payload,**additional):
            nonlocal sequence
            value=dict(payload)
            compact=value
            if kind=='evaluation':
                compact={_MARKER:{'kind':'value',**self._value(objects,value,refs)}}
                if 'attempt_id' in value: compact['attempt_id']=value['attempt_id']
            item={'task_id':self.task_id,'sequence':sequence,'kind':kind,'payload':compact}
            changes.append({'kind':'event','event':item,**additional}); sequence+=1

        for index,segment in enumerate(segments):
            for kind,payload in segment['events']: event(kind,self._load(payload))
            kind=segment['kind']; payload=self._load(segment['payload'])
            for row_value in segment['rows']:
                row=self._load(row_value)
                if 'cost_f64' in row: costs[row['evaluation_ordinal']]=row['cost_f64']
            if kind=='baseline_ready':
                marker=self._value(objects,payload['baseline'],refs)
                if 'cost_f64' in payload['baseline']:
                    costs[payload['baseline'].get('evaluation_ordinal',0)]=payload['baseline']['cost_f64']
                anchors=None
                zero_generation_optimization = (
                    terminal is not None
                    and terminal.get('type') == 'optimization'
                    and terminal.get('completed_generations') == 0
                )
                if isinstance(payload.get('anchors'),Mapping) and not zero_generation_optimization:
                    anchors=objects.put({'schema':'scnsim.benchmark_anchor_values','schema_version':3,
                        'task_id':self.task_id,'anchors':payload['anchors']},'benchmark_anchor_values')
                attributes={k:v for k,v in payload.items() if k not in {'attempt_id','baseline','anchors'}}
                block={'schema':'scnsim.benchmark_baseline_evidence','schema_version':3,'task_id':self.task_id,
                    'attempt_id':self.attempt_id,'baseline':marker['reference'],
                    'baseline_occurrence':marker['occurrence'],'anchors':anchors,'attributes':attributes}
                baseline=objects.put(block,'benchmark_baseline_evidence'); generation=None; evidence=baseline
                cpgen=int(payload['baseline'].get('generation',0))
                nextordinal=int(payload['baseline'].get('evaluation_ordinal',0))+1
                bestordinal=int(payload['baseline'].get('evaluation_ordinal',0))
            else:
                if baseline is None: raise _error('Generation evidence lacks its committed baseline.')
                row_values = [self._load(row) for row in segment['rows']]
                row_markers = [self._value(objects, row, refs) for row in row_values]
                rows=[{'value':marker['reference'],'occurrence':marker['occurrence']}
                      for marker in row_markers]
                block={'schema':'scnsim.benchmark_generation_evidence','schema_version':3,
                    'task_id':self.task_id,'attempt_id':self.attempt_id,'baseline':baseline,
                    'previous':generation,'rows':rows,
                    'attributes':{k:v for k,v in payload.items() if k!='attempt_id'}}
                generation=objects.put(block,'benchmark_generation_evidence'); evidence=generation
                cpgen=payload['generation']; nextordinal=payload['next_ordinal']; bestordinal=payload['best_ordinal']
                latest=dict(payload)
                first_ordinal = row_values[0]['evaluation_ordinal'] if row_values else nextordinal
                candidate_rows = []
                for offset, (row_value, marker) in enumerate(zip(row_values, row_markers)):
                    candidate_rows.append({
                        'ordinal': row_value['evaluation_ordinal'],
                        'generation': cpgen,
                        'row_offset': offset,
                        'block_ref': generation,
                        'value_ref': marker['reference'],
                        'summary_json': _candidate_summary(row_value, cpgen,
                                                           row_value['evaluation_ordinal']),
                    })
                candidate_index = {
                    'schema': 'scnsim.workspace_candidate_index', 'schema_version': 1,
                    'workspace_instance_id': self.store.workspace_instance_id,
                    'plan_sha256': self.store.plan_sha256,
                    'request_sha256': self.descriptor['request_sha256'],
                    'task_id': self.task_id, 'attempt_id': self.attempt_id,
                    'generation': cpgen, 'block_ref': generation,
                    'first_ordinal': first_ordinal, 'row_count': len(candidate_rows),
                    'summary_json': {
                        'generation': cpgen, 'first_ordinal': first_ordinal,
                        'row_count': len(candidate_rows), 'next_ordinal': nextordinal,
                        'best_ordinal': bestordinal,
                    },
                    'candidates': candidate_rows,
                }
                candidate_index_ref = objects.put(candidate_index, 'workspace_candidate_index')
                generation_row = {
                    'generation': cpgen, 'block_ref': generation, 'first_ordinal': first_ordinal,
                    'row_count': len(candidate_rows), 'summary_json': candidate_index['summary_json'],
                    'candidate_index_ref': candidate_index_ref,
                }
                staged_generation_indexes[generation['sha256']] = (generation_row, candidate_index,
                                                                     candidate_index_ref, block['previous'], block)
                generation_index_updates.append((
                    f'generation_index/{self.task_id}/{self.attempt_id}/{cpgen}', candidate_index_ref
                ))
                completion_generation_rows.append(generation_row)
                completion_candidate_rows.extend(candidate_rows)
            artifacts=[]
            if index==len(segments)-1 and isinstance(self.latest_state,Mapping):
                cp={'schema':'scnsim.benchmark_cma_checkpoint','schema_version':3,'task_id':self.task_id,
                    'request_sha256':self.descriptor['request_sha256'],'arm':self.descriptor['arm'],
                    'sample':self.descriptor['sample'],'environment_sha256':self.descriptor['environment']['environment_sha256'],
                    'attempt_id':self.attempt_id,'generation':cpgen,'next_ordinal':nextordinal,'best_ordinal':bestordinal,
                    'baseline_evidence':baseline,'generation_evidence':generation,'cma':self.latest_state['cma']}
                details={'generation':cpgen}
                with (self.phase_scope('checkpoint_state_publish',details) if self.phase_scope else nullcontext()):
                    cpref=objects.put(cp,'checkpoint'); details['checkpoint_bytes']=cpref['byte_length']
                    seal=checkpoint_seal(task_id=self.task_id,request_sha256=cp['request_sha256'],arm=cp['arm'],sample=cp['sample'],
                        environment_sha256=cp['environment_sha256'],attempt_id=self.attempt_id,
                        checkpoint_sha256=cpref['sha256'],byte_length=cpref['byte_length'])
                    sealref=objects.put(seal,'checkpoint_seal')
                    checkpoint={k:cp[k] for k in ('task_id','request_sha256','arm','sample','environment_sha256','attempt_id')}
                    checkpoint.update(checkpoint=cpref,seal=sealref,seal_sha256=seal['seal_sha256'])
                    artifacts=[cpref,sealref]
            event(kind,{'attempt_id':self.attempt_id,_MARKER:{'kind':'barrier','evidence':evidence,'checkpoint':checkpoint}},
                  evidence=evidence,checkpoint=checkpoint,artifacts=artifacts)
        for kind,payload in extra_events: event(kind,self._load(payload))
        result_ref=None; numerical_evidence=None; completion_ref=None; attempt_reference=None
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
            if terminal.get('type') == 'optimization':
                all_generation_rows, all_candidate_rows = _completion_indexes(
                    self, generation, staged_generation_indexes, completion_generation_rows,
                    completion_candidate_rows,
                )
                generation_count = len(all_generation_rows)
                if generation_count != terminal.get('completed_generations'):
                    raise _error('Sealed generation index differs from terminal generation count.',
                                 indexed=generation_count, terminal=terminal.get('completed_generations'))
                candidate_count = sum(row['row_count'] for row in all_generation_rows)
                request_ref = next((ref for ref in self.descriptor.get('artifacts', ())
                                    if ref.get('role') == 'operation_request'), None)
                if request_ref is None or request_ref.get('sha256') != self.descriptor.get('request_sha256'):
                    raise _error('Optimization task lacks its canonical request reference.', task_id=self.task_id)
                index_root = {
                    'schema': 'scnsim.workspace_completion_index_root', 'schema_version': 1,
                    'workspace_instance_id': self.store.workspace_instance_id,
                    'plan_sha256': self.store.plan_sha256,
                    'request_sha256': self.descriptor['request_sha256'],
                    'task_id': self.task_id, 'attempt_id': self.attempt_id,
                    'result_ref': result_ref,
                    'request_ref': request_ref,
                    'baseline_ref': baseline,
                    'generation_ancestry_ref': generation,
                    'generation_count': generation_count,
                    'candidate_count': candidate_count,
                    'best_ordinal': terminal['best_ordinal'],
                    'generations': all_generation_rows,
                }
                index_root_ref = objects.put(index_root, 'workspace_completion_index_root')
                sealed = {
                    'schema': 'scnsim.workspace_completion', 'schema_version': 1,
                    'workspace_instance_id': self.store.workspace_instance_id,
                    'plan_sha256': self.store.plan_sha256,
                    'request_sha256': self.descriptor['request_sha256'],
                    'task_id': self.task_id, 'attempt_id': self.attempt_id,
                    'result_ref': result_ref,
                    'terminal_attempt_ref': attempt_reference,
                    'baseline_ref': baseline,
                    'generation_ancestry_ref': generation,
                    'generation_count': generation_count,
                    'candidate_count': candidate_count,
                    'best_ordinal': terminal['best_ordinal'],
                    'index_root_ref': index_root_ref,
                }
                completion_ref = objects.put(sealed, 'workspace_completion')
                numerical_evidence={'baseline_evidence':baseline,'generation_evidence':generation,
                                    'terminal_event_sequence':sequence,
                                    'workspace_completion':completion_ref}
                completion['numerical_evidence']=numerical_evidence
                completion['workspace_completion'] = completion_ref
            event('completed',completion)
            changes.append({'kind':'attempt_update','attempt_id':self.attempt_id,'status':'success',
                  'failure':None,'interruption':None,'checkpoint':None,'artifacts':[result_ref]})
        checkpoint_selection=None if checkpoint is None else objects.put(checkpoint,'checkpoint_selection')
        checkpoint_selection_bytes=None if checkpoint is None else record_bytes(checkpoint)
        success_selection=None
        if result_ref is not None:
            selected={'event':'request_success_selected','request_sha256':self.descriptor['request_sha256'],
                      'task_id':self.task_id,'attempt_id':self.attempt_id,'result_ref':result_ref}
            if numerical_evidence is not None:
                selected['numerical_evidence']=numerical_evidence
                selected['workspace_completion'] = completion_ref
            success_selection=objects.put(selected,'success_selection')
        nextref=objects.put({'next_sequence':sequence},'event_sequence')
        txid=str(uuid4()); store=self.store
        committed_attempt = terminal_attempt if terminal is not None else _attempt_after_changes(
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
                for pointer_name, index_reference in generation_index_updates:
                    tx.set_pointer(pointer_name, index_reference)
                for change in changes:
                    raw=record_bytes(change)
                    _, change_reference = tx.append_with_ref(_stream(self.task_id),'task_change',raw)
                    if change.get('kind') == 'attempt_update':
                        tx.set_pointer(_attempt_state_pointer(self.task_id, change['attempt_id']), change_reference)
                    elif change.get('kind') == 'attempt_begin':
                        tx.set_pointer(_attempt_state_pointer(self.task_id, change['attempt']['attempt_id']), change_reference)
                tx.set_pointer('event_sequence/'+self.task_id,nextref)
                if checkpoint is not None:
                    tx.set_pointer('checkpoint/'+self.task_id+'/'+self.attempt_id,checkpoint_selection)
                    tx.append('checkpoints/'+self.task_id+'/'+self.attempt_id,'checkpoint_selection',checkpoint_selection_bytes)
                if result_ref is not None:
                    tx.select_first('success/'+self.descriptor['request_sha256'],success_selection)
                if completion_ref is not None:
                    tx.index_completion(completion_ref['sha256'], all_generation_rows, all_candidate_rows)
                    tx.set_pointer('completion/'+self.task_id+'/'+self.attempt_id, completion_ref)
        except BaseException as error:
            self.publication_uncertain=getattr(error,'operation_transaction_outcome',None) or (
                transaction.outcome if transaction is not None else {'status':'unknown','transaction_id':txid})
            raise
        self.revision=revision+len(changes); self.value_refs=refs; self.known_costs=costs
        self.attempt_document=committed_attempt
        self.baseline_block=baseline; self.generation_block=generation
        self.sequence_hint=sequence+max(0,len(self.pending)-len(extra_events))
        self.last_ack={'committed':True,'checkpoint':checkpoint,'evidence':generation or baseline,
                      'latest_generation':None if latest is None else latest['generation'],
                      'latest_generation_payload':latest,
                      'best_cost_f64':None if latest is None else costs[latest['best_ordinal']],
                      'transaction_id':txid}
        return result_ref,self.last_ack


def begin_task_writer(workspace,*,binding,operation_id,task_id,attempt_id,diagnostics,checkpoint_document=None,phase_scope=None,commit_every_generations=1,spool=None):
    return TaskWriter(_root(workspace),task_id,attempt_id,binding=binding,operation_id=operation_id,diagnostics=diagnostics,
        checkpoint_document=checkpoint_document,phase_scope=phase_scope,commit_every_generations=commit_every_generations,
        spool=spool)


def append_event(workspace,*,task_id,kind,payload,writer,force=False):
    return writer.append_event(kind=kind,payload=payload,force=force)


def commit_barrier(workspace,*,writer,kind,payload):
    return writer.commit_barrier(kind,payload)


def flush_completed(workspace,*,writer,reason):
    return writer.flush_completed(reason)


def complete_operation(binding,*,writer,terminal_bytes):
    if (operation_workspace(binding)!=writer.root or binding.plan_sha256!=writer.store.plan_sha256
            or binding.workspace_instance_id!=writer.store.workspace_instance_id):
        raise _error('Operation completion belongs to another bound Plan leaf.')
    if writer.completed:
        raise _error('Completed generation tail must be committed before terminal publication.')
    terminal=record_document(terminal_bytes)
    if record_bytes(terminal)!=terminal_bytes:
        raise _error('Operation terminal bytes are not canonical.')
    # A valid zero-generation optimization has a baseline and terminal result,
    # but no population checkpoint. Commit those together rather than creating
    # a baseline-only recovery boundary.
    segments=([writer.baseline_pending] if writer.baseline_pending is not None else [])
    result,ack=writer._publish(segments,extra_events=list(writer.pending),terminal=terminal)
    writer.pending.clear(); writer.generation_rows.clear()
    writer.baseline_pending=None
    return result,ack


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


def read_checkpoint(workspace,reference,*,binding,expected_task_id,expected_request_sha256,expected_arm,expected_sample,expected_environment_sha256,return_selection=False,spool=None):
    selection_reference = None
    if reference.get('role') == 'checkpoint_selection':
        if (reference.get('storage') != 'sqlite'
                or reference.get('schema_version') != _OPERATION_STORE_VERSION):
            raise UnsupportedEvidenceVersionError(
                'Checkpoint evidence uses an older storage format; use a new Workspace and recompute.',
                stage='checkpoint_read',
                evidence={'reference_storage':reference.get('storage'),
                          'action':'use a new Workspace and recompute'},
            )
        selection_reference = dict(reference)
        selection = None
    elif reference.get('checkpoint',{}).get('storage') == 'sqlite':
        selection = dict(reference)
    else:
        raise UnsupportedEvidenceVersionError(
            'Checkpoint evidence uses an older storage format; use a new Workspace and recompute.',
            stage='checkpoint_read',
            evidence={'reference_storage':reference.get('checkpoint',{}).get('storage'),
                      'action':'use a new Workspace and recompute'},
        )
    root=_root(workspace)
    expected={'task_id':expected_task_id,'request_sha256':expected_request_sha256,'arm':expected_arm,
              'sample':expected_sample,'environment_sha256':expected_environment_sha256}
    if selection is not None and any(selection.get(k)!=v for k,v in expected.items()):
        raise _error('Checkpoint binding differs from requested task.')
    with _snapshot(_writer_store(root,binding)) as snapshot:
        if selection_reference is not None:
            selection = _decode(snapshot, selection_reference)
            if not isinstance(selection, dict):
                raise _error('Checkpoint selection object is malformed.')
            if any(selection.get(k)!=v for k,v in expected.items()):
                raise _error('Checkpoint binding differs from requested task.')
        descriptor=_descriptor(snapshot,expected_task_id)
        if any(descriptor.get(k)!=v for k,v in expected.items() if k!='environment_sha256') or descriptor['environment']['environment_sha256']!=expected_environment_sha256:
            raise _error('Checkpoint task/environment identity changed.')
        selections=snapshot.read_stream('checkpoints/'+expected_task_id+'/'+selection['attempt_id'])
        if selection_reference is not None:
            selected = any(row['reference'] == selection_reference for row in selections['entries'])
        else:
            selected = any(_decode(snapshot,row['reference']) == selection for row in selections['entries'])
        if not selected:
            raise _error('Checkpoint is not the selected committed state for its attempt.')
        from .task_history import _verify_checkpoint_file, _hydrate_cma_checkpoint
        cp,_=_verify_checkpoint_file(root, selection, task_binding=descriptor)
        if any(cp.get(k)!=v for k,v in expected.items()) or cp.get('attempt_id')!=selection['attempt_id']:
            raise _error('Checkpoint content differs from sealed identity.')
        hydrated_document = _hydrate_cma_checkpoint(
            root, expected_task_id, cp, spool=spool
        )
        if spool is None:
            hydrated = record_bytes(hydrated_document)
        else:
            # The task history remains the durable complete ledger. The live
            # coordinator needs only anchor bodies, candidate cache identities,
            # the best scalar/ordinal and the exact committed CMA/RNG state.
            from ..execution.operation_spool import prepare_resume_checkpoint
            hydrated = prepare_resume_checkpoint(hydrated_document, spool)
        if return_selection:
            return hydrated, dict(selection)
        return hydrated
