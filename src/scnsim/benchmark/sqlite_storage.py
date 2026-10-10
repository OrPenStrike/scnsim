"""Current operation-domain writer and verified SQLite projections.

The core owns native transactions and byte objects; this adapter owns task,
occurrence, checkpoint and selected-result meanings. Completed generation groups
publish atomically. Only their latest CMA/RNG state is retained; an unfinished
population can be archived diagnostically but never enters resumable evidence.
Historical file records are read through storage.py and are never rewritten.
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
from ..workspace.operation_store import OperationStore, Snapshot, _reference
from .identity import checkpoint_seal
from .models import BenchmarkResult
from .prepared import record_bytes, record_document

_ACTIVE: ContextVar[tuple[Path, Snapshot] | None] = ContextVar('operation_sql_snapshot', default=None)
_MARKER = '$scnsim_benchmark_journal'
_OCCURRENCE = frozenset({'attempt_id','candidate_key','cache_hit','evaluation_ordinal','generation',
                        'population_column','latent_coordinates','source_index','origin','continuation_t_f64'})
_DIAGNOSTIC = frozenset({'population_observed','evaluation','progress'})


def _legacy():
    from . import storage
    return storage


def _error(message, **evidence):
    return _legacy()._integrity(message, **evidence)


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


def initialize_operation_record(binding, *, clock_binding):
    store = _bound(binding)
    store.initialize()
    declaration = {'schema':'scnsim.operation_trace','schema_version':1,
                   'plan_sha256':binding.plan_sha256,'workspace_instance_id':binding.workspace_instance_id}
    manifest = {'schema':'scnsim.benchmark_record','schema_version':4,
                'benchmark_sha256':sha256(record_bytes(declaration)).hexdigest(),
                'declaration':declaration,'plan_sha256':binding.plan_sha256,
                'workspace_instance_id':binding.workspace_instance_id,'clock':dict(clock_binding)}
    raw = record_bytes(manifest)
    with store.reader() as snapshot:
        present = snapshot.read_pointer('manifest')
    if present is None:
        with store.transaction(str(uuid4()), expected_revisions={}) as tx:
            tx.select_first('manifest', tx.put_object(raw, role='operation_manifest'))
    return store.root


def recover_operation_workspace(binding):
    store = _bound(binding)
    if not store.path.exists() and not store.path.is_symlink():
        return
    store.recover()


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


def start_operation(binding, row):
    return _publish_operation(binding, row)


def bind_operation(binding, *, row):
    return _publish_operation(binding, row)


def bind_operation_attempt(binding, *, row):
    return _publish_operation(binding, row)


def finish_operation(binding, row, *, failure=None):
    value = dict(row)
    if failure is not None:
        value['details'] = {**value.get('details',{}), 'failure':_legacy()._error_evidence(failure)}
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
    from .operations import project_indexed_operation_rows
    with _snapshot(store) as snapshot:
        indexed = _query_operation_snapshot(snapshot)
    current = project_indexed_operation_rows(indexed, workspace=root,
                plan_sha256=store.plan_sha256,workspace_instance_id=store.workspace_instance_id)
    return current


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
    descriptor = _legacy()._new_task_descriptor(task)
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
    events=[row['event'] for row in changes if row['kind']=='event']
    head={'task_id':descriptor['task_id'],'event_sequence':len(events)}
    task,_=_legacy()._task_document_from_chain(root,descriptor,head,
                       [{'operation':row} for row in changes],_manifest(snapshot))
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
    raw=(_legacy()._read_immutable(operation_workspace(binding),request,role='operation_request'))
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
    """Encode one publication delta before entering native SQL."""
    def __init__(self):
        self.objects={}
    def put(self,value,role):
        raw=record_bytes(value)
        digest=sha256(raw).hexdigest()
        ref=_reference(digest, role, len(raw))
        self.objects[(digest,role)]=(raw,role)
        return ref
    def publish(self,tx):
        for raw,role in self.objects.values():
            tx.put_object(raw,role=role)


class TaskWriter:
    """One attempt's staged completed prefix and latest exact CMA snapshot."""
    def __init__(self,root,task_id,attempt_id,*,binding,operation_id,diagnostics,checkpoint_document=None,
                 phase_scope=None,commit_every_generations=1):
        self.root=Path(root); self.task_id=task_id; self.attempt_id=attempt_id
        self.operation_id=operation_id
        self.store=_writer_store(self.root,binding,phase_scope=phase_scope)
        self.diagnostics=diagnostics; self.phase_scope=phase_scope
        self.commit_every_generations=commit_every_generations
        self.pending=[]; self.completed=[]; self.generation_rows=[]
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

    @staticmethod
    def _empty_ack():
        return {'committed':False,'checkpoint':None,'latest_generation':None}

    def _ensure_publishable(self):
        if self.publication_uncertain is not None:
            raise _error('Task writer cannot replay an unacknowledged publication.',
                         task_id=self.task_id,publication=self.publication_uncertain)

    def append_event(self,*,kind,payload,force=False):
        self._ensure_publishable(); self.last_ack=self._empty_ack()
        if kind in {"timing", "operation_span"}:
            raise _error("Timing diagnostics cannot be appended to numerical task history.", kind=kind)
        value=record_document(record_bytes(payload))
        if kind=='evaluation' and isinstance(value.get('generation'),int) and value['generation']>0:
            self.generation_rows.append(value)
        self.pending.append((kind,value))
        event={'task_id':self.task_id,'sequence':self.sequence_hint,'kind':kind,'payload':value}
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
        self.latest_state=state
        segment={'kind':kind,'payload':value,'events':list(self.pending),'rows':list(self.generation_rows)}
        self.pending.clear(); self.generation_rows.clear()
        self.completed.append(segment)
        event={'task_id':self.task_id,'sequence':self.sequence_hint,'kind':kind,'payload':value}
        self.sequence_hint+=1
        if kind=='baseline_ready' or len(self.completed)>=self.commit_every_generations:
            self._publish(self.completed)
            self.completed.clear(); self.latest_state=None
        return event,self.last_ack

    def flush_completed(self,reason):
        self._ensure_publishable(); self.last_ack=self._empty_ack()
        if self.completed:
            self._publish(self.completed)
            self.completed.clear(); self.latest_state=None
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
                    inherited=read_document(self.root,reference,schema='scnsim.benchmark_value',
                                            role='benchmark_value',bind={'task_id':self.task_id})
                if inherited.get('key_kind')!='candidate_key' or inherited.get('key')!=key:
                    raise _error('Inherited optimizer value does not bind its cache key.',task_id=self.task_id)
        else:
            key_kind='candidate_key' if isinstance(key,str) else 'numerical_source_id'
            value_key=key if isinstance(key,str) else value.get('numerical_source_id')
            if value_key is None:
                key_kind='event'
                value_key=str(value.get('evaluation_ordinal',value.get('source_index','baseline')))
            reference=objects.put({'schema':'scnsim.benchmark_value','schema_version':3,
                'task_id':self.task_id,'key_kind':key_kind,'key':value_key,'value':body},'benchmark_value')
            if isinstance(key,str): refs[key]=reference
        return {'reference':reference,'occurrence':occurrence}

    def _publish(self,segments,*,extra_events=(),terminal=None):
        self._ensure_publishable()
        objects=_Objects(); refs=dict(self.value_refs); changes=[]
        costs=dict(self.known_costs)
        baseline=self.baseline_block; generation=self.generation_block; checkpoint=None
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
            for kind,payload in segment['events']: event(kind,payload)
            kind=segment['kind']; payload=segment['payload']
            for row in segment['rows']:
                if 'cost_f64' in row: costs[row['evaluation_ordinal']]=row['cost_f64']
            if kind=='baseline_ready':
                marker=self._value(objects,payload['baseline'],refs)
                if 'cost_f64' in payload['baseline']:
                    costs[payload['baseline'].get('evaluation_ordinal',0)]=payload['baseline']['cost_f64']
                anchors=None
                if isinstance(payload.get('anchors'),Mapping):
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
                rows=[{'value':m['reference'],'occurrence':m['occurrence']} for m in
                      (self._value(objects,row,refs) for row in segment['rows'])]
                block={'schema':'scnsim.benchmark_generation_evidence','schema_version':3,
                    'task_id':self.task_id,'attempt_id':self.attempt_id,'baseline':baseline,
                    'previous':generation,'rows':rows,
                    'attributes':{k:v for k,v in payload.items() if k!='attempt_id'}}
                generation=objects.put(block,'benchmark_generation_evidence'); evidence=generation
                cpgen=payload['generation']; nextordinal=payload['next_ordinal']; bestordinal=payload['best_ordinal']
                latest=dict(payload)
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
        for kind,payload in extra_events: event(kind,payload)
        result_ref=None; numerical_evidence=None
        if terminal is not None:
            result_ref=objects.put(terminal,'operation_result')
            completion={'attempt_id':self.attempt_id,'result':result_ref}
            if terminal.get('type') == 'optimization':
                numerical_evidence={'baseline_evidence':baseline,'generation_evidence':generation,
                                    'terminal_event_sequence':sequence}
                completion['numerical_evidence']=numerical_evidence
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
            success_selection=objects.put(selected,'success_selection')
        change_bytes=[record_bytes(change) for change in changes]
        nextref=objects.put({'next_sequence':sequence},'event_sequence')
        txid=str(uuid4()); store=self.store
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
                for change,raw in zip(changes,change_bytes):
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
        except BaseException as error:
            self.publication_uncertain=getattr(error,'operation_transaction_outcome',None) or (
                transaction.outcome if transaction is not None else {'status':'unknown','transaction_id':txid})
            raise
        self.revision=revision+len(changes); self.value_refs=refs; self.known_costs=costs
        self.baseline_block=baseline; self.generation_block=generation
        self.sequence_hint=sequence+len(self.pending)
        self.last_ack={'committed':True,'checkpoint':checkpoint,'evidence':generation or baseline,
                      'latest_generation':None if latest is None else latest['generation'],
                      'latest_generation_payload':latest,
                      'best_cost_f64':None if latest is None else costs[latest['best_ordinal']],
                      'transaction_id':txid}
        return result_ref,self.last_ack


def begin_task_writer(workspace,*,binding,operation_id,task_id,attempt_id,diagnostics,checkpoint_document=None,phase_scope=None,commit_every_generations=1):
    return TaskWriter(_root(workspace),task_id,attempt_id,binding=binding,operation_id=operation_id,diagnostics=diagnostics,
        checkpoint_document=checkpoint_document,phase_scope=phase_scope,commit_every_generations=commit_every_generations)


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
    result,ack=writer._publish([],extra_events=list(writer.pending),terminal=terminal)
    writer.pending.clear(); writer.generation_rows.clear()
    return result,ack


def read_operation_success(binding,task_id,*,attempt_id=None):
    from .evidence_reader import read_success
    store = _bound(binding)
    if store.path.exists() or store.path.is_symlink():
        with _snapshot(store) as snapshot:
            descriptor = _pointer(snapshot, 'task/'+task_id)
            if descriptor is not None:
                selected = _pointer(snapshot, 'success/'+descriptor['request_sha256'])
                if selected is not None and attempt_id is not None and selected['attempt_id'] != attempt_id:
                    selected = None
                return read_success(store.root, snapshot, descriptor, attempt_id=attempt_id, selection=selected)
    raise _error('Operation task identity is not recorded.', task_id=task_id)


def find_operation_success(binding,request_sha256):
    store=_bound(binding)
    if store.path.exists() or store.path.is_symlink():
        with _snapshot(store) as snapshot:
            selected=_pointer(snapshot,'success/'+request_sha256)
            if selected is not None:
                descriptor=_descriptor(snapshot,selected['task_id'])
                from .evidence_reader import read_success
                success=read_success(store.root,snapshot,descriptor,
                                     attempt_id=selected['attempt_id'],selection=selected)
                if success is None or success['result_ref']!=selected['result_ref'] or descriptor['request_sha256']!=request_sha256:
                    raise _error('Selected operation success differs from committed task evidence.')
                return success
    legacy_path=operation_workspace(binding)/'benchmark.json'
    if legacy_path.exists():
        raise _unsupported_old_record(legacy_path)
    return None


def read_checkpoint(workspace,reference,*,binding,expected_task_id,expected_request_sha256,expected_arm,expected_sample,expected_environment_sha256):
    if reference.get('checkpoint',{}).get('storage')!='sqlite':
        raise UnsupportedEvidenceVersionError(
            'Checkpoint evidence uses an older storage format; use a new Workspace and recompute.',
            stage='checkpoint_read',
            evidence={'reference_storage':reference.get('checkpoint',{}).get('storage'),
                      'action':'use a new Workspace and recompute'},
        )
    root=_root(workspace)
    expected={'task_id':expected_task_id,'request_sha256':expected_request_sha256,'arm':expected_arm,
              'sample':expected_sample,'environment_sha256':expected_environment_sha256}
    if any(reference.get(k)!=v for k,v in expected.items()):
        raise _error('Checkpoint binding differs from requested task.')
    with _snapshot(_writer_store(root,binding)) as snapshot:
        descriptor=_descriptor(snapshot,expected_task_id)
        if any(descriptor.get(k)!=v for k,v in expected.items() if k!='environment_sha256') or descriptor['environment']['environment_sha256']!=expected_environment_sha256:
            raise _error('Checkpoint task/environment identity changed.')
        selections=snapshot.read_stream('checkpoints/'+expected_task_id+'/'+reference['attempt_id'])
        if not any(_decode(snapshot,row['reference'])==dict(reference) for row in selections['entries']):
            raise _error('Checkpoint is not the selected committed state for its attempt.')
        cp,_=_legacy()._verify_checkpoint_file(root,reference,task_binding=descriptor)
        if any(cp.get(k)!=v for k,v in expected.items()) or cp.get('attempt_id')!=reference['attempt_id']:
            raise _error('Checkpoint content differs from sealed identity.')
        return record_bytes(_legacy()._hydrate_cma_checkpoint(root,expected_task_id,cp))
