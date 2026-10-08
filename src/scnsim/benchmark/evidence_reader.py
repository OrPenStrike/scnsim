"""Verified detached numerical projection, separate from diagnostic task replay.

SQLite selection and the outer Plan/read snapshot remain the authority. A
completed Optimization follows immutable baseline/generation links once; control
rows establish its exact attempt and resume boundaries. Checkpoint-off uses the
same generation evidence, without needing CMA state. No solver or writer is
called here. A completion must name its numerical evidence explicitly; missing
locators are unsupported, never reconstructed from historical barriers.
"""
from collections.abc import Mapping

from .prepared import record_bytes, record_document


def read_success(root, snapshot, descriptor, *, attempt_id=None, selection=None):
    from . import storage
    from .sqlite_storage import _manifest, _materialize_task, _stream

    task_id = descriptor['task_id']
    request_ref = next((r for r in descriptor['artifacts'] if r.get('role') == 'operation_request'), None)
    if request_ref is None or request_ref['sha256'] != descriptor['request_sha256']:
        raise storage._integrity('Operation task lacks its canonical request.', task_id=task_id)
    request_raw = storage._read_immutable(root, request_ref, role='operation_request')
    request = record_document(request_raw)
    if record_bytes(request) != request_raw:
        raise storage._integrity('Operation request bytes are not canonical.', task_id=task_id)
    if request['operation'] != 'optimize_direct':
        return storage._operation_success_from_task(
            _materialize_task(root, snapshot, descriptor), attempt_id=attempt_id)
    def expand(root, task_id, binding, kind, compact):
        marker = compact.get(storage._JOURNAL_MARKER)
        if not isinstance(marker, Mapping) or marker.get('kind') != 'barrier':
            return storage._expand_payload(root, task_id, binding, kind, compact)
        # Control replay retains checkpoint/artifact links from the change itself.
        # Numerical tips come exclusively from the committed completion locator.
        return {key: value for key, value in compact.items()
                if key != storage._JOURNAL_MARKER}, None

    with snapshot.reuse_verified_objects():
        changes = []
        for sequence, reference, raw in snapshot.iter_task_changes(_stream(task_id), include_evaluations=False):
            if reference['role'] != 'task_change':
                raise storage._integrity('Task change has the wrong object role.', reference=reference)
            change = record_document(raw)
            if record_bytes(change) != raw:
                raise storage._integrity('Task change bytes are not canonical.', reference=reference)
            changes.append({'operation':change})
        task, _ = storage._task_document_from_chain(root, descriptor, {}, changes, _manifest(snapshot),
                payload_expander=expand, project_observations=False, sparse_events=True)
        completed = [event for event in task['events'] if event['kind'] == 'completed'
                     and (attempt_id is None or event['payload'].get('attempt_id') == attempt_id)]
        for event in completed:
            payload = event['payload']; selected_id = payload['attempt_id']
            attempt = next((row for row in task['attempts'] if row['attempt_id'] == selected_id), None)
            if attempt is None or attempt['status'] != 'success':
                raise storage._integrity('Selected completion lacks its successful committed attempt.', attempt_id=selected_id)
            terminal_ref = payload['result']
            if selection is not None and (selection['attempt_id'] != selected_id or selection['result_ref'] != terminal_ref):
                raise storage._integrity('Selected success differs from its completion.', task_id=task_id)
            terminal_raw = storage._read_immutable(root, terminal_ref, role='operation_result')
            terminal = record_document(terminal_raw)
            if record_bytes(terminal) != terminal_raw:
                raise storage._integrity('Operation terminal bytes are not canonical.', task_id=task_id)
            locator = payload.get('numerical_evidence')
            if not isinstance(locator, Mapping) or set(locator) != {
                    'baseline_evidence', 'generation_evidence', 'terminal_event_sequence'}:
                raise storage._integrity('Optimization completion requires an explicit numerical evidence locator.',
                                         task_id=task_id)
            if locator['terminal_event_sequence'] != event['sequence'] or (
                    selection is not None and selection.get('numerical_evidence') != locator):
                raise storage._integrity('Selected numerical locator differs from its completion.', task_id=task_id)
            values = _optimization_values(root, snapshot, descriptor, task, event, terminal, locator)
            observations = {'result_kind':'optimization', 'terminal_summary':terminal,
                            'records':[{'role':'baseline' if row['evaluation_ordinal']==0 else 'candidate',
                                        'value':row} for row in values]}
            payload['numerical_observations'] = observations
        if not completed:
            raise storage._integrity('Selected successful operation has no committed completion.', task_id=task_id)
        return storage._operation_success_from_task(task, attempt_id=attempt_id)


def _optimization_values(root, snapshot, descriptor, task, terminal_event, terminal, locator):
    from . import storage
    from .sqlite_storage import _decode
    task_id = descriptor['task_id']
    attempts = {row['attempt_id']:row for row in task['attempts']}
    attempt_id = terminal_event['payload']['attempt_id']
    boundaries = {}
    resume_boundaries = {}
    limits = {attempt_id: terminal['completed_generations']}
    def checkpoint_state(reference):
        checkpoint, _ = storage._verify_checkpoint_file(root, reference, task_binding=descriptor)
        expected = {key: descriptor[key] for key in ('task_id', 'request_sha256', 'arm', 'sample')}
        expected['environment_sha256'] = descriptor['environment']['environment_sha256']
        expected.update(schema='scnsim.benchmark_cma_checkpoint', schema_version=3,
                        attempt_id=reference['attempt_id'])
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise storage._integrity('Checkpoint content differs from its task binding.', task_id=task_id)
        selections = snapshot.read_stream('checkpoints/'+task_id+'/'+reference['attempt_id'])
        if not any(_decode(snapshot, row['reference']) == reference for row in selections['entries']):
            raise storage._integrity('Checkpoint is not committed for its attempt.', task_id=task_id)
        return checkpoint

    current = attempt_id
    while attempts[current].get('resume_from') is not None:
        reference = attempts[current]['resume_from']
        checkpoint = checkpoint_state(reference)
        parent = reference['attempt_id']
        resume_boundaries[current] = checkpoint
        boundaries[parent] = checkpoint
        if parent in limits or parent not in attempts:
            raise storage._integrity('Optimization resume ancestry is malformed.', task_id=task_id)
        limits[parent] = checkpoint['generation']
        current = parent
    selected_checkpoint = attempts[attempt_id].get('checkpoint')
    if selected_checkpoint is not None:
        checkpoint = checkpoint_state(selected_checkpoint)
        if checkpoint['generation'] != terminal['completed_generations']:
            raise storage._integrity('Successful checkpoint generation differs from terminal.', task_id=task_id)
        boundaries[attempt_id] = checkpoint
    baseline_ref = locator['baseline_evidence']
    baseline_block = storage._read_evidence_block(root, task_id, baseline_ref,
            schema='scnsim.benchmark_baseline_evidence', role='benchmark_baseline_evidence')
    if baseline_block['attempt_id'] not in limits:
        raise storage._integrity('Optimization baseline differs from selected ancestry.', task_id=task_id)
    value_bodies = {}

    def materialize(reference, occurrence):
        key = tuple(sorted(reference.items()))
        if key not in value_bodies:
            value_bodies[key] = storage._read_value(root, task_id, reference)
        value = dict(value_bodies[key])
        value.update(occurrence)
        return value

    values = [materialize(baseline_block['baseline'], baseline_block.get('baseline_occurrence', {}))]
    if values[0].get('evaluation_ordinal') != 0 or values[0].get('generation') != 0:
        raise storage._integrity('Optimization baseline ordinal is malformed.', task_id=task_id)
    for checkpoint in boundaries.values():
        if checkpoint['baseline_evidence'] != baseline_ref or (
                checkpoint['generation'] == 0 and checkpoint['generation_evidence'] is not None):
            raise storage._integrity('Resume checkpoint baseline differs from selected evidence.', task_id=task_id)
    reversed_blocks = []; seen = set()
    chain_references = {}
    reference = locator['generation_evidence']
    while reference is not None:
        digest = reference['sha256']
        if digest in seen:
            raise storage._integrity('Optimization generation evidence contains a cycle.', task_id=task_id)
        seen.add(digest)
        block = storage._read_evidence_block(root, task_id, reference,
                schema='scnsim.benchmark_generation_evidence', role='benchmark_generation_evidence')
        if block['baseline'] != baseline_ref or block['attempt_id'] not in limits:
            raise storage._integrity('Optimization generation evidence differs from its selected ancestry.', task_id=task_id)
        chain_references[block['attributes']['generation']] = reference
        reversed_blocks.append(block)
        reference = block['previous']
    # A checkpoint may point at an ancestor's tip when its attempt completed
    # no new generation. Match the exact tip, not just the tip's attempt ID.
    boundary_occurrences = {0: (None, 1)}
    for generation, block in enumerate(reversed(reversed_blocks), 1):
        if block['attributes']['generation'] != generation or generation > limits[block['attempt_id']]:
            raise storage._integrity('Optimization generation evidence order differs from its committed ancestry.', task_id=task_id)
        resume = resume_boundaries.get(block['attempt_id'])
        if resume is not None and generation <= resume['generation']:
            raise storage._integrity('Child generation precedes its selected resume boundary.', task_id=task_id)
        # Fetch one generation's unique byte authority through a JOIN; decoding
        # is reused by content reference, occurrence overlays remain detached.
        for _ in snapshot.get_objects([row['value'] for row in block['rows']]):
            pass
        for row in block['rows']:
            value = materialize(row['value'], row['occurrence'])
            if value['generation'] != generation or value['evaluation_ordinal'] != len(values):
                raise storage._integrity('Optimization numerical ledger order is not contiguous.', task_id=task_id)
            values.append(value)
        if block['attributes']['next_ordinal'] != len(values):
            raise storage._integrity('Generation boundary ordinal differs from numerical evidence.', task_id=task_id)
        boundary_occurrences[generation] = (chain_references[generation], len(values))
    for checkpoint in boundaries.values():
        if boundary_occurrences.get(checkpoint['generation']) != (
                checkpoint['generation_evidence'], checkpoint['next_ordinal']):
            raise storage._integrity('Selected checkpoint boundary is absent from numerical evidence.', task_id=task_id)
    if len(reversed_blocks) != terminal['completed_generations'] or terminal['best_ordinal'] not in range(len(values)):
        raise storage._integrity('Optimization terminal differs from its complete numerical ledger.', task_id=task_id)
    return values
