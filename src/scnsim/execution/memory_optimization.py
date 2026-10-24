"""Ephemeral record/publication ownership for the shared optimization coordinator.

The same physical Objective/root/stage/controller functions perform all work.
This owner retains synchronized host values and numerical evidence once in RAM;
keys are operation-local ordinals, never saved object identities. No Workspace,
SQLite, spool, checkpoint, transaction, task event or persistent codec is used.
Run owns active execution and the caller owns the detached terminal payload.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from dataclasses import fields, is_dataclass, replace
from typing import TypedDict

import numpy as np

from ..errors import RuntimePreparationError


class MemoryTrial(TypedDict):
    """Detached final trial fields consumed by the in-memory Results owner."""
    stage_id: str
    ordinal: int
    evaluation_ordinal: int
    kind: str
    point: Mapping
    parameters: Mapping
    status: str
    cost: float
    objective_components: tuple
    failure: Mapping | None
    gradient: np.ndarray | None
    model_identity: Mapping
    optimizer_unavailability: Mapping | None
    certificates: tuple
    bodies: Mapping
    observable_metadata: tuple


def _detach(value, memo=None):
    """Freeze host metadata once, preserving shared candidate/body references."""
    memo = {} if memo is None else memo
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, Mapping):
        detached = MappingProxyType({key: _detach(item, memo) for key, item in value.items()})
    elif isinstance(value, (tuple, list)):
        detached = tuple(_detach(item, memo) for item in value)
    elif is_dataclass(value) and not isinstance(value,type):
        detached = replace(value, **{field.name:_detach(getattr(value,field.name),memo) for field in fields(value)})
    elif isinstance(value, np.ndarray) or (hasattr(value, '__array__') and
                                         hasattr(value, 'dtype') and hasattr(value, 'shape')):
        # Actual checked certificates may still be device arrays. Ownership
        # crosses to detached host bytes here, never to a retained device frame.
        array = np.asarray(value)
        detached = np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)
    else:
        # Numerical EvaluationResult/Failure are already immutable host models;
        # no ProgramOutcome, partial function, backend or native factor is stored.
        detached = value
    memo[identity] = detached
    return detached


class MemoryCandidateRecords:
    """Typed same-frame projections, independent of durable candidate envelopes."""

    def success(self, *, objective_program, outcome, parameters, objective_records, affine_evidence=()):
        from ..numerics.evidence import numerical_error
        from .physical_analysis import point_contexts, project_primal_bodies
        from .quantities import expression_leaves
        from .quantity_policy import diagonal_passive_policy, project_quantity

        program = objective_program.quantity_program
        metadata = program.observable_metadata()
        inputs = program.inputs(parameters)
        contexts = point_contexts(program, inputs)
        bodies = project_primal_bodies(program, outcome.primal, inputs, ephemeral=True)
        for row in metadata:
            if row['kind'] == 'diagonal_root':
                result = diagonal_passive_policy(bodies[row['observable_index']],
                    node_count=len(program.program.views[
                        program.program.observables[row['observable_index']].view_index].node_ids),
                    precision=program.program.precision)
                if result.failure is not None:
                    raise numerical_error(result.failure)
                bodies[row['observable_index']] = result
        objective_units = {row['key']:row['si_unit'] for row in program.quantity_metadata()}
        objectives = []
        for ordinal, declaration in enumerate(objective_records):
            rows = [row for row in metadata if row['quantity_key'] == declaration['id'] and not row['hidden']]
            terms = []
            for term_ordinal, (selector, row) in enumerate(zip(expression_leaves(declaration['quantity']), rows, strict=True)):
                index = row['observable_index']
                body = bodies[index]
                actual = (body.root_omega_rad_s if body.root_omega_rad_s is not None else
                          body.response_value if body.response_value is not None else body.coupling_rad_s)
                terms.append({'term_ordinal': term_ordinal, 'selector': selector, 'status': 'success',
                    'value': float(project_quantity(selector, body)), 'actual_complex': actual,
                    'observable_index': index, 'unit':row['si_unit'], 'raw_unit':row['raw_si_unit'], 'context': contexts[index]})
            objectives.append({'id': declaration['id'], 'status': 'success',
                'value': float(outcome.values[ordinal]),
                'normalized_residual': float(outcome.normalized_residuals[ordinal]),
                'weighted_cost': float(outcome.contributions[ordinal]), 'unit':objective_units[declaration['id']],
                'residual_unit':'dimensionless', 'cost_unit':'dimensionless', 'terms': terms})
        if affine_evidence:
            for objective in objectives:
                objective['affine_evidence'] = affine_evidence
        return {**({'affine_evidence':affine_evidence} if affine_evidence else {}),
            'parameters': parameters, 'objectives': objectives, 'cost': float(outcome.total_cost),
            'failure': None, 'bodies': bodies, 'observable_metadata': metadata,
            'gradient': None if outcome.gradient is None else np.frombuffer(np.asarray(outcome.gradient).tobytes(),dtype=np.asarray(outcome.gradient).dtype).reshape(np.asarray(outcome.gradient).shape),
            'derivative_reasons': outcome.aux.observable_reasons,
            'certificates': _detach(outcome.aux.certificates)}

    def failure(self, *, parameters, objective_records, error, affine_evidence=()):
        from .quantities import expression_leaves
        failure = {'kind': error.kind, 'stage': error.stage, 'detail': str(error),
                   'evidence': error.evidence, 'phase':'quantity_evaluation'}
        original = getattr(error, 'failure', None)
        if original is not None:
            failure['numerical_failure'] = original
        # The shared aggregate failure has no objective/term occurrence unless
        # explicitly supplied. Retain declaration order without inventing which
        # leaf failed or claiming that later leaves were evaluated.
        objectives = []
        for declaration in objective_records:
            terms = [{'term_ordinal':index,'selector':selector,'status':'unevaluated'}
                     for index,selector in enumerate(expression_leaves(declaration['quantity']))]
            objectives.append({'id':declaration['id'],'status':'unevaluated','terms':terms})
        if affine_evidence:
            for objective in objectives:
                objective['affine_evidence'] = affine_evidence
        return {**({'affine_evidence':affine_evidence} if affine_evidence else {}),
                'parameters': parameters, 'cost': float('inf'), 'failure': failure,
                'gradient':None, 'objectives':objectives, 'certificates':(),
                'bodies': {}, 'observable_metadata': ()}

    def summary(self, candidate):
        return {'evaluation_ordinal': candidate['evaluation_ordinal'],
                'parameters': candidate['parameters'], 'cost': candidate['cost'],
                'status': candidate['status'], 'objectives': candidate['objectives'],
                **({'affine_evidence':candidate['affine_evidence']} if 'affine_evidence' in candidate else {}),
                'failure': candidate['failure'],
                'optimizer_unavailability': candidate.get('optimizer_unavailability')}

    def cost(self, candidate):
        return candidate['cost']

    def continuation(self, observed, *, parameters, t, start_index):
        return {'result': observed, 'parameters': parameters, 'continuation_t': t, 'start_index': start_index}


class MemoryOptimizationRecords:
    """RAM record keys and observed batches; neither constitutes a durable ACK."""

    def __init__(self, execution_id):
        self.execution_id = execution_id
        self._rows = []
        self._keys = {}
        self._batches = []

    def put_record(self, row):
        key = len(self._rows)
        self._rows.append(row)
        self._keys[id(row)] = key
        return key

    def get_record(self, key):
        return self._rows[key]

    def publish_batch(self, batch):
        if batch['checkpoint'] is not None:
            raise RuntimeError('in-memory batch cannot publish a checkpoint')
        keys = tuple(self._keys[id(row)] for row in batch['rows'])
        ordinal = len(self._batches)
        self._batches.append({'batch_ordinal': ordinal, 'stage_id': batch['stage_id'],
            'model_identity': batch['model_identity'], 'record_keys': keys,
            'stage_state': _detach(batch['stage_state']), 'transitions': _detach(batch['transitions'])})
        return MappingProxyType({'execution_id': self.execution_id, 'batch_ordinal': ordinal,
                                 'observed': True, 'durable': False})

    def finish(self, terminal):
        from ..compilation.compiler import parameter_key
        trials = []
        for row in self._rows:
            candidate = row['candidate']
            candidate.update(point=candidate['parameters'], kind=row['kind'],
                ordinal=candidate['evaluation_ordinal'], objective_components=candidate['objectives'],
                optimizer_unavailability=candidate.get('optimizer_unavailability'))
            trials.append(candidate)
        stages = []
        for state in terminal['stages']:
            best, accepted = state['observed_best'], state['accepted_iterate']
            stages.append(dict(state, variable_keys=tuple(parameter_key(ref) for ref in state['variable_refs']),
                eval_count=state['evaluation_count'], iter_count=state['iteration_count'],
                best_ordinal=None if best is None else best['evaluation_ordinal'],
                accepted_ordinal=None if accepted is None else accepted['evaluation_ordinal'],
                trials=tuple(trial for trial in trials if trial['stage_id'] == state['stage_id'])))
        comparison = dict(terminal['result']['comparison'])
        def owner_ref(locator):
            if locator is None:
                return None
            if locator['kind'] == 'baseline':
                baseline = terminal['baseline']
                return {'stage_id':baseline['stage_id'],'ordinal':baseline['evaluation_ordinal']}
            return {'stage_id':locator['stage_id'],'ordinal':locator['evaluation_ordinal']}
        comparison['baseline_ref'] = owner_ref(comparison['baseline'])
        comparison['winner_ref'] = owner_ref(comparison['winner'])
        payload = _detach({'execution_id': self.execution_id, 'persistence': 'memory',
            'baseline': terminal['baseline'], 'manifest': terminal['manifest'],
            'stages': stages, 'trials': trials, 'transitions': terminal['transitions'],
            'comparison': comparison, 'outcome': terminal['outcome'], 'summary': terminal['summary'],
            'batches': self._batches})
        self.clear()
        return payload

    def clear(self):
        self._rows.clear()
        self._keys.clear()
        self._batches.clear()


def execute_memory_optimization(*, plan_document, prepared_analysis, trace,
                                program_owner, execution_policy, commit_every_generations,
                                on_progress=None, progress_observer=None):
    """Use the ordinary optimization coordinator with RAM-only record owners.

    Runtime owns the closed/active Run boundary and Benchmark completion. Callbacks
    observe actual synchronized batches; their failures remain fatal. Resources
    drain before operation-local compilation/record ownership is released.
    """
    request = prepared_analysis.request()
    if request.get('operation') != 'optimize_direct' or request.get('runtime_semantic', {}).get('backend') != 'jax':
        raise RuntimePreparationError('In-memory optimization requires the prepared JAX optimization request',
                                      stage='memory_optimization_prepare')
    from .config import get_runtime_configuration, runtime_resource_identity
    if runtime_resource_identity() != request['runtime_semantic']['resources']:
        raise RuntimePreparationError('Prepared JAX resources changed before in-memory execution', stage='runtime_prepare')
    from .compile_owner import CompileOwner
    from .jax_backend import get_jax_backend
    from .physical_optimization import run_physical_optimization
    from .resources import operation_resources

    records = MemoryOptimizationRecords(trace.execution_id)
    frame_owner = CompileOwner(policy=execution_policy)
    from .jax_operation import _physical_progress
    latest = {}
    last_notified = {}
    display_enabled = progress_observer is not None
    def observe_progress(**values):
        nonlocal display_enabled
        latest.update(values)
        if not display_enabled:
            return
        try:
            progress_observer.update(phase=values['phase'], stage_id=values['stage_id'],
                strategy='cmaes' if values['strategy'] == 'cma_es' else values['strategy'],
                completed_generations=values.get('completed_generations'),
                total_generations=values.get('total_generations'),
                completed_iterations=values.get('completed_iterations'),
                total_iterations=values.get('total_iterations'), evaluated_count=values.get('evaluated_count',0),
                best_cost=values.get('best_cost'), committed_generations=values.get('committed_generations'),
                committed_iterations=values.get('committed_iterations'), reused=values.get('reused',False))
        except Exception as error:
            display_enabled = False
            import warnings
            warnings.warn('Optimization progress display disabled after '+type(error).__name__, RuntimeWarning, stacklevel=2)
    def on_observed(state,ack):
        stage_id, count = state['stage_id'], state['iteration_count']
        if count <= 0 or count <= last_notified.get(stage_id,0):
            return
        last_notified[stage_id] = count
        values = dict(latest, stage_id=stage_id)
        cma = state['strategy'] == 'cma_es'
        values['completed_generations' if cma else 'completed_iterations'] = count
        values['committed_generations' if cma else 'committed_iterations'] = 0
        _physical_progress(on_progress,request,values,phase='generation' if cma else 'iteration',trace=trace,persistence='memory')
    optimizer = request['spec']['optimizer']
    first_stage = optimizer['stages'][0] if optimizer['type'] == 'hybrid' else {'id':'main','optimizer':optimizer}
    controls = first_stage['optimizer']
    cma = controls['type'] == 'cma_es'
    backend = None
    primary = None
    try:
        observe_progress(phase='preparing',stage_id=first_stage['id'],strategy=controls['type'],
            completed_generations=0 if cma else None,total_generations=controls.get('complete_generations'),
            completed_iterations=None if cma else 0,total_iterations=controls.get('max_iterations'),
            committed_generations=0 if cma else None,committed_iterations=None if cma else 0,
            evaluated_count=0,best_cost=None,reused=False)
        _physical_progress(on_progress,request,latest,phase='initial',trace=trace,persistence='memory')
        configuration = get_runtime_configuration()
        with operation_resources(cpu_threads=configuration.cpu_threads) as resources:
            try:
                with trace.span('backend_prepare'):
                    backend = get_jax_backend(precision=request['runtime_semantic']['precision'],
                        resources=configuration, trace=trace, operation_resources=resources, diagnostics=True)
                terminal = run_physical_optimization(plan_document=plan_document, request=request,
                    effective_point=prepared_analysis.effective_point(), precision=backend.precision,
                    parameter_authorizations=prepared_analysis.parameter_authorizations(),
                    backend=backend, operation_resource=resources, trace=trace, program_owner=program_owner,
                    checkpoint_policy='off', resume_state=None, commit_every_generations=commit_every_generations,
                    publish_batch=records.publish_batch,
                    observe_progress=observe_progress, on_acknowledged=on_observed,
                    spool=records, candidate_records=MemoryCandidateRecords(), publication_mode='memory',
                    execution_policy=execution_policy, frame_owner=frame_owner)
            except BaseException as error:
                primary = error
                raise
            finally:
                if backend is not None:
                    try:
                        backend.close()
                    except BaseException as error:
                        if primary is None:
                            raise
                        primary.add_note('In-memory optimization backend cleanup also failed: '+repr(error))
        final_phase = 'complete' if terminal['outcome'] == 'success' else 'partial'
        _physical_progress(on_progress,request,latest,phase=final_phase,trace=trace,persistence='memory')
        observe_progress(**dict(latest,phase=final_phase))
        return records.finish(terminal)
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup_failure = None
        for cleanup in (frame_owner.clear, records.clear):
            try:
                cleanup()
            except BaseException as error:
                if primary is not None:
                    primary.add_note('In-memory optimization ownership cleanup also failed: '+repr(error))
                elif cleanup_failure is None:
                    cleanup_failure = error
                else:
                    cleanup_failure.add_note('Additional in-memory ownership cleanup failed: '+repr(error))
        if primary is None and cleanup_failure is not None:
            raise cleanup_failure
