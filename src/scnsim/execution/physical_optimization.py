"""Fixed-model optimization stages on the shared checked physical program.

The operation coordinator owns leases, durable publication and terminal decoding.
This module owns controller/stage state and synchronized numerical work only.
The record/publication owners choose durable spool or ephemeral RAM references.
The numerical and controller coordinator is shared; checkpoint state contains
detached locators, never frames, factors or optimizer curvature for restart.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from ..compilation.compiler import parameter_key, parameter_values
from ..compilation.mesh import quantity
from ..compilation.objective_program import compile_objective_program
from ..errors import SCNSimError, NumericalResolutionUnresolved
from ..numeric_encoding import bits, complex_record, complex_value, unbits
from ..numerics.evidence import numerical_error
from .cma_optimizer import CMAController, physical_seed_coordinates
from .local_coordinates import LocalCoordinateMap
from .local_optimizer import LocalEvaluation, LocalUnavailable, run_local_optimizer
from .physical_analysis import _point_at, point_identity, project_primal_bodies
from .physical_candidate import encode_physical_candidate, encode_physical_candidate_failure
from .quantities import EvaluationFailure, continue_root, quantity_record

_CANDIDATE_FAILURES = frozenset(('invalid_candidate_physical_parameter',
    'eliminated_block_solve_failure', 'root_slope_unresolved',
    'numerical_resolution_unresolved'))


def _candidate_error(error):
    if error.kind == 'direct_response_formation':
        return NumericalResolutionUnresolved(
            'candidate selected response is numerically unresolved',
            stage=error.stage, evidence=error.evidence)
    return error


def _native_root_branch(program, inputs, start_index, *, backend, owner, stage_id, compact_evidence=False):
    """Evaluate one declared branch through its exact native provider."""
    from contextlib import nullcontext
    from ..numerics.evidence import require_native_candidate

    jax = backend.jax
    jnp = jax.numpy
    native = program.quantity_program.native_program(
        inputs, program_owner=owner, admit_core=False,
    )
    with owner.hold_exact_core(native.core) if owner is not None else nullcontext():
        with jax.enable_x64(True), jax.default_device(backend.device):
            theta = jnp.asarray(inputs.theta, dtype=jnp.float64)
            seed = jnp.asarray(inputs.branch_starts_rad_s[start_index],
                               dtype=backend.complex_dtype)
        candidate_operands = (theta,)
        candidate_key = native.variant_key(
            'candidate', device=backend.device, operands=candidate_operands,
        )
        candidate = backend.execute_native(
            native.candidate_function, *candidate_operands,
            precision=backend.precision, compile_key=candidate_key,
            owner=owner, phase='physical_optimization_root_candidate',
            details={'stage_id': stage_id, 'start_index': start_index},
        )
        require_native_candidate(candidate)
        from .physical_analysis import execute_native_branch_packet
        result,packet,error = execute_native_branch_packet(native,candidate,start_index,seed,
            backend=backend,program_owner=owner,operation_resources=backend.operation_resources,
            compact_evidence=compact_evidence)
        return result,packet,error



def _compact_summary(candidate):
    return {'evaluation_ordinal': candidate['evaluation_ordinal'],
        'parameters':candidate['parameters'], 'cost_f64':candidate['cost_f64'],
        'status':candidate['status'],
        **({'affine_evidence':candidate['affine_evidence']} if 'affine_evidence' in candidate else {}),
        **({'failure':candidate['failure']} if candidate.get('failure') is not None else {}),
        **({'optimizer_unavailability':candidate['optimizer_unavailability']}
           if candidate.get('optimizer_unavailability') is not None else {}),
        'components':[{'objective_id':row['id'], 'status':row['status'],
            **({'weighted_cost_f64':row['cost_f64'],
                'normalized_residual_f64':row['normalized_residual_f64']}
               if row['status'] == 'success' else {}),
            'terms':[{key:value for key,value in term.items()
                      if key in ('term_ordinal','selector','status','value_f64','actual_complex','unit')}
                     for term in row['terms']]} for row in candidate['objectives']]}


class _SavedCandidateRecords:
    """Existing durable envelopes; the storage owner still seals publications."""

    def success(self, **kwargs):
        return encode_physical_candidate(**kwargs)

    def failure(self, **kwargs):
        return encode_physical_candidate_failure(**kwargs)

    def summary(self, candidate):
        return _compact_summary(candidate)

    def cost(self, candidate):
        return unbits(candidate['cost_f64'])

    def continuation(self, observed, *, parameters, t, start_index):
        row = quantity_record(observed)
        row.update(parameters=parameters, continuation_t_f64=bits(t), start_index=start_index)
        return row


def _checkpoint_cache_locator(row, stage_id):
    # An imported row is a source occurrence, not a new attempt occurrence.
    # Preserve its complete verified locator (task/attempt/ref included).
    if 'source_locator' in row:
        return row['source_locator']
    return {'stage_id':stage_id,
            'evaluation_ordinal':row['candidate']['evaluation_ordinal'],
            'kind':row['kind']}


def run_physical_optimization(*, plan_document, request, effective_point, precision,
                             backend, operation_resource, trace, program_owner,
                             checkpoint_policy, resume_state,
                             commit_every_generations, publish_batch,
                             observe_progress, on_acknowledged, spool,
                             execution_policy, frame_owner, candidate_records=None,
                             publication_mode="durable", parameter_authorizations=()):
    """Compute real stage batches; publication ACKs are supplied by the caller.

    ``resume_state`` contains exact checkpoint state and verified ``cache_records``
    hydrated by the storage owner. Its independent ``best_candidate`` is the
    canonical candidate named by the sealed best locator, not a cache lookup.
    An accepted-point restart supplies
    ``restart_point``; it creates fresh local curvature rather than restoring it.
    Callback, publication and infrastructure errors propagate unchanged.
    """
    if publication_mode not in ('durable', 'memory'):
        raise ValueError('unknown optimization publication mode')
    if publication_mode == 'memory' and (checkpoint_policy != 'off' or resume_state):
        raise ValueError('in-memory optimization has no checkpoints or saved restart')
    records = candidate_records if candidate_records is not None else _SavedCandidateRecords()
    spec = request['spec']
    from ..compilation.program import _AffineEvidence
    def new_affine_evidence():
        return _AffineEvidence(parameter_set=parameter_authorizations,
                              optimization_spec=spec['allow_extrapolation'])
    controls = spec['optimizer']
    declarations = (controls['stages'] if controls['type'] == 'hybrid' else
        ({'id': 'main', 'optimizer': controls, 'discretization': spec.get('discretization'),
          'seed': None},))
    comparison_id = controls.get('comparison_stage_id', 'main')
    original_values = parameter_values(effective_point)
    owner = program_owner
    stages, transitions = [], []
    global_baseline = None
    comparison_baseline = comparison_best = comparison_model = None
    selected_point = effective_point
    terminal_outcome = 'success'
    history_ordinal = 0
    resume = dict(resume_state or {})
    resumed_stage = resume.get('stage_id')
    restart_provenance = None
    if resume and resumed_stage is None:
        selected_point = resume['restart_point']
        restart_provenance = {key:value for key,value in resume.items() if key != 'restart_point'}
        resume = {}
    if resume:
        global_baseline = resume['baseline_body']
        transitions = list(resume.get('transitions', ()))
        comparison = resume.get('comparison', {})
        comparison_baseline = comparison.get('baseline')
        comparison_best = comparison.get('winner')
        comparison_model = comparison.get('comparison_model_identity')

    for declaration in declarations:
        stage_id = declaration['id']
        if resumed_stage is not None and stage_id != resumed_stage:
            # Previous entered stages are verified detached terminal facts from
            # the named checkpoint, not reconstructed by replaying their work.
            previous = next(row for row in resume['stages'] if row['stage_id'] == stage_id)
            stages.append(previous)
            continue
        stage_resume = resume if resumed_stage == stage_id else {}
        stage_controls = declaration['optimizer']
        stage_spec = dict(spec, optimizer=stage_controls)
        mesh = declaration.get('discretization')
        if mesh is not None:
            stage_spec['discretization'] = mesh
        seed = (stage_resume.get('restart_point') or stage_resume.get('stage_seed')
                or declaration.get('seed') or selected_point)
        variables = tuple(dict(row, si_unit=next(binding['value']['si_unit']
            for binding in effective_point['bindings']
            if parameter_key(binding['parameter']) == parameter_key(row['parameter'])))
            for row in spec['variables'])
        bounds = {parameter_key(row['parameter']): (quantity(row['lower']), quantity(row['upper']))
                  for row in variables if row.get('domain') is None}
        with trace.span('physical_program_prepare', details={'stage_id': stage_id}), operation_resource.exclusive_assembly():
            program = compile_objective_program(plan_document, stage_spec, effective_point,
                default_view=request.get('view'), mesh=mesh, bounds=bounds,
                precision=precision, authorized=frozenset(
                    parameter_key(row) for row in effective_point['allow_extrapolation']),
                program_owner=owner)
        if program.precision != backend.precision:
            raise ValueError('program precision differs from its execution backend')
        model_identity = {'program_sha256': program.program_sha256,
                          'precision': precision,
                          'discretization': program.quantity_program.point_metadata(
                              program.inputs(effective_point))['discretization']}
        metadata = program.quantity_program.observable_metadata()
        roots = {row['start_index']: row for row in metadata if row['start_index'] is not None}
        has_root_search = any(row['spec'].get('search') for row in roots.values())
        root_search_metadata = ()
        starts = list(program.quantity_program.branch_starts_rad_s)
        if stage_resume:
            starts = [complex_value(row) for row in stage_resume['anchors']]
            global_baseline = stage_resume['baseline_body']
        cache = {}
        pending = []
        next_ordinal = stage_resume.get('next_evaluation_ordinal', 0)
        history_ordinal = stage_resume.get('next_history_ordinal', history_ordinal)
        for key, row in stage_resume.get('cache_records', ()):
            cache[key] = spool.put_record(row)
        best = stage_resume.get('best')
        accepted = stage_resume.get('accepted_iterate')
        accepted_iterations = stage_resume.get('iteration_count', 0)
        preparations = stage_resume.get('preparation_evaluations', 0)
        evaluations = stage_resume.get('evaluation_count', 0)
        generations = stage_resume.get('generation', 0)
        committed_generations = generations if stage_resume else 0
        committed_iterations = accepted_iterations if stage_resume else 0
        baseline_locator = stage_resume.get('baseline_locator')
        baseline_ref = None
        if stage_resume:
            # Rebind one verified baseline body to the new attempt without
            # numerical work or a generation-zero checkpoint. Its scientific
            # stage/model identity and source ancestry remain immutable.
            baseline_ref = spool.put_record({'kind':'baseline',
                'candidate':global_baseline, 'summary_json':records.summary(global_baseline),
                'source_locator':stage_resume['baseline'],
                'source_task_id':stage_resume['source_task_id'],
                'source_attempt_id':stage_resume['source_attempt_id']})
            pending.append(baseline_ref)
            if baseline_locator.get('kind') == 'baseline':
                baseline_locator = {'kind':'baseline'}
        controller = None
        partial = False
        controller_facts = {}

        def progress(phase, *, reused=False):
            observe_progress(phase=phase, stage_id=stage_id,
                strategy=stage_controls['type'], requested_budget=stage_controls['max_evaluations'],
                achievable_evaluations=(1 + stage_controls['complete_generations'] *
                    stage_controls['resolved_population_size'] if stage_controls['type'] == 'cma_es' else None),
                completed_generations=generations if stage_controls['type'] == 'cma_es' else None,
                total_generations=stage_controls.get('complete_generations'),
                committed_generations=(committed_generations if publication_mode == 'durable' else 0) if stage_controls['type'] == 'cma_es' else None,
                completed_iterations=accepted_iterations if stage_controls['type'] in ('lbfgsb', 'projected_lbfgs') else None,
                total_iterations=stage_controls.get('max_iterations'),
                committed_iterations=(committed_iterations if publication_mode == 'durable' else 0) if stage_controls['type'] in ('lbfgsb', 'projected_lbfgs') else None,
                evaluated_count=evaluations,
                best_cost=None if best is None else unbits(best['cost_f64']), reused=reused)

        continuation_rows = {}
        affine_observations = {}

        def continue_starts(point, error):
            # The original-point anchors and the existing continuation policy
            # remain authoritative. Every intermediate uses this same program.
            corrected = list(starts)
            endpoint_frames = {}
            for index, row in roots.items():
                inputs = program.inputs(point, branch_starts_rad_s=corrected)
                result,packet,checked_error = _native_root_branch(
                    program, inputs, index, backend=backend, owner=owner,
                    stage_id=stage_id, compact_evidence=publication_mode == 'memory',
                )
                if result.failure is None:
                    endpoint_frames[index] = (packet,checked_error)
                if result.failure is None:
                    corrected[index] = result.root_omega_rad_s
                    continue
                def evaluate(jobs):
                    results = []
                    for job in jobs:
                        actual,packet,checked_error = _native_root_branch(
                            program, program.inputs(job.point,branch_starts_rad_s=[
                                job.omega if i == index else omega for i,omega in enumerate(corrected)]),
                            index,backend=backend,owner=owner,stage_id=stage_id,
                            compact_evidence=publication_mode == 'memory')
                        if actual.failure is None and point_identity(job.point) == point_identity(point):
                            endpoint_frames[index] = (packet,checked_error)
                        results.append(actual)
                    return tuple(results)
                def observe(values, actual, observed, t):
                    row_record = records.continuation(observed, parameters=actual, t=t, start_index=index)
                    continuation_rows.setdefault(point_identity(point).hex(), []).append(row_record)
                with trace.span('root_continuation', details={'stage_id': stage_id, 'start_index': index}):
                    try:
                        result = continue_root(baseline=original_values, values=parameter_values(point),
                    baseline_root=starts[index], endpoint_view=point, initial_result=result,
                    candidate_view=lambda values: _point_at(effective_point, values),
                    make_job=lambda identity, actual, omega: SimpleNamespace(id=identity, point=actual, omega=omega),
                    evaluate=evaluate, identity=next_ordinal,
                        observe=observe)
                    except EvaluationFailure as failed:
                        raise _candidate_error(numerical_error(failed.failure)) from failed
                if result.failure is not None:
                    raise numerical_error(result.failure)
                corrected[index] = result.root_omega_rad_s
            completed = (tuple((index,endpoint_frames[index][0]) for index in roots),
                         tuple(endpoint_frames[index][1] for index in roots))
            return corrected,completed

        def compute(point, *, gradient=False, prepared=None, anchor=False,
                    evaluation_role='candidate'):
            nonlocal root_search_metadata
            point_key = point_identity(point).hex()
            if prepared is None:
                affine_observations[point_key] = new_affine_evidence()
            affine_evidence = affine_observations[point_key]
            inputs = program.inputs(point, branch_starts_rad_s=starts)
            options = {'active_keys': tuple(parameter_key(row['parameter']) for row in variables)} if gradient else {'prepared': prepared}
            completed = None
            def native_ad_runner(native, arguments, loss, *, active_keys, compact_evidence=False, completed_branches=None, completed_branch_errors=None):
                from .native_ad import objective_value_and_gradient
                return objective_value_and_gradient(
                    native, arguments, loss,
                    coefficients=program.coefficient_operands(),
                    active_keys=active_keys,
                    backend=backend, program_owner=owner,
                    execution_policy=execution_policy,
                    frame_owner=frame_owner, compact_evidence=compact_evidence,
                    completed_branches=completed_branches,
                    completed_branch_errors=completed_branch_errors,
                )

            try:
                if has_root_search:
                    from .physical_analysis import acquire_completed_root_frame
                    _,completed,_,searches = acquire_completed_root_frame(program.quantity_program,inputs,
                        backend=backend,program_owner=owner,operation_resources=operation_resource,
                        compact_evidence=publication_mode == 'memory',discover=anchor,
                        affine_evidence=affine_evidence)
                    if anchor:
                        root_search_metadata = searches
                with trace.span('physical_value_gradient' if gradient else 'physical_value',
                                details={'stage_id': stage_id,
                                         'evaluation_role': evaluation_role}):
                    if gradient:
                        result = program.value_and_gradient(
                            inputs, active_keys=options['active_keys'],
                            native_executor=backend.execute_native,
                            program_owner=owner, device=backend.device,
                            native_ad_runner=native_ad_runner, compact_evidence=publication_mode == 'memory', completed=completed,
                            affine_evidence=affine_evidence,
                        )
                    else:
                        result = program.value(
                            inputs, prepared=prepared,
                            native_executor=backend.execute_native,
                            program_owner=owner, device=backend.device,
                            compact_evidence=publication_mode == 'memory', completed=completed,
                            affine_evidence=affine_evidence,
                        )
            except SCNSimError as error:
                if anchor:
                    raise
                # Candidate +Inf classification does not authorize root repair.
                continuation_eligible = error.kind == 'numerical_resolution_unresolved' and bool(roots)
                error = _candidate_error(error)
                if not continuation_eligible:
                    raise error
                corrected,completed = continue_starts(point, error)
                # A prepared packet binds exact hints; never reuse it after
                # continuation changes those operands.
                corrected_inputs = program.inputs(point, branch_starts_rad_s=corrected)
                if gradient:
                    result = program.value_and_gradient(
                        corrected_inputs, active_keys=options['active_keys'],
                        native_executor=backend.execute_native,
                        program_owner=owner, device=backend.device,
                        native_ad_runner=native_ad_runner, compact_evidence=publication_mode == 'memory', completed=completed,
                        affine_evidence=affine_evidence,
                    )
                else:
                    result = program.value(
                        corrected_inputs, native_executor=backend.execute_native,
                        program_owner=owner, device=backend.device,
                        compact_evidence=publication_mode == 'memory', completed=completed,
                        affine_evidence=affine_evidence,
                    )
            # These are the established objective-resolution conditions, not
            # controller acceptance or a new global finiteness policy.
            if not all(math.isfinite(float(value)) for values in
                       (result.values, result.normalized_residuals, result.contributions)
                       for value in values) or not math.isfinite(float(result.total_cost)):
                raise NumericalResolutionUnresolved('objective arithmetic is non-finite', stage='objective')
            if root_search_metadata:
                from ..numerics.evidence import attach_root_search_metadata
                result = attach_root_search_metadata(result,root_search_metadata)
            if anchor:
                bodies = project_primal_bodies(program.quantity_program, result.primal, inputs,
                    ephemeral=publication_mode == 'memory')
                for row in metadata:
                    if row['start_index'] is not None:
                        starts[row['start_index']] = bodies[row['observable_index']].root_omega_rad_s
            return result

        def record(point, outcome=None, *, error=None, kind='evaluation', optimizer_unavailability=None):
            nonlocal next_ordinal, history_ordinal, best, evaluations
            observation = affine_observations.get(point_identity(point).hex())
            affine_rows = () if observation is None else tuple(observation.rows)
            candidate = (records.success(objective_program=program,
                outcome=outcome, parameters=point, objective_records=spec['objectives'],
                affine_evidence=affine_rows)
                if error is None else records.failure(parameters=point,
                    objective_records=spec['objectives'], error=error, affine_evidence=affine_rows))
            affine_observations.pop(point_identity(point).hex(), None)
            if optimizer_unavailability is not None:
                candidate['optimizer_unavailability'] = {'kind':optimizer_unavailability.code,
                    'detail':optimizer_unavailability.message, 'role':'candidate_value_gradient',
                    'evidence':dict(optimizer_unavailability.evidence)}
            continuation = continuation_rows.pop(point_identity(point).hex(), ())
            if continuation:
                candidate['continuation'] = continuation
            candidate.update(stage_id=stage_id, model_identity=model_identity)
            if outcome is not None:
                candidate['discretization'] = program.quantity_program.point_metadata(
                    program.inputs(point))['discretization']
            candidate.update(evaluation_ordinal=next_ordinal,
                             status='success' if error is None else 'failure')
            locator = {'stage_id': stage_id, 'evaluation_ordinal': next_ordinal, 'kind': kind}
            row = {'kind': kind, 'candidate': candidate,
                   'summary_json': records.summary(candidate)}
            if kind == 'evaluation':
                row['history_ordinal'] = history_ordinal
                history_ordinal += 1
            ref = spool.put_record(row)
            pending.append(ref)
            cache[point_identity(point).hex()] = ref
            next_ordinal += 1
            evaluations += 1
            cost = records.cost(candidate)
            optimizer_eligible = (stage_controls['type'] != 'projected_lbfgs' or
                (optimizer_unavailability is None and outcome is not None and outcome.gradient is not None))
            if error is None and optimizer_eligible and (best is None or cost < unbits(best['cost_f64'])):
                best = dict(locator, cost_f64=bits(cost), row_ref=ref)
            return locator, ref, cost

        def state():
            return {'stage_id': stage_id, 'model_identity': model_identity,
                'strategy': stage_controls['type'], 'status': 'partial' if partial else 'running',
                'objective_ids': [row['id'] for row in spec['objectives']],
                'variable_refs': [row['parameter'] for row in variables],
                'evaluation_count': evaluations, 'iteration_count': accepted_iterations if stage_controls['type'] in ('lbfgsb', 'projected_lbfgs') else generations,
                'preparation_evaluations': preparations,
                'termination_reason': None,
                'observed_best': None if best is None else {key:value for key,value in best.items() if key not in ('cost_f64','row_ref')},
                'controller': controller_facts,
                'restart_provenance': restart_provenance,
                'accepted_iterate': accepted}

        def flush(*, tail=False):
            nonlocal committed_generations, committed_iterations
            if not pending:
                return
            checkpoint = None
            if checkpoint_policy != 'off' and controller is not None and generations > 0:
                checkpoint = {'stage_id': stage_id, 'model_identity': model_identity,
                    'strategy': stage_controls['type'], 'generation': generations,
                    'iteration_count': accepted_iterations, 'next_evaluation_ordinal': next_ordinal,
                    'next_history_ordinal': history_ordinal, 'anchors': [complex_record(value) for value in starts],
                    'cache': {key:_checkpoint_cache_locator(spool.get_record(ref), stage_id)
                              for key, ref in cache.items()},
                    'best': None if best is None else {key: value for key,value in best.items() if key != 'row_ref'},
                    'baseline_locator': baseline_locator, 'accepted_iterate': accepted,
                    'stage_seed': seed, 'transitions': transitions,
                    'comparison': {'baseline':comparison_baseline,'winner':comparison_best,
                                   'comparison_model_identity':comparison_model},
                    'preparation_evaluations': preparations, 'evaluation_count': evaluations,
                    'stages': stages, 'baseline': {'kind':'baseline'},
                    **({'cma': controller.snapshot()} if controller is not None else {})}
            ack = publish_batch({'stage_id': stage_id, 'model_identity': model_identity,
                'rows': (spool.get_record(ref) for ref in pending), 'stage_state': state(),
                'checkpoint': checkpoint, 'transitions': tuple(transitions)})
            pending.clear()
            if publication_mode == 'durable':
                committed_generations, committed_iterations = generations, accepted_iterations
            with trace.span('optimization_callback', details={'stage_id': stage_id}):
                on_acknowledged(state(), ack)
            progress('committed' if publication_mode == 'durable' else 'observed')

        if not stage_resume:
            original = compute(effective_point, anchor=True,
                evaluation_role='baseline' if global_baseline is None else 'stage_baseline')
            kind = 'baseline' if global_baseline is None else 'evaluation'
            baseline_locator, baseline_ref, _ = record(effective_point, original, kind=kind)
            baseline_locator['kind'] = kind
            if kind == 'baseline':
                baseline_locator = {'kind':'baseline'}
            preparations += 1
            del original
            if global_baseline is None:
                global_baseline = spool.get_record(baseline_ref)['candidate']
                global_baseline.update(stage_id=stage_id, model_identity=model_identity)
        if stage_id == comparison_id:
            comparison_baseline, comparison_model = baseline_locator, model_identity
        reevaluation_locator = baseline_locator
        if point_identity(seed) != point_identity(effective_point) and not stage_resume:
            seed_outcome = compute(seed, evaluation_role='stage_seed')
            reevaluation_locator, _, _ = record(seed, seed_outcome)
            preparations += 1
            del seed_outcome
        if stages and not stage_resume:
            source = stages[-1]
            source_locator = source['observed_best']
            reason = 'declared_stage_seed' if declaration.get('seed') is not None else 'previous_stage_selected_point'
            transitions.append({
                'transition_id': f'{source["stage_id"]}:{stage_id}',
                'source_stage_id': source['stage_id'], 'target_stage_id': stage_id,
                'source_model_identity': source['model_identity'],
                'target_model_identity': model_identity,
                'source_candidate_locator': source_locator,
                'selection_reason': reason,
                # Publication resolves this abstract projection to the actual
                # immutable target candidate/parameter reference atomically.
                'seed_parameters_ref': {'candidate_locator': reevaluation_locator, 'field':'parameters'},
                'reevaluation_locator': reevaluation_locator,
                'reevaluation_status': 'success', 'state':'entered',
                'ancestry': {'source_candidate_locator': source_locator,
                             'declared_seed': declaration.get('seed') is not None,
                             'restart': restart_provenance},
            })
        seed_values = parameter_values(seed)

        if stage_controls['type'] == 'cma_es':
            if stage_resume:
                # The encoded transition seed is nullable until stage entry.
                # Initialize from the verified stage seed before restoring the
                # complete post-tell state; do not mutate the sealed request.
                stage_spec = dict(stage_spec, optimizer=dict(stage_controls,
                    baseline_optimizer_coordinates_f64=physical_seed_coordinates(stage_spec, seed_values)))
            controller = (CMAController(stage_spec, seed_values, restore_state=stage_resume['cma'])
                if stage_resume else CMAController.from_physical_seed(stage_spec, seed_values))
            while controller.generation < stage_controls['complete_generations']:
                progress('population_evaluating')
                with trace.span('cma_ask', details={'stage_id':stage_id}):
                    raw = controller.ask_population()
                # Mapping errors have always been fatal before physical
                # candidate evaluation; do not invent a realized failed point.
                points = [_point_at(effective_point, controller.map_coordinates(value)) for value in raw]
                uncached, wave_keys = [], set()
                for i, point in enumerate(points):
                    key = point_identity(point).hex()
                    if key not in cache and key not in wave_keys:
                        uncached.append((i, point))
                        wave_keys.add(key)
                def worker(item):
                    index, point, packet = item
                    try:
                        with operation_resource.candidate_compute():
                            if isinstance(packet, SCNSimError):
                                raise packet
                            return index, point, compute(point, prepared=packet,
                                evaluation_role='candidate_value'), None
                    except SCNSimError as error:
                        if error.kind not in _CANDIDATE_FAILURES:
                            raise
                        return index, point, None, error
                def ready_results():
                    capacity = operation_resource.worker_capacity
                    for offset in range(0, len(uncached), capacity):
                        wave = uncached[offset:offset + capacity]
                        with trace.span('physical_prepare_batch', details={'stage_id':stage_id}), operation_resource.exclusive_assembly():
                            with backend.jax.enable_x64(precision == 'float64'), backend.jax.default_device(backend.device):
                                observations = tuple(new_affine_evidence() for _ in wave)
                                for (_, point), observation in zip(wave, observations, strict=True):
                                    affine_observations[point_identity(point).hex()] = observation
                                packets = []
                                for (_, point), observation in zip(wave, observations, strict=True):
                                    try:
                                        packet, = program.prepare_batch(
                                            (program.inputs(point, branch_starts_rad_s=starts),),
                                            program_owner=owner, affine_evidence=(observation,))
                                    except SCNSimError as error:
                                        # The documented rejected affine candidate still
                                        # belongs to this population occurrence, not to
                                        # operation-level failure or root continuation.
                                        if (error.kind != 'invalid_candidate_physical_parameter'
                                                or error.stage != 'affine_support'):
                                            raise
                                        packet = error
                                    packets.append(packet)
                        try:
                            yield from operation_resource.map_ordered(worker,
                                ((index,point,packet) for (index,point),packet in zip(wave,packets,strict=True)),
                                identity=lambda item: f'{stage_id}:{item[0]}')
                        finally:
                            del packets
                observed = ready_results()
                costs = []
                uncached_indices = {index for index, point in uncached}
                for index, point in enumerate(points):
                    key = point_identity(point).hex()
                    if index in uncached_indices:
                        actual_index, actual_point, outcome, error = next(observed)
                        try:
                            locator, ref, cost = record(actual_point, outcome, error=error)
                        except SCNSimError as encoding_error:
                            if encoding_error.kind not in _CANDIDATE_FAILURES:
                                raise
                            locator, ref, cost = record(actual_point, error=encoding_error)
                        del outcome, error
                    else:
                        # A new occurrence keeps its controller ordinal while
                        # reusing the exact actual scientific candidate body.
                        source = spool.get_record(cache[key])
                        candidate = dict(source['candidate'], evaluation_ordinal=next_ordinal)
                        summary = dict(source['summary_json'], evaluation_ordinal=next_ordinal)
                        row = {'kind':'evaluation', 'history_ordinal':history_ordinal,
                               'candidate':candidate, 'summary_json':summary}
                        ref = spool.put_record(row)
                        pending.append(ref)
                        cache[key] = ref
                        next_ordinal += 1
                        history_ordinal += 1
                        evaluations += 1
                        cost = records.cost(candidate)
                        del source, candidate, summary, row
                    costs.append(cost)
                # Exhaustion lets the resource owner drain/release its futures.
                for unused in observed:
                    raise RuntimeError('ordered numerical wave exceeded its declared points')
                del observed
                with trace.span('cma_tell', details={'stage_id':stage_id}):
                    controller.tell(raw, costs)
                generations = controller.generation
                progress('generation_complete')
                if generations % commit_every_generations == 0:
                    flush()
            termination = 'evaluation_budget'
            controller_facts = {'generation':generations, 'population_size':controller.population_size,
                                'controller_evaluations': 1 + generations * controller.population_size}
        else:
            coordinate_map = LocalCoordinateMap(variables, seed_values,
                coordinate_scales=stage_controls.get('coordinate_scales'))
            trial_rows = {}
            def evaluate(coordinates, *, need_gradient=True):
                point = _point_at(effective_point, coordinate_map.physical_values(coordinates))
                try:
                    with operation_resource.candidate_compute():
                        outcome = compute(point, gradient=True,
                            evaluation_role='candidate_value_gradient')
                    unavailable = (LocalUnavailable('derivative_unavailable', 'checked physical derivative unavailable',
                        {'reasons': outcome.aux.observable_reasons}) if outcome.gradient is None else None)
                    locator, ref, cost = record(point, outcome, optimizer_unavailability=unavailable)
                    trial_rows[tuple(coordinates)] = (locator, ref)
                    if unavailable is not None:
                        return LocalEvaluation(point, locator, cost, None, unavailable)
                    gradient = coordinate_map.gradient(tuple(float(value) for value in outcome.gradient), coordinates)
                    return LocalEvaluation(point, locator, cost, gradient)
                except SCNSimError as error:
                    if error.kind not in _CANDIDATE_FAILURES:
                        raise
                    unavailable = LocalUnavailable(error.kind, str(error), error.evidence)
                    locator,ref,cost = record(point, error=error, optimizer_unavailability=unavailable)
                    return LocalEvaluation(point, locator, cost, None, unavailable)
            def on_accepted(coordinates):
                nonlocal accepted, accepted_iterations
                locator,ref = trial_rows[tuple(coordinates)]
                accepted = dict(locator)
                accepted_iterations += 1
                trial_rows.clear()
                progress('iteration_complete')
                if accepted_iterations % stage_controls['commit_every_iterations'] == 0:
                    flush()
            with trace.span('local_controller', details={'stage_id': stage_id}):
                local = run_local_optimizer(initial=coordinate_map.coordinates(seed_values),
                bounds=coordinate_map.bounds, controls=stage_controls,
                evaluate=evaluate, on_accepted=on_accepted)
            controller_facts = {'algorithm':stage_controls['type'],
                **({'procedure':{'memory':10,'initial_normalized_step':1,'step_contraction':.5,
                    'acceptance':'actual strict cost decrease and available actual gradient'}}
                   if stage_controls['type'] == 'projected_lbfgs' else {}),
                'coordinate_map': coordinate_map.canonical_record(),
                'message':local.message, 'scipy_success':local.scipy_success,
                'scipy_status':local.scipy_status,'nfev':local.scipy_nfev,
                'njev':local.scipy_njev,'nit':local.scipy_nit,
                'evaluation_requests':local.evaluation_requests}
            termination = local.termination
            partial = local.unavailable is not None
            del local
            trial_rows.clear()
        flush(tail=True)
        stage = state()
        stage.update(status='partial' if partial else 'complete', termination_reason=termination)
        stages.append(stage)
        if best is not None:
            if 'row_ref' in best:
                selected_point = spool.get_record(best['row_ref'])['candidate']['parameters']
            else:
                # The point cache deduplicates operands, not occurrences. A
                # strict-less winner may precede a later identical cache row.
                selected_point = stage_resume['best_candidate']['parameters']
            if stage_id == comparison_id:
                comparison_best = stage['observed_best']
        if partial:
            terminal_outcome = 'partial'
            break
        resume = {}
        resumed_stage = None
        with trace.span('optimization_stage_cleanup', details={'stage_id':stage_id}):
            cache.clear()
            pending.clear()
            continuation_rows.clear()
    result = {'comparison': {'comparison_model_identity': comparison_model,
        'comparison_stage_id': comparison_id if comparison_model is not None else None,
        'declarations': spec, 'winner': comparison_best, 'baseline': comparison_baseline}}
    return {'baseline': global_baseline, 'manifest': spec, 'stages': stages,
        'transitions': transitions, 'outcome': terminal_outcome, 'result': result,
        'summary': {'entered_stages': len(stages), 'outcome': terminal_outcome}}
