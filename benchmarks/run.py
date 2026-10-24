"""M0 public-operation measurement and raw-byte capture frontend.

Each cold observation starts in a new interpreter. Warm observations reuse its
Run and declared request structure. Golden capture is a separate instrumented operation, excluded
from timing summaries. No performance target is a pass/fail condition here.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time
import traceback


def machine():
    cpu = None
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None)
    return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,
            'cpu_model':cpu,'logical_cpus':os.cpu_count(),
            'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'load_average':list(os.getloadavg()) if hasattr(os,'getloadavg') else None}


def _optimization_report(result, elapsed_ns, native_calls, benchmark, prior_ids):
    """Project public counts and recorded native phases after the stopwatch."""
    stages = [] if result is None else [
        dict(stage_id=x.stage_id, strategy=x.strategy, evaluation_count=x.evaluation_count,
             iteration_count=x.iteration_count, status=x.status, termination_reason=x.termination_reason)
        for x in result.stages]
    reason = ('no returned public Optimization stages' if result is None or not stages else
              'no current native dispatch; stage counts may describe readback' if not native_calls else None)
    nfev = sum(x['evaluation_count'] for x in stages) if reason is None else None
    nit = sum(x['iteration_count'] for x in stages) if reason is None else None
    phases = ('physical_objective_value', 'physical_objective_value_gradient')
    rows = []
    if prior_ids is None:
        rows = [dict(phase=x.phase, count=x.count, elapsed_ns=x.elapsed_ns,
                     status_counts=dict(x.status_counts)) for x in benchmark['timings'] if x.phase in phases]
    else:
        for batch in benchmark['timing_batches']:
            if batch['operation_id'] in prior_ids:
                continue
            rows.extend(dict(phase=x['kind'], count=x['count'], elapsed_ns=x['elapsed_ns_sum'],
                             status_counts=x['status_counts']) for x in batch['aggregates'] if x['kind'] in phases)
            rows.extend(dict(phase=x['kind'], count=1, elapsed_ns=x['end_ns']-x['start_ns'],
                             status_counts={x['status']:1}) for x in batch['spans'] if x['kind'] in phases)
    missing = reason or ('actual stage evaluation_count is zero' if nfev == 0 else None)
    return dict(stages=stages, evaluation_count=nfev, iteration_count=nit,
        counts_unavailable_reason=reason, api_wall_ns=elapsed_ns,
        api_wall_mean_ns_per_evaluation=elapsed_ns/nfev if missing is None else None,
        mean_unavailable_reason=missing, native_objective_phase_timings=rows,
        native_objective_phase_mean_ns_per_evaluation=sum(x['elapsed_ns'] for x in rows)/nfev if missing is None and rows else None,
        native_mean_unavailable_reason=missing or (None if rows else 'no recorded objective native phase'),
        native_timing_basis='existing synchronized physical_objective_value/value_gradient phases; includes root work inside that objective executable; excludes separate native_compilation and host/publication work outside those phases',
        api_timing_basis='public optimize wall time / actual public stage evaluations; excludes fixture construction and report projection')


@contextmanager
def _observe_local_initial(enabled):
    """Retain the first real controller evaluation without adding evaluations."""
    rows = []
    if not enabled:
        yield rows
        return
    from scnsim.execution import physical_optimization
    original = physical_optimization.run_local_optimizer
    def observe(*, initial, bounds, controls, evaluate, on_accepted):
        row = {'initial':initial, 'bounds':bounds, 'controls':controls,
               'evaluate':evaluate, 'first':None}
        rows.append(row)
        def evaluated(*args, **kwargs):
            observed = evaluate(*args, **kwargs)
            if row['first'] is None:
                row['first'] = observed
                row['evaluation_coordinates'] = args[0] if args else kwargs['coordinates']
            return observed
        return original(initial=initial,bounds=bounds,controls=controls,
                        evaluate=evaluated,on_accepted=on_accepted)
    physical_optimization.run_local_optimizer = observe
    try:
        yield rows
    finally:
        physical_optimization.run_local_optimizer = original


def _local_initial_report(rows, result, capture):
    """Project observed operands after timing; never infer optimality from status."""
    import inspect
    records = []
    for ordinal,row in enumerate(rows):
        observed = row['first']
        coordinate_map = inspect.getclosurevars(row['evaluate']).nonlocals.get('coordinate_map')
        record = {'controller_ordinal':ordinal,
            'source':'same-call physical_optimization.run_local_optimizer/evaluate observer',
            'initial_coordinates':capture.project(row['initial']),
            'bounds':capture.project(row['bounds']),
            'controls':capture.project(row['controls']),
            'coordinate_map':None if coordinate_map is None else capture.project(coordinate_map.canonical_record()),
            'coordinate_map_unavailable_reason':None if coordinate_map is not None else 'actual evaluation closure has no coordinate_map',
            'initial_objective':None if observed is None else capture.project(observed.cost),
            'optimizer_gradient':None if observed is None else capture.project(observed.gradient),
            'evaluation_coordinates':capture.project(row.get('evaluation_coordinates')),
            'evaluation_parameters':None if observed is None else capture.project(observed.point),
            'evaluation_unavailability':None if observed is None else capture.project(observed.unavailable),
            'observation_unavailable_reason':None if observed is not None else 'no returned controller evaluation',
            'projected_gradient':None,
            'projected_gradient_unavailable_reason':'no returned gradient' if observed is None or observed.gradient is None else None,
            'projection_basis':'L-BFGS-B bound projection in actual controller coordinates: g<0 -> max(g,x-upper); g>=0 -> min(g,x-lower), for declared finite bounds; no stopping classification'}
        if observed is not None and observed.gradient is not None:
            projected = []
            for x,g,(lower,upper) in zip(row['evaluation_coordinates'],observed.gradient,row['bounds'],strict=True):
                projected.append(max(g,x-upper) if g < 0 and upper is not None else
                                 min(g,x-lower) if g >= 0 and lower is not None else g)
            record['projected_gradient'] = capture.project(tuple(projected))
        records.append(record)
    physical = []
    if result is not None and type(result).__module__ == 'scnsim.results.memory_optimization':
        for stage in result.stages:
            first = next((trial for trial in stage.history if trial.gradient is not None),None)
            physical.append({'stage_id':stage.stage_id,
                'source':'returned Memory Result public stage.history first gradient-bearing trial',
                'evaluation_ordinal':None if first is None else first.evaluation_ordinal,
                'parameters':None if first is None else capture.project(first.parameters),
                'cost':None if first is None else capture.project(first.cost),
                'gradient':None if first is None else capture.project(first.gradient),
                'unavailable_reason':None if first is not None else 'no public gradient-bearing trial'})
    return {'controller_observations':records,'physical_gradient_observations':physical,
            'unavailable_reason':None if rows else 'no local controller execution; cached readback or operation failed before controller entry',
            'physical_gradient_unavailable_reason':None if physical else 'no returned Memory public gradient trial; saved Result history does not expose physical gradients',
            'observer_overhead':'forwarding wrapper and one retained LocalEvaluation within measured call; projection and bound-projection arithmetic after stopwatch; no extra objective or AD calls'}


def _root_failure_report(spec, outcome, capture):
    """Name existing error operands, without reconstructing an unreturned packet."""
    error = outcome.get('exception')
    evidence = None if error is None else error.get('evidence')
    actual = {} if evidence is None else dict(evidence['mapping'])
    return {'request':capture.project(spec),
            'source':'original same-call structured exception evidence',
            'final_iterate_rad_s':actual.get('omega'),
            'final_iterate_unavailable_reason':None if 'omega' in actual else 'original exception did not expose omega',
            'slope':actual.get('slope'),
            'slope_unavailable_reason':None if 'slope' in actual else 'original exception did not expose slope',
            'status':actual.get('status'),'steps':actual.get('steps'),
            'closure':actual.get('closure')}


@contextmanager
def _observe_jax_trace(output):
    """Start after the first existing backend acquisition, without adding one."""
    if output is None:
        yield
        return
    from scnsim.execution import jax_backend
    original = jax_backend.get_jax_backend
    contexts = ExitStack()
    started = False
    primary = None
    def acquired(*args, **kwargs):
        nonlocal started
        backend = original(*args, **kwargs)
        if not started:
            try:
                from jax.profiler import trace
                contexts.enter_context(trace(str(Path(output).expanduser().resolve())))
            except BaseException as error:
                # The caller has not received this real backend to own its cleanup.
                try:
                    backend.close()
                except BaseException as cleanup_error:
                    try:
                        error.add_note('JAX trace startup backend cleanup also failed: '
                                       +type(cleanup_error).__module__+'.'
                                       +type(cleanup_error).__qualname__+': '+str(cleanup_error))
                    except BaseException:
                        pass  # A secondary diagnostic must not replace trace startup failure.
                raise
            started = True
        return backend
    jax_backend.get_jax_backend = acquired
    try:
        yield lambda: started
    except BaseException as error:
        primary = error
        raise
    finally:
        jax_backend.get_jax_backend = original
        try:
            contexts.close()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            try:
                primary.add_note('JAX trace finalization also failed: '
                                 +type(cleanup_error).__module__+'.'
                                 +type(cleanup_error).__qualname__+': '+str(cleanup_error))
            except BaseException:
                pass  # Keep the original worker exception when trace cleanup also fails.


def run_worker(args):
    # Runtime policy must be set before any JAX import or case construction.
    from scnsim import configure_runtime
    if args.cpu_threads is not None:
        configure_runtime(cpu_threads=args.cpu_threads)
    # Excludes first host/backend bootstrap; includes subsequent original work and close.
    with _observe_jax_trace(args.jax_trace_output) as trace_started:
        result = _run_worker_configured(args)
        if args.jax_trace_output is not None and result == 0 and not trace_started():
            raise RuntimeError('JAX trace unavailable: completed worker acquired no numerical backend; worker results are retained')
        return result


def _installed_wheel_identity(module):
    """Bind this worker to an ordinary installed distribution, never an overlay."""
    import importlib.metadata
    dist = importlib.metadata.distribution('scnsim')
    origin = Path(module.__file__).resolve()
    expected = Path(dist.locate_file('scnsim/__init__.py')).resolve()
    direct_url = json.loads(dist.read_text('direct_url.json') or '{}')
    if origin != expected or direct_url.get('dir_info', {}).get('editable', False):
        raise RuntimeError('installed-wheel-only worker requires the ordinary installed package')
    if os.environ.get('PYTHONPATH'):
        raise RuntimeError('installed-wheel-only worker requires PYTHONPATH unset')
    record = dist.read_text('RECORD')
    if record is None:
        raise RuntimeError('installed-wheel-only worker requires installed RECORD')
    return {'origin':str(origin), 'version':dist.version, 'direct_url':direct_url,
            'record_sha256':hashlib.sha256(record.encode()).hexdigest(), 'editable':False}


def _sample_control(args, phase, iteration):
    """Optional maintainer pipe handshake outside the measured public call."""
    if args.sample_control_fds is None:
        return
    events_fd, release_fd = args.sample_control_fds
    event = {'phase':phase, 'iteration':iteration, 'machine':machine(),
             'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
    with os.fdopen(os.dup(events_fd), 'w', encoding='utf-8') as events:
        events.write(json.dumps(event)+'\n'); events.flush()
    if phase == 'ready' and os.read(release_fd, 1) != b'G':
        raise RuntimeError('sample controller closed before releasing this call')


def _complete_sample_control(args, iteration, primary, operation_error, record):
    """Return the original operation failure if a secondary handshake fails."""
    try:
        _sample_control(args, 'complete', iteration)
    except BaseException as error:
        record['sample_control_error']={'type':type(error).__module__+'.'+type(error).__qualname__,
                                        'message':str(error)}
        original = primary if primary is not None else operation_error
        if original is None:
            raise
        original.add_note('secondary complete-handshake failure: '+type(error).__name__+': '+str(error))
        return original
    return None


def _run_worker_configured(args):
    # The optional trace encloses the original workload and its Run.close lifecycle.
    import importlib.metadata
    import scnsim
    installed_identity = _installed_wheel_identity(scnsim) if args.installed_wheel_only else None
    import jax
    from circuits import build_case
    from golden import Capture, observe_direct
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    capture=Capture(out/'golden');record={'input':vars(args).copy(),'machine_before':machine(),
            'source_commit':args.source_commit,'scnsim_source':scnsim.__file__,
            'installed_wheel_identity':installed_identity,
            'harness_sources':{name:{'path':str(Path(path).resolve()),
                'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                for name,path in {'run':__file__, 'circuits':sys.modules[build_case.__module__].__file__,
                                  'golden':sys.modules[Capture.__module__].__file__}.items()},
            'versions':{n:importlib.metadata.version(n) for n in ('scnsim','jax','jaxlib','numpy','scipy')},
            'calls':[],'compilation_events':[],'jax_duration_events':[]}
    def compilation_event(name,**kwargs):
        record['compilation_events'].append({'name':name,'metadata':{key:str(value) for key,value in kwargs.items()}})
    def duration_event(name, duration, **kwargs):
        record['jax_duration_events'].append({'name':name,'duration_seconds':duration,'metadata':{key:str(value) for key,value in kwargs.items()}})
    jax.monitoring.register_event_listener(compilation_event)
    jax.monitoring.register_event_duration_secs_listener(duration_event)
    from scnsim.execution.jax_backend import JaxBackend
    execution_count = 0
    native_shapes = []
    original_execute = JaxBackend.execute_native
    def counted_execute(backend, *operands, **kwargs):
        nonlocal execution_count
        execution_count += 1
        if args.point_counts:
            native_shapes.append([dict(shape=list(x.shape), dtype=str(x.dtype))
                                  for x in jax.tree_util.tree_leaves(operands[1:]) if hasattr(x, 'shape')])
        return original_execute(backend, *operands, **kwargs)
    JaxBackend.execute_native = counted_execute
    case=None;primary=None
    previous_term = signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt("benchmark worker terminated")))
    try:
        case=build_case(args.case,storage=args.storage,workspace=out/'workspace' if args.storage=='workspace' else None,
                        chunk_size=args.chunk_size,points=args.points,cma_evaluations=args.cma_evaluations,tail_failure=args.tail_failure)
        record['case_metadata']=capture.project(case.metadata)
        operation=case.operations[args.operation]
        if args.point_counts and args.operation != 'direct':
            raise ValueError('--point-counts is the declared Direct grid-reuse observation')
        iterations = len(args.point_counts) if args.point_counts else 1+args.warm_repetitions
        for iteration in range(iterations):
            selected_point = case.select_point(0 if args.point_counts or args.operation in ('cma', 'lbfgsb') else iteration)
            selected_grid = case.select_grid(args.point_counts[iteration]) if args.point_counts else None
            compile_before = len(record['compilation_events'])
            duration_before = len(record['jax_duration_events'])
            shape_before = len(native_shapes)
            execution_before = execution_count
            prior_ids = None
            if args.operation in ('cma', 'lbfgsb') and args.storage == 'workspace':
                prior_ids = {x['operation_id'] for x in case.run.benchmark().document()['operations']}
            result = None
            operation_error = None
            context=observe_direct(capture) if args.capture_only else nullcontext([])
            with context as staged, _observe_local_initial(args.operation == 'lbfgsb') as local_initial:
                _sample_control(args, 'ready', iteration)
                started=time.perf_counter_ns()
                try:
                    result=operation()
                except BaseException as error:
                    operation_error = error
                    elapsed=time.perf_counter_ns()-started
                    outcome={'exception':capture.exception(error)}
                    record['calls'].append({'iteration':iteration,'elapsed_ns':elapsed,'outcome':'raised','exception_type':type(error).__module__+'.'+type(error).__qualname__,'exception':outcome['exception']})
                    # Expected physical errors are observations, never fabricated success.
                    physical_kinds = {"direct_response_formation", "eliminated_block_solve_failure",
                                      "root_slope_unresolved", "numerical_resolution_unresolved",
                                      "invalid_candidate_physical_parameter"}
                    if getattr(error, "kind", None) not in physical_kinds:
                        primary=error
                else:
                    elapsed=time.perf_counter_ns()-started
                    record['calls'].append({'iteration':iteration,'elapsed_ns':elapsed,'outcome':'returned','result_type':type(result).__module__+'.'+type(result).__qualname__})
                    outcome={'result':capture.project(result)} if args.capture_only else {'result_type':type(result).__module__+'.'+type(result).__qualname__}
                record['calls'][-1].update(parameter_point=capture.project(selected_point),
                    native_execute_calls=execution_count-execution_before,
                    observer_active=args.capture_only, native_counter_active=True,
                    grid=capture.project(selected_grid), native_operand_shapes=native_shapes[shape_before:],
                    compilation_events=record['compilation_events'][compile_before:],
                    jax_duration_events=record['jax_duration_events'][duration_before:],
                    request_identity=capture.project(getattr(result, "identity", None)) if record['calls'][-1]['outcome']=='returned' else None,
                    request_sha256=getattr(result, "request_sha256", None) if record['calls'][-1]['outcome']=='returned' else None)
                if args.operation in ('cma', 'lbfgsb'):
                    record['calls'][-1]['optimization_initial_parameters'] = capture.project(case.metadata['optimization_initial_parameters'])
                    record['calls'][-1]['optimization_fixed_parameters'] = capture.project(case.metadata['optimization_parameter_set'])
                    record['calls'][-1]['optimization'] = _optimization_report(
                        result, elapsed, execution_count-execution_before,
                        case.run.benchmark().document(), prior_ids)
                if args.operation == 'lbfgsb':
                    record['calls'][-1]['local_initial_diagnostic'] = _local_initial_report(local_initial,result,capture)
                if args.operation == 'transfer_zero' and record['calls'][-1]['outcome'] == 'raised':
                    record['calls'][-1]['root_failure_diagnostic'] = _root_failure_report(
                        case.metadata['specs']['transfer_zero'],outcome,capture)
                if args.capture_only:
                    capture.save('observation-'+str(iteration)+'.json' if args.point_counts else 'observation.json',{'case_metadata':capture.project(case.metadata),'outcome':outcome,'direct_source_order':capture.materialize(staged)})
            completion_error = _complete_sample_control(args, iteration, primary, operation_error, record)
            if completion_error is not None:raise completion_error
            if primary is not None:raise primary
            if args.capture_only and not args.point_counts:break
    except BaseException as error:
        record['collector_error']={'type':type(error).__module__+'.'+type(error).__qualname__,'message':str(error),'traceback':traceback.format_exc()}
        primary=error
    finally:
        if case is not None:
            try:case.close();record['run_closed']=True
            except BaseException as close_error:
                record['close_error']={'type':type(close_error).__module__+'.'+type(close_error).__qualname__,'message':str(close_error)}
                if primary is None:primary=close_error
        JaxBackend.execute_native = original_execute
        signal.signal(signal.SIGTERM, previous_term)
        jax.monitoring.unregister_event_listener(compilation_event)
        jax.monitoring.unregister_event_duration_listener(duration_event)
        record['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        record['machine_after']=machine()
        if args.installed_wheel_only:
            record['scnsim_module_origins']={name:str(Path(module.__file__).resolve())
                for name,module in tuple(sys.modules.items())
                if (name=='scnsim' or name.startswith('scnsim.')) and getattr(module,'__file__',None)}
        (out/'worker.json').write_text(json.dumps(record,indent=2)+'\n')
    return 1 if primary is not None else 0


def _summary(values):
    if not values:
        return {"count":0,"samples_ns":[],"median_ns":None,"q1_ns":None,"q3_ns":None,"iqr_ns":None}
    quartiles=statistics.quantiles(values,n=4,method='inclusive') if len(values)>1 else [values[0]]*3
    return {'count':len(values),'samples_ns':values,'median_ns':statistics.median(values),'q1_ns':quartiles[0],'q3_ns':quartiles[2],'iqr_ns':quartiles[2]-quartiles[0]}


def run(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    records=[];cold=[];warm=[];warm_compute=[];warm_no_native=[];optimization_calls=[]
    base=[sys.executable,str(Path(__file__).resolve()),'--worker','--case',args.case,'--operation',args.operation,
          '--storage',args.storage,'--points',str(args.points),'--chunk-size',str(args.chunk_size),
          '--cma-evaluations',str(args.cma_evaluations),'--source-commit',args.source_commit]
    if args.cpu_threads is not None:base+=['--cpu-threads',str(args.cpu_threads)]
    if args.installed_wheel_only:base+=['--installed-wheel-only']
    if args.tail_failure:base+=['--tail-failure']
    if args.point_counts:base+=['--point-counts', *map(str,args.point_counts)]
    env=os.environ.copy();env['JAX_LOG_COMPILES']='1'
    for index in range(args.repetitions+1):
        target=root/('capture' if index==args.repetitions else 'timing-'+str(index))
        argv=base+['--output',str(target),'--warm-repetitions',str(args.repetitions if index==0 else 0)]
        if index==args.repetitions:argv+=['--capture-only']
        with (root/(target.name+'.stdout')).open('wb') as stdout, (root/(target.name+'.stderr')).open('wb') as stderr:
            child = None
            primary = None
            process = {'argv':argv, 'state':'not_started'}
            try:
                child = subprocess.Popen(argv,env=env,stdout=stdout,stderr=stderr)
                process.update(pid=child.pid, state='running', returncode=None, waited=False)
                records.append(process)
                (root/'processes.json').write_text(json.dumps(records,indent=2)+'\n')
                rc = child.wait()
            except BaseException as error:
                primary = error
                process['primary_exception'] = {'type':type(error).__module__+'.'+type(error).__qualname__, 'message':str(error)}
                raise
            finally:
                if child is not None:
                    secondary = []
                    try:
                        if child.poll() is None:
                            child.terminate()
                    except BaseException as termination_error:
                        secondary.append(('termination', termination_error))
                    # Reaping is independent: a failed termination request does
                    # not relinquish ownership of the spawned child.
                    try:
                        rc = child.wait()
                        process.update(state='terminal', returncode=rc, waited=True)
                    except BaseException as reap_error:
                        secondary.append(('reap', reap_error))
                    process['cleanup_errors'] = [
                        {'phase':phase, 'type':type(error).__module__+'.'+type(error).__qualname__, 'message':str(error)}
                        for phase, error in secondary
                    ]
                    try:
                        (root/'processes.json').write_text(json.dumps(records,indent=2)+'\n')
                    except BaseException as persistence_error:
                        secondary.append(('persistence', persistence_error))
                    retained = primary if primary is not None else (secondary[0][1] if secondary else None)
                    if retained is not None:
                        for phase, error in secondary:
                            if error is not retained:
                                try:
                                    retained.add_note('Secondary benchmark '+phase+' error: '+type(error).__module__+'.'+type(error).__qualname__+': '+str(error))
                                except BaseException:
                                    pass  # Diagnostic notes must not replace the owned primary error.
                        if primary is None:
                            raise retained
        if rc:return rc
        worker=json.loads((target/'worker.json').read_text())
        if index<args.repetitions:
            optimization_calls.extend(dict(call['optimization'], sample='cold' if call['iteration']==0 else 'warm', worker=target.name)
                                      for call in worker['calls'] if 'optimization' in call)
            cold.append(worker['calls'][0]['elapsed_ns'])
            warm.extend(call['elapsed_ns'] for call in worker['calls'][1:] if not args.point_counts)
            warm_compute.extend(call['elapsed_ns'] for call in worker['calls'][1:] if call['native_execute_calls']>0 and not args.point_counts)
            warm_no_native.extend(call['elapsed_ns'] for call in worker['calls'][1:] if call['native_execute_calls']==0 and not args.point_counts)
    summary={'input':vars(args),'cold':_summary(cold),'warm':_summary(warm),
             'warm_compute':_summary(warm_compute),'warm_no_native_dispatch':_summary(warm_no_native),
             'machine':machine(),'processes':records,'timing_note':'Public operation wall time with forwarding native counter; separate certificate-capture operation excluded. Cold fresh interpreter; warm_compute requires actual native dispatch. Any warm_no_native_dispatch samples remain separate, never compute claims.',
             'compile_events':[json.loads((root/('timing-'+str(i))/'worker.json').read_text())['compilation_events'] for i in range(args.repetitions)]}
    if args.point_counts:
        summary['same_run_grid_reuse'] = [json.loads((root/('timing-'+str(i))/'worker.json').read_text())['calls'] for i in range(args.repetitions)]
    if args.operation in ('cma', 'lbfgsb'):
        summary['optimization_executions'] = optimization_calls
        summary['comparison_metric'] = 'mean public API wall nanoseconds per actual evaluation'
        summary['optimization_per_evaluation'] = {}
        for label in ('cold', 'warm'):
            rows = [x for x in optimization_calls if x['sample']==label and x['api_wall_mean_ns_per_evaluation'] is not None]
            summary['optimization_per_evaluation'][label] = dict(
                per_execution_means=_summary([x['api_wall_mean_ns_per_evaluation'] for x in rows]),
                pooled_mean_ns_per_evaluation=sum(x['api_wall_ns'] for x in rows)/sum(x['evaluation_count'] for x in rows) if rows else None,
                unavailable_execution_count=sum(x['sample']==label for x in optimization_calls)-len(rows))
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True);parser.add_argument('--operation',required=True)
    parser.add_argument('--storage',choices=('memory','workspace'),default='memory')
    parser.add_argument('--point-counts',type=int,nargs='+');parser.add_argument('--points',type=int,default=401);parser.add_argument('--chunk-size',type=int,default=64)
    parser.add_argument('--cma-evaluations',type=int,choices=(64,512),default=64)
    parser.add_argument('--tail-failure',action='store_true');parser.add_argument('--repetitions',type=int,default=5)
    parser.add_argument('--output',required=True);parser.add_argument('--source-commit',default='8346af0')
    parser.add_argument('--cpu-threads',type=int,default=10);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--warm-repetitions',type=int,default=0);parser.add_argument('--capture-only',action='store_true')
    parser.add_argument('--jax-trace-output',help='JAX trace directory for this worker only')
    parser.add_argument('--installed-wheel-only',action='store_true',help='require an ordinary installed wheel and unset PYTHONPATH')
    parser.add_argument('--sample-control-fds',type=int,nargs=2,metavar=('EVENTS_FD','RELEASE_FD'),help='maintainer worker handshake outside timed calls')
    args=parser.parse_args(argv)
    if args.sample_control_fds is not None and (not args.worker or args.capture_only):
        parser.error('--sample-control-fds requires a timing worker')
    if args.jax_trace_output is not None and not args.worker:
        parser.error('--jax-trace-output requires --worker')
    return run_worker(args) if args.worker else run(args)


if __name__=='__main__':
    raise SystemExit(main())
