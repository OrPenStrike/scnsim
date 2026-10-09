"""Operation-owned candidate threads and managed numerical resource allocation.

The coordinator alone owns the process operation lease and BLAS restoration.
Workers receive this context, keep mutable handles thread-local, and never write
Workspace state. Shared candidate work excludes exclusive JAX assembly; the
existing process JAX pool is unchanged. This is not a Notebook-wide CPU quota.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition, RLock, Thread, local
from time import perf_counter_ns

_OPERATION_LOCK = RLock()
_OPERATION_CONTEXT = local()

def current_operation_resources():
    return getattr(_OPERATION_CONTEXT, "resource", None)


@dataclass
class CandidateContext:
    identity: str
    diagnostics: list = field(default_factory=list)

    def emit(self, kind, payload):
        self.diagnostics.append((kind, dict(payload)))


def _secondary(original, message, error):
    """Cleanup/diagnostic failure must not replace an active native exception."""
    try:
        original.add_note(f'{message}: {error!r}')
    except BaseException:
        pass


class _NumericalGate:
    """Queued readers may share capacity; no reader bypasses an earlier writer."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.condition = Condition()
        self.queue = deque()
        self.shared = 0
        self.exclusive = False

    def acquire(self, exclusive):
        request = (object(), exclusive)
        with self.condition:
            self.queue.append(request)
            try:
                while True:
                    earlier = []
                    for queued in self.queue:
                        if queued is request:
                            break
                        earlier.append(queued)
                    ready = (not self.exclusive and self.shared == 0 and not earlier) if exclusive else (
                        not self.exclusive and self.shared < self.capacity
                        and not any(item[1] for item in earlier))
                    if ready:
                        self.queue.remove(request)
                        if exclusive:
                            self.exclusive = True
                        else:
                            self.shared += 1
                        self.condition.notify_all()
                        return
                    self.condition.wait()
            except BaseException:
                self.queue.remove(request)
                self.condition.notify_all()
                raise

    def release(self, exclusive):
        with self.condition:
            if exclusive:
                self.exclusive = False
            else:
                self.shared -= 1
            self.condition.notify_all()


class OperationResources:
    """One coordinator lifetime; one lazy persistent pool across generation waves."""
    def __init__(self, cpu_threads, controller=None):
        self.cpu_threads = cpu_threads
        self.worker_capacity = 1 if cpu_threads is None else cpu_threads
        self._controller = controller
        self._gate = _NumericalGate(self.worker_capacity)
        self._local = local()
        self._pool = None
        self._futures = []
        self._closed = False
        self._worker_states = []
        self._states_lock = RLock()
        self._broker = None
        self.assembly_batching = False
        self._statistics = {}
        self._counts = dict(assembly_calls=0, new_shape_count=0, executable_cache_hits=0)

    def _initialize_worker(self):
        if self._controller is not None:
            # Keep the limiter alive for this worker; do not nest restoring
            # contexts around jobs. The coordinator restores after all joins.
            self._local.blas_limit = self._controller.limit(limits=1, user_api='blas')

    def statistic(self, name, amount=1):
        with self._states_lock:
            self._statistics[name] = self._statistics.get(name, 0) + amount

    def snapshot_statistics(self):
        with self._states_lock:
            return dict(self._statistics, **self.counters())

    @contextmanager
    def phase(self, name):
        started = perf_counter_ns()
        try:
            yield
        finally:
            self.statistic(name + '_wall_ns', perf_counter_ns() - started)
            self.statistic(name + '_count')

    def stop_assembly(self, error):
        with self._states_lock:
            broker = self._broker
        if broker is not None:
            broker._fail(error)

    def assemble(self, request, execute_group):
        # Waiting for the broker must never retain the shared permit it needs.
        suspended = getattr(self._local, 'shared_owned', False)
        if suspended:
            self._local.shared_owned = False
            self._gate.release(False)
        original = None
        try:
            with self._states_lock:
                if self._broker is None:
                    self._broker = AssemblyBroker(self)
                broker = self._broker
            return broker.submit(request, execute_group).result()
        except BaseException as error:
            original = error
            raise
        finally:
            if suspended:
                try:
                    started = perf_counter_ns()
                    self._gate.acquire(False)
                    self.statistic('shared_reacquisition_wait_ns', perf_counter_ns() - started)
                    self.statistic('shared_reacquisition_count')
                    self._local.shared_owned = True
                except BaseException as error:
                    if original is None:
                        raise
                    _secondary(original, 'Shared numerical reacquisition also failed', error)

    def current_candidate(self):
        return getattr(self._local, 'candidate', None)

    def worker_state(self, key, factory):
        states = getattr(self._local, 'states', None)
        if states is None:
            states = {}
            self._local.states = states
            with self._states_lock:
                self._worker_states.append(states)
        if key not in states:
            states[key] = factory()
        return states[key]

    @contextmanager
    def candidate_compute(self):
        depth = getattr(self._local, 'depth', 0)
        if depth == 0:
            started = perf_counter_ns()
            self._gate.acquire(False)
            self.statistic("shared_wait_ns", perf_counter_ns() - started)
            self.statistic("shared_acquisition_count")
            self._local.shared_owned = True
        self._local.depth = depth + 1
        try:
            yield
        finally:
            self._local.depth = depth
            if depth == 0 and self._local.shared_owned:
                self._local.shared_owned = False
                self._gate.release(False)

    @contextmanager
    def exclusive_assembly(self):
        suspended = getattr(self._local, 'shared_owned', False)
        if suspended:
            self._local.shared_owned = False
            self._gate.release(False)
        acquired = False
        original = None
        try:
            started = perf_counter_ns()
            self._gate.acquire(True)
            self.statistic("exclusive_wait_ns", perf_counter_ns() - started)
            self.statistic("exclusive_acquisition_count")
            acquired = True
            held_start = perf_counter_ns()
            yield
        except BaseException as error:
            original = error
            raise
        finally:
            try:
                if acquired:
                    self._gate.release(True)
                    self.statistic("exclusive_held_ns", perf_counter_ns() - held_start)
            except BaseException as error:
                if original is None:
                    raise
                _secondary(original, 'Exclusive numerical release also failed', error)
            finally:
                if suspended:
                    try:
                        self._gate.acquire(False)
                        self._local.shared_owned = True
                    except BaseException as error:
                        if original is None:
                            raise
                        _secondary(original, 'Shared numerical reacquisition also failed', error)

    def record_assembly(self, *, new_shape):
        with self._gate.condition:
            self._counts['assembly_calls'] += 1
            self._counts['new_shape_count'] += int(new_shape)
            self._counts['executable_cache_hits'] += int(not new_shape)

    def counters(self):
        with self._gate.condition:
            return dict(self._counts)

    def _invoke(self, function, item, identity):
        context = CandidateContext(identity)
        previous = self.current_candidate()
        self._local.candidate = context
        try:
            with self.candidate_compute():
                return function(item)
        except BaseException as original:
            try:
                original.scnsim_candidate_diagnostics = tuple(context.diagnostics)
            except BaseException as secondary:
                _secondary(original, 'Candidate diagnostic attachment also failed', secondary)
            raise
        finally:
            self._local.candidate = previous

    def map_ordered(self, function, items, *, identity):
        if self._closed:
            raise RuntimeError('operation resource context is closed')
        if self.worker_capacity == 1:
            for item in items:
                yield self._invoke(function, item, identity(item))
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.worker_capacity,
                thread_name_prefix='scnsim-candidate', initializer=self._initialize_worker)
        futures = []
        try:
            for item in items:
                future = self._pool.submit(self._invoke, function, item, identity(item))
                futures.append(future)
                self._futures.append(future)
            for future in futures:
                yield future.result()
            self._futures.clear()
        except BaseException as original:
            for future in futures:
                future.cancel()
            try:
                self.close(cancel=True)
            except BaseException as secondary:
                _secondary(original, 'Candidate pool cleanup also failed', secondary)
            raise

    def close(self, *, cancel=False):
        first_error = None
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=cancel)
            except BaseException as error:
                first_error = error
            # Completion Futures, rather than interrupted Thread.join state,
            # establish that native calls have ended before restoring BLAS.
            for future in self._futures:
                while not future.done():
                    try:
                        future.result()
                    except CancelledError:
                        break
                    except BaseException as error:
                        if first_error is None:
                            first_error = error
                        if future.done():
                            break
            while True:
                try:
                    self._pool.shutdown(wait=True, cancel_futures=cancel)
                    break
                except (KeyboardInterrupt, SystemExit) as error:
                    if first_error is None:
                        first_error = error
            self._pool = None
        if self._broker is not None:
            try:
                self._broker.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                else:
                    _secondary(first_error, "Assembly broker cleanup also failed", error)
            self._broker = None
        self._futures.clear()
        self._closed = True
        for states in self._worker_states:
            states.clear()
        self._worker_states.clear()
        if first_error is not None:
            raise first_error


@contextmanager
def operation_resources(*, cpu_threads):
    """Serialize SCNSim operations; restore native limits only after pool join."""
    with _OPERATION_LOCK:
        # Discover all currently used numerical runtimes before one controller
        # snapshots their settings; importing does not initialize a JAX device.
        import jax
        import numpy
        import scipy.linalg
        import scipy.sparse.linalg
        import cmaes
        from threadpoolctl import ThreadpoolController

        controller = None if cpu_threads is None else ThreadpoolController()
        # Snapshot each native library before limit() can mutate any of them.
        # Retain controllers individually: prefix-based restoration can conflate
        # two loaded libraries with different original limits.
        original_limits = () if controller is None else tuple(
            (library, library.num_threads)
            for library in controller.lib_controllers if library.user_api == 'blas')
        resource = OperationResources(cpu_threads, controller)
        original = None
        try:
            if controller is not None:
                controller.limit(limits=1, user_api='blas')
            _OPERATION_CONTEXT.resource = resource
            yield resource
        except BaseException as error:
            original = error
            raise
        finally:
            _OPERATION_CONTEXT.resource = None
            try:
                resource.close(cancel=original is not None)
            except BaseException as error:
                if original is None:
                    original = error
                    raise
                _secondary(original, 'Operation resource cleanup also failed', error)
            finally:
                restoration_error = None
                for library, num_threads in original_limits:
                    try:
                        library.set_num_threads(num_threads)
                    except BaseException as error:
                        if original is not None:
                            _secondary(original, 'BLAS restoration also failed', error)
                        elif restoration_error is None:
                            restoration_error = error
                        else:
                            _secondary(restoration_error, 'Another BLAS restoration also failed', error)
                if restoration_error is not None:
                    raise restoration_error


class AssemblyBroker:
    """Finite ready snapshot after exclusive acquisition; no fill barrier/padding."""
    def __init__(self, resource):
        self.resource = resource
        self.condition = Condition()
        self.queue = []
        self.active_snapshot = ()
        self.stopping = False
        self.failure = None
        self.thread = Thread(target=self._run, name='scnsim-assembly', daemon=False)
        self.thread.start()

    @staticmethod
    def _failure_copy(error):
        # Future consumers attach their own candidate diagnostics/traceback.
        from copy import copy
        try:
            return copy(error)
        except BaseException:
            cloned = RuntimeError(f'Assembly broker failed: {type(error).__name__}: {error}')
            cloned.__cause__ = error
            return cloned

    def submit(self, request, execute_group):
        future = Future()
        with self.condition:
            if self.failure is not None:
                future.set_exception(self._failure_copy(self.failure))
            elif self.stopping:
                future.set_exception(RuntimeError('assembly broker is closed'))
            else:
                self.queue.append((request, execute_group, future))
                self.condition.notify_all()
        return future

    def _fail(self, error, snapshot=()):
        with self.condition:
            self.failure = error
            self.stopping = True
            pending, self.queue = self.queue, []
            active = self.active_snapshot
            self.condition.notify_all()
        for _, _, future in (*snapshot, *active, *pending):
            if not future.done():
                future.set_exception(self._failure_copy(error))

    def _run(self):
        snapshot = []
        completed = []
        try:
            self.resource._initialize_worker()
            while True:
                with self.condition:
                    while not self.queue and not self.stopping:
                        self.condition.wait()
                    if not self.queue and self.stopping:
                        return
                completed = []
                with self.resource.exclusive_assembly():
                    # Include arrivals during the gate wait; snapshot is finite.
                    with self.condition:
                        snapshot, self.queue = self.queue, []
                        self.active_snapshot = tuple(snapshot)
                    groups = {}
                    for item in snapshot:
                        groups.setdefault(item[0]['compatibility'], []).append(item)
                    for items in groups.values():
                        try:
                            outputs = items[0][1]([item[0] for item in items])
                            if len(outputs) != len(items):
                                raise RuntimeError('assembly batch changed request count')
                        except BaseException as error:
                            # This group's numerical call owns its error. Other
                            # candidates may still need subsequent Newton calls;
                            # only ordered coordinator consumption cancels them.
                            completed.extend((item[2], None, error) for item in items)
                        else:
                            completed.extend((item[2], output, None) for item, output in zip(items, outputs))
                        self.resource.statistic('ready_batch_count')
                        self.resource.statistic('ready_batch_items', len(items))
                        self.resource.statistic('ready_batch_size_' + str(len(items)))
                # Exclusive is released before any worker becomes runnable.
                for future, output, error in completed:
                    if not future.done():
                        if error is None: future.set_result(output)
                        else: future.set_exception(self._failure_copy(error))
                with self.condition:
                    self.active_snapshot = ()
                snapshot = []
        except BaseException as error:
            # Infrastructure failure is broker-wide. Preserve already owned
            # group outcomes after exclusive unwind before failing other work.
            for future, output, group_error in completed:
                if not future.done():
                    if group_error is None: future.set_result(output)
                    else: future.set_exception(self._failure_copy(group_error))
            self._fail(error, snapshot)

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        original = None
        while self.thread.is_alive():
            try:
                self.thread.join()
            except (KeyboardInterrupt, SystemExit) as error:
                if original is None:
                    original = error
                self._fail(error)
        if original is not None:
            raise original
