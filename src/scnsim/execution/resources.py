"""Operation-owned candidate threads and managed numerical resource allocation.

The coordinator alone owns the process operation lease and BLAS restoration.
Workers receive this context, keep mutable handles thread-local, and never write
Workspace state. Shared candidate work excludes exclusive JAX assembly; the
existing process JAX pool is unchanged. This is not a Notebook-wide CPU quota.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition, RLock, local

_OPERATION_LOCK = RLock()


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
        self._counts = dict(assembly_calls=0, new_shape_count=0, executable_cache_hits=0)

    def _initialize_worker(self):
        if self._controller is not None:
            # Keep the limiter alive for this worker; do not nest restoring
            # contexts around jobs. The coordinator restores after all joins.
            self._local.blas_limit = self._controller.limit(limits=1, user_api='blas')

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
            self._gate.acquire(False)
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
            self._gate.acquire(True)
            acquired = True
            yield
        except BaseException as error:
            original = error
            raise
        finally:
            try:
                if acquired:
                    self._gate.release(True)
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
            yield resource
        except BaseException as error:
            original = error
            raise
        finally:
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
