"""Persistent ordered candidate actors; coordinator remains state authority.

Candidate actors share the operation resource gate and ready-batch broker.
The coordinator alone owns ordered failures, state publication and cancellation.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, CancelledError
import os
from time import perf_counter_ns
from collections.abc import Iterator

from .candidate_worker import _codec, create_backend_factory
from .resources import current_operation_resources, _secondary


def _identity(document):
    return (document['operation_id'], document['generation'], document['population_column'], document['parameter_digest'])


def _completed(function, *args, **kwargs):
    future = Future()
    try:
        future.set_result(function(*args, **kwargs))
    except BaseException as error:
        future.set_exception(error)
    return future


def _drain_executor(executor, futures):
    """Future completion proves native return before joins/restoration."""
    first = None
    executor.shutdown(wait=False, cancel_futures=True)
    for future in futures:
        while not future.done():
            try:
                future.result()
            except CancelledError:
                break
            except BaseException as error:
                if first is None:
                    first = error
    while True:
        try:
            executor.shutdown(wait=True)
            break
        except (KeyboardInterrupt, SystemExit) as error:
            if first is None:
                first = error
    if first is not None:
        raise first


class _ThreadActor:
    def __init__(self, pool, actor_id, *, serial=False):
        self.pool, self.actor_id, self.serial = pool, actor_id, serial
        self.executor = None if serial else ThreadPoolExecutor(max_workers=1,
            thread_name_prefix=f'scnsim-actor-{actor_id}', initializer=pool.resource._initialize_worker)
        self.futures = []
        self.actor = None
        try:
            self.ready = self.submit('initialize', None).result()
        except BaseException as original:
            try: self.close()
            except BaseException as secondary: _secondary(original,'Actor initialization cleanup also failed',secondary)
            raise

    def _handle(self, kind, payload, generation):
        if kind == 'initialize':
            from .candidate_state import create_actor
            self.actor = create_actor(self.pool.declaration_bytes,
                backend_factory=create_backend_factory(self.pool.declaration_bytes,
                    operation_resources=self.pool.resource), emit=None)
            return {'backend_identity': self.actor.backend_identity, 'pid': os.getpid()}
        if kind == 'prepare':
            return self.actor.prepare(payload, generation=generation)
        if kind == 'evaluate':
            return self.actor.evaluate(payload)
        if kind == 'anchors':
            return self.actor.install_anchors(payload)
        if kind == 'release':
            return self.actor.release_generation(generation)
        if kind == 'close':
            return None if self.actor is None else self.actor.close()
        raise RuntimeError(f'unknown candidate actor command {kind}')

    def submit(self, kind, payload, generation=None):
        def invoke():
            identity = f'actor:{self.actor_id}:{kind}'
            if payload is not None and kind in ('prepare', 'evaluate'):
                identity = repr(_identity(_codec()[1](payload)))
            return self.pool.resource._invoke(lambda _: self._handle(kind,payload,generation), None, identity)
        future = _completed(invoke) if self.serial else self.executor.submit(invoke)
        self.futures.append(future)
        return future

    def close(self):
        for future in self.futures:
            future.cancel()
        first = None
        # Close is queued after all currently running work; no force stop.
        try:
            self.submit('close', None).result()
        except BaseException as error:
            first = error
        if self.executor is not None:
            try:
                _drain_executor(self.executor, self.futures)
            except BaseException as error:
                if first is None: first = error
                else: _secondary(first,'Candidate actor join also failed',error)
        if first is not None:
            raise first


class CandidatePool:
    """One operation's persistent actors and exact ordered byte handoffs."""
    def __init__(self, declaration_bytes: bytes, *, capacity: int):
        self.declaration_bytes = declaration_bytes
        self.declaration = _codec()[1](declaration_bytes)
        self.capacity = min(capacity,self.declaration['population_size'])
        self.resource = current_operation_resources()
        if self.resource is None:
            raise RuntimeError('CandidatePool requires the owning operation resource lease')
        self.actors = []
        self.residency = {}
        self.anchors = None
        self.closed = False

    def __enter__(self):
        if self.capacity > 1:
            self.resource.assembly_batching = True
        try:
            self._ensure_actors(1)
        except BaseException as original:
            try: self.close()
            except BaseException as secondary: _secondary(original,'Candidate initialization cleanup also failed',secondary)
            raise
        return self

    def _ensure_actors(self, count):
        while len(self.actors) < count:
            number = len(self.actors)
            actor = _ThreadActor(self,number,serial=self.capacity<=1)
            self.actors.append(actor)
            if self.anchors is not None:
                actor.submit('anchors',self.anchors).result()

    def _ordered_results(self, submitted, dispatch_error=None):
        """Yield one outcome before touching the next ordered Future.

        Encoded semantic failures belong to the parent. A later native/IPC or
        dispatch error cannot overtake its classification of an earlier reply.
        """
        try:
            for future in submitted:
                started = perf_counter_ns()
                try:
                    reply = future.result()
                finally:
                    self.resource.statistic('ordered_collection_wall_ns',perf_counter_ns()-started)
                yield reply
            if dispatch_error is not None:
                raise dispatch_error
        except BaseException as original:
            try: self.close(cancel=True)
            except BaseException as secondary: _secondary(original,'Candidate pool cleanup also failed',secondary)
            raise
        finally:
            # Completed replies belong to parent consumers, not actor history.
            for actor in self.actors:
                actor.futures[:] = [future for future in actor.futures if not future.done()]

    def _collect(self, submitted):
        # Control commands carry no encoded candidate outcomes.
        return tuple(self._ordered_results(submitted))

    def prepare_candidates(self, points: tuple[bytes,...], *, generation: int) -> Iterator[bytes]:
        if self.closed: raise RuntimeError('candidate pool is closed')
        if generation != 0: self._ensure_actors(self.capacity)
        submitted = []
        dispatch_error = None
        start = perf_counter_ns()
        try:
            for index, point in enumerate(points):
                document = _codec()[1](point)
                identity = _identity(document)
                if identity[0] != self.declaration['operation_id'] or identity[1] != generation:
                    raise RuntimeError('candidate preparation routing conflicts with operation/generation')
                actor = self.actors[index % len(self.actors)]
                self.residency[identity] = actor
                submitted.append(actor.submit('prepare',point,generation))
        except BaseException as error:
            dispatch_error = error
        replies = self._ordered_results(submitted,dispatch_error)
        try:
            # Advance owned replies first so final yield is resumed to normal
            # exhaustion; early parent abort still closes/cancels the pool.
            for reply,point in zip(replies,points):
                if _identity(_codec()[1](reply)) != _identity(_codec()[1](point)):
                    raise RuntimeError('candidate preparation reply routing mismatch')
                self.resource.statistic('prepared_candidates')
                yield reply
        finally:
            replies.close()
            self.resource.statistic('preparation_wall_ns',perf_counter_ns()-start)

    def evaluate_wave(self, commands: tuple[bytes,...]) -> Iterator[bytes]:
        if self.closed: raise RuntimeError('candidate pool is closed')
        submitted=[]
        dispatch_error=None
        try:
            for command in commands:
                identity = _identity(_codec()[1](command))
                actor = self.residency[identity]
                submitted.append(actor.submit('evaluate',command))
        except BaseException as error:
            dispatch_error=error
        replies=self._ordered_results(submitted,dispatch_error)
        try:
            # Owned replies must reach normal exhaustion before close().
            for reply,command in zip(replies,commands):
                expected,observed = _codec()[1](command),_codec()[1](reply)
                if _identity(observed) != _identity(expected) or any(
                    observed[key] != expected[key] for key in ('objective_index','term_index')):
                    raise RuntimeError('candidate numerical reply routing mismatch')
                actor = self.residency[_identity(expected)]
                self.resource.statistic(f'numeric_reply_actor_{actor.actor_id}_count')
                yield reply
        finally:
            replies.close()

    def install_anchors(self, anchor_bytes: bytes) -> None:
        self.anchors=anchor_bytes
        self._collect([actor.submit('anchors',anchor_bytes) for actor in self.actors])

    def release_generation(self,generation: int) -> None:
        self._collect([actor.submit('release',None,generation) for actor in self.actors])
        self.residency={key:value for key,value in self.residency.items() if key[1]!=generation}

    def snapshot_statistics(self):
        result=self.resource.snapshot_statistics()
        result['actor_readiness']=[dict(actor_id=i, **actor.ready) for i,actor in enumerate(self.actors)]
        result['candidate_capacity']=self.capacity
        result['scheduler_id']='scnsim.candidate-thread-ready-batch.v1'
        return result

    def close(self,*,cancel=False):
        if self.closed:return
        self.closed=True
        if cancel:
            self.resource.stop_assembly(CancelledError("candidate operation interrupted or failed"))
        first=None
        # Cancel every queued actor command before any actor join. Running work
        # still drains to actual native return; no process kill/replay.
        for actor in self.actors:
            for future in actor.futures: future.cancel()
        for actor in self.actors:
            try:actor.close()
            except BaseException as error:
                if first is None:first=error
                else:_secondary(first,'Another candidate actor cleanup also failed',error)
        self.residency.clear()
        self.resource.assembly_batching=False
        if first is not None:raise first

    def __exit__(self,kind,original,tb):
        try:self.close(cancel=original is not None)
        except BaseException as error:
            if original is None:raise
            _secondary(original,'Candidate pool cleanup also failed',error)
