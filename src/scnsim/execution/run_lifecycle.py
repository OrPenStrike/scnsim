"""Atomic Run authorization, separate from operation resources and storage locks.

Activity includes preparation, callbacks, decoding and cleanup. The short state
lock never surrounds those operations; close rejects active work, never cancels.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from threading import Lock

from ..errors import SCNSimStateError


class RunLifecycle:
    __slots__ = ('_lock', '_active', '_closed')

    def __init__(self):
        self._lock = Lock()
        self._active = 0
        self._closed = False

    @contextmanager
    def activity(self):
        with self._lock:
            if self._closed:
                raise SCNSimStateError('CircuitRun is closed', stage='run_lifecycle')
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1

    def close(self, release):
        with self._lock:
            if self._closed:
                return
            if self._active:
                raise SCNSimStateError('CircuitRun has an active operation', stage='run_lifecycle')
            self._closed = True
        # Even failed release cannot reopen authorization.
        release()


def run_activity(method):
    """Register a facade call without a closure retaining any Run instance."""
    @wraps(method)
    def authorized(self, *args, **kwargs):
        with self._lifecycle.activity():
            return method(self, *args, **kwargs)
    return authorized
