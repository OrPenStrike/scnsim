"""Explicit operation-scoped native execution without a live Run or Workspace.

One guardian owns startup probes and native work. Kernel liveness is separate
from native stdin/ACK; only the Kernel retains its write end. The stable lease
is duplicated into the guard and released by descriptor close after OS drain.
No global subprocess patch, preexec callback, or permanent daemon is installed.
"""
from __future__ import annotations
import builtins
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from .native_guard import receive_record, send_record, restore_error


def _retain_error(original, secondary, label):
    if original is None:
        return secondary
    original.add_note(f'{label}: {secondary!r}')
    return original


def _complete_wait(process, original=None):
    # Interrupted close still owns/drains native work before propagating the
    # interruption; it must not mark the operation closed while guard is alive.
    while True:
        try:
            process.wait()
            return original
        except (KeyboardInterrupt, SystemExit) as error:
            original = _retain_error(original, error, 'Native drain interruption')
        except BaseException as error:
            return _retain_error(original, error, 'Native drain failure')


class NativeProcess:
    """Popen-compatible stdio/termination handle; guardian owns native reaping."""
    def __init__(self, supervisor, argv, pid, streams):
        self._supervisor = supervisor
        self.args, self.pid = argv, pid
        self.stdin, self.stdout, self.stderr = streams
        self.returncode = None
        self._readers = None
        self._outputs = [None, None]
        self._reader_errors = []

    def poll(self):
        if self.returncode is None:
            self.returncode = self._supervisor._request({'action': 'poll', 'pid': self.pid})['returncode']
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(.02)
        return self.returncode

    def send_signal(self, number):
        if self.poll() is None:
            self._supervisor._request({'action': 'signal', 'pid': self.pid, 'signal': number})

    def terminate(self): self.send_signal(signal.SIGTERM)
    def kill(self): self.send_signal(signal.SIGKILL)

    def communicate(self, input=None, timeout=None):
        if input is not None and self.stdin is None:
            raise ValueError('stdin must be PIPE to send input')
        if self._readers is not None and input is not None:
            raise ValueError('cannot send input after communication started')
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            if self._readers is None:
                self._readers = []
                def io_task(index, stream, writing=False):
                    original = None
                    try:
                        if writing:
                            if input is not None: stream.write(input)
                        else:
                            self._outputs[index] = stream.read()
                    except BrokenPipeError as error:
                        if not writing: original = error
                    except BaseException as error:
                        original = error
                    finally:
                        try: stream.close()
                        except BrokenPipeError as error:
                            if not writing: original = _retain_error(original, error, 'Native IO close')
                        except BaseException as error:
                            original = _retain_error(original, error, 'Native IO close')
                        if original is not None: self._reader_errors.append(original)
                for index, stream in enumerate((self.stdout, self.stderr)):
                    if stream is not None:
                        thread = threading.Thread(target=io_task, args=(index, stream), daemon=True)
                        self._readers.append(thread); thread.start()
                if self.stdin is not None:
                    writer = threading.Thread(target=io_task, args=(0, self.stdin, True), daemon=True)
                    self._readers.append(writer); writer.start()
            drained = False
            while True:
                # Failed text decoding/writing cannot wait for an EOF child.
                # Its stream is closed by the worker; all owned descendants are
                # drained on this fatal exit before returning the original error.
                if self._reader_errors: raise self._reader_errors[0]
                exited = self.poll() is not None
                if exited and not drained:
                    # A root can exit while a detached descendant still owns a
                    # pipe. Drain the owned tree before waiting for stream EOF.
                    self._supervisor.drain()
                    drained = True
                if exited and all(not reader.is_alive() for reader in self._readers):
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(self.args, timeout, *self._outputs)
                time.sleep(.02)
            if self._reader_errors: raise self._reader_errors[0]
            return tuple(self._outputs)
        except BaseException as original:
            try: self._supervisor.drain()
            except BaseException as secondary:
                original.add_note(f'Native communication drain: {secondary!r}')
            raise

    def __enter__(self): return self
    def __exit__(self, kind, error, traceback):
        original = error
        if error is not None:
            try: self._supervisor.drain()
            except BaseException as secondary:
                error.add_note(f'Native process cleanup: {secondary!r}')
            return
        for stream in (self.stdin, self.stdout, self.stderr):
            if stream is not None:
                try: stream.close()
                except BaseException as secondary:
                    original = _retain_error(original, secondary, 'Native stream cleanup')
        try: self.wait()
        except BaseException as secondary:
            original = _retain_error(original, secondary, 'Native process wait')
        try: self._supervisor.drain()
        except BaseException as cleanup:
            original = _retain_error(original, cleanup, 'Native process drain')
        if original is not None: raise original


class NativeSupervisor:
    def __init__(self, *, operation_id, lease_fds=()):
        self.operation_id = operation_id
        self._lock = threading.RLock()
        self._closed = False
        self._broken = False
        self._processes = []
        self._control = None
        self._guard = None
        self._alive_write = None
        parent = child = None
        alive_read = alive_write = None
        leases = []
        def abort_setup(error):
            for resource in (parent, child):
                if resource is not None:
                    try: resource.close()
                    except BaseException as secondary:
                        error.add_note(f'Guardian socket setup cleanup: {secondary!r}')
            if self._alive_write is not None:
                try: os.close(self._alive_write)
                except BaseException as secondary:
                    error.add_note(f'Guardian liveness setup cleanup: {secondary!r}')
                self._alive_write = None
            if self._guard is not None:
                _complete_wait(self._guard, error)

        try:
            parent, child = socket.socketpair()
            self._control = parent
            alive_read, alive_write = os.pipe()
            self._alive_write = alive_write
            for lease_fd in lease_fds:
                # Record each duplicate before acquiring the next one, so a
                # partial tuple duplication cannot strand an earlier lease.
                leases.append(os.dup(lease_fd))
            self._control = parent
            self._alive_write = alive_write
            fds = (child.fileno(), alive_read, *leases)
            self._guard = subprocess.Popen(
                [sys.executable, '-I', '-B', str(Path(__file__).with_name('native_guard.py')),
                 str(child.fileno()), str(alive_read), *map(str, leases)],
                pass_fds=fds, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
            child.close(); os.close(alive_read); alive_read = None
            record, incoming = receive_record(parent)
            for fd in incoming: os.close(fd)
            if 'error' in record:
                raise restore_error(record['error'])
            if record != {'ready': True}:
                raise RuntimeError('native guardian did not establish ownership')
        except BaseException as error:
            abort_setup(error)
            raise
        finally:
            original = sys.exception()
            cleanup_error = None
            for fd in (alive_read, *leases):
                if fd is not None:
                    try: os.close(fd)
                    except BaseException as secondary:
                        if original is None:
                            cleanup_error = _retain_error(cleanup_error, secondary, 'Guardian descriptor setup cleanup')
                        else:
                            original.add_note(f'Guardian descriptor setup cleanup: {secondary!r}')

            if original is None and cleanup_error is not None:
                abort_setup(cleanup_error)
                raise cleanup_error

    def _request(self, command, fds=(), *, timeout_args=None, timeout=None):
        with self._lock:
            if self._closed:
                raise RuntimeError('native supervisor is closed')
            if self._broken:
                raise RuntimeError('native supervisor control was interrupted')
            try:
                send_record(self._control, command, fds)
                record, incoming = receive_record(self._control)
                for fd in incoming: os.close(fd)
            except BaseException as original:
                # A partially consumed frame cannot be reused. Kernel liveness
                # closure forces guardian drain rather than issuing a blind RPC.
                self._broken = True
                if self._alive_write is not None:
                    try: os.close(self._alive_write)
                    except BaseException as secondary:
                        original.add_note(f'Native interrupted transport cleanup: {secondary!r}')
                    self._alive_write = None
                raise
            if 'error' in record:
                raise restore_error(record['error'])
            return record

    def retain_lease(self, fd: int) -> None:
        """Pin an acquired caller lease in the guardian until final drain.

        The caller registers its lease cleanup before this handoff. A failed or
        interrupted acknowledgement does not prove the guardian lacks the FD;
        the outer operation owner must still close/drain its supervisor. Neither
        side unlocks or unlinks the caller's lease inode.
        """
        duplicate = os.dup(fd)
        try:
            os.set_inheritable(duplicate, False)
            answer = self._request({'action': 'retain_lease'}, (duplicate,))
            if answer != {'retained': True}:
                raise RuntimeError('native guardian did not acknowledge lease retention')
        finally:
            original = sys.exception()
            try: os.close(duplicate)
            except BaseException as secondary:
                if original is None: raise
                original.add_note(f'Native lease handoff descriptor cleanup: {secondary!r}')

    def popen(self, argv, *, cwd=None, env=None, stdin=None, stdout=None, stderr=None,
              text=False, encoding=None, errors=None, bufsize=-1, shell=False,
              start_new_session=True, pass_fds=(), _fd_arguments=None):
        if not start_new_session:
            raise ValueError('native supervisor requires operation-owned native process groups')
        if shell:
            raise ValueError('native supervisor requires an explicit argv, not shell=True')
        argv = [os.fspath(item) for item in argv]
        child_fds, opened, streams = [], [], []
        try:
            for index, value in enumerate((stdin, stdout, stderr)):
                if value == subprocess.PIPE:
                    read, write = os.pipe()
                    child_fd, parent_fd = (read, write) if index == 0 else (write, read)
                    opened.append(child_fd)
                    mode = 'w' if index == 0 else 'r'
                    try:
                        stream = os.fdopen(parent_fd, mode if text or encoding is not None else mode + 'b',
                                           buffering=bufsize, **({'encoding': encoding or 'utf-8', 'errors': errors or 'strict'}
                                            if text or encoding is not None else {}))
                    except BaseException:
                        os.close(parent_fd); raise
                    streams.append(stream)
                else:
                    streams.append(None)
                    if value == subprocess.DEVNULL:
                        child_fd = os.open(os.devnull, os.O_RDONLY if index == 0 else os.O_WRONLY)
                        opened.append(child_fd)
                    elif value == subprocess.STDOUT and index == 2:
                        child_fd = child_fds[1]
                    elif value is None:
                        child_fd = index
                    else:
                        child_fd = value if isinstance(value, int) else value.fileno()
                child_fds.append(child_fd)
            fd_arguments = {} if _fd_arguments is None else _fd_arguments
            command = {'action': 'launch', 'argv': argv, 'cwd': None if cwd is None else os.fspath(cwd),
                       'env': dict(os.environ if env is None else env),
                       'fd_arguments': {str(position): 3 + tuple(pass_fds).index(fd)
                                        for position, fd in fd_arguments.items()}}
            record = self._request(command, (*child_fds, *pass_fds))
            process = NativeProcess(self, argv, record['pid'], streams)
            self._processes.append(process)
            return process
        except BaseException as original:
            for stream in streams:
                if stream is not None:
                    try: stream.close()
                    except BaseException as secondary:
                        original.add_note(f'Native stdio setup cleanup: {secondary!r}')
            raise
        finally:
            original = sys.exception()
            for fd in opened:
                try: os.close(fd)
                except BaseException as secondary:
                    if original is None: raise
                    original.add_note(f'Native stdio descriptor cleanup: {secondary!r}')

    def run(self, argv, *, input=None, capture_output=False, timeout=None, check=False, **kwargs):
        if input is not None:
            if kwargs.get('stdin') is not None: raise ValueError('stdin and input arguments may not both be used')
            kwargs['stdin'] = subprocess.PIPE
        if capture_output:
            if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
                raise ValueError('stdout and stderr arguments may not be used with capture_output')
            kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process = self.popen(argv, **kwargs)
        try:
            stdout, stderr = process.communicate(input, timeout)
        except BaseException as error:
            try: self.drain()
            except BaseException as secondary:
                error.add_note(f'Native run cleanup: {secondary!r}')
            raise
        if check and process.returncode:
            raise subprocess.CalledProcessError(process.returncode, argv, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)

    def discover_julia(self, version):
        read, write = os.pipe()
        process = None
        try:
            argv = [sys.executable, '-I', '-B', str(Path(__file__).with_name('native_helper.py')),
                    version, str(write)]
            process = self.popen(argv, stdin=subprocess.DEVNULL,
                                 pass_fds=(write,), _fd_arguments={5: write})
            os.close(write); write = None
            with os.fdopen(read, 'r', encoding='utf-8') as stream:
                read = None
                reply = json.load(stream)
            process.wait()
            if 'error' in reply:
                error = restore_error(reply['error'])
                error._scnsim_discovery_phase = reply['phase']
                raise error
            self.drain()
            return tuple(reply['result'])
        except BaseException as error:
            if process is not None:
                try: self.drain()
                except BaseException as secondary:
                    error.add_note(f'Julia discovery cleanup: {secondary!r}')
            raise
        finally:
            original = sys.exception()
            for fd in (read, write):
                if fd is not None:
                    try: os.close(fd)
                    except BaseException as secondary:
                        if original is None: raise
                        original.add_note(f'Julia discovery descriptor cleanup: {secondary!r}')

    def drain(self):
        """Drain current native work while retaining the guardian and leases.

        Subsequent preparation/inspection launches reuse the same descendant
        tracker. Only broken control forces the owning guardian to close.
        """
        with self._lock:
            if self._closed:
                return 'terminated'
            try:
                answer = self._request({'action': 'drain'})
            except BaseException as original:
                if self._broken:
                    try: self.close()
                    except BaseException as secondary:
                        original.add_note(f'Broken native control drain: {secondary!r}')
                raise
            original = None
            for process in self._processes:
                code = answer['returncodes'].get(str(process.pid))
                if code is not None: process.returncode = code
                if process._readers is not None:
                    for reader in process._readers:
                        while reader.is_alive():
                            try: reader.join()
                            except (KeyboardInterrupt, SystemExit) as secondary:
                                original = _retain_error(original, secondary, 'Native IO drain interruption')
            if original is not None: raise original
            return answer['termination']

    def close(self):
        with self._lock:
            if self._closed: return
            original = None
            try:
                if not self._broken:
                    send_record(self._control, {'action': 'close'})
            except BaseException as error:
                original = error
            # Liveness close is the independent shutdown path if control send
            # failed/interrupted. Cleanup always proceeds through actual drain.
            if self._alive_write is not None:
                try: os.close(self._alive_write)
                except BaseException as error:
                    original = _retain_error(original, error, 'Native liveness close')
                self._alive_write = None
            try: self._control.close()
            except BaseException as error:
                original = _retain_error(original, error, 'Native control close')
            original = _complete_wait(self._guard, original)
            for process in self._processes:
                if process._readers is not None:
                    for reader in process._readers:
                        while reader.is_alive():
                            try: reader.join()
                            except (KeyboardInterrupt, SystemExit) as error:
                                original = _retain_error(original, error, 'Native IO drain interruption')
                            except BaseException as error:
                                original = _retain_error(original, error, 'Native IO drain')
                                break
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None and not stream.closed:
                        try: stream.close()
                        except BaseException as error:
                            original = _retain_error(original, error, 'Native stream cleanup')
            self._processes.clear()
            self._closed = self._guard.returncode is not None
            if original is not None: raise original
            if self._guard.returncode:
                raise RuntimeError(f'native guardian exited with status {self._guard.returncode}')

    def __enter__(self): return self
    def __exit__(self, kind, error, traceback):
        try: self.close()
        except BaseException as cleanup:
            if error is None: raise
            error.add_note(f'Native supervisor cleanup: {cleanup!r}')
