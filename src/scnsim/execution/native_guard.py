"""Operation guardian: descriptor liveness, OS descendant ownership, native drain.

No SQL/checkpoint authority. Linux subreaping owns detached descendants; macOS
kqueue NOTE_TRACK follows forks before the exec barrier opens. Owned processes
are never inferred from an unrelated PID or an age/PID cleanup heuristic.
"""
from __future__ import annotations
import array
import base64
import builtins
import traceback
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
import time


def _exact(sock, size):
    chunks = []
    while size:
        data = sock.recv(size)
        if not data:
            raise EOFError('native guardian control closed')
        chunks.append(data)
        size -= len(data)
    return b''.join(chunks)


def send_record(sock, record, fds=()):
    payload = json.dumps(record).encode('utf-8')
    ancillary = ([(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', fds))]
                 if fds else [])
    sock.sendmsg([b'R'], ancillary)
    sock.sendall(struct.pack('!Q', len(payload)) + payload)


def receive_record(sock):
    fds = []
    try:
        data, ancillary, flags, address = sock.recvmsg(1, socket.CMSG_SPACE(256 * array.array('i').itemsize))
        if not data:
            raise EOFError('native guardian control closed')
        for level, kind, body in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                values = array.array('i')
                values.frombytes(body[:len(body) - len(body) % values.itemsize])
                fds.extend(values)
        for fd in fds:
            os.set_inheritable(fd, False)
        if flags & socket.MSG_CTRUNC:
            raise RuntimeError('native descriptor transfer was truncated')
        length = struct.unpack('!Q', _exact(sock, 8))[0]
        return json.loads(_exact(sock, length)), fds
    except BaseException as original:
        for fd in fds:
            try: os.close(fd)
            except BaseException as secondary:
                original.add_note(f'Native received descriptor cleanup: {secondary!r}')
        raise


class NativeCommunicationError(RuntimeError):
    """A remote failure could not be transported without changing its type."""


def _encode_argument(value):
    if value is None or type(value) in (bool, int, str):
        return {'kind': 'scalar', 'value': value}
    if type(value) is float:
        return {'kind': 'float', 'value': value.hex()}
    if type(value) is bytes:
        return {'kind': 'bytes', 'value': base64.b64encode(value).decode('ascii')}
    if type(value) in (tuple, list):
        return {'kind': 'tuple' if type(value) is tuple else 'list',
                'value': [_encode_argument(item) for item in value]}
    raise NativeCommunicationError(f'Unsupported native exception argument type: {type(value).__module__}.{type(value).__name__}')


def _decode_argument(value):
    kind = value['kind']
    if kind == 'scalar': return value['value']
    if kind == 'float': return float.fromhex(value['value'])
    if kind == 'bytes': return base64.b64decode(value['value'], validate=True)
    if kind in ('tuple', 'list'):
        items = [_decode_argument(item) for item in value['value']]
        return tuple(items) if kind == 'tuple' else items
    raise NativeCommunicationError('Unknown native exception argument encoding')


def _encode_error(error, seen):
    if id(error) in seen:
        raise NativeCommunicationError('Cyclic native exception cause cannot be transported')
    seen.add(id(error))
    cls = type(error)
    if cls.__module__ != 'builtins' or getattr(builtins, cls.__name__, None) is not cls:
        raise NativeCommunicationError(f'Untransportable native exception class: {cls.__module__}.{cls.__name__}')
    record = {'type': cls.__name__, 'args': _encode_argument(error.args),
              'attributes': {name: _encode_argument(getattr(error, name))
                             for name in ('errno', 'strerror', 'filename', 'filename2', 'winerror', 'name', 'path')
                             if hasattr(error, name)},
              'traceback': ''.join(traceback.format_exception(type(error), error, error.__traceback__)),
              'notes': list(getattr(error, '__notes__', ())),
              'suppress_context': error.__suppress_context__,
              'cause': None, 'context': None}
    if error.__cause__ is not None:
        record['cause'] = _encode_error(error.__cause__, seen)
    elif error.__context__ is not None and not error.__suppress_context__:
        record['context'] = _encode_error(error.__context__, seen)
    seen.remove(id(error))
    return record


def error_record(error):
    try:
        return _encode_error(error, set())
    except BaseException as transport:
        return {'transport_error': str(transport),
                'remote_type': f'{type(error).__module__}.{type(error).__name__}',
                'traceback': ''.join(traceback.format_exception(type(error), error, error.__traceback__))}


def restore_error(record):
    if 'transport_error' in record:
        error = NativeCommunicationError(record['transport_error'])
        error.add_note('Original remote failure (' + record['remote_type'] + '):\n' + record['traceback'])
        return error
    try:
        cls = getattr(builtins, record['type'])
        if not isinstance(cls, type) or not issubclass(cls, BaseException):
            raise NativeCommunicationError('Invalid native exception class')
        error = cls(*_decode_argument(record['args']))
        if type(error) is not cls:
            raise NativeCommunicationError('Native exception constructor changed class')
        for name, value in record['attributes'].items():
            setattr(error, name, _decode_argument(value))
        for note in record['notes']: error.add_note(note)
        error.add_note('Managed native traceback:\n' + record['traceback'])
        error.__cause__ = None if record['cause'] is None else restore_error(record['cause'])
        error.__context__ = None if record['context'] is None else restore_error(record['context'])
        error.__suppress_context__ = record['suppress_context']
        return error
    except BaseException as failure:
        error = NativeCommunicationError('Native exception reconstruction failed')
        error.__cause__ = failure
        error.add_note('Original remote traceback:\n' + record.get('traceback', ''))
        return error


class Descendants:
    def __init__(self):
        self.owned = {}
        self.lock = threading.RLock()
        self.failure = None
        self.kqueue = None
        if sys.platform.startswith('linux'):
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
                errno = ctypes.get_errno()
                raise OSError(errno, os.strerror(errno))
        elif sys.platform == 'darwin':
            self.kqueue = select.kqueue()
        else:
            raise RuntimeError('native guardian requires Linux or macOS process ownership')

    @staticmethod
    def _linux_identity(pid):
        try:
            text = Path(f'/proc/{pid}/stat').read_text()
        except FileNotFoundError:
            return None
        fields = text[text.rfind(')') + 2:].split()
        return int(fields[1]), fields[19], fields[0]  # ppid, starttime, state

    def register(self, pid):
        with self.lock:
            if self.kqueue is None:
                identity = self._linux_identity(pid)
                if identity is None:
                    raise ProcessLookupError(pid)
                self.owned[pid] = identity[1]
            else:
                event = select.kevent(pid, filter=select.KQ_FILTER_PROC,
                    flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE,
                    fflags=select.KQ_NOTE_EXIT | select.KQ_NOTE_FORK | select.KQ_NOTE_TRACK)
                self.kqueue.control([event], 0, 0)
                self.owned[pid] = True

    def scan(self):
        with self.lock:
            if self.kqueue is not None:
                while True:
                    events = self.kqueue.control(None, max(1, len(self.owned) + 1), 0)
                    if not events:
                        break
                    for event in events:
                        if event.fflags & select.KQ_NOTE_TRACKERR:
                            raise RuntimeError('native descendant tracking failed')
                        if event.fflags & select.KQ_NOTE_CHILD:
                            self.owned[event.ident] = True
                        if event.fflags & select.KQ_NOTE_EXIT:
                            self.owned.pop(event.ident, None)
                return
            # Direct children include subreaper-adopted double-fork descendants.
            pending = [os.getpid(), *self.owned]
            seen = set()
            while pending:
                pid = pending.pop()
                if pid in seen:
                    continue
                seen.add(pid)
                if pid != os.getpid():
                    identity = self._linux_identity(pid)
                    if identity is None or identity[1] != self.owned.get(pid):
                        continue
                try:
                    children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
                except FileNotFoundError:
                    continue
                for value in children:
                    child = int(value)
                    identity = self._linux_identity(child)
                    if identity is not None:
                        self.owned[child] = identity[1]
                        pending.append(child)
            for pid, token in list(self.owned.items()):
                identity = self._linux_identity(pid)
                if identity is None or identity[1] != token or identity[2] == 'Z':
                    self.owned.pop(pid, None)

    def signal_all(self, number):
        self.scan()
        with self.lock:
            for pid, token in list(self.owned.items()):
                if self.kqueue is None:
                    identity = self._linux_identity(pid)
                    if identity is None or identity[1] != token:
                        continue
                try:
                    os.kill(pid, number)
                except ProcessLookupError:
                    pass

    def drain(self, processes):
        # Preserve existing TERM, five-second grace, KILL policy. Refresh fork
        # ownership while draining; separately-created sessions remain owned.
        self.signal_all(signal.SIGTERM)
        deadline = time.monotonic() + 5
        disposition = "terminated"
        while True:
            for process in processes.values():
                process.poll()
            self.scan()
            if not self.owned:
                break
            if time.monotonic() >= deadline:
                disposition = "killed_after_grace"
                self.signal_all(signal.SIGKILL)
            time.sleep(.02)
        for process in processes.values():
            process.wait()
        # Reap adopted descendants only after Popen has collected its roots.
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
        return disposition

    def close(self):
        if self.kqueue is not None:
            self.kqueue.close()


def launch(command, fds, tracker):
    config_read = config_write = barrier_read = barrier_write = error_read = error_write = None
    process = None
    try:
        argv = list(command['argv'])
        for position, index in command['fd_arguments'].items():
            argv[int(position)] = str(fds[index])
        config_read, config_write = os.pipe()
        barrier_read, barrier_write = os.pipe()
        error_read, error_write = os.pipe()
        # Large environments must not block before the bootstrap reads them.
        config = json.dumps({'argv': argv, 'env': command['env']}).encode('utf-8')
        process = subprocess.Popen(
            [sys.executable, '-I', '-B', str(Path(__file__).with_name('native_bootstrap.py')),
             str(config_read), str(barrier_read), str(error_write)],
            stdin=fds[0], stdout=fds[1], stderr=fds[2], cwd=command['cwd'],
            start_new_session=True,
            pass_fds=(config_read, barrier_read, error_write, *fds[3:]))
        os.close(config_read); config_read = None
        os.close(barrier_read); barrier_read = None
        os.close(error_write); error_write = None
        tracker.register(process.pid)
        with os.fdopen(config_write, 'wb') as stream:
            config_write = None
            stream.write(config)
        os.write(barrier_write, b'G')
        os.close(barrier_write); barrier_write = None
        with os.fdopen(error_read, 'rb') as stream:
            error_read = None
            failure = stream.read()
        if failure:
            raise restore_error(json.loads(failure))
        return process
    except BaseException as original:
        if process is not None:
            try:
                process.kill()
                process.wait()
            except BaseException as secondary:
                original.add_note(f'Native bootstrap failure cleanup: {secondary!r}')
        raise
    finally:
        original = sys.exception()
        for fd in (config_read, config_write, barrier_read, barrier_write, error_read, error_write, *fds):
            if fd is not None:
                try: os.close(fd)
                except BaseException as secondary:
                    if original is None: raise
                    original.add_note(f'Native launch descriptor cleanup: {secondary!r}')


def main():
    control_fd, alive_fd = map(int, sys.argv[1:3])
    lease_fds = list(map(int, sys.argv[3:]))
    control = socket.socket(fileno=control_fd)
    # These descriptors belong only to the guardian; exec children cannot keep
    # Kernel liveness or the operation lease alive independently of this guard.
    for fd in (control_fd, alive_fd, *lease_fds):
        if fd >= 0:
            os.set_inheritable(fd, False)
    processes = {}
    tracker = None
    stop = threading.Event()
    ownership_failure = []
    try:
        try:
            tracker = Descendants()
        except BaseException as error:
            send_record(control, {'error': error_record(error)})
            raise
        send_record(control, {'ready': True})
        def monitor():
            try:
                while not stop.is_set():
                    readable, _, _ = select.select([alive_fd], [], [], .02)
                    if readable and not os.read(alive_fd, 1):
                        stop.set()
                        control.shutdown(socket.SHUT_RDWR)
                        return
                    tracker.scan()
            except BaseException as error:
                ownership_failure.append(error)
                stop.set()
                try:
                    control.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        watcher = threading.Thread(target=monitor, name='scnsim-native-owner', daemon=True)
        watcher.start()
        try:
            while not stop.is_set():
                try:
                    command, fds = receive_record(control)
                except EOFError:
                    break
                try:
                    action = command['action']
                    if action == 'retain_lease':
                        if len(fds) != 1:
                            raise ValueError('native lease handoff requires one descriptor')
                        # receive_record already made this descriptor
                        # noninheritable. Transfer cleanup ownership before ACK;
                        # an interrupted reply still leaves final drain owning it.
                        lease_fds.append(fds[0])
                        fds = []
                        answer = {'retained': True}
                    elif action == 'launch':
                        launching_fds, fds = fds, []
                        process = launch(command, launching_fds, tracker)
                        processes[process.pid] = process
                        answer = {'pid': process.pid}
                    elif action == 'drain':
                        disposition = tracker.drain(processes)
                        answer = {'termination': disposition,
                                  'returncodes': {str(pid): process.returncode
                                                  for pid, process in processes.items()}}
                        processes.clear()
                    elif action == 'close':
                        break
                    else:
                        process = processes[command['pid']]
                        if action == 'poll': answer = {'returncode': process.poll()}
                        elif action == 'wait':
                            # Poll so Kernel death interrupts a long native wait.
                            end = None if command['timeout'] is None else time.monotonic() + command['timeout']
                            while process.poll() is None and not stop.wait(.02):
                                if end is not None and time.monotonic() >= end:
                                    raise subprocess.TimeoutExpired(process.args, command['timeout'])
                            if stop.is_set(): break
                            answer = {'returncode': process.returncode}
                        elif action == 'signal':
                            process.send_signal(command['signal']); answer = {}
                        else: raise RuntimeError('unknown native guardian command')
                    send_record(control, answer)
                except BaseException as error:
                    send_record(control, {'error': error_record(error)})
                finally:
                    original = sys.exception()
                    for fd in fds:
                        try: os.close(fd)
                        except BaseException as secondary:
                            if original is None: raise
                            original.add_note(f'Native command descriptor cleanup: {secondary!r}')
        finally:
            stop.set()
            watcher.join()
        if ownership_failure:
            raise ownership_failure[0]
    finally:
        original = sys.exception()
        cleanup_error = None
        try:
            if tracker is not None:
                tracker.drain(processes)
        except BaseException as secondary:
            if original is None:
                cleanup_error = secondary
            else:
                original.add_note(f'Native guardian drain cleanup: {secondary!r}')
        if tracker is not None:
            try: tracker.close()
            except BaseException as secondary:
                target = original if original is not None else cleanup_error
                if target is None: cleanup_error = secondary
                else: target.add_note(f'Native tracker cleanup: {secondary!r}')
        for resource in (control, alive_fd, *lease_fds):
            try:
                if isinstance(resource, socket.socket): resource.close()
                else: os.close(resource)
                # Lease descriptor close only, never LOCK_UN: the guard owns
                # both the shared root gate and operation open-file description.
            except BaseException as secondary:
                target = original if original is not None else cleanup_error
                if target is None: cleanup_error = secondary
                else: target.add_note(f'Native guardian descriptor cleanup: {secondary!r}')
        if original is None and cleanup_error is not None:
            raise cleanup_error


if __name__ == '__main__':
    main()
