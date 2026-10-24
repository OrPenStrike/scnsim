"""Installed Windows-wheel evidence for the existing finite native/JAX CI cases.

This CI-only consumer never changes numerical rules or supplies a fallback.
Build audit and installed public use have separate records; exceptions remain
fatal after preserving completed observations. The isolated consumer runs plain
uv sync against the downloaded wheel with wheel-only dependencies. No native
compiler is used by consumer installation or smoke.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from email.parser import Parser
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile
from uuid import uuid4


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _host(value):
    """Keep numeric dtype/bits and fail on unassigned report payload types."""
    import numpy as np
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, bytes):
        return {'bytes_hex': value.hex()}
    if isinstance(value, float):
        return {'f64': struct.pack('>d', value).hex()}
    if isinstance(value, complex):
        return {'real': _host(value.real), 'imag': _host(value.imag)}
    if isinstance(value, np.ndarray):
        return {'dtype': value.dtype.str, 'shape': list(value.shape),
                'data_hex': value.tobytes(order='C').hex()}
    if isinstance(value, np.generic):
        return _host(np.asarray(value))
    jax = sys.modules.get('jax')
    if jax is not None and isinstance(value, jax.Array):
        return _host(np.asarray(value))
    if hasattr(value, 'magnitude') and hasattr(value, 'units'):
        return {'magnitude': _host(np.asarray(value.magnitude)), 'unit': str(value.units)}
    if isinstance(value, Mapping):
        return {key: _host(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_host(item) for item in value]
    if is_dataclass(value):
        return {field.name: _host(getattr(value, field.name)) for field in fields(value)}
    raise TypeError(f'unassigned CI evidence type: {type(value).__module__}.{type(value).__name__}')


def _identity_facts(identity):
    """Publish receipt hashes, not ResultIdentity's private verification token."""
    return {'plan_sha256': identity.plan_sha256,
            'request_sha256': identity.request_sha256,
            'attempt_sha256': identity.attempt_sha256,
            'result_sha256': identity.result_sha256}


def _failure_facts(failure):
    """Project the public SCNSim root error without encoding exception ownership."""
    if failure is None:
        return None
    return {'type': type(failure).__name__, 'module': type(failure).__module__,
            'category': failure.category, 'kind': failure.kind,
            'stage': failure.stage, 'message': str(failure),
            'evidence': failure.evidence}


def _platform():
    if sys.platform != 'win32' or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise RuntimeError('Windows wheel observation requires actual Windows x86_64')
    if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8:
        raise RuntimeError('Windows wheel observation requires 64-bit CPython 3.12')
    return {'os': platform.platform(), 'machine': platform.machine(),
            'python': sys.version, 'executable': sys.executable}


def _busy(call):
    """Observe only actual native contention; unrelated failures escape."""
    try:
        call()
    except BlockingIOError as error:
        return {'type': type(error).__name__, 'errno': error.errno,
                'cause_winerror': getattr(error.__cause__, 'winerror', None)}
    raise RuntimeError('exclusive/shared conflict unexpectedly acquired its real file lock')


def _emit(value):
    print(json.dumps(value), flush=True)


def lock_child(args):
    from scnsim.workspace._os_io import lock_fd, unlock_fd
    descriptor = os.open(args.path, os.O_RDWR | os.O_CREAT, 0o600)
    held = False
    try:
        lock_fd(descriptor, exclusive=False, nonblocking=True)
        held = True
        _emit({'phase': 'shared-acquired'})
        if sys.stdin.readline().strip() != 'release-shared':
            raise RuntimeError('missing parent shared-lock release handshake')
        unlock_fd(descriptor)
        held = False
        _emit({'phase': 'exclusive-busy', 'error': _busy(
            lambda: lock_fd(descriptor, exclusive=True, nonblocking=True))})
        if sys.stdin.readline().strip() != 'acquire-exclusive':
            raise RuntimeError('missing parent exclusive-lock acquisition handshake')
        lock_fd(descriptor, exclusive=True, nonblocking=True)
        held = True
        _emit({'phase': 'exclusive-acquired'})
        if sys.stdin.readline().strip() != 'release-exclusive':
            raise RuntimeError('missing parent exclusive-lock release handshake')
        unlock_fd(descriptor)
        held = False
        _emit({'phase': 'released'})
    finally:
        try:
            if held:
                unlock_fd(descriptor)
        finally:
            os.close(descriptor)


def idle_child(args):
    from scnsim.workspace.operation_lease import OperationLease, _idle_operation_lease
    # Each process owns independent real descriptors; no duplicated-handle
    # inheritance guarantee is assumed. The root SH gate stays held here.
    with OperationLease(args.root, str(uuid4())):
        with _idle_operation_lease(args.root, args.operation_id) as idle:
            _emit({'phase': 'idle-while-owned', 'acquired': idle.acquired})
        if sys.stdin.readline().strip() != 'owner-released':
            raise RuntimeError('missing parent operation-lease release handshake')
        with _idle_operation_lease(args.root, args.operation_id) as idle:
            _emit({'phase': 'idle-after-release', 'acquired': idle.acquired})


def _lock_observation(root):
    from scnsim.workspace._os_io import lock_fd, unlock_fd
    from scnsim.workspace.operation_lease import OperationLease
    root.mkdir(parents=True)
    events = []
    def start(*arguments):
        return subprocess.Popen([sys.executable, '-I', str(Path(__file__).resolve()), *arguments],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    def read(process, phase):
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f'lock child ended before {phase}; returncode={process.poll()}')
        row = json.loads(line)
        if row['phase'] != phase:
            raise RuntimeError(f'unexpected lock child phase: {row}')
        events.append(row)
        return row
    def send(process, message):
        process.stdin.write(message + '\n')
        process.stdin.flush()
    def finish(process):
        original = sys.exception()
        failure = original
        actions = ([process.terminate] if process.poll() is None else [])
        actions += [process.wait, process.stdin.close, process.stdout.close]
        for action in actions:
            try:
                action()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(f'Lock child cleanup: {error!r}')
        if original is None and failure is not None:
            raise failure
    descriptor = os.open(root / 'shared exclusive.lock', os.O_RDWR | os.O_CREAT, 0o600)
    held = False
    child = None
    try:
        lock_fd(descriptor, exclusive=False)
        held = True
        child = start('lock-child', '--path', str(root / 'shared exclusive.lock'))
        read(child, 'shared-acquired')
        send(child, 'release-shared')
        read(child, 'exclusive-busy')
        unlock_fd(descriptor)
        held = False
        send(child, 'acquire-exclusive')
        read(child, 'exclusive-acquired')
        for exclusive in (False, True):
            events.append({'phase': 'parent-busy', 'exclusive': exclusive,
                'error': _busy(lambda: lock_fd(descriptor, exclusive=exclusive, nonblocking=True))})
        send(child, 'release-exclusive')
        read(child, 'released')
        if child.wait() != 0:
            raise RuntimeError('shared/exclusive lock child failed')
        lock_fd(descriptor, exclusive=True, nonblocking=True)
        held = True
        events.append({'phase': 'parent-exclusive-after-release'})
        unlock_fd(descriptor)
        held = False
    finally:
        try:
            if child is not None:
                finish(child)
        finally:
            try:
                if held:
                    unlock_fd(descriptor)
            finally:
                os.close(descriptor)
    operation_id = str(uuid4())
    child = None
    try:
        with OperationLease(root, operation_id):
            child = start('idle-child', '--root', str(root), '--operation-id', operation_id)
            if read(child, 'idle-while-owned')['acquired']:
                raise RuntimeError('idle-operation probe acquired an actively held operation')
        send(child, 'owner-released')
        if not read(child, 'idle-after-release')['acquired']:
            raise RuntimeError('idle-operation probe failed to acquire released operation')
        if child.wait() != 0:
            raise RuntimeError('idle-operation child failed')
    finally:
        if child is not None:
            finish(child)
    return events


def audit(args):
    report = {'platform': _platform()}
    source_path = 'src/scnsim/_native/compensated_cpu.cpp'
    git_source = subprocess.check_output(['git', 'show', f'HEAD:{source_path}'])
    if args.source_cpp.read_bytes() != git_source:
        raise RuntimeError('checkout kernel source differs from its exact Git blob')
    report['git_source'] = {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'path': source_path,
        'blob_id': subprocess.check_output(['git', 'rev-parse', f'HEAD:{source_path}'], text=True).strip(),
        'sha256': _sha(git_source),
        'checkout_eol': subprocess.check_output(['git', 'ls-files', '--eol', '--', source_path], text=True).strip(),
        'core_autocrlf': subprocess.check_output(['git', 'config', '--show-origin', '--get', 'core.autocrlf'], text=True).strip(),
        'core_eol': subprocess.check_output(['git', 'config', '--show-origin', '--get', 'core.eol'], text=True).strip(),
    }
    wheels = tuple(args.wheelhouse.glob('*.whl'))
    if len(wheels) != 1:
        raise RuntimeError('Windows build must produce exactly one wheel')
    wheel, = wheels
    if not wheel.name.endswith('-cp312-cp312-win_amd64.whl'):
        raise RuntimeError(f'unexpected Windows wheel tag: {wheel.name}')
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if any(name.startswith('scnsim/_julia/') for name in names):
            raise RuntimeError('JAX-only wheel contains retired Julia resources')
        binaries = [name for name in names if name.startswith('scnsim/_native/_compensated_cpu') and name.endswith('.pyd')]
        if len(binaries) != 1:
            raise RuntimeError('wheel must contain exactly one compensated .pyd')
        binary, = binaries
        data = archive.read(binary)
        pe = struct.unpack_from('<I', data, 0x3c)[0]
        if data[:2] != b'MZ' or data[pe:pe + 4] != b'PE\0\0' or struct.unpack_from('<H', data, pe + 4)[0] != 0x8664:
            raise RuntimeError('native extension is not a PE AMD64 binary')
        source = 'scnsim/_native/compensated_cpu.cpp'
        if archive.read(source) != git_source:
            raise RuntimeError('wheel kernel source differs from its exact Git blob')
        wheel_record, = [name for name in names if name.endswith('.dist-info/WHEEL')]
        if 'Tag: cp312-cp312-win_amd64' not in archive.read(wheel_record).decode():
            raise RuntimeError('wheel metadata lacks the actual Windows CP312 tag')
        record, = [name for name in names if name.endswith('.dist-info/RECORD')]
        metadata, = [name for name in names if name.endswith('.dist-info/METADATA')]
        package_metadata = Parser().parsestr(archive.read(metadata).decode())
        report.update(wheel=wheel.name, wheel_sha256=_sha(wheel.read_bytes()),
                      package_version=package_metadata['Version'],
                      metadata_sha256=_sha(archive.read(metadata)),
                      wheel_metadata=archive.read(wheel_record).decode(),
                      record_sha256=_sha(archive.read(record)),
                      resources=[{'path': name, 'sha256': _sha(archive.read(name)),
                                  'bytes': len(archive.read(name))} for name in (source, binary)])
        extracted = args.wheelhouse / Path(binary).name
        extracted.write_bytes(data)
    # Audit the actual compiler and imported DLL names; no DLL is bundled or
    # inferred available merely from its name in the dependency table.
    vswhere = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    installation = subprocess.check_output([str(vswhere), '-latest', '-products', '*',
        '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    toolsets = sorted((Path(installation) / 'VC/Tools/MSVC').iterdir())
    tools = toolsets[-1] / 'bin/Hostx64/x64'
    result = subprocess.run([str(tools / 'cl.exe'), '/Bv'], capture_output=True, text=True)
    report['compiler_inventory'] = {'argv': [str(tools / 'cl.exe'), '/Bv'],
        'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    report['compiler_inventory_role'] = 'Hosted tool inventory; actual compilation commands and selected flags are retained in BUILD.log.'
    dlls = subprocess.check_output([str(tools / 'dumpbin.exe'), '/DEPENDENTS', str(extracted)], text=True)
    (args.wheelhouse / 'PE-DEPENDENCIES.txt').write_text(dlls, encoding='utf-8')
    log = args.build_log.read_text(encoding='utf-8-sig')
    marker = "SCNSim ordered kernel compiler=msvc flags=['/std:c++17', '/fp:strict']"
    if marker not in log:
        raise RuntimeError('build log lacks actual MSVC strict-arithmetic flag selection')
    report.update(build_log_sha256=_sha(args.build_log.read_bytes()),
                  compiler_selection=marker, pe_dependencies=dlls,
                  runtime_prerequisite='Imported MSVC runtime DLLs must be available; hosted install is recorded separately, not all-host proof.')
    _write(args.output, report)
    extracted.unlink()


def smoke(args):
    report = {'platform': _platform(), 'completed': []}
    def save():
        _write(args.output, _host(report))
    primary = None
    run = None
    try:
        import numpy as np
        import scnsim
        from scnsim import (CircuitRun, DirectSolveSpec, DiagonalRootSpec, ReductionPipeline,
            SensitivitySpec, JVP, VJP, OutputCoordinate, configure_runtime,
            OptimizationSpec, OptimizationVariable, CostObjective, CMAESSpec,
            ParameterSet, units as u)
        from scnsim._native import _compensated_cpu
        package = Path(scnsim.__file__).resolve()
        if package.is_relative_to((args.repository / 'src').resolve()):
            raise RuntimeError('Windows smoke imported checkout instead of installed wheel')
        report['installed'] = {'path': str(package), 'extension': _compensated_cpu.__file__,
            'versions': {name: version(name) for name in ('scnsim', 'jax', 'jaxlib', 'scipy', 'numpy')}}
        binding = json.loads((args.wheelhouse / 'NATIVE-RESOURCE-MANIFEST.json').read_text(encoding='utf-8'))
        if version('scnsim') != binding['package_version']:
            raise RuntimeError('installed package version differs from the audited wheel')
        wheel = args.wheelhouse / binding['wheel']
        if _sha(wheel.read_bytes()) != binding['wheel_sha256']:
            raise RuntimeError('downloaded wheel differs from its build manifest')
        extension = Path(_compensated_cpu.__file__)
        extension_record, = [row for row in binding['resources'] if row['path'].endswith('.pyd')]
        if _sha(extension.read_bytes()) != extension_record['sha256']:
            raise RuntimeError('installed native extension differs from the audited wheel')
        report['wheel_binding'] = binding
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            # Exact identity buffer smoke, not a physical accuracy criterion.
            a = np.ones(1, dtype=dtype); low = np.zeros_like(a)
            index = np.array([0], dtype=np.int32); ptr = np.array([0, 1], dtype=np.int32)
            rhs = np.ones((1, 1), dtype=dtype); rhs_low = np.zeros_like(rhs)
            high = np.empty_like(rhs); tail = np.empty_like(rhs)
            _compensated_cpu.csr_pair_product(a, index, ptr, low, index, ptr, rhs, rhs_low, high, tail)
            if not np.array_equal(high, rhs) or not np.array_equal(tail, rhs_low):
                raise RuntimeError('native identity product differs from exact operands')
            report['completed'].append({'kernel_dtype': np.dtype(dtype).name, 'high': high, 'low': tail})
        configure_runtime(cpu_threads=10)
        blocks = re.findall(r'```\{python\}\n(.*?)\n```', args.chapter.read_text(encoding='utf-8'), re.S)
        namespace = {}
        for cell_id in ('ch2-import-and-define-parameters', 'ch2-build-grounded-lc-child', 'ch2-assemble-coupler-and-port'):
            block, = [block for block in blocks if f'#| id: {cell_id}' in block]
            block = block.replace('from IPython.display import display\n', '')
            exec(compile(block, str(args.chapter) + ':' + cell_id, 'exec'), namespace)
        report['fixture'] = {'path': str(args.chapter), 'sha256': _sha(args.chapter.read_bytes()),
                             'origin': 'existing CI Chapter2 three construction cells; unchanged inputs'}
        capacitance, inductance = namespace['capacitance'], namespace['inductance']
        # Same authored Plan/requests, an actual rooted Windows Path containing
        # spaces and non-ASCII text; no separator or URI is synthesized here.
        with TemporaryDirectory(prefix='scnsim Windows 空間 ') as temporary:
            report['completed'].append({'operation': 'native crossprocess locks',
                'events': _lock_observation(Path(temporary) / 'lock ownership 鎖')})
            save()
            workspace = Path(temporary) / 'saved LC 工作區'
            saved_run = CircuitRun(plan=namespace['plan'], workspace=workspace)
            saved_error = None
            try:
                direct_spec = DirectSolveSpec(frequencies=[5.5, 6.0, 6.5]*u.GHz)
                saved_direct = saved_run.solve(saved_run.original, direct_spec, precision='float64')
                resolved_direct = saved_run.resolve(saved_run.original, direct_spec, precision='float64')
                if saved_direct.identity != resolved_direct.identity:
                    raise RuntimeError('Workspace Direct resolve changed exact Result identity')
                saved_view = saved_run.original.reduce(ReductionPipeline().retain(namespace['resonator_coordinate']))
                saved_root_spec = DiagonalRootSpec(coordinate=namespace['resonator_coordinate'], root_hint=6.0*u.GHz)
                saved_root = saved_run.evaluate(saved_view, saved_root_spec, precision='float64')
                resolved_root = saved_run.resolve(saved_view, saved_root_spec, precision='float64')
                if saved_root.identity != resolved_root.identity:
                    raise RuntimeError('Workspace root resolve changed exact Result identity')
                report['completed'].append({'operation': 'Workspace Direct/root/resolve',
                    'workspace_path': str(workspace), 'rooted_drive': workspace.drive,
                    'direct_identity': _identity_facts(saved_direct.identity),
                    'root_identity': _identity_facts(saved_root.identity),
                    's': resolved_direct.s.view.matrix, 'frequency': resolved_root.frequency,
                    'benchmark': saved_run.benchmark().document()})
                save()
                saved_run.close()
                report['completed'].append({'operation': 'Workspace postclose',
                    's': resolved_direct.s.view.matrix, 'frequency': resolved_root.frequency})
                save()
            except BaseException as error:
                saved_error = error
                raise
            finally:
                try:
                    saved_run.close()
                except BaseException as error:
                    if saved_error is None:
                        raise
                    saved_error.add_note(f'Windows Workspace Run cleanup: {error!r}')
        run = CircuitRun(plan=namespace['plan'], storage='memory')
        retained = []
        for precision in ('float64', 'float32'):
            direct = run.solve(run.original, DirectSolveSpec(frequencies=[5.5, 6.0, 6.5]*u.GHz), precision=precision)
            report['completed'].append({'operation': 'Direct', 'precision': precision,
                'request': direct.request_sha256, 's': direct.s.view.matrix,
                'y': direct.y.view.matrix, 'z': direct.z.view.matrix, 'frequencies': direct.frequencies})
            retained.append(direct)
            save()
        view = run.original.reduce(ReductionPipeline().retain(namespace['resonator_coordinate']))
        root = DiagonalRootSpec(coordinate=namespace['resonator_coordinate'], root_hint=6.0*u.GHz)
        value = run.evaluate(view, root, precision='float64')
        report['completed'].append({'operation': 'root', 'status': value.status,
            'frequency': value.frequency, 'failure': _failure_facts(value.failure), 'evaluation': value.evaluation})
        save()
        if value.failure is not None:
            raise value.failure
        retained.append(value)
        projections = []
        for derivative in (JVP(direction={capacitance: 1.0*u.fF}),
                           VJP(weights={OutputCoordinate('frequency'): 1.0/u.Hz})):
            spec = SensitivitySpec(quantities={'frequency': root.frequency.on(view)},
                                   wrt=(capacitance,), derivative=derivative)
            result = run.sensitivity(spec, precision='float64')
            projection = result.jvp if isinstance(derivative, JVP) else result.vjp
            block = projection.block(projection.block_ids[0])
            projections.append(projection)
            report['completed'].append({'operation': type(derivative).__name__, 'status': block.status,
                'failure': block.failure, 'value': block.value if isinstance(derivative, JVP) else block.values,
                'certificates': result.certificates})
            save()
            if block.status != 'complete' or block.failure is not None:
                raise RuntimeError('existing LC derivative unavailable; original evidence retained')
        spec = OptimizationSpec(variables=(OptimizationVariable(parameter=capacitance,
            bounds=(80.0*u.fF, 140.0*u.fF)),), objectives=(CostObjective(id='resonance_frequency',
            quantity=root.frequency, target=6.2*u.GHz, weight=1.0*u.dimensionless),),
            optimizer=CMAESSpec(seed=17, population_size=6, max_evaluations=7))
        optimization = run.optimize(view, spec, parameters=ParameterSet({inductance: 5.8*u.nH}),
                                    precision='float64', progress=False)
        report['completed'].append({'operation': 'CMA', 'outcome': optimization.outcome,
            'best_cost': None if optimization.best is None else optimization.best.cost,
            'trials': [{'ordinal': trial.ordinal, 'status': trial.status, 'cost': trial.cost,
                        'parameters': [{'definitions_id': parameter.definitions_id,
                                        'id': parameter.id, 'value': point_value}
                                       for parameter, point_value in trial.parameters.values.items()],
                        'failure': trial.failure, 'optimizer_unavailability': trial.optimizer_unavailability,
                        'certificates': trial.certificates}
                       for trial in optimization.trials],
            'stages': [{'strategy': stage.strategy, 'status': stage.status,
                        'termination_reason': stage.termination_reason} for stage in optimization.stages]})
        save()
        if optimization.best is None:
            raise RuntimeError('existing bounded CMA produced no available best trial; history retained')
        benchmark = run.benchmark()
        run.close()
        run = None
        report['detached'] = {'s': retained[0].s.view.matrix, 'frequency': value.frequency,
            'blocks': [{'status': p.block(p.block_ids[0]).status,
                        'failure': p.block(p.block_ids[0]).failure} for p in projections],
            'best_cost': None if optimization.best is None else optimization.best.cost,
            'benchmark': benchmark.document()}
        report['state'] = 'returned'
        save()
    except BaseException as error:
        primary = error
        report['state'] = 'raised'
        report['exception'] = {'type': type(error).__name__, 'message': str(error)}
        try:
            save()
        except BaseException as secondary:
            error.add_note(f'Windows evidence serialization: {secondary!r}')
        raise
    finally:
        if run is not None:
            try:
                run.close()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(f'Windows Run cleanup: {error!r}')


def sync_consumer(args):
    """Own one isolated virtual consumer and retain install facts, not smoke success."""
    consumer = args.consumer.resolve()
    environment = args.environment.resolve()
    output = args.output.resolve()
    wheels = tuple(args.wheelhouse.resolve().glob('*.whl'))
    wheel, = wheels
    with zipfile.ZipFile(wheel) as archive:
        metadata, = [name for name in archive.namelist()
                     if name.endswith('.dist-info/METADATA')]
        package = Parser().parsestr(archive.read(metadata).decode('utf-8'))
    if package['Name'].lower() != 'scnsim':
        raise ValueError('consumer source must be the actual SCNSim wheel')
    # JSON basic strings are also valid TOML strings; POSIX spelling preserves
    # Windows drives, spaces and non-ASCII without TOML backslash ambiguity.
    quote = lambda value: json.dumps(value, ensure_ascii=False)
    document = (
        '[project]\nname = "scnsim-windows-wheel-consumer"\n'
        'version = "0.0.0"\nrequires-python = ">=3.12,<3.13"\n'
        f'dependencies = [{quote("scnsim==" + package["Version"])}]\n\n'
        '[tool.uv]\npackage = false\nno-build = true\n'
        '\n'
        '[tool.uv.sources]\n'
        f'scnsim = {{ path = {quote(wheel.as_posix())} }}\n'
    )
    consumer.mkdir(parents=True, exist_ok=False)
    (consumer / 'pyproject.toml').write_text(document, encoding='utf-8')
    (consumer / '.python-version').write_text('3.12\n', encoding='utf-8')
    output.mkdir(parents=True, exist_ok=True)
    environment_variables = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(environment))
    report = {'kind': 'consumer_install_facts', 'state': 'prepared',
              'consumer': str(consumer), 'environment': str(environment),
              'wheel': str(wheel), 'wheel_sha256': _sha(wheel.read_bytes()),
              'candidate_version': package['Version'], 'command': ['uv', 'sync'],
              'cwd': str(consumer),
              'uv_version': subprocess.check_output(['uv', '--version'], text=True).strip(),
              'source_build_exception': [],
              'all_other_dependencies': 'wheel-only', 'sync_exit_code': None}

    def retain():
        files = {}
        for name in ('pyproject.toml', '.python-version', 'uv.lock'):
            source = consumer / name
            if source.is_file():
                data = source.read_bytes()
                (output / name).write_bytes(data)
                files[name] = {'sha256': _sha(data), 'bytes': len(data)}
            else:
                files[name] = {'available': False}
        report['consumer_files'] = files
        _write(output / 'CONSUMER-INSTALL.json', report)

    retain()
    primary = None
    try:
        completed = subprocess.run(['uv', 'sync'], cwd=consumer,
                                   env=environment_variables, check=False)
        report['sync_exit_code'] = completed.returncode
        report['state'] = 'sync_returned' if completed.returncode == 0 else 'sync_failed'
        return completed.returncode
    except BaseException as error:
        primary = error
        report['state'] = 'sync_raised'
        report['exception'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        try:
            retain()
        except BaseException as error:
            if primary is not None:
                primary.add_note(f'Consumer install evidence retention: {error!r}')
            elif report['sync_exit_code'] not in (None, 0):
                # Preserve the actual failed sync exit even if copying evidence
                # also fails. Reporting that secondary cause is best-effort.
                try:
                    print(f'Consumer install evidence retention failed: {error!r}', file=sys.stderr)
                except Exception:
                    pass  # A failed diagnostic stream must not replace sync's exit.
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('audit')
    build.add_argument('--wheelhouse', type=Path, required=True)
    build.add_argument('--source-cpp', type=Path, required=True)
    build.add_argument('--build-log', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    use = sub.add_parser('smoke')
    use.add_argument('--repository', type=Path, required=True)
    use.add_argument('--chapter', type=Path, required=True)
    use.add_argument('--wheelhouse', type=Path, required=True)
    use.add_argument('--output', type=Path, required=True)
    consumer = sub.add_parser('sync-consumer')
    consumer.add_argument('--consumer', type=Path, required=True)
    consumer.add_argument('--environment', type=Path, required=True)
    consumer.add_argument('--wheelhouse', type=Path, required=True)
    consumer.add_argument('--output', type=Path, required=True)
    child = sub.add_parser('lock-child')
    child.add_argument('--path', type=Path, required=True)
    idle = sub.add_parser('idle-child')
    idle.add_argument('--root', type=Path, required=True)
    idle.add_argument('--operation-id', required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        audit(args)
    elif args.command == 'smoke':
        smoke(args)
    elif args.command == 'sync-consumer':
        return sync_consumer(args)
    elif args.command == 'lock-child':
        lock_child(args)
    else:
        idle_child(args)


if __name__ == '__main__':
    sys.exit(main())
