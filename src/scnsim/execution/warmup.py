"""Explicit Run-owned executable preparation, never an analysis attempt.

The existing request/compiler/View authorities select real declared shapes.
Only lowering and compilation run: no assembly execution, SuperLU, Newton,
CMA, Workspace publication, checkpoint or numerical result cache participates.
Strong ownership is the Run's warm holder; shared executable lookup is weak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .prepared import PreparedAnalysis
from hashlib import sha256
from time import perf_counter_ns


@dataclass(frozen=True, slots=True)
class WarmupKernel:
    signature_sha256: str
    kernel: str
    argument_shapes: tuple[tuple[tuple[int, ...], str], ...]
    reused: bool
    lookup_and_compile_ns: int
    compile_ns: int | None


@dataclass(frozen=True, slots=True)
class WarmupSummary:
    backend: str
    precision: str
    request_sha256: str
    parameter_points: int
    kernels: tuple[WarmupKernel, ...]
    elapsed_ns: int
    preparation_ns: int
    unknown_shapes_lazy: bool = True
    lazy_work: tuple[str, ...] = ('future_parameter_or_continuation_meshes',
                                 'future_candidate_ready_batch_widths')

    @property
    def compiled_count(self):
        return sum(not kernel.reused for kernel in self.kernels)

    @property
    def reused_count(self):
        return sum(kernel.reused for kernel in self.kernels)

    @property
    def compile_ns(self):
        return sum(kernel.compile_ns for kernel in self.kernels if kernel.compile_ns is not None)


def prepare_warmup(*, plan_document, analysis: PreparedAnalysis, precision,
                   executable_owner, baseline_parameters):
    """Compile known request signatures and retain only their executables.

    Optimization uses its declared baseline point and objective Views. Literal
    sweep points use their actual meshes; unrequested future shapes and ready
    candidate batch widths are deliberately not predicted or padded.
    """
    started_ns = perf_counter_ns()
    from ..compilation.compiler import compile_model, parameter_key, parameter_values
    from ..compilation.views import realize_view
    from .jax_backend import get_jax_backend
    from ..canonical import canonical_json_bytes
    from .config import get_runtime_configuration
    from .python_host import resolved_points
    from .quantities import expression_leaves
    from .resources import operation_resources
    from .resources import _secondary

    preparation_start = perf_counter_ns()

    request = analysis.request()
    operation = request['operation']
    points = list(resolved_points(request['parameter_source']))
    if (operation == 'evaluate_direct' and request['spec']['type'] in
            ('diagonal_root', 'operator_element_root', 'hybridized_pole',
             'transfer_zero', 'residue_normalized_coupling')):
        baseline = baseline_parameters
        if canonical_json_bytes(baseline) not in {canonical_json_bytes(point) for point in points}:
            points.insert(0, baseline)
    declarations = []
    if operation == 'optimize_direct':
        for objective in request['spec']['objectives']:
            for selector in expression_leaves(objective['quantity']):
                declarations.append((selector['view'], selector['spec']))
    else:
        declarations.append((request['view'], request['spec']))

    configuration = get_runtime_configuration()
    candidate_batch = (operation == 'optimize_direct' and
                       min(configuration.optimization_workers,
                           request['spec']['optimizer']['resolved_population_size']) > 1)
    preparation_ns = perf_counter_ns() - preparation_start
    kernels = []
    with operation_resources(cpu_threads=configuration.cpu_threads) as resource:
        backend = get_jax_backend(precision=precision, resources=configuration,
                                 trace=None, operation_resources=resource, diagnostics=False)
        original = None
        try:
            for point in points:
                preparation_start = perf_counter_ns()
                authorized = {parameter_key(item) for item in point['allow_extrapolation']}
                authorized.update(parameter_key(item) for item in request['spec'].get('allow_extrapolation', ()))
                raw = compile_model(plan_document, parameter_values(point),
                                    authorized=authorized, preparation_cache={})
                preparation_ns += perf_counter_ns() - preparation_start
                views = {}
                for declaration, quantity in declarations:
                    key = canonical_json_bytes(declaration)
                    if key not in views:
                        preparation_start = perf_counter_ns()
                        views[key] = realize_view(raw, declaration)
                        preparation_ns += perf_counter_ns() - preparation_start
                    realized = views[key]
                    pending = [quantity]
                    while pending:
                        current = pending.pop(0)
                        kind = current['type']
                        if kind == 'residue_normalized_coupling':
                            pending.extend((current['branch_a'], current['branch_b']))
                        kind = 'direct' if kind == 'direct_solve' else kind
                        frequency_count = (len(current['frequencies'])
                                           if kind in ('direct', 'operator') else 1)
                        observations = backend.compile_warm_view(
                            realized, kind=kind, frequency_count=frequency_count,
                            family=current.get('family'), candidate_batch=candidate_batch)
                        for signature, name, shapes, reused, elapsed, compile_time in observations:
                            # Signature digest is descriptive, not a result identity.
                            digest = sha256(repr(signature).encode('utf-8')).hexdigest()
                            kernels.append(WarmupKernel(digest, name, shapes, reused,
                                                        elapsed, compile_time))
        except BaseException as error:
            original = error
            raise
        finally:
            # Also retain successful compilations preceding a native compile
            # error, without manufacturing a completed warmup summary.
            try:
                executable_owner.update(backend._executables)
            except BaseException as error:
                if original is None:
                    original = error
                    raise
                _secondary(original, 'Warm executable retention also failed', error)
            finally:
                try:
                    backend.close()
                except BaseException as error:
                    if original is None:
                        raise
                    _secondary(original, 'Warm backend cleanup also failed', error)
    return WarmupSummary('jax', precision, analysis.request_sha256,
                         len(points), tuple(kernels), perf_counter_ns() - started_ns,
                         preparation_ns)
