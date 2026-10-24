"""All-storage-off execution for one native JAX Direct point.

This path reuses the ordinary PhysicalAnalysis candidate and frequency
providers, but its sink owns only detached in-memory response arrays and
failure-only certificates. It has no Workspace, operation lease, task journal,
spool, persistent codec, or result-selection authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from ..errors import RuntimePreparationError


def direct_result_metadata(metadata, spec, frequencies_hz):
    """Project Direct axes and units for both point and Sweep RAM owners."""
    result = dict(metadata)
    terminal_ids = tuple(result["terminal_ids"])
    channels = tuple((identifier, ()) for identifier in terminal_ids)
    result.update({
        "operation": "solve_direct",
        "result_kind": "direct_solve",
        "point_count": len(frequencies_hz),
        "terminal_count": len(terminal_ids),
        "input_channels": channels,
        "output_channels": channels,
        "traces": tuple(spec.get("traces", ())),
        "frequencies_hz": frequencies_hz,
        "frequency_count": len(frequencies_hz),
        "frequency_unit": "hertz",
        "family_units": {"S": "dimensionless", "Y": "siemens", "Z": "ohm"},
    })
    return result


def _freeze_metadata(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array
    return value


class InMemoryDirectSink:
    """Direct-only sink that gathers source-ordered outputs without encoding."""

    domain = "direct"
    in_memory_direct = True

    def __init__(self, request: Mapping[str, object]):
        self._request = request
        self._metadata = None
        self._frequencies = None
        self._families = None
        self._filled = None
        self._destinations = None
        self._failure_certificates = {}

    def set_metadata(self, metadata):
        if not isinstance(metadata, Mapping):
            raise TypeError("in-memory Direct metadata must be a mapping")
        if self._metadata is not None:
            raise RuntimeError("in-memory Direct metadata was supplied more than once")
        self._metadata = dict(metadata)

    def set_affine_evidence(self, rows):
        """Attach the actual call's facts before the payload is detached at finish."""
        self._metadata["affine_evidence"] = rows

    def prepare_direct_output(self, *, frequencies_hz, work_indices,
                              terminal_count, complex_dtype):
        if self._families is not None:
            raise RuntimeError("in-memory Direct output was prepared more than once")
        frequencies = np.array(frequencies_hz, dtype=np.float64, copy=True)
        destinations = {}
        for output_index, work_index in enumerate(work_indices):
            destinations.setdefault(work_index, []).append(output_index)
        self._frequencies = frequencies
        self._destinations = {key: tuple(value) for key, value in destinations.items()}
        shape = (len(frequencies), terminal_count, terminal_count)
        dtype = np.dtype(complex_dtype)
        self._families = {
            family: np.empty(shape, dtype=dtype) for family in ("S", "Y", "Z")
        }
        self._filled = np.zeros(len(frequencies), dtype=np.bool_)

    def stage_direct_work(self, work_index, record):
        if self._families is None or self._destinations is None:
            raise RuntimeError("in-memory Direct work arrived before its output layout")
        destinations = self._destinations.get(work_index)
        if destinations is None:
            raise RuntimeError("native Direct work is absent from the declared occurrence layout")
        families = record.get("families")
        if not isinstance(families, Mapping) or set(families) != {"S", "Y", "Z"}:
            raise TypeError("in-memory Direct work lacks its synchronized S/Y/Z values")
        for family in ("S", "Y", "Z"):
            value = np.asarray(families[family])
            for output_index in destinations:
                self._families[family][output_index] = value
                self._filled[output_index] = True
        certificates = tuple(record.get("certificates", ()))
        if certificates:
            self._failure_certificates[work_index] = certificates

    def direct_work(self, work_index):
        """Expose only actual failure certificates to the existing reducer."""
        return {"certificates": self._failure_certificates.get(work_index, ())}

    def finish(self, terminal):
        if not isinstance(terminal, Mapping) or terminal.get("type") != "direct_solve":
            raise TypeError("in-memory Direct terminal is malformed")
        if self._metadata is None or self._families is None or self._filled is None:
            raise RuntimeError("in-memory Direct terminal is missing its compiled output layout")
        if int(terminal.get("point_count", -1)) != len(self._frequencies):
            raise RuntimeError("in-memory Direct terminal count differs from its output axis")
        if not bool(np.all(self._filled)):
            raise RuntimeError("in-memory Direct occurrence layout was not fully populated")

        metadata = direct_result_metadata(
            self._metadata, self._request["spec"], self._frequencies,
        )
        frequencies = self._frequencies
        frequencies.setflags(write=False)
        families = {}
        for family, values in self._families.items():
            values.setflags(write=False)
            families[family] = values
        payload = MappingProxyType({
            "terminal": _freeze_metadata(dict(terminal)),
            "frequencies_hz": frequencies,
            "families": MappingProxyType(families),
            "metadata": _freeze_metadata(metadata),
            "precision": metadata["precision"],
        })
        self._failure_certificates.clear()
        self._destinations.clear()
        self._filled = None
        self._frequencies = None
        self._families = None
        self._metadata = None
        self._request = None
        return payload


def _reject_task_event(*_args, **_kwargs):
    raise RuntimeError("in-memory Direct execution cannot publish task events")


def execute_memory_direct(*, plan_document, prepared_analysis, trace,
                          program_owner, execution_policy):
    """Run one JAX Direct point without any storage or persistence boundary."""
    request = prepared_analysis.request()
    source = request.get("parameter_source")
    if (request.get("operation") != "solve_direct"
            or not isinstance(source, Mapping)
            or source.get("kind") != "point"
            or request.get("runtime_semantic", {}).get("backend") != "jax"):
        raise RuntimePreparationError(
            "In-memory execution supports one JAX Direct point only",
            stage="memory_direct_prepare",
        )

    from .config import get_runtime_configuration, runtime_resource_identity

    resources = request["runtime_semantic"]["resources"]
    if runtime_resource_identity() != resources:
        raise RuntimePreparationError(
            "Prepared JAX resources changed before in-memory execution",
            stage="runtime_prepare",
        )
    configuration = get_runtime_configuration()
    from .jax_backend import get_jax_backend
    from .python_host import execute_analysis
    from .resources import operation_resources

    sink = InMemoryDirectSink(request)
    backend = None
    primary = None
    with operation_resources(cpu_threads=configuration.cpu_threads) as operation_resource:
        try:
            with trace.span("backend_prepare"):
                backend = get_jax_backend(
                    precision=request["runtime_semantic"]["precision"],
                    resources=configuration,
                    trace=trace,
                    operation_resources=operation_resource,
                    diagnostics=True,
                )
            terminal = execute_analysis(
                plan_document,
                prepared_analysis,
                backend=backend,
                emit=_reject_task_event,
                trace=trace,
                operation_resources=operation_resource,
                program_owner=program_owner,
                execution_policy=execution_policy,
                result_sink=sink,
            )
            return sink.finish(terminal)
        except BaseException as error:
            primary = error
            raise
        finally:
            if backend is not None:
                try:
                    backend.close()
                except BaseException as cleanup_error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "In-memory Direct backend cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
