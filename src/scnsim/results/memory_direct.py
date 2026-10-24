"""Materialized Direct results returned by the all-storage-off execution path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import UUID

import numpy as np
from pint import Quantity

from .. import units
from ..construction import unavailable
from .base import (
    LineDiscretization,
    MatrixFamilyResult,
    MatrixView,
    Result,
    _fresh_quantity_attribute,
    _sha256,
)
from .derived import TraceResult
from .factory import _MEMORY_RESULT_TOKEN, _verified_result
from .matrix import ScatteringMatrixResult, _fresh_evidence_attribute

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


@dataclass(frozen=True, slots=True)
class InMemoryDirectSolveResult(Result):
    """Detached all-S/Y/Z response with in-memory provenance, not a receipt.

    The object has no ``identity`` or Workspace selection token. Its Plan and
    request hashes and execution UUID describe only the live in-memory call.
    """

    frequencies: Quantity
    s: ScatteringMatrixResult
    y: MatrixFamilyResult
    z: MatrixFamilyResult
    traces: Mapping[str, TraceResult]
    plan_sha256: str
    request_sha256: str
    execution_id: str
    discretization: tuple[LineDiscretization, ...] | None
    metadata: Mapping[str, object]
    evidence: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)
    _benchmark: InMemoryBenchmarkResult | None = field(default=None, repr=False, compare=False)
    _memory_report_token: object = field(default=None, init=False, repr=False, compare=False)
    _quantity_fields = frozenset({"frequencies"})

    def __init__(self) -> None:
        unavailable("InMemoryDirectSolveResult construction")

    def __getattribute__(self, name: str) -> object:
        if name == "evidence":
            return _fresh_evidence_attribute(object.__getattribute__(self, name))
        if name == "benchmark":
            benchmark = object.__getattribute__(self, "_benchmark")
            if benchmark is None:
                raise RuntimeError("in-memory Direct result finalization is incomplete")
            return benchmark
        return _fresh_quantity_attribute(self, name)

    @property
    def benchmark(self) -> InMemoryBenchmarkResult:
        return object.__getattribute__(self, "_benchmark")  # type: ignore[return-value]

    @property
    def persistence_kind(self) -> Literal["memory"]:
        return "memory"

    def _family(self, family: Literal["S", "Y", "Z"]) -> MatrixFamilyResult:
        if family == "S":
            return self.s
        if family == "Y":
            return self.y
        if family == "Z":
            return self.z
        raise ValueError("family must be 'S', 'Y', or 'Z'")

    def plot(self, *, family: Literal["S", "Y", "Z"] = "S", **presentation: object) -> Figure:
        return self._family(family).plot(**presentation)

    def show(self, *, family: Literal["S", "Y", "Z"] = "S", **presentation: object) -> None:
        from ..visualization.plots.common import show_figure

        return show_figure(self.plot(family=family, **presentation))

    def add_to(
        self,
        fig: Figure,
        *,
        family: Literal["S", "Y", "Z"] = "S",
        **presentation: object,
    ) -> Figure:
        return self._family(family).add_to(fig, **presentation)

    def __repr__(self) -> str:
        frequencies = object.__getattribute__(self, "frequencies")
        return (
            "InMemoryDirectSolveResult("
            f"execution_id={self.execution_id!r}, "
            f"frequency_count={len(frequencies.magnitude)}, "
            f"frequency_unit={frequencies.units}, "
            f"coordinates={self.s.view.coordinates!r})"
        )


def _memory_discretization(rows: object) -> tuple[LineDiscretization, ...] | None:
    if rows is None:
        return None
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("in-memory Direct discretization must be an ordered sequence")
    result: list[LineDiscretization] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("in-memory Direct discretization row must be a mapping")
        result.append(LineDiscretization(
            component_path=tuple(row["component_path"]),
            kind=row["kind"],
            length=units.registry.Quantity(row["length_m"], "meter"),
            n_sections=row["n_sections"],
            dx=units.registry.Quantity(row["dx_m"], "meter"),
            modal_velocities=tuple(
                units.registry.Quantity(value, "meter / second")
                for value in row.get("modal_velocities_m_s", ())
            ),
            hmax=(None if "hmax_m" not in row
                  else units.registry.Quantity(row["hmax_m"], "meter")),
            policy=row.get("policy"),
        ))
    return tuple(result)


def build_memory_direct_result(
    *,
    payload: Mapping[str, object],
    request: Mapping[str, object],
    plan_sha256: str,
    request_sha256: str,
    execution_id: str,
    benchmark: InMemoryBenchmarkResult | None = None,
) -> InMemoryDirectSolveResult:
    """Build immutable result leaves directly from the live numerical arrays.

    This adapter deliberately bypasses receipt identities, numeric codecs,
    serialization, fixed readers, and Workspace selection.
    """

    terminal = payload.get("terminal")
    if not isinstance(terminal, Mapping) or terminal.get("type") != "direct_solve":
        raise TypeError("in-memory Direct payload lacks its actual Direct terminal")
    if request.get("operation") != "solve_direct":
        raise TypeError("in-memory Direct result requires a solve_direct request")
    spec = request.get("spec")
    if not isinstance(spec, Mapping):
        raise TypeError("in-memory Direct request has no Spec mapping")
    _sha256(plan_sha256, name="plan_sha256")
    _sha256(request_sha256, name="request_sha256")
    UUID(execution_id)

    raw_metadata = payload.get("metadata")
    families = payload.get("families")
    if not isinstance(raw_metadata, Mapping) or not isinstance(families, Mapping):
        raise TypeError("in-memory Direct payload lacks metadata or matrix families")
    if set(families) != {"S", "Y", "Z"}:
        raise TypeError("in-memory Direct payload must contain S, Y, and Z")

    frequency_values = np.asarray(payload["frequencies_hz"])
    if frequency_values.ndim != 1:
        raise ValueError("in-memory Direct frequency axis must be one-dimensional")
    if terminal.get("point_count") != frequency_values.size:
        raise ValueError("in-memory Direct terminal count differs from its frequency axis")
    frequencies = units.registry.Quantity(frequency_values, "hertz")

    coordinates = tuple(raw_metadata["terminal_ids"])
    input_channels = tuple(
        (channel[0], tuple(channel[1])) for channel in raw_metadata["input_channels"]
    )
    output_channels = tuple(
        (channel[0], tuple(channel[1])) for channel in raw_metadata["output_channels"]
    )
    family_units = raw_metadata["family_units"]
    if not isinstance(family_units, Mapping):
        raise TypeError("in-memory Direct family units must be a mapping")
    presentation = {
        "view": raw_metadata["view"],
        "ref_lineage": raw_metadata["lineage"],
        "persistence": "memory",
        "plan_sha256": plan_sha256,
        "request_sha256": request_sha256,
        "execution_id": execution_id,
    }
    probe_loads = raw_metadata["probe_loads"]

    def make_family(name: str, result_type: type[MatrixFamilyResult]) -> MatrixFamilyResult:
        matrix = np.asarray(families[name])
        expected_shape = (frequency_values.size, len(output_channels), len(input_channels))
        if matrix.shape != expected_shape:
            raise ValueError(f"in-memory {name} family shape differs from its declared axes")
        view = _verified_result(
            MatrixView,
            matrix=units.registry.Quantity(matrix, family_units[name]),
            frequencies=frequencies,
            coordinates=coordinates,
            input_channels=input_channels,
            output_channels=output_channels,
            probe_loads=probe_loads,
        )
        return _verified_result(
            result_type,
            view=view,
            _parent_identity=None,
            _presentation=dict(presentation, family=name),
        )

    s_result = make_family("S", ScatteringMatrixResult)
    y_result = make_family("Y", MatrixFamilyResult)
    z_result = make_family("Z", MatrixFamilyResult)
    s_values = np.asarray(s_result.view.matrix.magnitude)
    traces: dict[str, TraceResult] = {}
    for declaration in spec["traces"]:
        output = coordinates.index(declaration["output_port"])
        input_ = coordinates.index(declaration["input_port"])
        traces[declaration["id"]] = _verified_result(
            TraceResult,
            frequencies=frequencies,
            value=units.registry.Quantity(s_values[:, output, input_], "dimensionless"),
            _parent_identity=None,
            _presentation=dict(
                presentation,
                id=declaration["id"],
                family="S",
                input_channel={"coordinate": coordinates[input_], "mode": []},
                output_channel={"coordinate": coordinates[output], "mode": []},
            ),
        )

    # Preserve compact compiled metadata; the numeric response arrays stay
    # solely in the materialized MatrixViews above.
    metadata = dict(raw_metadata)
    metadata.pop("frequencies_hz", None)
    metadata.update({
        "plan_sha256": plan_sha256,
        "request_sha256": request_sha256,
        "execution_id": execution_id,
        "terminal": terminal,
    })
    result = _verified_result(
        InMemoryDirectSolveResult,
        frequencies=frequencies,
        s=s_result,
        y=y_result,
        z=z_result,
        traces=traces,
        plan_sha256=plan_sha256,
        request_sha256=request_sha256,
        execution_id=execution_id,
        discretization=_memory_discretization(raw_metadata.get("discretization")),
        metadata=metadata,
        evidence=({"affine_evidence": raw_metadata["affine_evidence"]}
                  if raw_metadata.get("affine_evidence") else {}),
        _benchmark=benchmark,
    )
    object.__setattr__(result, "_memory_report_token", _MEMORY_RESULT_TOKEN)
    return result


def attach_memory_benchmark(
    result: InMemoryDirectSolveResult,
    benchmark: InMemoryBenchmarkResult,
) -> InMemoryDirectSolveResult:
    """Attach the post-resource recorder snapshot before exposing the result."""

    # Keep the diagnostics/result package boundary acyclic at import time;
    # resolve the runtime type only when finalization is actually requested.
    from ..diagnostics.memory_operation import InMemoryBenchmarkResult

    if type(result) is not InMemoryDirectSolveResult:
        raise TypeError("benchmark can be attached only to an in-memory Direct result")
    if not isinstance(benchmark, InMemoryBenchmarkResult):
        raise TypeError("in-memory Direct result requires its in-memory benchmark")
    if object.__getattribute__(result, "_benchmark") is not None:
        raise RuntimeError("in-memory Direct benchmark was already finalized")
    object.__setattr__(result, "_benchmark", benchmark)
    return result


__all__ = ["InMemoryDirectSolveResult", "attach_memory_benchmark", "build_memory_direct_result"]
