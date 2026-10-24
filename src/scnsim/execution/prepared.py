"""Pure prepared declarations between a sealed Run and canonical execution.

The records in this module deliberately carry no ``CircuitRun``, workspace,
process callback, or candidate realization. Runtime ownership checks and
parameter resolution happen before construction; the canonical request
encoder consumes only these finalized bytes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
import numpy as np
from .. import units
from ..canonical import (
    canonical_json_bytes,
    complex_quantity_envelope,
    float64_hex,
    quantity_envelope,
)
from .request import canonical_request_document
from ..authoring.physical_values import RLGCParameterSpec
from ..authoring import CoordinateRef, ElectricNodeRef, ParameterSet
from ..errors import CompilerInvariantError, InvalidOptimizationSpec
from ..specs import (
    DiagonalRootSpec,
    OperatorElementRootSpec,
    DirectSolveSpec,
    HybridizedPoleSpec,
    OperatorSpec,
    StaticCapacitanceSpec,
    StaticCapacitanceElementSpec,
    OptimizationSpec,
    CMAESSpec,
    LBFGSBSpec,
    ProjectedLBFGSSpec,
    HybridSpec,
    QuantityAbsolute,
    QuantityDifference,
    QuantitySelector,
    QuantitySum,
    ResidueNormalizedCouplingSpec,
    ResponseElementSpec,
    SParameterTrace,
    TransferZeroSpec,
    _expression_unit,
)


@dataclass(frozen=True, slots=True)
class BoundOptimizationLeaf:
    """One validated objective leaf's position in the ordered declaration."""

    objective_ordinal: int
    term_ordinal: int

    @classmethod
    def create(
        cls,
        *,
        objective_id: str,
        objective_ordinal: int,
        term_ordinal: int,
        declaration: Mapping[str, object],
    ) -> BoundOptimizationLeaf:
        if (
            not isinstance(objective_id, str)
            or not objective_id
            or not isinstance(objective_ordinal, int)
            or isinstance(objective_ordinal, bool)
            or objective_ordinal < 0
            or not isinstance(term_ordinal, int)
            or isinstance(term_ordinal, bool)
            or term_ordinal < 0
        ):
            raise CompilerInvariantError(
                "bound optimization leaf identity is malformed",
                stage="request_encode",
            )
        projection = declaration.get("projection")
        selector_type = declaration.get("type")
        spec = declaration.get("spec")
        view = declaration.get("view")
        if (
            not isinstance(projection, str)
            or not projection
            or not isinstance(selector_type, str)
            or not selector_type
            or not isinstance(spec, Mapping)
            or not isinstance(view, Mapping)
        ):
            raise CompilerInvariantError(
                "bound optimization leaf declaration is malformed",
                stage="request_encode",
            )
        # Preserve declaration encoding failures before optimization Spec lowering.
        # The final request owns these bytes; leaf ordering needs only positions.
        canonical_json_bytes(declaration)
        return cls(
            objective_ordinal=objective_ordinal,
            term_ordinal=term_ordinal,
        )


@dataclass(frozen=True, slots=True)
class BoundOptimization:
    """One normalized optimization declaration after leaf-order validation."""

    spec_bytes: bytes

    @classmethod
    def create(
        cls,
        *,
        spec: Mapping[str, object],
        leaves: Sequence[BoundOptimizationLeaf],
    ) -> BoundOptimization:
        checked = tuple(leaves)
        if not checked or not all(
            isinstance(leaf, BoundOptimizationLeaf) for leaf in checked
        ):
            raise CompilerInvariantError(
                "bound optimization requires ordered leaves",
                stage="request_encode",
            )
        expected = sorted(
            checked,
            key=lambda leaf: (leaf.objective_ordinal, leaf.term_ordinal),
        )
        if list(checked) != expected or len(
            {(leaf.objective_ordinal, leaf.term_ordinal) for leaf in checked}
        ) != len(checked):
            raise CompilerInvariantError(
                "bound optimization leaf order is not canonical",
                stage="request_encode",
            )
        return cls(canonical_json_bytes(spec))

    def spec(self) -> dict[str, object]:
        return json.loads(self.spec_bytes)


@dataclass(frozen=True, slots=True)
class SelectorBinding:
    """Detached public-alias ownership for readonly materialized selection.

    The integer owner identity and immutable canonical tables mirror the saved
    decoder boundary. No Plan, Run, reader, numerical program or closure is
    retained; scalar selector encoding remains the existing pure authority.
    """

    plan_owner_id: int
    coordinate_lookup: Mapping[str, str | None]
    view_bytes: bytes
    derived_coordinates: frozenset[str]

    def coordinate_id(self, value) -> str:
        if isinstance(value, str) and value in self.derived_coordinates:
            return value
        if isinstance(value, ElectricNodeRef) and id(value.plan) != self.plan_owner_id:
            raise ValueError("coordinate belongs to another Plan")
        if isinstance(value, CoordinateRef) and id(value.scope.root) != self.plan_owner_id:
            raise ValueError("coordinate belongs to another Plan")
        key = _coordinate_binding_key(value)[1]
        resolved = self.coordinate_lookup.get(key)
        if resolved is None:
            raise ValueError("coordinate is not a public alias in this Plan")
        return resolved

    def encode_selector(self, selector) -> bytes:
        if not isinstance(selector, QuantitySelector):
            raise TypeError("quantity must be a QuantitySelector")
        return canonical_json_bytes(_encode_scalar_expression(
            selector,
            coordinate_bindings={_coordinate_binding_key(coordinate): self.coordinate_id(coordinate)
                                 for coordinate in _quantity_coordinates(selector.spec)},
        ))


def capture_selector_binding(*, plan_owner_id: int,
                             coordinate_lookup: Mapping[str, str | None],
                             view: Mapping[str, object]) -> SelectorBinding:
    """Detach the same captured alias table and exact prepared View namespace."""
    derived = frozenset(coordinate for transform in view.get("transforms", ())
                        for coordinate in transform.get("output_coordinates", ())
                        if isinstance(coordinate, str))
    return SelectorBinding(plan_owner_id, MappingProxyType(dict(coordinate_lookup)),
                           canonical_json_bytes(view), derived)


@dataclass(frozen=True, slots=True)
class PreparedAnalysis:
    """Final immutable request declaration with non-executable source evidence."""

    request_bytes: bytes
    source_unit_bytes: tuple[bytes, ...]
    parameter_authorization_bytes: tuple[bytes, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        plan_sha256: str,
        operation: str,
        view: Mapping[str, object],
        spec: Mapping[str, object],
        parameter_source: Mapping[str, object],
        runtime_semantic: Mapping[str, object],
        source_units: Sequence[Mapping[str, object]],
        parameter_authorizations: Sequence[Mapping[str, object]] = (),
    ) -> PreparedAnalysis:
        if runtime_semantic.get("backend") == "jax" and operation == "optimize_direct":
            # Backend identity and transform naming share one canonical owner;
            # the existing host mapping math is unchanged.
            def jax_controls(record):
                optimizer = dict(record)
                if optimizer["type"] == "cma_es":
                    optimizer["box_transform_id"] = (
                        "scnsim-python-cmaes-0.13.1-native-domains.v1"
                        if optimizer["box_transform_id"].endswith("native-domains.v1")
                        else "scnsim-python-cmaes-0.13.1-linquad-unit-box.v1"
                    )
                elif optimizer["type"] == "hybrid":
                    optimizer["stages"] = [
                        {**stage, "optimizer": jax_controls(stage["optimizer"])}
                        for stage in optimizer["stages"]
                    ]
                return optimizer
            spec = {**spec, "optimizer": jax_controls(spec["optimizer"])}
        request = canonical_request_document(
            plan_sha256=plan_sha256,
            operation=operation,
            view=view,
            spec=spec,
            parameter_source=parameter_source,
            runtime_semantic=runtime_semantic,
        )
        return cls(
            request_bytes=canonical_json_bytes(request),
            source_unit_bytes=tuple(
                canonical_json_bytes(record) for record in source_units
            ),
            parameter_authorization_bytes=tuple(canonical_json_bytes(ref)
                                                for ref in parameter_authorizations),
        )

    def request(self) -> dict[str, object]:
        return json.loads(self.request_bytes)

    def effective_point(self) -> dict[str, object]:
        """Detach the encoded single effective point without resolving it again.

        Optimization/single-point consumers borrow this canonical declaration;
        grids and point sequences remain the ordinary source-expansion owner's
        responsibility, not an implicit baseline or first-point selection.
        """
        source = self.request()['parameter_source']
        if source['kind'] != 'point':
            raise TypeError('effective_point requires a single-point parameter source')
        return source['parameters']

    def parameter_authorizations(self) -> tuple[dict[str, object], ...]:
        """Actual caller authorization before optimization's baseline union."""
        return tuple(json.loads(ref) for ref in self.parameter_authorization_bytes)

    def source_units(self) -> tuple[dict[str, object], ...]:
        return tuple(json.loads(record) for record in self.source_unit_bytes)

    @property
    def request_sha256(self) -> str:
        return sha256(self.request_bytes).hexdigest()


def _coordinate_binding_key(
    value: str | ElectricNodeRef | CoordinateRef,
) -> tuple[str, str]:
    if isinstance(value, str):
        if not value:
            raise ValueError("coordinate IDs must not be empty")
        return "string", value
    if isinstance(value, ElectricNodeRef):
        return "electric_node", value.id
    if isinstance(value, CoordinateRef):
        return (
            "coordinate",
            canonical_json_bytes(
                {"scope": list(value.scope.path()), "id": value.id}
            ).decode("utf-8"),
        )
    raise TypeError("coordinate must be a public SCNSim coordinate handle or ID")


def _quantity_coordinates(
    spec: object,
) -> tuple[str | ElectricNodeRef | CoordinateRef, ...]:
    if isinstance(spec, DiagonalRootSpec):
        return (spec.coordinate,)
    if isinstance(spec, OperatorElementRootSpec):
        return spec.row, spec.column
    if isinstance(spec, HybridizedPoleSpec):
        return tuple(spec.coordinates)
    if isinstance(spec, TransferZeroSpec):
        return spec.input_coordinate, spec.output_coordinate
    if isinstance(spec, ResidueNormalizedCouplingSpec):
        return (
            *_quantity_coordinates(spec.branch_a),
            *_quantity_coordinates(spec.branch_b),
        )
    if isinstance(spec, ResponseElementSpec):
        return spec.input_coordinate, spec.output_coordinate
    if isinstance(spec, StaticCapacitanceElementSpec):
        return spec.row, spec.column
    if isinstance(spec, StaticCapacitanceSpec):
        return ()
    if isinstance(spec, OperatorSpec):
        return ()
    raise TypeError("unsupported Direct quantity Spec")


def _bound_coordinate_id(
    value: str | ElectricNodeRef | CoordinateRef,
    coordinate_bindings: Mapping[tuple[str, str], str],
) -> str:
    try:
        return coordinate_bindings[_coordinate_binding_key(value)]
    except KeyError as error:
        raise CompilerInvariantError(
            "prepared coordinate binding is incomplete",
            stage="request_encode",
        ) from error


def _frequency_grid(value: object) -> list[dict[str, str]]:
    converted = value.to("hertz")
    magnitudes = np.asarray(converted.magnitude, dtype=np.float64)
    return [
        quantity_envelope(
            units.registry.Quantity(float(item), "hertz"),
            si_unit="hertz",
            registry=units.registry,
        )
        for item in magnitudes
    ]


def _frequency_anchor_envelope(value: object) -> dict[str, str]:
    """Preserve an authored complex seed, including an explicit ``x + 0j``."""

    magnitude = getattr(value, "magnitude", None)
    authored_complex = (
        isinstance(magnitude, complex)
        or getattr(getattr(magnitude, "dtype", None), "kind", None) == "c"
    )
    encoder = complex_quantity_envelope if authored_complex else quantity_envelope
    return encoder(value, si_unit="hertz", registry=units.registry)


def _encoded_root_search(spec) -> dict[str, object]:
    search = spec.search
    if search is None:
        return {}
    region = ({"real_frequency_interval": [quantity_envelope(value, si_unit="hertz", registry=units.registry)
                                           for value in search.real_frequency_interval]}
              if search.hint_frequency_factor is None else
              {"hint_frequency_factor_f64": float64_hex(search.hint_frequency_factor)})
    if search.imag_frequency_interval is not None:
        region["imag_frequency_interval"] = [quantity_envelope(value, si_unit="hertz", registry=units.registry)
                                             for value in search.imag_frequency_interval]
    record = {"method": search.method, "region": region,
              "selection": {"kind": "nearest_hint", "metric": "complex_frequency_euclidean", "tie": search.tie},
              "tracking": {"kind": "baseline_anchor"}}
    if search.method == "finite_seeds":
        record["seeds"] = [complex_quantity_envelope(value, si_unit="hertz", registry=units.registry)
                           for value in search.seeds]
    return {"search": record}


def _encode_root(
    spec: DiagonalRootSpec,
    *,
    coordinate_bindings: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    return {
        "type": "diagonal_root",
        "coordinate": _bound_coordinate_id(spec.coordinate, coordinate_bindings),
        "root_hint": _frequency_anchor_envelope(spec.root_hint),
        **_encoded_root_search(spec),
    }


def _encode_element_root(
    spec: OperatorElementRootSpec,
    *,
    coordinate_bindings: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    return {
        "type": "operator_element_root",
        "row": _bound_coordinate_id(spec.row, coordinate_bindings),
        "column": _bound_coordinate_id(spec.column, coordinate_bindings),
        "root_hint": quantity_envelope(spec.root_hint, si_unit="hertz", registry=units.registry),
        **_encoded_root_search(spec),
    }


def _encode_scalar_expression(
    value: object,
    *,
    coordinate_bindings: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    if isinstance(value, QuantitySelector):
        return {
            "type": value.type,
            "spec": _encode_direct_quantity(
                value.spec, coordinate_bindings=coordinate_bindings
            ),
            "projection": value.projection,
        }
    if isinstance(value, QuantitySum):
        return {
            "type": "quantity_sum",
            "terms": [
                _encode_scalar_expression(
                    term,
                    coordinate_bindings=coordinate_bindings,
                )
                for term in value.terms
            ],
        }
    if isinstance(value, QuantityDifference):
        return {
            "type": "quantity_difference",
            "left": _encode_scalar_expression(
                value.left, coordinate_bindings=coordinate_bindings
            ),
            "right": _encode_scalar_expression(
                value.right, coordinate_bindings=coordinate_bindings
            ),
        }
    if isinstance(value, QuantityAbsolute):
        return {
            "type": "quantity_absolute",
            "operand": _encode_scalar_expression(
                value.operand, coordinate_bindings=coordinate_bindings
            ),
        }
    raise InvalidOptimizationSpec(
        "objective quantity must be a supported scalar expression",
        stage="spec_validation",
    )


def _encode_direct_quantity(
    spec: DiagonalRootSpec
    | OperatorElementRootSpec
    | HybridizedPoleSpec
    | TransferZeroSpec
    | ResidueNormalizedCouplingSpec
    | ResponseElementSpec
    | OperatorSpec
    | StaticCapacitanceSpec
    | StaticCapacitanceElementSpec,
    *,
    coordinate_bindings: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    if isinstance(spec, DiagonalRootSpec):
        return _encode_root(spec, coordinate_bindings=coordinate_bindings)
    if isinstance(spec, OperatorElementRootSpec):
        return _encode_element_root(spec, coordinate_bindings=coordinate_bindings)
    if isinstance(spec, HybridizedPoleSpec):
        return {
            "type": "hybridized_pole",
            "coordinates": [
                _bound_coordinate_id(value, coordinate_bindings)
                for value in spec.coordinates
            ],
            "anchor": _frequency_anchor_envelope(spec.anchor),
            **_encoded_root_search(spec),
        }
    if isinstance(spec, TransferZeroSpec):
        return {
            "type": "transfer_zero",
            "anchor": _frequency_anchor_envelope(spec.anchor),
            "family": spec.family,
            "input_coordinate": _bound_coordinate_id(
                spec.input_coordinate, coordinate_bindings
            ),
            "output_coordinate": _bound_coordinate_id(
                spec.output_coordinate, coordinate_bindings
            ),
            **_encoded_root_search(spec),
        }
    if isinstance(spec, ResidueNormalizedCouplingSpec):
        return {
            "type": "residue_normalized_coupling",
            "branch_a": _encode_direct_quantity(
                spec.branch_a, coordinate_bindings=coordinate_bindings
            ),
            "branch_b": _encode_direct_quantity(
                spec.branch_b, coordinate_bindings=coordinate_bindings
            ),
            "frequency": (
                spec.frequency
                if isinstance(spec.frequency, str)
                else quantity_envelope(
                    spec.frequency, si_unit="hertz", registry=units.registry
                )
            ),
        }
    if isinstance(spec, ResponseElementSpec):
        return {
            "type": "response_element",
            "family": spec.family,
            "input_coordinate": _bound_coordinate_id(
                spec.input_coordinate, coordinate_bindings
            ),
            "output_coordinate": _bound_coordinate_id(
                spec.output_coordinate, coordinate_bindings
            ),
            "frequency": quantity_envelope(
                spec.frequency, si_unit="hertz", registry=units.registry
            ),
        }
    if isinstance(spec, (StaticCapacitanceSpec, StaticCapacitanceElementSpec)):
        record = {"type": "static_capacitance"}
        if isinstance(spec, StaticCapacitanceElementSpec):
            record.update(type="static_capacitance_element",
                          row=_bound_coordinate_id(spec.row, coordinate_bindings),
                          column=_bound_coordinate_id(spec.column, coordinate_bindings))
        return record
    if isinstance(spec, OperatorSpec):
        return {"type": "operator", "frequencies": _frequency_grid(spec.frequencies)}
    raise TypeError("unsupported Direct quantity Spec")


def _optimization_variables(declared_variables, parameters):
    """Resolve the existing physical variable/domain declaration at a seed."""
    variables: list[dict[str, object]] = []
    baseline_optimizer_coordinates: list[str] = []
    for variable in declared_variables:
        parameter = variable.parameter
        if isinstance(parameter.spec, RLGCParameterSpec):
            raise InvalidOptimizationSpec(
                "RLGC parameters cannot be continuous optimization variables",
                stage="spec_validation",
            )
        parameter_unit = parameter.spec.si_unit
        baseline_value = float(parameters.values[parameter].to(parameter_unit).magnitude)
        if variable.domain is not None:
            domain = variable.domain.value
            if domain in {"POSITIVE", "NEGATIVE"} and (
                baseline_value == 0.0 or (baseline_value > 0.0) != (domain == "POSITIVE")
            ):
                raise InvalidOptimizationSpec("signed log domain requires a strictly signed initial value", stage="spec_validation")
            if domain == "NONNEGATIVE" and baseline_value < 0.0 or domain == "NONPOSITIVE" and baseline_value > 0.0:
                raise InvalidOptimizationSpec("initial value is outside optimization domain", stage="spec_validation")
            scale = None
            if variable.scale is not None:
                try:
                    scale_value = float(variable.scale.to(parameter_unit).magnitude)
                except Exception as error:
                    raise InvalidOptimizationSpec("domain scale must match the parameter unit", stage="spec_validation") from error
                if not math.isfinite(scale_value) or scale_value <= 0.0 or (
                    domain != "UNBOUNDED" and not math.isfinite(baseline_value / scale_value)
                ):
                    raise InvalidOptimizationSpec("domain scale must be finite, positive, and same-unit", stage="spec_validation")
                scale = quantity_envelope(variable.scale, si_unit=parameter_unit, registry=units.registry)
            variables.append({"parameter": parameter._key_record(), "domain": domain,
                              "transform": variable.transform, "scale": scale})
            baseline_optimizer_coordinates.append(float64_hex(0.0))
            continue
        assert variable.bounds is not None and variable.model_default_bounds is not None
        lower, upper = variable.bounds
        low = quantity_envelope(lower, si_unit=parameter_unit, registry=units.registry)
        high = quantity_envelope(upper, si_unit=parameter_unit, registry=units.registry)
        low_value = float(lower.to(parameter_unit).magnitude)
        high_value = float(upper.to(parameter_unit).magnitude)
        if low_value >= high_value:
            raise InvalidOptimizationSpec(
                "optimization lower bound must be below upper bound",
                stage="spec_validation",
            )
        if not low_value <= baseline_value <= high_value:
            raise InvalidOptimizationSpec(
                "sealed baseline must lie within resolved variable bounds",
                stage="spec_validation",
            )
        if variable.transform == "log" and low_value <= 0.0:
            raise InvalidOptimizationSpec(
                "log optimization bounds must be strictly positive",
                stage="spec_validation",
            )
        coordinate = (
            (baseline_value - low_value) / (high_value - low_value)
            if variable.transform == "linear"
            else math.log(baseline_value / low_value)
            / math.log(high_value / low_value)
        )
        if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
            raise InvalidOptimizationSpec(
                "sealed baseline has no finite unit-box coordinate",
                stage="spec_validation",
            )
        baseline_optimizer_coordinates.append(float64_hex(coordinate))
        default = [
            quantity_envelope(item, si_unit=parameter_unit, registry=units.registry)
            for item in variable.model_default_bounds
        ]
        override = (
            None
            if variable.consumer_override_bounds is None
            else [
                quantity_envelope(item, si_unit=parameter_unit, registry=units.registry)
                for item in variable.consumer_override_bounds
            ]
        )
        variables.append(
            {
                "parameter": parameter._key_record(),
                "model_default_bounds": default,
                "consumer_override_bounds": override,
                "lower": low,
                "upper": high,
                "transform": variable.transform,
            }
        )
    return variables, baseline_optimizer_coordinates


def _optimizer_controls(optimizer, declared_variables, parameters, *, discretization, baseline_coordinates):
    """Encode explicit controller controls; compilation owns realized meshes."""
    if isinstance(optimizer, CMAESSpec):
        population = optimizer.population_size or (4 + math.floor(3 * math.log(len(declared_variables))))
        generations = (optimizer.max_evaluations - 1) // population
        if generations < 1:
            raise ValueError("CMA-ES budget must fit the baseline and one complete generation")
        unused = optimizer.max_evaluations - (1 + generations * population)
        return {
            "type": "cma_es", "seed": optimizer.seed,
            "max_evaluations": optimizer.max_evaluations,
            "population_size": optimizer.population_size,
            "resolved_population_size": population,
            "initial_sigma_f64": float64_hex(optimizer.initial_sigma),
            "baseline_optimizer_coordinates_f64": baseline_coordinates,
            "box_transform_id": ("cmaes-jl-0.2.6-native-domains.v1"
                                 if any(v.domain is not None for v in declared_variables)
                                 else "cmaes-jl-0.2.6-linquad-unit-box.v1"),
            "complete_generations": generations, "unused_evaluations": unused,
            "hidden_stops": "disabled",
        }
    if isinstance(optimizer, (LBFGSBSpec, ProjectedLBFGSSpec)):
        scales = optimizer.coordinate_scales
        scale_rows = None
        if scales is not None:
            declared_keys = {(v.parameter.definitions_id, v.parameter.id) for v in declared_variables}
            supplied_keys = {(p.definitions_id, p.id) for p in scales}
            if declared_keys != supplied_keys:
                raise InvalidOptimizationSpec("coordinate_scales must bind every local optimization variable", stage="request_encode")
            by_key = {(p.definitions_id, p.id): value for p, value in scales.items()}
            scale_rows = [{"parameter": v.parameter._key_record(),
                           "value": quantity_envelope(by_key[(v.parameter.definitions_id, v.parameter.id)],
                                                      si_unit=v.parameter.spec.si_unit,
                                                      registry=units.registry)}
                          for v in declared_variables]
        return {"type": "projected_lbfgs" if isinstance(optimizer, ProjectedLBFGSSpec) else "lbfgsb",
                "max_evaluations": optimizer.max_evaluations,
                "max_iterations": optimizer.max_iterations,
                "ftol_f64": float64_hex(optimizer.ftol), "gtol_f64": float64_hex(optimizer.gtol),
                "max_line_search": optimizer.max_line_search,
                "commit_every_iterations": optimizer.commit_every_iterations,
                "coordinate_scales": scale_rows}
    if isinstance(optimizer, HybridSpec):
        stages = []
        for ordinal, stage in enumerate(optimizer.stages):
            seed_values = dict(parameters.values)
            if stage.seed is not None:
                captured = {(p.definitions_id, p.id): p for p in seed_values}
                for parameter, value in stage.seed.values.items():
                    seed_values[captured[(parameter.definitions_id, parameter.id)]] = value
            seed = ParameterSet(seed_values, allow_extrapolation=parameters.allow_extrapolation)
            _, stage_coordinates = _optimization_variables(declared_variables, seed)
            if ordinal > 0 and stage.seed is None:
                stage_coordinates = None  # Actual selected point is sealed at transition.
            mesh = stage.discretization if stage.discretization is not None else discretization
            stages.append({"id": stage.id,
                           "optimizer": _optimizer_controls(stage.optimizer, declared_variables, seed, discretization=mesh, baseline_coordinates=stage_coordinates),
                           "discretization": None if mesh is None else mesh._canonical_record(),
                           "seed": None if stage.seed is None else seed._record()})
        return {"type": "hybrid", "stages": stages,
                "comparison_stage_id": optimizer.comparison_stage_id}
    raise TypeError("optimizer must be CMAESSpec, LBFGSBSpec, ProjectedLBFGSSpec or HybridSpec")


def _encode_spec(
    spec: DirectSolveSpec

    | DiagonalRootSpec
    | OperatorElementRootSpec
    | HybridizedPoleSpec
    | TransferZeroSpec
    | ResidueNormalizedCouplingSpec
    | ResponseElementSpec
    | OperatorSpec
    | StaticCapacitanceSpec
    | StaticCapacitanceElementSpec
    | OptimizationSpec,
    parameters: ParameterSet,
    *,
    coordinate_bindings: Mapping[tuple[str, str], str],
    trace_channels: Mapping[str, str],
    optimization_quantities: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    def trace_record(trace: SParameterTrace) -> dict[str, object]:
        record = dict(trace._canonical_record())
        try:
            record["input_port"] = trace_channels[trace.input_port]
            record["output_port"] = trace_channels[trace.output_port]
        except KeyError as error:
            raise CompilerInvariantError(
                "prepared trace-channel binding is incomplete",
                stage="request_encode",
            ) from error
        return record

    if isinstance(spec, DirectSolveSpec):
        return {
            "type": "direct_solve",
            "frequencies": _frequency_grid(spec.frequencies),
            "traces": [trace_record(trace) for trace in spec.traces],
        }
    if isinstance(
        spec,
        (
            DiagonalRootSpec,
            OperatorElementRootSpec,
            HybridizedPoleSpec,
            TransferZeroSpec,
            ResidueNormalizedCouplingSpec,
            ResponseElementSpec,
            OperatorSpec,
            StaticCapacitanceSpec,
            StaticCapacitanceElementSpec,
        ),
    ):
        return _encode_direct_quantity(spec, coordinate_bindings=coordinate_bindings)
    if optimization_quantities is None or len(optimization_quantities) != len(
        spec.objectives
    ):
        raise CompilerInvariantError(
            "optimization quantity normalization is incomplete",
            stage="request_encode",
        )
    variables, baseline_optimizer_coordinates = _optimization_variables(spec.variables, parameters)
    objectives: list[dict[str, object]] = []
    for objective, quantity in zip(spec.objectives, optimization_quantities):
        objective_unit = _expression_unit(objective.quantity)
        target = quantity_envelope(
            objective.target, si_unit=objective_unit, registry=units.registry
        )
        target_value = abs(float(objective.target.to(objective_unit).magnitude))
        if objective.scale is None:
            if target_value == 0.0:
                if objective.target.dimensionless:
                    scale_value = units.registry.Quantity(1.0, "dimensionless")
                    scale_source = "dimensionless_unity"
                else:
                    raise ValueError(
                        "a dimensional zero target requires an explicit objective scale"
                    )
            else:
                scale_value = units.registry.Quantity(target_value, objective_unit)
                scale_source = "relative_target"
        else:
            scale_value = objective.scale
            scale_source = "explicit"
        scale_magnitude = float(scale_value.to(objective_unit).magnitude)
        if not math.isfinite(scale_magnitude) or scale_magnitude <= 0.0:
            raise InvalidOptimizationSpec(
                "objective scale must be finite and strictly positive",
                stage="spec_validation",
            )
        weight = float(objective.weight.to("dimensionless").magnitude)
        if not math.isfinite(weight) or weight <= 0.0:
            raise InvalidOptimizationSpec(
                "objective weight must be finite and strictly positive",
                stage="spec_validation",
            )
        objectives.append(
            {
                "id": objective.id,
                "quantity": dict(quantity),
                "comparison": objective.comparison,
                "target": target,
                "weight_f64": float64_hex(weight),
                "resolved_scale": quantity_envelope(
                    scale_value, si_unit=objective_unit, registry=units.registry
                ),
                "scale_source": scale_source,
            }
        )
    return {
        "type": "optimization",
        "variables": variables,
        "objectives": objectives,
        "optimizer": _optimizer_controls(spec.optimizer, spec.variables, parameters,
                                          discretization=spec.discretization, baseline_coordinates=baseline_optimizer_coordinates),
        **({"discretization": None if spec.discretization is None else spec.discretization._canonical_record()}
           if spec.discretization is not None or not isinstance(spec.optimizer, CMAESSpec) else {}),
        "allow_extrapolation": [
            parameter._key_record() for parameter in spec.allow_extrapolation
        ],
    }
