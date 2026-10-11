"""Worker-local prepared candidate and continuation state.

The coordinator owns candidate admission, ordering, objective aggregation and
all durable state. This module owns only one actor's current candidate slots
and the actual numerical evaluation of an ordered selector wave.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Mapping

from ..canonical import canonical_json_bytes
from ..errors import CompilerInvariantError, SCNSimError
from .quantities import (
    EvaluationFailure,
    QuantityEvaluator,
    expression_leaves,
    quantity_body_id,
    quantity_record,
    result_from_record,
)
from ..compilation.compiler import compile_model, parameter_key, parameter_values
from ..numerics.models import EvaluationResult, NumericalBackend
from ..compilation.mesh import mesh_from_record
from ..numeric_encoding import record_bytes, record_document
from ..compilation.views import realize_view


DECLARATION_SCHEMA = "scnsim.candidate-actor-declaration.v1"
POINT_SCHEMA = "scnsim.candidate-point.v1"
PREPARATION_SCHEMA = "scnsim.candidate-preparation.v1"
WAVE_SCHEMA = "scnsim.candidate-wave.v1"
EVALUATION_SCHEMA = "scnsim.candidate-evaluation.v1"
ANCHOR_SCHEMA = "scnsim.candidate-anchors.v1"


def _read_record(payload: bytes, *, expected_schema: str) -> dict[str, object]:
    document = record_document(payload)
    if document.get("schema") != expected_schema or document.get("schema_version") != 1:
        raise CompilerInvariantError(
            "candidate actor received an unsupported record",
            stage="candidate_protocol",
            evidence={"expected_schema": expected_schema, "actual_schema": document.get("schema")},
        )
    return document


def encode_point(
    *,
    operation_id: str,
    generation: int,
    population_column: int,
    parameter_set: dict[str, object],
    baseline: bool = False,
) -> bytes:
    """Encode one already-admitted point using the existing parameter record."""
    parameter_bytes = canonical_json_bytes(parameter_set)
    return record_bytes({
        "schema": POINT_SCHEMA,
        "schema_version": 1,
        "operation_id": operation_id,
        "generation": generation,
        "population_column": population_column,
        "candidate_key": parameter_bytes.decode("utf-8"),
        "parameter_digest": sha256(parameter_bytes).hexdigest(),
        "parameter_set": parameter_set,
        "baseline": baseline,
    })


def encode_wave(
    *,
    operation_id: str,
    generation: int,
    population_column: int,
    parameter_set: dict[str, object],
    objective_index: int,
    term_index: int,
    selector: dict[str, object],
    baseline: bool = False,
) -> bytes:
    """Encode one ordered objective-term request for a prepared candidate."""
    parameter_bytes = canonical_json_bytes(parameter_set)
    return record_bytes({
        "schema": WAVE_SCHEMA,
        "schema_version": 1,
        "operation_id": operation_id,
        "generation": generation,
        "population_column": population_column,
        "candidate_key": parameter_bytes.decode("utf-8"),
        "parameter_digest": sha256(parameter_bytes).hexdigest(),
        "parameter_set": parameter_set,
        "objective_index": objective_index,
        "term_index": term_index,
        "selector": selector,
        "baseline": baseline,
    })


def encode_anchors(
    anchor_references: Mapping[str, str],
    dependency_bodies: Mapping[str, dict[str, object]],
) -> bytes:
    """Transport the parent's committed anchor references and their real bodies."""
    return record_bytes({
        "schema": ANCHOR_SCHEMA,
        "schema_version": 1,
        "anchors": dict(anchor_references),
        "bodies": dict(dependency_bodies),
    })


@dataclass(slots=True)
class _CandidateState:
    operation_id: str
    generation: int
    population_column: int
    candidate_key: str
    parameter_sha256: str
    parameter_set: dict[str, object]
    values: dict
    baseline: bool
    discretization: object
    views: dict[str, object]
    preparation_cache: dict
    result_cache: dict = field(default_factory=dict)
    dependency_bodies: dict = field(default_factory=dict)
    delivered_body_ids: set[str] = field(default_factory=set)

    @property
    def residency_key(self) -> tuple[str, int, int, str]:
        return (self.operation_id, self.generation, self.population_column, self.parameter_sha256)


class CandidateActor:
    """One pool-owned actor, created inside its numerical owner context."""

    def __init__(self, declaration_bytes: bytes, *, backend: NumericalBackend, emit=None):
        self.declaration = _read_record(
            declaration_bytes, expected_schema=DECLARATION_SCHEMA
        )
        self.operation_id = self.declaration["operation_id"]
        plan = self.declaration["plan"]
        request = self.declaration["request"]
        self.plan = plan["document"]
        self.analysis = request["document"]
        self.plan_sha256 = plan["sha256"]
        self.request_sha256 = request["sha256"]
        self.source_units = tuple(self.declaration["source_units"])
        self.mesh = mesh_from_record(self.declaration["mesh"])
        self.precision = self.declaration["precision"]
        self.resources = self.declaration["resources"]
        self.population_size = self.declaration["population_size"]
        self.backend = backend
        self.backend_identity = backend.identity()
        self.emit = emit
        self.spec = self.analysis["spec"]
        source = self.analysis["parameter_source"]
        baseline_set = (
            source["parameters"] if source["kind"] == "point"
            else source["baseline_parameters"] if source["kind"] == "points"
            else source["base_parameters"]
        )
        self.baseline_values = parameter_values(baseline_set)
        self.authorized = {
            parameter_key(record)
            for record in self.spec.get(
                "allow_extrapolation", baseline_set.get("allow_extrapolation", [])
            )
        }
        self.quantity_evaluator = QuantityEvaluator(
            backend, emit=None, continuation_step_scope=self._continuation_step_scope
        )
        self.anchors: dict[str, EvaluationResult] = {}
        self._candidates: dict[tuple[str, int, int, str], _CandidateState] = {}
        self._template_cache: dict = {}
        self._continuation_caches = None
        self._closed = False

    def _state_from_record(self, document: dict[str, object]) -> _CandidateState:
        parameters = document["parameter_set"]
        parameter_bytes = canonical_json_bytes(parameters)
        key = parameter_bytes.decode("utf-8")
        digest = sha256(parameter_bytes).hexdigest()
        if key != document["candidate_key"] or digest != document["parameter_digest"]:
            raise CompilerInvariantError(
                "candidate actor routing identity disagrees with parameters",
                stage="candidate_protocol",
            )
        values = parameter_values(parameters)
        # Material preparation belongs to this candidate and its continuation,
        # rather than retaining every material encountered by the actor.
        preparation_cache: dict = {}
        raw = compile_model(
            self.plan, values, mesh=self.mesh, authorized=self.authorized,
            preparation_cache=preparation_cache, template_cache=self._template_cache,
        )
        views = {}
        for objective in self.spec["objectives"]:
            for selector in expression_leaves(objective["quantity"]):
                view_document = selector["view"]
                view_key = canonical_json_bytes(view_document).decode("utf-8")
                if view_key not in views:
                    views[view_key] = realize_view(
                        raw, view_document, template_cache=self._template_cache
                    )
        return _CandidateState(
            operation_id=document["operation_id"],
            generation=document["generation"],
            population_column=document["population_column"],
            candidate_key=key,
            parameter_sha256=digest,
            parameter_set=parameters,
            values=values,
            baseline=bool(document.get("baseline", False)),
            discretization=json.loads(raw.evidence_bytes)["discretization"],
            views=views,
            preparation_cache=preparation_cache,
        )

    def _route(self, document: dict[str, object]) -> tuple[str, int, int, str]:
        if document["operation_id"] != self.operation_id:
            raise CompilerInvariantError(
                "candidate actor received a different operation",
                stage="candidate_protocol",
            )
        parameters = document["parameter_set"]
        encoded = canonical_json_bytes(parameters)
        key = encoded.decode("utf-8")
        digest = sha256(encoded).hexdigest()
        if key != document["candidate_key"] or digest != document["parameter_digest"]:
            raise CompilerInvariantError(
                "candidate actor routing identity disagrees with parameters",
                stage="candidate_protocol",
            )
        return (self.operation_id, document["generation"], document["population_column"], digest)

    @staticmethod
    def _error_fields(error: SCNSimError) -> dict[str, object]:
        return {
            "error_type": type(error).__name__,
            "kind": error.kind,
            "stage": error.stage,
            "detail": str(error),
            "evidence": dict(error.evidence),
        }

    @staticmethod
    def _numerical_failure(failure) -> dict[str, object]:
        return {
            "kind": failure.kind,
            "stage": failure.stage,
            "detail": failure.detail,
            "evidence": record_document(failure.evidence_bytes),
        }

    def prepare(self, point_bytes: bytes, *, generation: int) -> bytes:
        point = _read_record(point_bytes, expected_schema=POINT_SCHEMA)
        if point["operation_id"] != self.operation_id or point["generation"] != generation:
            raise CompilerInvariantError(
                "candidate preparation routing identity disagrees",
                stage="candidate_protocol",
            )
        route = self._route(point)
        response = {
            "schema": PREPARATION_SCHEMA,
            "schema_version": 1,
            "operation_id": route[0],
            "generation": route[1],
            "population_column": route[2],
            "candidate_key": point["candidate_key"],
            "parameter_digest": route[3],
        }
        try:
            state = self._state_from_record(point)
        except SCNSimError as error:
            response.update(status="error", failure=self._error_fields(error))
            return record_bytes(response)
        self._candidates[state.residency_key] = state
        response.update(
            status="prepared",
            discretization=state.discretization,
            views={
                key: {
                    "lineage": json.loads(view.lineage_bytes),
                    "terminal_ids": list(view.terminal_ids),
                }
                for key, view in state.views.items()
            },
        )
        return record_bytes(response)

    @contextmanager
    def _continuation_step_scope(self):
        """Keep one intermediate geometry alive only through its numerical call."""
        previous = self._continuation_caches
        caches = ({}, {})  # Material preparation and compiler/View templates.
        self._continuation_caches = caches
        try:
            with self.backend.continuation_step_scope():
                yield
        finally:
            self._continuation_caches = previous
            for cache in caches:
                cache.clear()

    def _candidate_view(self, state: _CandidateState, values: dict, declaration: dict):
        preparation, templates = (
            (state.preparation_cache, self._template_cache)
            if self._continuation_caches is None else self._continuation_caches
        )
        raw = compile_model(
            self.plan, values, mesh=self.mesh, authorized=self.authorized,
            preparation_cache=preparation, template_cache=templates,
        )
        return realize_view(raw, declaration, template_cache=templates)

    def _anchor_references(self) -> dict[str, str]:
        return {key: quantity_body_id(result)[0] for key, result in self.anchors.items()}

    def evaluate(self, command_bytes: bytes) -> bytes:
        command = _read_record(command_bytes, expected_schema=WAVE_SCHEMA)
        route = self._route(command)
        state = self._candidates[route]
        objective_index = command["objective_index"]
        term_index = command["term_index"]
        selector = command["selector"]
        identity = f"candidate:{route[2]}:objective:{objective_index}:term:{term_index}"
        view = state.views[canonical_json_bytes(selector["view"]).decode("utf-8")]
        response = {
            "schema": EVALUATION_SCHEMA,
            "schema_version": 1,
            "operation_id": route[0],
            "generation": route[1],
            "population_column": route[2],
            "candidate_key": state.candidate_key,
            "parameter_digest": route[3],
            "objective_index": objective_index,
            "term_index": term_index,
        }
        try:
            evaluated = self.quantity_evaluator.evaluate(
                selector["spec"], view, view_declaration=selector["view"],
                identity=identity, baseline_values=self.baseline_values,
                values=state.values, anchors=self.anchors,
                result_cache=state.result_cache, baseline=state.baseline,
                candidate_view=lambda values_at: self._candidate_view(
                    state, values_at, selector["view"]
                ),
                selector=selector, dependency_bodies=state.dependency_bodies,
                candidate_failure_conversion=True,
            )
        except EvaluationFailure as error:
            state.dependency_bodies.update(error.dependencies)
            response.update(
                status="numerical_failure",
                failure=self._numerical_failure(error.failure),
                result=None if error.result is None else quantity_record(error.result),
                dependencies=self._new_dependencies(state),
                lineage=json.loads(view.lineage_bytes),
                discretization=state.discretization,
                terminal_ids=list(view.terminal_ids),
            )
        except SCNSimError as error:
            response.update(
                status="error",
                failure=self._error_fields(error),
                dependencies=self._new_dependencies(state),
                lineage=json.loads(view.lineage_bytes),
                discretization=state.discretization,
                terminal_ids=list(view.terminal_ids),
            )
        else:
            state.dependency_bodies.update(evaluated["dependencies"])
            body_id = evaluated["body_id"]
            dependencies = self._new_dependencies(state)
            result = quantity_record(evaluated["result"])
            response.update(
                status="evaluated",
                value=evaluated["value"],
                body_id=body_id,
                result=result if body_id in dependencies else None,
                dependencies=dependencies,
                lineage=json.loads(view.lineage_bytes),
                discretization=state.discretization,
                terminal_ids=list(view.terminal_ids),
            )
        if state.baseline:
            response["anchor_references"] = self._anchor_references()
        return record_bytes(response)

    @staticmethod
    def _new_dependencies(state: _CandidateState) -> dict[str, dict[str, object]]:
        new = {
            body_id: body
            for body_id, body in state.dependency_bodies.items()
            if body_id not in state.delivered_body_ids
        }
        state.delivered_body_ids.update(new)
        return new

    def install_anchors(self, anchor_bytes: bytes) -> None:
        document = _read_record(anchor_bytes, expected_schema=ANCHOR_SCHEMA)
        bodies = document["bodies"]
        self.anchors = {
            key: result_from_record(bodies[body_id])
            for key, body_id in document["anchors"].items()
        }

    def release_generation(self, generation: int) -> None:
        self._candidates = {
            key: state for key, state in self._candidates.items()
            if state.generation != generation
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.backend.close()
        finally:
            # Drained actors cannot retain a numerical ledger or lowering
            # operands through their pool/Future references after the operation.
            self._candidates.clear()
            self.anchors.clear()
            self._template_cache.clear()
            self.baseline_values.clear()
            self.declaration = self.plan = self.analysis = self.spec = None
            self.source_units = ()
            self.quantity_evaluator = None
            self.emit = None


def create_actor(declaration_bytes: bytes, *, backend_factory, emit=None) -> CandidateActor:
    """Build actual backend and candidate state inside the pool-owned actor."""
    _read_record(declaration_bytes, expected_schema=DECLARATION_SCHEMA)
    backend = backend_factory()
    # Ownership transfers only after the actor is fully constructed. Until then
    # this factory must release the backend if declaration/identity setup fails.
    try:
        return CandidateActor(declaration_bytes, backend=backend, emit=emit)
    except BaseException as original:
        try:
            backend.close()
        except BaseException as cleanup:
            original.add_note(f'Candidate backend construction cleanup: {cleanup!r}')
        raise
