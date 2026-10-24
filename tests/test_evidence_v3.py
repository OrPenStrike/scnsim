"""Assigned M5 evidence integrity, ordering and exact saved-bit regressions."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import json
from operator import index as ordinal_index
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scnsim import (CircuitRun, DirectSolveSpec, EvidenceIntegrityError,
                    DiagonalRootSpec, JVP, OperatorSpec, ParameterSpace, ReductionPipeline,
                    ResponseElementSpec, SensitivitySpec,
                    UnsupportedEvidenceVersionError, units as u)
from scnsim.numeric_encoding import array_from_record, array_record
from scnsim.results.memory_analysis import InMemoryAnalysisResult
from scnsim.results.matrix import OperatorResult
from scnsim.workspace.binary_blocks import MAGIC, record_bytes, record_document
from scnsim.workspace.operation_store import OperationStore
from tests.test_workspace_readback import _primitive_plan
from tests.test_workspace_optimization_readback import _fixture as _optimization_fixture, _saved_values


def _bits(array):
    value = np.asarray(array)
    return {"dtype": value.dtype.str, "shape": list(value.shape),
            "bytes": value.tobytes(order="C").hex()}


def _tree(value):
    if isinstance(value, Mapping):
        if set(value) == {"dtype", "shape", "data_hex"}:
            return _bits(array_from_record(dict(value)))
        return {key: _tree(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tree(child) for child in value]
    if isinstance(value, np.ndarray):
        return _bits(value)
    if isinstance(value, (float, np.floating)):
        return {"f64": struct.pack(">d", float(value)).hex()}
    if isinstance(value, np.generic):
        return _tree(value.item())
    return value


def _saved(result):
    values = {}
    for name in ("S", "Y", "Z"):
        view = getattr(result, name.lower()).view
        matrix = view.matrix
        values[name] = {"matrix": _bits(matrix.magnitude), "unit": str(matrix.units),
                        "frequencies": _bits(view.frequencies.magnitude),
                        "frequency_unit": str(view.frequencies.units),
                        "coordinates": list(view.coordinates),
                        "input_channels": _tree(view.input_channels),
                        "output_channels": _tree(view.output_channels)}
    return {"families": values, "evidence": _tree(result.evidence)}


def _operator_saved(result):
    if isinstance(result, InMemoryAnalysisResult):
        points = result.operator_points
    elif isinstance(result, OperatorResult):
        points = result.points
    else:
        raise TypeError("Unexpected public Operator Result type")
    return [{"matrix": _bits(point.matrix.magnitude), "unit": str(point.matrix.units),
             "frequency": _bits(point.frequency.magnitude),
             "frequency_unit": str(point.frequency.units),
             "coordinates": list(point.coordinates)} for point in points]


_CERTIFICATE_ARRAYS = ("status", "eta", "residual_numerator", "residual_denominator",
                       "performed", "present")
_CERTIFICATE_META = ("site", "role", "stage", "derivative_order")
_ROOT_FIELDS = ("omega", "slope", "status", "failure_stage", "steps", "closure", "certificate",
                "right_vector", "left_vector", "singular_values", "determinant_mantissa",
                "determinant_exponent", "determinant_derivative_mantissa",
                "determinant_derivative_exponent", "row_shifts", "normalizer_mantissa",
                "normalizer_exponent", "check_status", "certificate_status", "certificates")


def _typed(value):
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return array_record(np.asarray(value), exact_dtype=True)
    if is_dataclass(value):
        return {field.name: _typed(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _typed(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_typed(child) for child in value]
    return value


def _root_typed(value):
    # RootPacket facts recursively name their existing NamedTuple fields;
    # this differs from generic evidence tuple encoding. Build from the raw
    # packet, including Workspace's saved selected-state numerical arrays.
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return array_record(np.asarray(value), exact_dtype=True)
    if hasattr(value, "_fields"):
        return {key: _root_typed(getattr(value, key)) for key in value._fields}
    if isinstance(value, Mapping):
        return {key: _root_typed(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_root_typed(child) for child in value]
    return value


def _certificate_records(value):
    """Select actual certificate records without dropping their other fields."""
    if isinstance(value, Mapping):
        if set(_ROOT_FIELDS) <= set(value):
            yield value
        elif set(_CERTIFICATE_ARRAYS) <= set(value) and set(_CERTIFICATE_META) <= set(value):
            yield value
        else:
            for child in value.values():
                yield from _certificate_records(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _certificate_records(child)


def _certificate_signature(row, expected=None):
    # The envelope retains all fields; this signature matches a certificate to
    # its same-call native payload across existing enclosing record layouts.
    if set(_ROOT_FIELDS) <= set(row):
        return _tree({name: row[name] for name in (_ROOT_FIELDS if expected is None else expected)})
    return _tree(row)


def _saved_certificates(result):
    # M5's assigned readonly access uses the existing fixed selection and
    # integrity-verified Snapshot. No product observation helper is added.
    reader = result.s.view._fixed_reader
    rows = []
    with reader._store().reader() as snapshot:
        _, _, _, index_root, _ = reader._selection_snapshot(snapshot)
        for row in index_root["chunks"]:
            summary = row["summary_json"]
            if set(summary) != {"summary_ref"}:
                raise AssertionError("Direct certificate summary is not a sealed locator")
            body = reader._canonical(snapshot, summary["summary_ref"], "direct", "chunk_summary")
            selected = reader.selection
            for key in ("workspace_instance_id", "plan_sha256", "request_sha256", "task_id", "attempt_id"):
                if body[key] != getattr(selected, key):
                    raise AssertionError("Direct certificate belongs to another selected execution")
            if body["ordinal"] != row["ordinal"]:
                raise AssertionError("Direct certificate chunk order changed")
            rows.extend(_certificate_records(body["evidence"]))
    return _tree(rows)


# The assigned private runner may persist these test observations immediately.
# Ordinary unittest execution keeps them on the TestCase without a private path.
_OBSERVATION_SINK = None


def _compact(value):
    if isinstance(value, Mapping):
        if set(value) in ({"dtype", "shape", "bytes"}, {"dtype", "shape", "data_hex"}):
            raw = bytes.fromhex(value["bytes"] if "bytes" in value else value["data_hex"])
            return {"dtype": value["dtype"], "shape": value["shape"], "order": "C",
                    "byte_length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        return {key: _compact(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_compact(child) for child in value]
    return value


def _checkpoint(test):
    if _OBSERVATION_SINK is not None:
        try:
            _OBSERVATION_SINK(test.id(), test.observations)
        except Exception as error:
            test.observation_write_errors = getattr(test, "observation_write_errors", []) + [
                {"type": type(error).__name__, "message": str(error)}]


def _differences(expected, actual):
    differences = []
    def visit(left, right, path):
        if len(differences) == 20:
            return
        if type(left) is not type(right):
            differences.append({"path": path, "reason": "type", "expected": type(left).__name__,
                                "actual": type(right).__name__})
        elif isinstance(left, Mapping):
            if set(left) != set(right):
                differences.append({"path": path, "reason": "keys",
                                    "expected": list(left), "actual": list(right)})
            for key in left:
                if key in right:
                    visit(left[key], right[key], path + "/" + str(key))
        elif isinstance(left, list):
            if len(left) != len(right):
                differences.append({"path": path, "reason": "length", "expected": len(left), "actual": len(right)})
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, path + "/" + str(index))
                if len(differences) == 20:
                    break
        elif left != right:
            if path.endswith(("/bytes", "/data_hex")):
                a, b = bytes.fromhex(left), bytes.fromhex(right)
                offset = next((index for index, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
                differences.append({"path": path, "reason": "bits", "byte_offset": offset,
                    "expected_bytes": len(a), "actual_bytes": len(b),
                    "expected_excerpt": a[offset:offset+16].hex(), "actual_excerpt": b[offset:offset+16].hex()})
            else:
                differences.append({"path": path, "reason": "value", "expected": left, "actual": right})
    visit(expected, actual, "")
    return differences


def _assert_same(test, label, expected, actual):
    differences = _differences(expected, actual)
    test.observations.setdefault("comparisons", []).append({"label": label,
        "expected": _compact(expected), "actual": _compact(actual), "first20_differences": differences})
    _checkpoint(test)
    test.assertFalse(differences, (label, differences))
    test.assertEqual(getattr(test, "observation_write_errors", []), [])


def _retain_observer(test, key, observed, **extra):
    test.observations[key] = _compact({"raw_native": observed.raw_native,
        "direct_expected": observed.direct_expected, "primal_expected": observed.primal_expected,
        "primal_contexts": observed.primal_contexts, "derivative_expected": observed.derivative_expected,
        "claims": observed.claims, "stored_write_objects": list(observed.objects.values()),
        "candidate_reused": observed.reused, "observer_errors": observed.errors, **extra})
    _checkpoint(test)


class _SameCallObservation:
    def __init__(self):
        self.primal_expected = []
        self.primal_contexts = []
        self.derivative_expected = []
        self.raw_native = []
        self.direct_expected = []
        self.objects = {}
        self.claims = []
        self.errors = []
        self.authorities = {}
        self.candidates = {}
        self.candidate_rows = []
        self.body_routes = {}
        self.record_routes = {}
        self.raw_roots = {}
        self.reused = []
        self.publication = []

    def collect(self, callback, *args):
        try:
            callback(*args)
        except Exception as error:
            self.errors.append({"type": type(error).__name__, "message": str(error)})

    def remember(self, evidence, expected):
        # Retain the actual object to prevent Python id reuse. This association
        # is local routing, never a cross-storage identity or numerical oracle.
        previous = self.authorities.get(id(evidence))
        if previous is not None and previous[1] != expected:
            raise AssertionError("One evidence object has ambiguous native authority")
        self.authorities[id(evidence)] = (evidence, expected)

    def authority(self, evidence):
        found = self.authorities.get(id(evidence))
        if found is None or found[0] is not evidence:
            raise AssertionError("Missing same-call evidence object authority")
        return found[1]

    def checked(self, test):
        test.assertEqual(self.errors, [], self.errors)


@contextmanager
def _same_call_certificates(family):
    """Forward actual calls once; associate raw inputs through existing edges."""
    from contextvars import ContextVar
    from scnsim.execution import physical_analysis, jax_operation, quantities
    from scnsim.execution import physical_candidate, physical_optimization, memory_optimization
    from scnsim.execution.jax_backend import JaxBackend
    from scnsim.execution.operation_spool import OperationSpool
    from scnsim.numerics import evidence as native_evidence
    from scnsim.numerics.native_sparse import sparse_certificates
    from scnsim.workspace import sqlite_storage
    from scnsim.workspace.operation_store import StoreTransaction
    observed = _SameCallObservation()
    originals = []
    chunk_context = ContextVar("m5_actual_chunk", default=None)

    def wrap(owner, name, callback, *, static=False):
        original = getattr(owner, name)
        originals.append((owner, name, original, static))
        def forwarded(*args, **kwargs):
            result = original(*args, **kwargs)
            observed.collect(callback, args, kwargs, result)
            return result
        setattr(owner, name, staticmethod(forwarded) if static else forwarded)

    def direct_packet(args, kwargs, packet):
        if family not in ("direct", "operator") or kwargs.get("phase") != "native_frequency_chunk":
            return
        certificates = sparse_certificates(packet.errors)
        context = chunk_context.get()
        if context is None:
            raise AssertionError("Frequency packet has no actual work-row context")
        observed.raw_native.append({"work": context,
            "executed": _bits(packet.executed),
            "check_status": _bits(packet.outcomes.check_status),
            "certificate_status": _bits(packet.outcomes.certificate_status),
            "certificates": _tree(_typed(certificates))})
        for lane, executed in enumerate(np.asarray(packet.executed)):
            if not executed:
                continue
            for row in certificates:
                expected = {key: _tree(getattr(row, key)) for key in _CERTIFICATE_META}
                # Historical Direct detachment, independently expressed.
                for key in _CERTIFICATE_ARRAYS:
                    value = np.asarray(getattr(row, key))[lane]
                    dtype = ("<c8" if value.dtype == np.dtype("complex64") else
                             "<f4" if value.dtype == np.dtype("float32") else
                             "<c16" if np.iscomplexobj(value) else "<f8")
                    expected[key] = _bits(np.asarray(value, dtype=dtype))
                observed.direct_expected.append(expected)

    def primal(args, kwargs, result):
        if family not in ("root", "optimization"):
            expected = _tree(_typed(args[0].aux.certificates))
            observed.raw_native.append({"primal_certificates": expected})
            # Same-call inputs are the oracle, not program_evidence's return.
            observed.primal_expected.append(expected)
            observed.primal_contexts.append(_tree(args[1]))

    def sensitivity_projection(value):
        if isinstance(value, np.ndarray):
            return sensitivity_projection(value.tolist())
        if isinstance(value, np.generic):
            return sensitivity_projection(value.item())
        if is_dataclass(value):
            return {field.name: sensitivity_projection(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, Mapping):
            return {key: sensitivity_projection(child) for key, child in value.items()}
        if isinstance(value, (tuple, list)):
            return [sensitivity_projection(child) for child in value]
        return _typed(value)

    def sensitivity(args, kwargs, result):
        outcome = args[2]
        observed.raw_native.append({"derivative_certificates": _tree(_typed(outcome.aux.certificates))})
        observed.derivative_expected.append(_tree(sensitivity_projection(outcome.aux.certificates)))

    def root_packet(args, kwargs, result):
        declaration, packet = args[:2]
        raw = _tree(_root_typed(packet))
        expected = dict(raw)
        if kwargs.get("compact_evidence", False):
            expected.pop("selected_state")
        observed.remember(result.evidence_bytes, expected)
        observed.raw_roots[id(result.evidence_bytes)] = raw
        if family == "root":
            # Declaration identity is stable; packet array dtypes remain raw.
            observed.raw_native.append({"quantity_key": declaration.quantity_key, "root_packet": raw})

    def affine(args, kwargs, result):
        before = args[0]
        if family in ("root", "optimization"):
            observed.remember(result.evidence_bytes, observed.authority(before.evidence_bytes))
            observed.raw_roots[id(result.evidence_bytes)] = observed.raw_roots[id(before.evidence_bytes)]

    def searched(args, kwargs, result):
        if family not in ("root", "optimization"):
            return
        for old_aux, new_aux in ((args[0].primal.aux, result.primal.aux), (args[0].aux, result.aux)):
            before = dict(old_aux.branch_results)
            for index, value in new_aux.branch_results:
                observed.remember(value.evidence_bytes, observed.authority(before[index].evidence_bytes))
                observed.raw_roots[id(value.evidence_bytes)] = observed.raw_roots[id(before[index].evidence_bytes)]

    def quantity(args, kwargs, result):
        value = args[0]
        if family in ("root", "optimization") and list(_certificate_records(result)):
            observed.remember(result["evidence"], observed.authority(value.evidence_bytes))

    def analysis_record(args, kwargs, result):
        if family != "root":
            return
        expected = observed.authority(args[2].evidence_bytes)
        key = result["id"]
        context = {name: _tree(result[name]) for name in
                   ("id", "source_index", "work_index", "view_index", "parameters") if name in result}
        if key in observed.record_routes and observed.record_routes[key][1] != expected:
            raise AssertionError("Root record identity has ambiguous native authority")
        observed.record_routes[key] = (context, expected)

    def body_id(args, kwargs, result):
        if family != "optimization":
            return
        identifier, body = result
        rows = list(_certificate_records(body))
        if rows:
            expected = observed.authority(args[0].evidence_bytes)
            previous = observed.body_routes.get(identifier)
            if previous is not None and previous != expected:
                raise AssertionError("One logical body id has ambiguous native certificates")
            observed.body_routes[identifier] = expected

    def task_object(args, kwargs, reference):
        if family != "root":
            return
        value, role = args[1:3]
        if role == "benchmark_value" and isinstance(value, Mapping) and "evidence" in value:
            context, expected = observed.record_routes[value["id"]]
            observed.claims.append({"context": context, "reference": dict(reference), "expected": expected})

    def success(args, kwargs, candidate):
        outcome = kwargs["outcome"]
        root_rows = []
        for start, body in outcome.primal.aux.branch_results:
            # The native ordinal may be a scalar JAX array. Preserve its
            # original dtype/shape/bits separately from its exact integer value.
            root_rows.append({"start_index": {"value": ordinal_index(np.asarray(start).item()),
                "native": _bits(start)}, "packet": observed.raw_roots[id(body.evidence_bytes)]})
        raw = {"primal_certificates": _tree(_typed(outcome.primal.aux.certificates)),
               "objective_certificates": _tree(_typed(outcome.aux.certificates)),
               "roots": root_rows}
        observed.candidates[id(candidate)] = (candidate, raw)

    def candidate_row(args, kwargs, result):
        if family != "optimization":
            return
        row = args[1]
        candidate = row["candidate"]
        found = observed.candidates.get(id(candidate))
        if found is None:
            # CMA reuses an existing exact point while assigning a new logical
            # ordinal. Bind the existing parameter/body edge, not arrival order.
            matches = [entry for entry in observed.candidate_rows
                       if entry["parameters"] == _tree(_typed(candidate["parameters"]))
                       and entry["stage_id"] == candidate["stage_id"]
                       and entry["model_identity"] == _tree(candidate["model_identity"])]
            if not matches or any(item["native"] != matches[0]["native"] for item in matches):
                raise AssertionError("Ambiguous or absent cached candidate native authority")
            raw = matches[0]["native"]
            reused = True
        else:
            if found[0] is not candidate:
                raise AssertionError("Candidate object identity changed")
            raw, reused = found[1], False
        logical = {"stage_id": candidate["stage_id"],
                   "evaluation_ordinal": candidate["evaluation_ordinal"], "kind": row["kind"],
                   "parameters": _tree(_typed(candidate["parameters"])),
                   "model_identity": _tree(candidate["model_identity"]), "native": raw}
        observed.candidate_rows.append(logical)
        # Reuse is recorded separately: it does not invent a native execution.
        observed.raw_native.append(logical)
        observed.candidates[id(candidate)] = (candidate, raw)
        observed.reused.append(reused)

    def stored(args, kwargs, reference):
        role = kwargs["role"]
        if family == "optimization" and role == "optimization_candidate":
            candidate = record_document(args[1])
            logical = {key: _tree(candidate[key]) for key in
                       ("stage_id", "evaluation_ordinal", "kind", "parameters", "model_identity")}
            matches = [row for row in observed.candidate_rows
                       if all(row[key] == value for key, value in logical.items())]
            if not matches or any(row["native"] != matches[0]["native"] for row in matches):
                raise AssertionError("Published candidate lacks unambiguous same-call native authority")
            for identifier, dependency_ref in candidate["dependencies"].items():
                written = observed.objects.get((dependency_ref["sha256"], dependency_ref["role"]))
                if written is None or identifier not in observed.body_routes:
                    raise AssertionError("Published dependency lacks its own-call certificate authority")
                claim = {"context": dict(logical, body_id=identifier),
                         "candidate_reference": dict(reference),
                         "reference": dict(dependency_ref),
                         "expected": observed.body_routes[identifier]}
                observed.publication.append(claim)
            return
        if role not in {"direct_chunk_summary", "benchmark_value", "sensitivity_equation_evidence"}:
            return
        body = record_document(args[1])
        rows = list(_certificate_records(body))
        if not rows:
            raise AssertionError("Declared certificate object omitted its records")
        observed.objects[(reference["sha256"], role)] = {
            "reference": dict(reference), "certificates": _tree(rows)}

    chunk_original = physical_analysis.PhysicalAnalysis._execute_native_chunk
    def chunk(self, native, candidate, rows, omega_source, **kwargs):
        token = chunk_context.set(_tree(rows))
        try:
            return chunk_original(self, native, candidate, rows, omega_source, **kwargs)
        finally:
            chunk_context.reset(token)

    publication_original = sqlite_storage.commit_physical_stage_batch
    def publication(*args, **kwargs):
        previous = observed.publication
        # Observe only real per-row put_object calls. Never iterate, replace
        # or change the publisher's original rows operand.
        pending = []
        observed.publication = pending
        try:
            returned = publication_original(*args, **kwargs)
            observed.collect(observed.claims.extend, pending)
            return returned
        finally:
            observed.publication = previous

    try:
        if family in ("direct", "operator"):
            physical_analysis.PhysicalAnalysis._execute_native_chunk = chunk
        if family == "optimization":
            sqlite_storage.commit_physical_stage_batch = publication
        wrap(JaxBackend, "execute_native", direct_packet)
        wrap(physical_analysis, "program_evidence", primal)
        wrap(jax_operation, "_sensitivity_equation_evidence", sensitivity)
        wrap(native_evidence, "project_native_root", root_packet)
        wrap(physical_analysis.PhysicalAnalysis, "_bind_affine_evidence", affine, static=True)
        wrap(native_evidence, "attach_root_search_metadata", searched)
        wrap(quantities, "quantity_record", quantity)
        wrap(physical_analysis, "quantity_record", quantity)
        wrap(physical_analysis.PhysicalAnalysis, "record", analysis_record)
        wrap(quantities, "quantity_body_id", body_id)
        wrap(physical_candidate, "quantity_body_id", body_id)
        wrap(sqlite_storage._Objects, "put", task_object)
        wrap(physical_optimization._SavedCandidateRecords, "success", success)
        wrap(memory_optimization.MemoryCandidateRecords, "success", success)
        wrap(OperationSpool, "put_record", candidate_row)
        wrap(memory_optimization.MemoryOptimizationRecords, "put_record", candidate_row)
        wrap(StoreTransaction, "put_object", stored)
        yield observed
    finally:
        primary = sys.exception()
        cleanup_errors = []
        restorations = [(physical_analysis.PhysicalAnalysis, "_execute_native_chunk", chunk_original, False),
                        (sqlite_storage, "commit_physical_stage_batch", publication_original, False)]
        restorations.extend(reversed(originals))
        for owner, name, original, static in restorations:
            try:
                setattr(owner, name, staticmethod(original) if static else original)
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            if primary is not None:
                for error in cleanup_errors:
                    primary.add_note("Observer restoration also failed: " + repr(error))
            else:
                first, *secondary = cleanup_errors
                for error in secondary:
                    first.add_note("Observer restoration also failed: " + repr(error))
                raise first


def _stored_observations(workspace, observations, test):
    # Each payload is fetched through the existing hash/length/role verifier.
    paths = list(Path(workspace).rglob("operations.sqlite3"))
    if len(paths) != 1:
        raise AssertionError("M5 fixture must have exactly one OperationStore")
    store = OperationStore.open_readonly(paths[0].parent)
    result = []
    with store.reader() as snapshot:
        for row in observations.objects.values():
            body = record_document(snapshot.get_object(row["reference"]))
            actual = _tree(list(_certificate_records(body)))
            _assert_same(test, "verified_object_vs_write/" + row["reference"]["sha256"], row["certificates"], actual)
            result.append(row)
    return result


def _public_values(family, result):
    if family == "direct":
        return _saved(result)
    if family == "operator":
        return _operator_saved(result)
    if family == "root":
        return {key: _saved_values(getattr(result, key))
                for key in ("root", "frequency", "linewidth", "slope")}
    if family == "sensitivity":
        return {"status": result.status, "quantities": _saved_values(result.quantities),
                "primal": _saved_values(tuple(result.primal)),
                "operations": {key: {"status": value.status,
                    "blocks": _saved_values(tuple(value.blocks))}
                    for key, value in result.operations.items()}}
    if family == "optimization":
        def best(value):
            return None if value is None else {key: _saved_values(getattr(value, key))
                for key in ("parameters", "cost", "evaluation_ordinal", "status")}
        def components(row):
            if isinstance(row, Mapping) and row.get("kind") == "baseline":
                return [{"id": value["id"], "status": value["status"],
                    **({"weighted_cost_f64": value["cost_f64"],
                        "normalized_residual_f64": value["normalized_residual_f64"]}
                        if value["status"] == "success" else {}),
                    "terms": [{"term_ordinal": term["term_ordinal"], "status": term["status"],
                        **({"value_f64": term["value_f64"], "actual_complex": dict(term["actual_complex"])}
                           if term["status"] == "success" else {})} for term in value["terms"]]}
                    for value in row["objectives"]]
            if isinstance(row, Mapping):
                return [{"id": value["objective_id"], "status": value["status"],
                    **({"weighted_cost_f64": value["weighted_cost_f64"],
                        "normalized_residual_f64": value["normalized_residual_f64"]}
                        if value["status"] == "success" else {}),
                    "terms": [{"term_ordinal": term["term_ordinal"], "status": term["status"],
                        **({"value_f64": term["value_f64"], "actual_complex": dict(term["actual_complex"])}
                           if term["status"] == "success" else {})} for term in value["terms"]]}
                    for value in row["components"]]
            return [{"id": value["id"], "status": value["status"],
                **({"weighted_cost_f64": struct.pack(">d", value["weighted_cost"]).hex(),
                    "normalized_residual_f64": struct.pack(">d", value["normalized_residual"]).hex()}
                    if value["status"] == "success" else {}),
                "terms": [{"term_ordinal": term["term_ordinal"], "status": term["status"],
                    **({"value_f64": struct.pack(">d", term["value"]).hex(),
                        "actual_complex": {"real_f64": struct.pack(">d", complex(term["actual_complex"]).real).hex(),
                            "imag_f64": struct.pack(">d", complex(term["actual_complex"]).imag).hex()}}
                       if term["status"] == "success" else {})} for term in value["terms"]]}
                for value in row.objective_components]
        def history(stage):
            rows = []
            for row in stage.history:
                if isinstance(row, Mapping):
                    # Public Workspace history is an encoded scalar record;
                    # decode its declared SI values explicitly for comparison.
                    parameters = [{"parameter": dict(binding["parameter"]),
                        "si_unit": binding["value"]["si_unit"],
                        "si_f64": binding["value"]["si_value_f64"]}
                        for binding in row["parameters"]["bindings"]]
                    rows.append({"evaluation_ordinal": row["evaluation_ordinal"],
                        "parameters": parameters, "status": row["status"], "cost_f64": row["cost_f64"],
                        "components": components(row)})
                else:
                    rows.append({"evaluation_ordinal": row.evaluation_ordinal,
                        "parameters": [{"parameter": {"definitions_id": ref.definitions_id,
                            "parameter_id": ref.id}, "si_unit": ref.spec.si_unit,
                            "si_f64": struct.pack(">d", float(value.to(ref.spec.si_unit).magnitude)).hex()}
                            for ref, value in row.parameters.values.items()],
                        "status": row.status, "cost_f64": struct.pack(">d", row.cost).hex(),
                        "components": components(row)})
            return rows
        return {"best": best(result.best), "stages": [{
            **{key: getattr(stage, key) for key in ("strategy", "status", "termination_reason",
                                                   "evaluation_count", "iteration_count")},
            "observed_best": best(stage.observed_best),
            "accepted_iterate": best(stage.accepted_iterate),
            "history": history(stage)} for stage in result.stages]}

    raise ValueError(family)


def _family_request(run, family, coordinate, capacitance, optimization=None):
    view = run.original.reduce(ReductionPipeline().retain(coordinate))
    if family == "direct":
        return run.original, DirectSolveSpec(frequencies=[5.5, 6.0, 6.0, 6.5]*u.GHz)
    if family == "operator":
        return view, OperatorSpec(frequencies=[5.5, 6.0]*u.GHz)
    root = DiagonalRootSpec(coordinate=coordinate, root_hint=6.0*u.GHz)
    if family == "root":
        return view, root
    if family == "sensitivity":
        response = ResponseElementSpec(family="Y", input_coordinate="signal_in",
                                      output_coordinate="signal_in", frequency=5.5*u.GHz)
        return run.original, SensitivitySpec(quantities={"response": response.real.on(run.original)},
            wrt=(capacitance,), derivative=JVP(direction={capacitance: 110.0*u.fF}))
    if family == "optimization":
        return view, optimization
    raise ValueError(family)


def _resolve_family(workspace, family, stored):
    if family == "optimization":
        plan, coordinate, optimization = _optimization_fixture("cma")
        capacitance = None
    else:
        plan, coordinate, capacitance = _primitive_plan()
        optimization = None
    run = CircuitRun(plan=plan, workspace=Path(workspace), precision="float64")
    try:
        view, spec = _family_request(run, family, coordinate, capacitance, optimization)
        result = run.resolve(spec) if family == "sensitivity" else run.resolve(view, spec)
    finally:
        run.close()
    paths = list(Path(workspace).rglob("operations.sqlite3"))
    store = OperationStore.open_readonly(paths[0].parent)
    certificates = []
    with store.reader() as snapshot:
        for row in stored:
            body = record_document(snapshot.get_object(row["reference"]))
            certificates.append(_tree(list(_certificate_records(body))))
    return {"values": _public_values(family, result), "certificates": certificates,
            "direct_detached": _saved_certificates(result) if family == "direct" else None,
            "full_saved_history": _saved_values(tuple(result.stage().history)) if family == "optimization" else None}


def _scalar_saved(result):
    return {"value": _bits(result.value.magnitude), "unit": str(result.value.units)}


def _partial_saved(result):
    rows = []
    for point in result.points:
        source_index = point.source_index
        if isinstance(source_index, int):
            source_index = (source_index,)
        row = {"source_index": list(source_index), "parameters": _saved_values(point.parameters),
               "succeeded": point.succeeded}
        if point.succeeded:
            row["result"] = _scalar_saved(point.result)
        else:
            failure = point.failure
            row["failure"] = {"type": type(failure).__module__ + "." + type(failure).__qualname__,
                "kind": failure.kind, "category": failure.category, "stage": failure.stage,
                "message": str(failure), "evidence": _tree(failure.evidence)}
        rows.append(row)
    return rows


def _partial_request(run, coordinate, capacitance):
    view = run.original.reduce(ReductionPipeline().retain(coordinate))
    parameters = ParameterSpace.grid(axes={capacitance: [110.0, 1e308, 110.0]*u.fF})
    spec = ResponseElementSpec(family="Z", input_coordinate=coordinate,
                               output_coordinate=coordinate, frequency=6.0*u.GHz)
    return view, spec, parameters


def _fresh_partial(workspace):
    plan, coordinate, capacitance = _primitive_plan()
    run = CircuitRun(plan=plan, workspace=Path(workspace), precision="float64")
    try:
        view, spec, parameters = _partial_request(run, coordinate, capacitance)
        result = run.resolve(view, spec, parameters=parameters)
    finally:
        run.close()
    return _partial_saved(result)


class EvidenceV3Tests(unittest.TestCase):
    def test_literal_columns_preserve_bits_and_reject_corruption(self):
        # Signed zero and a NaN payload are data, not normalization choices.
        source = np.array([0x8000000000000000, 0x7ff8000000000042], dtype="<u8").view("<f8")
        body = {"values": array_record(source, exact_dtype=True)}
        payload = record_bytes(body)
        self.assertTrue(payload.startswith(MAGIC))
        self.assertEqual(_bits(array_from_record(record_document(payload)["values"])), _bits(source))
        for offset in (0, 8, 16):
            damaged = bytearray(payload)
            damaged[offset] ^= 0xff
            with self.subTest(offset=offset), self.assertRaises(EvidenceIntegrityError):
                record_document(bytes(damaged))
        with TemporaryDirectory() as directory:
            store = OperationStore(Path(directory), plan_sha256="a"*64, workspace_instance_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
            store.initialize()
            with store.transaction("block", expected_revisions={}) as tx:
                reference = tx.put_object(payload, role="benchmark_value")
                self.assertEqual(tx.get_object(reference), payload)
            connection = sqlite3.connect(store.path)
            try:
                damaged = payload[:-1] + bytes([payload[-1] ^ 1])
                connection.execute("UPDATE objects SET payload=? WHERE sha256=?", (damaged, reference["sha256"]))
                connection.commit()
            finally:
                connection.close()
            with store.reader() as snapshot, self.assertRaises(EvidenceIntegrityError):
                snapshot.get_object(reference)

    def test_short_missing_and_interrupted_blocks_never_publish_success(self):
        payload = record_bytes({"values": array_record(np.array([1.0, 2.0]))})
        for truncated in (payload[:10], payload[:17], payload[:-1]):
            with self.subTest(length=len(truncated)), self.assertRaises(EvidenceIntegrityError):
                record_document(truncated)
        with TemporaryDirectory() as directory:
            store = OperationStore(Path(directory), plan_sha256="a"*64, workspace_instance_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
            store.initialize()
            with store.transaction("chunk", expected_revisions={}) as tx:
                reference = tx.put_object(payload, role="benchmark_value")
            with self.assertRaisesRegex(RuntimeError, "interrupted publication"):
                with store.transaction("terminal", expected_revisions={}) as tx:
                    tx.set_pointer("success/request", reference)
                    raise RuntimeError("interrupted publication")
            with store.reader() as snapshot:
                self.assertIsNone(snapshot.read_pointer("success/request"))
                self.assertEqual(snapshot.get_object(reference), payload)
            connection = sqlite3.connect(store.path)
            try:
                connection.execute("DELETE FROM objects WHERE sha256=?", (reference["sha256"],))
                connection.commit()
            finally:
                connection.close()
            with store.reader() as snapshot, self.assertRaises(EvidenceIntegrityError):
                snapshot.get_object(reference)

    def test_optimization_history_locator_storage_integrity(self):
        from scnsim.workspace.results import FixedPhysicalOptimizationReader
        from scnsim.workspace.sqlite_storage import (
            _physical_candidate_source_locator, _physical_stage_history_baseline_locator,
        )
        # These are storage metadata fixtures, not fabricated optimizer results.
        # Exercise the same occurrence guard used after the reader's verified
        # fetch, with real immutable objects and no numerical operation.
        model = {"fixture": "history-locator"}
        source = {"stage_id": "baseline", "model_identity": model,
                  "kind": "baseline", "evaluation_ordinal": 0,
                  "source_task_id": "fixture-task", "source_attempt_id": "fixture-attempt"}
        with TemporaryDirectory() as directory:
            store = OperationStore(Path(directory), plan_sha256="a"*64,
                workspace_instance_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
            store.initialize()
            with store.transaction("locator-fixture", expected_revisions={}) as tx:
                baseline_ref = tx.put_object(record_bytes(source), role="optimization_candidate")
                evaluation_ref = tx.put_object(record_bytes(dict(source, kind="evaluation")),
                                               role="optimization_candidate")
                later_ref = tx.put_object(record_bytes(dict(source, evaluation_ordinal=1)),
                                          role="optimization_candidate")
                batch_ref = tx.put_object(record_bytes({"fixture": "source-batch"}),
                                          role="optimization_stage_batch")
            baseline = dict(source, candidate_ref=baseline_ref)
            # Real evaluation locators deliberately omit model_identity;
            # their authored stage name must not select the baseline branch.
            locators = {("baseline", baseline_ref["sha256"]): baseline,
                        ("baseline", 1): {"kind": "evaluation", "stage_id": "baseline",
                                          "evaluation_ordinal": 1, "candidate_ref": later_ref}}
            locator = _physical_stage_history_baseline_locator(
                locators, baseline_ref, "baseline", model)
            expected = {key: baseline[key] for key in (
                "kind", "stage_id", "evaluation_ordinal", "candidate_ref",
                "source_task_id", "source_attempt_id")}
            self.assertEqual(locator, expected)
            self.assertEqual(_physical_candidate_source_locator(baseline), expected)
            self.assertNotIn("source_batch_ref", locator)
            batched = dict(baseline, source_batch_ref=batch_ref)
            self.assertEqual(_physical_candidate_source_locator(batched),
                             dict(expected, source_batch_ref=batch_ref))
            self.assertIsNone(_physical_stage_history_baseline_locator(
                locators, baseline_ref, "other-stage", model))
            with store.reader() as snapshot:
                actual = record_document(snapshot.get_object(baseline_ref))
                FixedPhysicalOptimizationReader._require_baseline_occurrence(actual, locator)
                for reference in (evaluation_ref, later_ref):
                    with self.subTest(reference=reference["sha256"]):
                        substituted = dict(locator, candidate_ref=reference, source_batch_ref=batch_ref)
                        fetched = record_document(snapshot.get_object(reference))
                        with self.assertRaises(EvidenceIntegrityError):
                            FixedPhysicalOptimizationReader._require_baseline_occurrence(
                                fetched, substituted)

    def test_old_evidence_is_rejected_before_any_workspace_mutation(self):
        from scnsim.numeric_encoding import record_bytes as old_record_bytes
        from scnsim.canonical import canonical_json_bytes
        with self.assertRaises(UnsupportedEvidenceVersionError):
            record_document(old_record_bytes({"values": array_record(np.array([1.0]))}))
        for kind in ("empty_old_envelope", "requests", "zarr", "sqlite7"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                root = Path(directory)
                if kind == "empty_old_envelope":
                    (root/"workspace.json").write_bytes(canonical_json_bytes(
                        {"schema": "scnsim.workspace", "schema_version": 1}))
                elif kind == "requests":
                    (root/"requests").mkdir()
                elif kind == "zarr":
                    (root/".zgroup").write_text('{"zarr_format":2}')
                else:
                    connection = sqlite3.connect(root/"operations.sqlite3")
                    try:
                        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
                        connection.execute("INSERT INTO metadata VALUES ('schema_version','7')")
                        connection.commit()
                    finally:
                        connection.close()
                def snapshot():
                    return {str(p.relative_to(root)): ("directory" if p.is_dir() else p.read_bytes())
                            for p in root.rglob("*")}
                before = snapshot()
                with self.assertRaises(UnsupportedEvidenceVersionError):
                    if kind == "sqlite7":
                        OperationStore.open_readonly(root)
                    else:
                        plan, _, _ = _primitive_plan()
                        CircuitRun(plan=plan, workspace=root, precision="float64")
                self.assertEqual(snapshot(), before)
                self.assertFalse((root/".scnsim.lock").exists())

    def test_partial_sweep_preserves_failure_and_repeated_source_order(self):
        # Retain the original operands. The existing scalar Z operation checks
        # its Y-to-Z inverse; Operator alone has no such inverse check.
        observed = []
        self.observations = {}
        for storage in ("memory", "workspace"):
            with TemporaryDirectory() as directory:
                plan, coordinate, capacitance = _primitive_plan()
                run = CircuitRun(plan=plan, storage=storage, precision="float64",
                                 **({"workspace": Path(directory)} if storage == "workspace" else {}))
                try:
                    view, spec, parameters = _partial_request(run, coordinate, capacitance)
                    result = run.evaluate(view, spec, parameters=parameters)
                finally:
                    run.close()
                points = list(result.points)
                self.assertEqual([(point.source_index,) if isinstance(point.source_index, int)
                                  else point.source_index for point in points], [(0,), (1,), (2,)])
                self.assertEqual([point.succeeded for point in points], [True, False, True])
                failure = points[1].failure
                self.assertIsNotNone(failure)
                self.assertEqual(failure.category, "execution")
                self.assertEqual(failure.kind, "direct_response_formation")
                self.assertEqual(failure.stage, "y_to_z")
                row = _partial_saved(result)
                self.assertEqual(row[0]["result"], row[2]["result"])
                observed.append(row)
                self.observations[storage] = row
                if storage == "workspace":
                    self.assertEqual(_fresh_partial(directory), row)
                    child = subprocess.run([sys.executable, "-c",
                        "import json,sys; from tests.test_evidence_v3 import _fresh_partial; "
                        "print(json.dumps(_fresh_partial(sys.argv[1])))", directory],
                        capture_output=True, text=True, cwd=Path(__file__).parents[1])
                    self.assertEqual(child.returncode, 0, child.stderr)
                    self.assertEqual(json.loads(child.stdout), row)
        self.assertEqual(observed[0], observed[1])

    def _assert_family_storage_exact_bits(self, family):
        self.observations = {}
        with self.subTest(family=family), TemporaryDirectory() as directory:
            values, native = [], []
            for storage in ("memory", "workspace"):
                if family == "optimization":
                    plan, coordinate, optimization = _optimization_fixture("cma")
                    capacitance = None
                else:
                    plan, coordinate, capacitance = _primitive_plan()
                    optimization = None
                run = CircuitRun(plan=plan, storage=storage, precision="float64",
                    **({"workspace": Path(directory)} if storage == "workspace" else {}))
                key = family + "/" + storage
                primary = None
                actual = None
                saved = None
                try:
                    view, spec = _family_request(run, family, coordinate, capacitance, optimization)
                    with _same_call_certificates(family) as actual:
                        if family == "direct":
                            result = run.solve(view, spec)
                        elif family == "sensitivity":
                            result = run.sensitivity(spec)
                        elif family == "optimization":
                            result = run.optimize(view, spec, progress=False)
                        else:
                            result = run.evaluate(view, spec)
                    saved = _public_values(family, result)
                except BaseException as error:
                    primary = error
                    raise
                finally:
                    try:
                        run.close()
                    except BaseException as error:
                        if primary is None:
                            raise
                        primary.add_note("Run.close also failed: " + type(error).__name__ + ": " + str(error))
                    finally:
                        if actual is not None:
                            _retain_observer(self, key, actual, public_values=saved,
                                original_exception=None if primary is None else {
                                    "type": type(primary).__name__, "message": str(primary)})
                # All expected/actual manifests above already survive the
                # following assertions and fixture-directory cleanup.
                actual.checked(self)
                self.assertTrue(actual.raw_native, "Missing same-call native certificates")
                _assert_same(self, key + "/postclose_public", saved, _public_values(family, result))
                values.append(saved)
                native.append(actual.raw_native)
                if storage != "workspace":
                    continue
                archived = _stored_observations(directory, actual, self)
                self.observations[key]["stored_objects"] = _compact(archived)
                _checkpoint(self)
                self.assertTrue(archived, "No own-call stored certificate objects")
                stored_rows = [certificate for obj in archived for certificate in obj["certificates"]]
                if family == "direct":
                    _assert_same(self, key + "/all_stored_certificates", actual.direct_expected, stored_rows)
                elif family == "operator":
                    # Operator's ordinary quantity program publishes the
                    # same full typed primal certificate sequence. It does
                    # not use Direct's historical lane detachment.
                    expected_rows = [row for frame in actual.primal_expected for row in frame]
                    _assert_same(self, key + "/all_stored_certificates", expected_rows, stored_rows)
                elif family == "sensitivity":
                    self.assertEqual(len(actual.derivative_expected), 1)
                    self.assertEqual(len(archived), 1)
                    self.assertEqual(archived[0]["reference"]["role"], "sensitivity_equation_evidence")
                    _assert_same(self, key + "/all_stored_certificates", actual.derivative_expected[0], stored_rows)
                else:
                    self.assertTrue(actual.claims, "Missing logical publication claims")
                    by_ref = {(row["reference"]["sha256"], row["reference"]["role"]): row for row in archived}
                    claimed = {(row["reference"]["sha256"], row["reference"]["role"]) for row in actual.claims}
                    self.assertEqual(claimed, set(by_ref), "Unclaimed or missing certificate object")
                    expected_positions, actual_positions = [], []
                    for claim in actual.claims:
                        reference = claim["reference"]
                        rows = by_ref[(reference["sha256"], reference["role"])]["certificates"]
                        expected_positions.append([claim["expected"]])
                        # Missing raw packet fields remain a strict error.
                        # selected_state is required only for a full saved
                        # packet, never for compact Memory root histories.
                        actual_positions.append([_certificate_signature(row, claim["expected"]) for row in rows])
                    _assert_same(self, key + "/logical_stored_certificates", expected_positions, actual_positions)
                detached = _saved_certificates(result) if family == "direct" else None
                if family == "direct":
                    _assert_same(self, key + "/fixed_reader_certificates", actual.direct_expected, detached)
                original = {"values": saved, "certificates": [row["certificates"] for row in archived],
                            "direct_detached": detached,
                            "full_saved_history": _saved_values(tuple(result.stage().history)) if family == "optimization" else None}
                self.observations[key]["saved_readback_expected"] = _compact(original)
                _checkpoint(self)
                _assert_same(self, key + "/reopen", original, _resolve_family(directory, family, archived))
                command = [sys.executable, "-c",
                    "import json,sys; from tests.test_evidence_v3 import _resolve_family; "
                    "print(json.dumps(_resolve_family(sys.argv[1],sys.argv[2],json.load(sys.stdin))))",
                    directory, family]
                child = subprocess.run(command, input=json.dumps(archived), capture_output=True,
                    text=True, cwd=Path(__file__).parents[1])
                self.observations[key]["fresh_process"] = {"returncode": child.returncode, "stderr": child.stderr}
                _checkpoint(self)
                self.assertEqual(child.returncode, 0, child.stderr)
                _assert_same(self, key + "/fresh_process", original, json.loads(child.stdout))
            if family == "optimization":
                self.assertEqual(values[0]["stages"][0]["history"][0]["evaluation_ordinal"], 0)
                self.assertEqual(values[1]["stages"][0]["history"][0]["evaluation_ordinal"], 0)
            _assert_same(self, family + "/memory_workspace_values", values[0], values[1])
            if family != "operator":
                _assert_same(self, family + "/memory_workspace_native", native[0], native[1])

    def test_memory_workspace_postclose_and_fresh_process_exact_bits_direct(self):
        self._assert_family_storage_exact_bits("direct")

    def test_memory_workspace_postclose_and_fresh_process_exact_bits_operator(self):
        self._assert_family_storage_exact_bits("operator")

    def test_memory_workspace_postclose_and_fresh_process_exact_bits_root(self):
        self._assert_family_storage_exact_bits("root")

    def test_memory_workspace_postclose_and_fresh_process_exact_bits_sensitivity(self):
        self._assert_family_storage_exact_bits("sensitivity")

    def test_memory_workspace_postclose_and_fresh_process_exact_bits_optimization(self):
        self._assert_family_storage_exact_bits("optimization")


if __name__ == "__main__":
    unittest.main()
