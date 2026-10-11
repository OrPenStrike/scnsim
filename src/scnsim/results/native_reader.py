"""Fixed, lazy access to the sealed native Optimization index and artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..canonical import canonical_json_bytes, sha256_hex
from ..errors import EvidenceIntegrityError
from .base import _freeze


def _integrity(message: str, **evidence: object) -> EvidenceIntegrityError:
    return EvidenceIntegrityError(message, stage="result_decode", evidence=evidence)


class NativeResultReader:
    """Read compact native projections and only explicitly selected bodies.

    Workspace supplies a verified immutable index root and the reference to
    that root. Artifact reads go back through Workspace so each short read is
    checked against the fixed binding, terminal attempt, index root, and role
    reference. This object retains metadata only; it has no Run, Workspace, or
    open storage handle.
    """

    __slots__ = (
        "_binding_identity", "_directory", "_index_ref", "_index_metadata",
        "_baseline", "_best_ordinal", "_generation_count", "_candidate_count",
        "_population_size", "_declarations", "_request_sha256", "_attempt_sha256",
        "_ledger_refs",
    )

    def __init__(
        self,
        binding_identity: Mapping[str, object],
        result: Mapping[str, object],
        request: Mapping[str, object],
        directory: Path,
        *,
        native_index_ref: Mapping[str, object],
        index_metadata: Mapping[str, object],
    ) -> None:
        if not isinstance(index_metadata, Mapping):
            raise _integrity("Native Optimization has no verified result index metadata.")
        if not isinstance(native_index_ref, Mapping) or native_index_ref.get("role") != "native_result_index":
            raise _integrity("Native Optimization has no bound result index reference.")
        if index_metadata.get("schema") != "scnsim.native_result_index" or index_metadata.get("schema_version") != 1:
            raise _integrity("Native Optimization result index has an unsupported schema.")

        request_sha256 = sha256_hex(canonical_json_bytes(dict(request)))
        result_sha256 = sha256_hex(canonical_json_bytes(dict(result)))
        attempt_sha256 = result.get("attempt_sha256")
        attempt_identity = index_metadata.get("attempt_identity")
        native_result_ref = index_metadata.get("native_result_ref")
        expected_binding = {
            "workspace_instance_id": binding_identity.get("workspace_instance_id"),
            "plan_sha256": binding_identity.get("plan_sha256"),
            "request_sha256": request_sha256,
        }
        if any(index_metadata.get(key) != value for key, value in expected_binding.items()):
            raise _integrity("Native result index belongs to a different Workspace or request.")
        if (not isinstance(attempt_identity, Mapping)
                or attempt_identity.get("sha256") != attempt_sha256
                or not isinstance(attempt_sha256, str)):
            raise _integrity("Native result index belongs to a different terminal attempt.")
        if (not isinstance(native_result_ref, Mapping)
                or native_result_ref.get("sha256") != result_sha256
                or native_result_ref.get("role") != "native_result"):
            raise _integrity("Native result index belongs to a different result artifact.")
        if (native_index_ref.get("sha256") is None
                or native_index_ref.get("path") != "artifacts/native-index/root.json"):
            raise _integrity("Native result index reference is malformed.")
        metadata_bytes = canonical_json_bytes(dict(index_metadata))
        if (native_index_ref.get("byte_length") != len(metadata_bytes)
                or native_index_ref.get("sha256") != sha256_hex(metadata_bytes)):
            raise _integrity("Native result index metadata differs from its sealed root reference.")

        generations = index_metadata.get("generation_index_refs")
        generation_count = index_metadata.get("generation_count")
        completed = result.get("completed_generations")
        artifacts = result.get("ledger_artifacts")
        optimizer = request.get("spec", {}).get("optimizer", {})
        population_size = optimizer.get("resolved_population_size")
        if (not isinstance(generations, Sequence) or isinstance(generations, (str, bytes))
                or not isinstance(generation_count, int) or isinstance(generation_count, bool)
                or generation_count != len(generations) or generation_count != completed
                or not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes))
                or len(artifacts) != generation_count
                or not isinstance(population_size, int) or isinstance(population_size, bool)
                or population_size < 1):
            raise _integrity("Native result index generation or population counts are malformed.")
        candidate_count = index_metadata.get("population_candidate_count")
        if candidate_count != generation_count * population_size:
            raise _integrity("Native result index population count is inconsistent.")
        if result.get("result_kind") != "optimization":
            raise _integrity("Native result index is attached to a non-Optimization result.")
        if (any(not isinstance(artifact, Mapping) for artifact in artifacts)
                or any(not isinstance(generation, Mapping) for generation in generations)
                or any(generation.get("native_ledger_ref") != dict(artifact)
                       for generation, artifact in zip(generations, artifacts, strict=True))):
            raise _integrity("Native result index ledger references differ from the result envelope.")

        best = result.get("best")
        baseline = result.get("baseline")
        baseline_locator = index_metadata.get("baseline_locator")
        best_locator = index_metadata.get("best_locator")
        if (not isinstance(best, Mapping) or not isinstance(baseline, Mapping)
                or not isinstance(baseline_locator, Mapping) or not isinstance(best_locator, Mapping)
                or baseline_locator.get("evaluation_ordinal") != 0
                or best_locator.get("evaluation_ordinal") != best.get("evaluation_ordinal")):
            raise _integrity("Native result index baseline or best locator is malformed.")
        if (baseline_locator.get("member") != "baseline"
                or baseline_locator.get("native_result_ref") != dict(native_result_ref)
                or not isinstance(baseline_locator.get("summary_ref"), Mapping)
                or baseline_locator["summary_ref"].get("role") != "native_baseline_index"):
            raise _integrity("Native baseline locator differs from its result artifact.")
        best_ordinal = best.get("evaluation_ordinal")
        if best_ordinal == 0:
            if best_locator != baseline_locator:
                raise _integrity("Native baseline winner differs from the sealed baseline locator.")
        else:
            if not isinstance(best_ordinal, int) or isinstance(best_ordinal, bool) or best_ordinal < 1 or best_ordinal > candidate_count:
                raise _integrity("Native best candidate ordinal is outside its sealed population.")
            best_generation_index, best_row_index = divmod(best_ordinal - 1, population_size)
            best_generation = generations[best_generation_index]
            if (best_locator.get("generation") != best_generation_index + 1
                    or best_locator.get("row_index") != best_row_index
                    or best_locator.get("native_ledger_ref") != best_generation.get("native_ledger_ref")
                    or best_locator.get("summary_ref") != best_generation.get("summary_ref")):
                raise _integrity("Native best locator differs from its sealed generation entry.")

        # The root is detached by Workspace's verified-success selection. Keep
        # only its index rows and references; never retain candidate history.
        object.__setattr__(self, "_binding_identity", MappingProxyType(dict(binding_identity)))
        object.__setattr__(self, "_directory", Path(directory))
        object.__setattr__(self, "_index_ref", MappingProxyType(dict(native_index_ref)))
        object.__setattr__(self, "_index_metadata", _freeze(index_metadata))
        object.__setattr__(self, "_baseline", _freeze(baseline))
        object.__setattr__(self, "_best_ordinal", best["evaluation_ordinal"])
        object.__setattr__(self, "_generation_count", generation_count)
        object.__setattr__(self, "_candidate_count", candidate_count)
        object.__setattr__(self, "_population_size", population_size)
        object.__setattr__(self, "_declarations", _freeze({
            "objectives": tuple(request["spec"]["objectives"]),
            "variables": tuple(request["spec"]["variables"]),
        }))
        object.__setattr__(self, "_request_sha256", request_sha256)
        object.__setattr__(self, "_attempt_sha256", attempt_sha256)
        object.__setattr__(self, "_ledger_refs", tuple(_freeze(item) for item in artifacts))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("native result readers are immutable")

    @property
    def generation_count(self) -> int:
        return self._generation_count

    @property
    def candidate_count(self) -> int:
        return self._candidate_count

    def _generation_entry(self, index: int) -> Mapping[str, Any]:
        if index < 0:
            index += self._generation_count
        if index < 0 or index >= self._generation_count:
            raise IndexError(index)
        entry = self._index_metadata["generation_index_refs"][index]
        if (not isinstance(entry, Mapping) or entry.get("generation") != index + 1
                or entry.get("row_count") != self._population_size):
            raise _integrity("Native generation index entry is malformed.", generation=index + 1)
        ledger_ref = entry.get("native_ledger_ref")
        summary_ref = entry.get("summary_ref")
        if (not isinstance(ledger_ref, Mapping)
                or not isinstance(ledger_ref.get("path"), str)
                or not isinstance(ledger_ref.get("sha256"), str)
                or not isinstance(summary_ref, Mapping)
                or summary_ref.get("role") != "native_generation_index"):
            raise _integrity("Native generation index references are malformed.", generation=index + 1)
        if dict(ledger_ref) != dict(self._ledger_refs[index]):
            raise _integrity("Native ledger reference differs from its verified result.", generation=index + 1)
        return entry

    def _read_artifact(self, reference: Mapping[str, object]) -> Mapping[str, Any]:
        # This Workspace primitive performs the short bound read and validates
        # the fixed success/index-root/attempt chain before returning detached
        # canonical bytes. There is deliberately no path-based fallback here.
        from ..workspace.evidence import read_native_result_artifact

        value = read_native_result_artifact(
            binding_identity=self._binding_identity,
            directory=self._directory,
            index_ref=self._index_ref,
            artifact_ref=reference,
        )
        if not isinstance(value, Mapping):
            raise _integrity("Verified native result artifact is not an object.")
        return value

    def _summary_block(self, index: int) -> Mapping[str, Any]:
        entry = self._generation_entry(index)
        summary_ref = entry["summary_ref"]
        block = self._read_artifact(summary_ref)
        generation = index + 1
        expected = {
            "schema": "scnsim.native_generation_index",
            "schema_version": 1,
            "workspace_instance_id": self._binding_identity["workspace_instance_id"],
            "plan_sha256": self._binding_identity["plan_sha256"],
            "request_sha256": self._request_sha256,
            "attempt_identity": self._index_metadata["attempt_identity"],
            "native_result_ref": self._index_metadata["native_result_ref"],
            "generation": generation,
            "native_ledger_ref": entry["native_ledger_ref"],
            "first_ordinal": index * self._population_size + 1,
            "row_count": self._population_size,
        }
        if any(block.get(key) != value for key, value in expected.items()):
            raise _integrity("Native generation summary differs from its sealed index entry.",
                             generation=generation)
        rows = block.get("rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != self._population_size:
            raise _integrity("Native generation summary has an invalid candidate count.",
                             generation=generation)
        return block

    def _read_generation(self, index: int, summary: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        entry = self._generation_entry(index)
        if summary is None:
            summary = self._summary_block(index)
        ledger = self._read_artifact(entry["native_ledger_ref"])
        generation = index + 1
        if (ledger.get("schema") != "scnsim.optimization_ledger"
                or ledger.get("schema_version") != 4
                or ledger.get("request_sha256") != self._request_sha256
                or ledger.get("generation") != generation
                or ledger.get("population_size") != self._population_size
                or ledger.get("attempt_sha256") != summary.get("source_attempt_sha256")
                or ledger.get("previous_ledger_sha256") != summary.get("previous_ledger_sha256")):
            raise _integrity("Selected native ledger differs from its sealed summary locator.",
                             generation=generation)
        candidates = ledger.get("candidates")
        summary_rows = summary["rows"]
        if (not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes))
                or len(candidates) != self._population_size):
            raise _integrity("Selected native ledger has an invalid candidate count.",
                             generation=generation)
        for row_index, (candidate, indexed) in enumerate(zip(candidates, summary_rows, strict=True)):
            locator = indexed.get("locator") if isinstance(indexed, Mapping) else None
            ordinal = index * self._population_size + row_index + 1
            if (not isinstance(locator, Mapping)
                    or locator.get("native_ledger_ref") != dict(entry["native_ledger_ref"])
                    or locator.get("row_index") != row_index
                    or locator.get("evaluation_ordinal") != ordinal
                    or locator.get("generation") != generation
                    or locator.get("source_attempt_sha256") != ledger.get("attempt_sha256")
                    or not isinstance(candidate, Mapping)
                    or candidate.get("evaluation_ordinal") != ordinal
                    or candidate.get("generation") != generation):
                raise _integrity("Selected native candidate differs from its sealed row locator.",
                                 generation=generation, evaluation_ordinal=ordinal)
        return _freeze(ledger)

    def read_generation(self, index: int) -> Mapping[str, Any]:
        return self._read_generation(index)

    def read_candidate(self, evaluation_ordinal: int) -> Mapping[str, Any]:
        if not isinstance(evaluation_ordinal, int) or isinstance(evaluation_ordinal, bool):
            raise TypeError("evaluation ordinal must be an integer")
        if evaluation_ordinal == 0:
            return self._baseline
        if evaluation_ordinal < 1 or evaluation_ordinal > self._candidate_count:
            raise IndexError(evaluation_ordinal)
        zero_index = evaluation_ordinal - 1
        generation_index, row_index = divmod(zero_index, self._population_size)
        summary = self._summary_block(generation_index)
        indexed = summary["rows"][row_index]
        locator = indexed.get("locator")
        if (not isinstance(locator, Mapping)
                or locator.get("evaluation_ordinal") != evaluation_ordinal
                or locator.get("generation") != generation_index + 1
                or locator.get("row_index") != row_index):
            raise _integrity("Selected native candidate locator is malformed.",
                             evaluation_ordinal=evaluation_ordinal)
        generation = self._read_generation(generation_index, summary)
        return generation["candidates"][row_index]

    def read_candidate_discretization(self, flat_population_index: int) -> object | None:
        return self.read_candidate(flat_population_index + 1).get("discretization")

    def project(self, kind: str, selector=None):
        if kind not in {"history", "objective", "residual", "parameter", "table", "comparison"}:
            raise ValueError(f"unsupported optimization projection: {kind}")
        if kind == "comparison":
            best = self._baseline if self._best_ordinal == 0 else self.read_candidate(self._best_ordinal)
            return _freeze({
                "baseline": self._baseline,
                "best": best,
                "declarations": self._declarations,
            })

        rows: list[Mapping[str, object]] = []
        for index in range(self._generation_count):
            block = self._summary_block(index)
            for indexed in block["rows"]:
                if not isinstance(indexed, Mapping) or not isinstance(indexed.get("summary"), Mapping):
                    raise _integrity("Native summary candidate is malformed.", generation=index + 1)
                summary = indexed["summary"]
                if kind in {"objective", "residual"} and selector is not None:
                    components = tuple(
                        component for component in summary.get("components", ())
                        if component.get("objective_id") == selector
                    )
                    summary = {**summary, "components": components}
                elif kind == "parameter" and selector is not None:
                    parameter_record = summary["parameters"]
                    selected_key = {
                        "definitions_id": selector["definitions_id"],
                        "parameter_id": selector["parameter_id"],
                    }
                    parameter_record = {
                        **parameter_record,
                        "bindings": [
                            binding for binding in parameter_record["bindings"]
                            if binding["parameter"] == selected_key
                        ],
                    }
                    summary = {**summary, "parameters": parameter_record}
                rows.append(summary)
        return _freeze(rows)
