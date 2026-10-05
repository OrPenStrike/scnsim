"""Optimization candidate, checkpoint, and generation evidence verification."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from ...canonical import canonical_json_bytes as _canonical_bytes, sha256_hex as _sha256
from ..primitives import _inside
from ..records import BaselineCheckpoint
from ..storage import _ATTEMPT, _STAGING, _decode_bytes, _load_canonical
from .common import (
    _SHA256,
    _f64_value,
    _finite_f64,
    _integrity,
    _required_extrapolation_rows,
    _valid_sha,
    _valid_utc_timestamp,
    _verify_discretization,
    _verify_extrapolation_evidence,
    _verify_parameter_set_document,
    _verify_quantity_role,
)
from .requests import (
    _is_leaf_local_passive_root_failure,
    _is_projection_only_optimization_failure,
    _is_shared_element_root_failure,
    _optimization_dependency,
    _optimization_failure_context,
    _optimization_leaf_catalog,
    _selector_terms,
    _verify_failure_document,
    _verify_request_document,
    _verify_selector,
    _verify_selector_lineage,
)
from .result_artifacts import _verify_residue_coupling_evidence

def _verify_generation_artifacts(
    directory: Path,
    artifacts: object,
    *,
    request_sha256: str,
    attempt_sha256: str,
    allow_other_artifacts: bool = False,
) -> list[tuple[int, str]]:
    if not isinstance(artifacts, list):
        raise _integrity("Attempt has no artifact inventory.")
    request_path = directory.parent.parent / "request.json"
    if request_path.is_symlink() or not request_path.is_file() or _sha256(request_path.read_bytes()) != request_sha256:
        raise _integrity("Optimization ledgers lack their exact request envelope.")
    request = _load_canonical(request_path)
    plan_path = directory.parents[3] / "plan.json"
    plan = _load_canonical(plan_path)
    plan_sha256 = request.get("plan_sha256")
    if not isinstance(plan_sha256, str) or _sha256(plan_path.read_bytes()) != plan_sha256:
        raise _integrity("Optimization ledger request does not bind its leaf Plan.")
    _verify_request_document(request, plan_sha256, plan)
    spec = request.get("spec")
    if request.get("operation") != "optimize_direct" and artifacts:
        raise _integrity("Only optimization attempts may retain generation ledgers.")
    if artifacts and (not isinstance(spec, dict) or spec.get("type") != "optimization"):
        raise _integrity("Optimization ledger request spec is malformed.")
    checkpoint = (
        _verify_baseline_checkpoint_directory(
            request_path.parent / "baseline-checkpoint",
            request_sha256=request_sha256, request=request, plan=plan,
        )
        if artifacts else None
    )
    ledgers: list[tuple[int, str, Mapping[str, object]]] = []
    identifiers: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"id", "sha256"}:
            raise _integrity("Optimization ledger artifact is malformed.")
        identifier = artifact.get("id")
        digest = artifact.get("sha256")
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"generation_[0-9]{6,}", identifier) is None
        ):
            raise _integrity("Non-success attempts may retain only generation ledgers.")
        if identifier in identifiers:
            raise _integrity("Optimization ledger inventory repeats an artifact ID.", artifact_id=identifier)
        identifiers.add(identifier)
        generation = int(identifier.removeprefix("generation_"))
        path = f"artifacts/generations/{generation:06d}.json"
        file_path = _inside(directory, path)
        if not file_path.is_file() or file_path.is_symlink():
            raise _integrity("Optimization ledger file is missing.", path=path)
        raw = file_path.read_bytes()
        if _sha256(raw) != _valid_sha(digest):
            raise _integrity("Optimization ledger digest does not match its bytes.", path=path)
        ledger = _decode_bytes(raw, "optimization ledger")
        if (
            ledger.get("schema") != "scnsim.optimization_ledger"
            or ledger.get("schema_version") != 4
            or ledger.get("request_sha256") != request_sha256
            or ledger.get("generation") != generation
        ):
            raise _integrity("Optimization ledger identity is inconsistent.", path=path)
        if (
            checkpoint is None
            or ledger.get("baseline_checkpoint_sha256") != checkpoint.checkpoint_sha256
            or ledger.get("baseline_checkpoint_seal_sha256") != checkpoint.seal_sha256
        ):
            raise _integrity("Optimization ledger does not bind the verified baseline checkpoint.")
        _verify_generation_ledger(ledger, spec, plan, generation)
        producer = ledger.get("attempt_sha256")
        if producer != attempt_sha256 and not _prior_ledger_is_receipt_backed(
            directory,
            request_sha256=request_sha256,
            attempt_sha256=producer,
            artifact_id=identifier,
            digest=digest,
        ):
            raise _integrity("Replayed ledger lacks its producing attempt evidence.", path=path)
        ledgers.append((generation, digest, ledger))
    ledgers.sort(key=lambda item: item[0])
    if [item[0] for item in ledgers] != list(range(1, len(ledgers) + 1)):
        raise _integrity("Optimization ledger generations are not contiguous.")
    previous: str | None = None
    complete_generations = None
    if ledgers:
        complete_generations = spec.get("optimizer", {}).get("complete_generations") if isinstance(spec.get("optimizer"), dict) else None
        if not isinstance(complete_generations, int) or isinstance(complete_generations, bool) or complete_generations < 1:
            raise _integrity("Optimization request has an invalid complete-generation count.")
    for index, (generation, digest, ledger) in enumerate(ledgers):
        if ledger.get("previous_ledger_sha256") != previous:
            raise _integrity("Optimization ledger hash chain is broken.")
        certificate = ledger["continuation_certificate"]
        expected_boundary = "terminal_post_update" if generation == complete_generations else "post_update_post_next_sample_pre_next_update"
        if generation > complete_generations or certificate.get("boundary") != expected_boundary:
            raise _integrity("Optimization ledger continuation boundary is inconsistent with its requested generation.")
        if index + 1 < len(ledgers):
            following = ledgers[index + 1][2]
            if (
                certificate.get("next_raw_optimizer_population_sha256") != following.get("raw_optimizer_population_sha256")
                or certificate.get("next_transformed_optimizer_population_sha256") != following.get("transformed_optimizer_population_sha256")
            ):
                raise _integrity("Optimization continuation certificate does not bind the next generation population.")
        previous = digest
    generation_root = _inside(directory, "artifacts/generations")
    if generation_root.is_symlink() or (generation_root.exists() and not generation_root.is_dir()):
        raise _integrity("Generation artifact directory is unsafe.")
    children = list(generation_root.iterdir()) if generation_root.exists() else []
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise _integrity("Generation artifact directory contains a non-regular entry.")
    actual = {path.relative_to(directory).as_posix() for path in children}
    declared = {str(artifact[2]["generation"]).zfill(6) for artifact in ledgers}
    expected = {f"artifacts/generations/{name}.json" for name in declared}
    if actual != expected:
        raise _integrity("Generation artifact directory contains undeclared files.")
    artifact_root = directory / "artifacts"
    if artifact_root.exists():
        if artifact_root.is_symlink() or not artifact_root.is_dir():
            raise _integrity("Attempt artifact directory is unsafe.")
        if not allow_other_artifacts and any(child.name != "generations" for child in artifact_root.iterdir()):
            raise _integrity("Non-success attempt contains undeclared solver artifacts.")
    result_path = directory / "result.json"
    if result_path.exists():
        result = _load_canonical(result_path)
        if result.get("result_kind") == "optimization":
            _verify_optimization_winner(result, spec, plan, [ledger for _, _, ledger in ledgers])
    return [(generation, digest) for generation, digest, _ in ledgers]

def _verify_generation_ledger(
    ledger: Mapping[str, object],
    spec: Mapping[str, object],
    plan: Mapping[str, object],
    generation: int,
) -> None:
    expected = {
        "schema", "schema_version", "request_sha256", "attempt_sha256",
        "algorithm_id", "generation", "previous_ledger_sha256", "population_size",
        "raw_optimizer_population_sha256", "transformed_optimizer_population_sha256",
        "continuation_certificate", "candidates", "baseline_checkpoint_sha256",
        "baseline_checkpoint_seal_sha256",
    }
    optimizer = spec.get("optimizer")
    variables = spec.get("variables")
    objectives = spec.get("objectives")
    if not isinstance(optimizer, dict) or not isinstance(variables, list) or not isinstance(objectives, list):
        raise _integrity("Optimization request controls are malformed.")
    population_size = optimizer.get("resolved_population_size")
    candidates = ledger.get("candidates")
    if (
        set(ledger) != expected
        or ledger.get("schema_version") != 4
        or ledger.get("algorithm_id") != "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v9"
        or _SHA256.fullmatch(str(ledger.get("baseline_checkpoint_sha256", ""))) is None
        or _SHA256.fullmatch(str(ledger.get("baseline_checkpoint_seal_sha256", ""))) is None
        or not isinstance(population_size, int)
        or isinstance(population_size, bool)
        or population_size < 2
        or ledger.get("population_size") != population_size
        or _SHA256.fullmatch(str(ledger.get("raw_optimizer_population_sha256", ""))) is None
        or _SHA256.fullmatch(str(ledger.get("transformed_optimizer_population_sha256", ""))) is None
        or not isinstance(candidates, list)
        or len(candidates) != population_size
    ):
        raise _integrity("Optimization ledger envelope is open or inconsistent.")
    _verify_continuation_certificate(ledger.get("continuation_certificate"), generation)
    for column, candidate in enumerate(candidates, 1):
        expected_ordinal = 1 + (generation - 1) * population_size + (column - 1)
        _verify_candidate_outcome(
            candidate,
            variables=variables,
            objectives=objectives,
            plan=plan,
            optimization_authorizations=spec.get("allow_extrapolation", []),
            generation=generation,
            column=column,
            evaluation_ordinal=expected_ordinal,
            baseline=False,
        )
    if (
        ledger.get("raw_optimizer_population_sha256")
        != _candidate_population_sha256(candidates, "optimizer_latent_coordinates_f64", len(variables))
        or ledger.get("transformed_optimizer_population_sha256")
        != _candidate_population_sha256(candidates, "optimizer_coordinates_f64", len(variables))
    ):
        raise _integrity("Optimization population hashes do not reproduce their candidate coordinate arrays.")

def _candidate_population_sha256(
    candidates: list[object],
    field: str,
    variables: int,
) -> str:
    values = [
        candidate[field][row]
        for row in range(variables)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    if len(values) != variables * len(candidates):
        raise _integrity("Optimization population matrix is incomplete.")
    return _sha256(_canonical_bytes({
        "shape": [variables, len(candidates)],
        "values_f64": values,
    }))

def _verify_continuation_certificate(value: object, generation: int) -> None:
    if not isinstance(value, dict):
        raise _integrity("CMA continuation certificate is missing.")
    common = {
        "schema", "schema_version", "projection_id", "boundary",
        "completed_generation", "state_sha256",
    }
    boundary = value.get("boundary")
    expected = common | (
        {"next_raw_optimizer_population_sha256", "next_transformed_optimizer_population_sha256"}
        if boundary == "post_update_post_next_sample_pre_next_update"
        else set()
    )
    if (
        set(value) != expected
        or value.get("schema") != "scnsim.cmaes_continuation_certificate"
        or value.get("schema_version") != 1
        or value.get("projection_id") != "cmaes-jl-0.2.6-julia-1.12.6-continuation-state.v1"
        or boundary not in {"post_update_post_next_sample_pre_next_update", "terminal_post_update"}
        or value.get("completed_generation") != generation
        or any(_SHA256.fullmatch(str(value.get(field, ""))) is None for field in expected if field.endswith("sha256"))
    ):
        raise _integrity("CMA continuation certificate is open or malformed.")

def _verify_candidate_failure_context(
    failure: object,
    *,
    objectives: list[object],
    candidate: Mapping[str, object],
    phase: str,
    owner: Mapping[str, object],
    affected: list[Mapping[str, object]],
    dependency: Mapping[str, object] | None = None,
) -> None:
    context = _optimization_failure_context(failure)
    expected_candidate = {
        "evaluation_ordinal": candidate["evaluation_ordinal"],
        "origin": candidate["origin"],
        "generation": candidate["generation"],
        "population_column": candidate["population_column"],
    }
    catalog = _optimization_leaf_catalog(objectives)
    known = [locator for locator, _ in catalog]
    if any(locator not in known for locator in affected):
        raise _integrity("Optimization failure names a leaf absent from its request.")
    expected = {
        "schema": "scnsim.optimization_failure_context",
        "schema_version": 1,
        "phase": phase,
        "candidate": expected_candidate,
        "owner": dict(owner),
        "affected_leaves": [dict(item) for item in affected],
    }
    if dependency is not None:
        expected["dependency"] = dict(dependency)
    if context != expected:
        raise _integrity("Optimization failure context disagrees with evaluated request order and dependencies.")

def _optimization_root_selectors(selector: Mapping[str, object]) -> list[Mapping[str, object]]:
    kind = selector.get("type")
    if kind in {"diagonal_root_projection", "operator_element_root_projection", "hybridized_pole_projection", "transfer_zero_projection"}:
        return [selector]
    if kind == "residue_coupling_projection":
        spec = selector.get("spec")
        view = selector.get("view")
        if not isinstance(spec, Mapping) or not isinstance(view, Mapping):
            raise _integrity("Residue selector root dependencies are malformed.")
        roots: list[Mapping[str, object]] = []
        for name in ("branch_a", "branch_b"):
            branch = spec.get(name)
            if not isinstance(branch, Mapping):
                raise _integrity("Residue selector branch is malformed.")
            branch_type = branch.get("type")
            selector_type = (
                "residue_diagonal_root_projection" if branch_type == "diagonal_root"
                else "hybridized_pole_projection" if branch_type == "hybridized_pole"
                else None
            )
            if selector_type is None:
                raise _integrity("Residue selector branch type is unsupported.")
            roots.append({"type": selector_type, "spec": branch, "projection": "frequency", "view": view})
        return roots
    return []

def _optimization_checkpoint_roots(objectives: object) -> list[dict[str, object]]:
    """Return source-ordered unique root dependencies from the sealed request."""

    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for _locator, selector in _optimization_leaf_catalog(objectives):
        for root in _optimization_root_selectors(selector):
            dependency = _optimization_dependency(root)
            key = str(dependency["dependency_sha256"])
            if key not in seen:
                seen.add(key)
                result.append(dependency)
    return result

def _verify_baseline_checkpoint_document(
    checkpoint: Mapping[str, object],
    *,
    request_sha256: str,
    request: Mapping[str, object],
    plan: Mapping[str, object],
) -> None:
    spec = request.get("spec")
    if (
        request.get("operation") != "optimize_direct"
        or not isinstance(spec, Mapping)
        or spec.get("type") != "optimization"
    ):
        raise _integrity("Baseline checkpoint request is not optimization.")
    expected_fields = {
        "schema", "schema_version", "request_sha256", "algorithm_id",
        "baseline", "baseline_roots",
    }
    if (
        set(checkpoint) != expected_fields
        or checkpoint.get("schema") != "scnsim.optimization_baseline_checkpoint"
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("request_sha256") != request_sha256
        or checkpoint.get("algorithm_id")
        != "scnsim.direct_cmaes.cmaes_jl_0_2_6_state_replay.v9"
    ):
        raise _integrity("Optimization baseline checkpoint envelope is open or inconsistent.")
    variables = spec.get("variables")
    objectives = spec.get("objectives")
    if not isinstance(variables, list) or not isinstance(objectives, list):
        raise _integrity("Optimization checkpoint request declarations are malformed.")
    baseline = checkpoint.get("baseline")
    _verify_candidate_outcome(
        baseline,
        variables=variables, objectives=objectives, plan=plan,
        optimization_authorizations=spec.get("allow_extrapolation", []),
        generation=0, column=None, evaluation_ordinal=0, baseline=True,
    )
    _verify_checkpoint_baseline_point(checkpoint["baseline"], request)
    if (
        not isinstance(baseline, Mapping)
        or baseline.get("cache_hit") is not False
        or not isinstance(baseline.get("outcome"), Mapping)
        or baseline["outcome"].get("status") != "success"
    ):
        raise _integrity(
            "Published optimization baseline must be a fresh, complete success."
        )
    _verify_checkpoint_primary_lineage(
        baseline, request_view=request.get("view"), plan=plan,
    )
    roots = checkpoint.get("baseline_roots")
    expected_dependencies = _optimization_checkpoint_roots(objectives)
    if not isinstance(roots, list) or len(roots) != len(expected_dependencies):
        raise _integrity("Optimization baseline root inventory is incomplete.")
    for row, dependency in zip(roots, expected_dependencies):
        value = row.get("value") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or set(row) != {"dependency", "value"}
            or row.get("dependency") != dependency
            or not isinstance(value, Mapping)
            or set(value) != {"real_f64", "imag_f64", "si_unit", "dimensionality"}
            or value.get("si_unit") != "radian / second"
            or value.get("dimensionality") != "inverse_time"
            or not _finite_f64(value.get("real_f64"))
            or not _finite_f64(value.get("imag_f64"))
        ):
            raise _integrity("Optimization baseline root evidence is malformed or out of order.")
    _verify_checkpoint_root_references(checkpoint, objectives)

def _verify_checkpoint_primary_lineage(
    baseline: object,
    *,
    request_view: object,
    plan: Mapping[str, object],
) -> Mapping[str, object]:
    """Require baseline term evidence to determine one primary Result lineage."""

    outcome = baseline.get("outcome") if isinstance(baseline, Mapping) else None
    components = outcome.get("objective_components") if isinstance(outcome, Mapping) else None
    if not isinstance(request_view, Mapping) or not isinstance(components, list):
        raise _integrity("Optimization checkpoint cannot determine its primary View lineage.")
    primary_key = _canonical_bytes(request_view)
    selected: Mapping[str, object] | None = None
    for component in components:
        terms = component.get("terms") if isinstance(component, Mapping) else None
        if not isinstance(terms, list):
            raise _integrity("Optimization checkpoint objective terms are malformed.")
        for term in terms:
            selector = term.get("selector") if isinstance(term, Mapping) else None
            if (
                not isinstance(selector, Mapping)
                or not isinstance(selector.get("view"), Mapping)
                or _canonical_bytes(selector["view"]) != primary_key
            ):
                continue
            if term.get("status") != "success" or not isinstance(term.get("ref_lineage"), Mapping):
                raise _integrity("Optimization checkpoint primary View term is not successful.")
            lineage = term["ref_lineage"]
            _verify_selector_lineage({"view": request_view}, lineage, plan)
            if selected is None:
                selected = lineage
            elif _canonical_bytes(selected) != _canonical_bytes(lineage):
                raise _integrity("Optimization checkpoint primary View lineages disagree.")
    if selected is None:
        raise _integrity("Optimization checkpoint does not realize its primary View.")
    return selected

def _verify_checkpoint_baseline_point(
    baseline: Mapping[str, object], request: Mapping[str, object]
) -> None:
    """Reconstruct the sealed baseline point and its unit-box coordinates."""

    source = request.get("parameter_source")
    spec = request.get("spec")
    if (
        not isinstance(source, Mapping)
        or source.get("kind") != "point"
        or not isinstance(source.get("parameters"), Mapping)
        or not isinstance(spec, Mapping)
        or not isinstance(spec.get("variables"), list)
    ):
        raise _integrity("Optimization checkpoint lacks one sealed baseline point.")
    expected_parameters = source["parameters"]
    parameters = baseline.get("parameters")
    if parameters != expected_parameters or not isinstance(parameters, Mapping):
        raise _integrity("Optimization checkpoint baseline differs from the sealed request point.")
    expected_coordinates = spec.get("optimizer", {}).get(
        "baseline_optimizer_coordinates_f64"
    )
    if baseline.get("optimizer_coordinates_f64") != expected_coordinates:
        raise _integrity("Optimization checkpoint baseline coordinates disagree with its request point.")

def _verify_checkpoint_root_references(
    checkpoint: Mapping[str, object], objectives: list[object]
) -> None:
    """Bind successful root terms to the sealed dependency inventory."""

    roots = checkpoint.get("baseline_roots")
    baseline = checkpoint.get("baseline")
    outcome = baseline.get("outcome") if isinstance(baseline, Mapping) else None
    components = outcome.get("objective_components") if isinstance(outcome, Mapping) else None
    if not isinstance(roots, list) or not isinstance(components, list):
        raise _integrity("Optimization checkpoint root reference evidence is malformed.")
    by_dependency = {
        row["dependency"]["dependency_sha256"]: row["value"]
        for row in roots
        if isinstance(row, Mapping)
        and isinstance(row.get("dependency"), Mapping)
        and isinstance(row.get("value"), Mapping)
    }
    for objective, component in zip(objectives, components):
        selectors = _selector_terms(objective.get("quantity")) if isinstance(objective, Mapping) else []
        terms = component.get("terms") if isinstance(component, Mapping) else None
        if not isinstance(terms, list) or len(terms) != len(selectors):
            raise _integrity("Optimization checkpoint objective terms are malformed.")
        for selector, term in zip(selectors, terms):
            kind = selector.get("type")
            if kind not in {
                "diagonal_root_projection", "operator_element_root_projection", "hybridized_pole_projection",
                "transfer_zero_projection",
            }:
                # Residue-coupling branch anchors are private dependencies.
                continue
            dependency = _optimization_dependency(selector)
            value = by_dependency.get(dependency["dependency_sha256"])
            term_value = term.get("value") if isinstance(term, Mapping) else None
            if not isinstance(value, Mapping) or not isinstance(term_value, Mapping):
                raise _integrity("Optimization checkpoint root has no successful public term.")
            _verify_quantity_role(
                term_value, complex_value=False, unit="hertz", dimensionality="inverse_time"
            )

def _verify_baseline_checkpoint_directory(
    directory: Path,
    *,
    request_sha256: str,
    request: Mapping[str, object],
    plan: Mapping[str, object],
) -> BaselineCheckpoint:
    if directory.is_symlink() or not directory.is_dir() or directory.parent.is_symlink():
        raise _integrity("Baseline checkpoint directory is missing or symlinked.")
    expected_names = {"checkpoint.json", "source-attempt.json", "seal.json"}
    if {item.name for item in directory.iterdir()} != expected_names:
        raise _integrity("Baseline checkpoint directory has an unknown inventory.")
    checkpoint_path = directory / "checkpoint.json"
    source_path = directory / "source-attempt.json"
    seal_path = directory / "seal.json"
    if any(path.is_symlink() or not path.is_file() for path in (checkpoint_path, source_path, seal_path)):
        raise _integrity("Baseline checkpoint files must be regular and unsymlinked.")
    checkpoint_bytes = checkpoint_path.read_bytes()
    source_bytes = source_path.read_bytes()
    checkpoint = _decode_bytes(checkpoint_bytes, "baseline checkpoint")
    source_attempt = _decode_bytes(source_bytes, "source attempt")
    seal_bytes = seal_path.read_bytes()
    seal = _decode_bytes(seal_bytes, "baseline checkpoint seal")
    checkpoint_sha = _sha256(checkpoint_bytes)
    source_sha = _sha256(source_bytes)
    expected_seal = {
        "schema", "schema_version", "request_sha256", "checkpoint_sha256",
        "checkpoint_byte_length", "source_attempt_sha256",
        "source_attempt_byte_length", "published_at_utc",
    }
    if (
        set(seal) != expected_seal
        or seal.get("schema") != "scnsim.optimization_checkpoint_seal"
        or seal.get("schema_version") != 1
        or seal.get("request_sha256") != request_sha256
        or seal.get("checkpoint_sha256") != checkpoint_sha
        or seal.get("checkpoint_byte_length") != len(checkpoint_bytes)
        or seal.get("source_attempt_sha256") != source_sha
        or seal.get("source_attempt_byte_length") != len(source_bytes)
        or not _valid_utc_timestamp(seal.get("published_at_utc"))
    ):
        raise _integrity("Baseline checkpoint seal does not bind its exact files.")
    _verify_checkpoint_source_attempt(source_attempt, request_sha256)
    _verify_baseline_checkpoint_document(
        checkpoint, request_sha256=request_sha256, request=request, plan=plan,
    )
    return BaselineCheckpoint(
        checkpoint_sha, _sha256(seal_bytes), checkpoint,
        source_attempt, directory,
    )

def _verify_checkpoint_source_attempt(
    attempt: Mapping[str, object], request_sha256: str
) -> None:
    fields = {
        "schema", "schema_version", "request_sha256", "ordinal", "ordinal_text",
        "directory", "staging_directory", "attempt_state", "started_at_utc",
        "julia_executable_sha256", "os", "architecture", "cpu", "julia_threads",
        "blas_threads", "blas_vendor",
    }
    ordinal = attempt.get("ordinal")
    ordinal_text = attempt.get("ordinal_text")
    staging = attempt.get("staging_directory")
    if (
        set(attempt) != fields
        or attempt.get("schema") != "scnsim.attempt"
        or attempt.get("schema_version") != 2
        or attempt.get("request_sha256") != request_sha256
        or attempt.get("attempt_state") != "launched"
        or not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1
        or ordinal_text != str(ordinal).zfill(6)
        or attempt.get("directory") != f"requests/{request_sha256}/attempts/{ordinal_text}"
        or not isinstance(staging, str)
        or not staging.startswith(f"requests/{request_sha256}/attempts/.staging-{ordinal_text}-")
        or _STAGING.fullmatch(Path(staging).name) is None
        or not isinstance(attempt.get("started_at_utc"), str)
        or not str(attempt["started_at_utc"]).endswith("Z")
        or _SHA256.fullmatch(str(attempt.get("julia_executable_sha256", ""))) is None
        or any(not isinstance(attempt.get(key), str) or not attempt[key] for key in ("os", "architecture", "cpu", "blas_vendor"))
        or attempt.get("julia_threads") != 1
        or attempt.get("blas_threads") != 1
    ):
        raise _integrity("Baseline checkpoint source attempt is not a closed producing optimization attempt.")

def _verify_attempt_checkpoint_consumption(
    attempt: Mapping[str, object],
    receipt: Mapping[str, object],
    *,
    attempt_sha256: str,
    checkpoint: BaselineCheckpoint | None,
) -> None:
    """Require recovery evidence exactly when a later attempt consumed it."""

    checkpoint_sha = attempt.get("baseline_checkpoint_sha256")
    seal_sha = attempt.get("baseline_checkpoint_seal_sha256")
    has_pair = checkpoint_sha is not None or seal_sha is not None
    if checkpoint is None:
        if has_pair:
            raise _integrity("Attempt references an absent baseline checkpoint.")
        return
    if has_pair and (
        checkpoint_sha != checkpoint.checkpoint_sha256
        or seal_sha != checkpoint.seal_sha256
    ):
        raise _integrity("Attempt baseline checkpoint reference is absent or corrupt.")

    source_attempt_sha = _sha256(_canonical_bytes(checkpoint.source_attempt))
    if attempt_sha256 == source_attempt_sha:
        return
    artifacts = receipt.get("artifacts")
    failure = receipt.get("failure")
    evidence = failure.get("evidence") if isinstance(failure, Mapping) else None
    context = evidence.get("optimization_context") if isinstance(evidence, Mapping) else None
    candidate = context.get("candidate") if isinstance(context, Mapping) else None
    consumed = (
        attempt.get("resume_ledger_sha256") is not None
        or receipt.get("outcome") == "success"
        or (
            isinstance(artifacts, list)
            and any(
                isinstance(row, Mapping)
                and str(row.get("id", "")).startswith("generation_")
                for row in artifacts
            )
        )
        or (
            isinstance(candidate, Mapping)
            and candidate.get("origin") == "population"
        )
    )
    if consumed and not has_pair:
        raise _integrity(
            "A later optimization attempt consumed baseline evidence without binding its checkpoint pair."
        )

def _optimization_quantity_failure_consumers(
    catalog: list[tuple[dict[str, object], Mapping[str, object]]],
    start: int,
    selector: Mapping[str, object],
    dependency: Mapping[str, object],
    failure: object,
) -> list[dict[str, object]]:
    """Derive later public consumers of the selector or its failed private root."""

    selector_dependency = _optimization_dependency(selector)
    root_dependencies = [
        _optimization_dependency(root)
        for root in _optimization_root_selectors(selector)
    ]
    if dependency in root_dependencies and (
        _is_shared_element_root_failure(failure, selector)
        or dependency != selector_dependency
    ):
        return [
            locator for locator, item in catalog[start:]
            if any(
                _optimization_dependency(root) == dependency
                for root in _optimization_root_selectors(item)
            )
        ]
    if dependency == selector_dependency:
        if _is_leaf_local_passive_root_failure(failure, selector):
            return [catalog[start][0]]
        return [
            locator for locator, item in catalog[start:]
            if _optimization_dependency(item) == dependency
        ]
    raise _integrity("Quantity failure dependency disagrees with its failed selector.")

def _verify_terminal_optimization_failure(
    failure: object,
    spec: object,
    plan: Mapping[str, object],
    *,
    completed_generations: int = 0,
) -> None:
    if not isinstance(spec, Mapping) or spec.get("type") != "optimization":
        raise _integrity("Optimization terminal failure lacks its request spec.")
    context = _optimization_failure_context(failure)
    candidate = context.get("candidate")
    if not isinstance(candidate, Mapping):
        raise _integrity("Terminal optimization failure has no candidate position.")
    if (
        not isinstance(completed_generations, int)
        or isinstance(completed_generations, bool)
        or completed_generations < 0
    ):
        raise _integrity("Terminal optimization ledger prefix is malformed.")
    if candidate.get("origin") == "population":
        optimizer = spec.get("optimizer")
        population = optimizer.get("resolved_population_size") if isinstance(optimizer, Mapping) else None
        complete = optimizer.get("complete_generations") if isinstance(optimizer, Mapping) else None
        generation = candidate.get("generation")
        column = candidate.get("population_column")
        if (
            not isinstance(population, int) or isinstance(population, bool) or population < 2
            or not isinstance(complete, int) or isinstance(complete, bool) or complete < 1
            or not isinstance(generation, int) or isinstance(generation, bool)
            or not isinstance(column, int) or isinstance(column, bool)
            or generation < 1 or generation > complete
            or column < 1 or column > population
            or generation != completed_generations + 1
            or candidate.get("evaluation_ordinal") != 1 + (generation - 1) * population + (column - 1)
        ):
            raise _integrity("Terminal optimization population position is inconsistent with its request.")
    elif completed_generations != 0:
        raise _integrity("Baseline failure follows completed population ledgers.")
    objectives = spec.get("objectives")
    catalog = _optimization_leaf_catalog(objectives)
    all_leaves = [locator for locator, _ in catalog]
    phase = context.get("phase")
    dependency = context.get("dependency")
    owner = context.get("owner")
    affected = context.get("affected_leaves")
    if phase in {"candidate_prepare", "candidate_compile"}:
        expected = ({"kind": "candidate"}, all_leaves, None)
    elif phase == "view_realization":
        matches = [(locator, selector) for locator, selector in catalog if _optimization_dependency(selector, kind="view") == dependency]
        if not matches:
            raise _integrity("Terminal View failure dependency is absent from the request.")
        expected = ({"kind": "dependency"}, [locator for locator, _ in matches], dependency)
    elif phase == "baseline_root_anchor":
        if candidate.get("origin") != "baseline":
            raise _integrity("Population failure claims baseline root-anchor ownership.")
        matches: list[Mapping[str, object]] = []
        for locator, selector in catalog:
            if any(_optimization_dependency(root) == dependency for root in _optimization_root_selectors(selector)):
                matches.append(locator)
        if not matches:
            raise _integrity("Baseline root-anchor dependency is absent from the request.")
        expected = ({"kind": "dependency"}, matches, dependency)
    elif phase == "quantity_evaluation":
        leaf = owner.get("leaf") if isinstance(owner, Mapping) else None
        if leaf not in all_leaves:
            raise _integrity("Baseline quantity failure owner is absent from the request.")
        index = all_leaves.index(leaf)
        selector = catalog[index][1]
        if dependency is None:
            if not _is_projection_only_optimization_failure(failure):
                raise _integrity("Shared quantity failure omits its dependency identity.")
            expected_affected = [leaf]
        else:
            if _is_projection_only_optimization_failure(failure):
                raise _integrity("Projection-only failure claims a shared dependency.")
            expected_affected = _optimization_quantity_failure_consumers(
                catalog, index, selector, dependency, failure,
            )
        expected = ({"kind": "leaf", "leaf": leaf}, expected_affected, dependency)
    elif phase == "objective_aggregation":
        objective_id = owner.get("objective_id") if isinstance(owner, Mapping) else None
        if objective_id not in [item.get("id") for item in objectives if isinstance(item, Mapping)]:
            raise _integrity("Baseline objective aggregation owner is absent from the request.")
        if candidate.get("origin") != "baseline":
            raise _integrity("Population objective aggregation failure escaped candidate evidence.")
        objective = next(
            item for item in objectives
            if isinstance(item, Mapping) and item.get("id") == objective_id
        )
        witness = context.get("aggregation_witness")
        if not isinstance(witness, Mapping) or witness.get("objective_id") != objective_id:
            raise _integrity("Baseline objective aggregation failure lacks its witness.")
        _verify_objective_component(
            {
                "objective_id": objective_id,
                "quantity": objective["quantity"],
                "status": "failure",
                "terms": witness.get("terms"),
                "failure": failure,
            },
            objective,
            plan,
            expected_status="failure",
        )
        expected = ({"kind": "objective", "objective_id": objective_id}, [], None)
    elif phase == "total_aggregation":
        if candidate.get("origin") != "baseline":
            raise _integrity("Population total aggregation failure escaped candidate evidence.")
        witness = context.get("aggregation_witness")
        components = (
            witness.get("objective_components")
            if isinstance(witness, Mapping)
            else None
        )
        if not isinstance(components, list) or len(components) != len(objectives):
            raise _integrity("Baseline total aggregation failure lacks its witness.")
        for objective, component in zip(objectives, components):
            _verify_objective_component(
                component, objective, plan, expected_status="success"
            )
        expected = ({"kind": "candidate"}, [], None)
    else:
        raise _integrity("Terminal optimization failure uses an invalid baseline phase.")
    if owner != expected[0] or affected != expected[1] or dependency != expected[2]:
        raise _integrity("Terminal optimization failure context disagrees with its sealed request.")

def _verify_objective_component(
    component: object,
    objective: object,
    plan: Mapping[str, object],
    *,
    expected_status: str,
) -> None:
    if not isinstance(component, dict) or not isinstance(objective, dict):
        raise _integrity("Optimization objective component is malformed.")
    quantity = objective.get("quantity")
    expected_terms = _selector_terms(quantity)
    terms = component.get("terms")
    common = {"objective_id", "quantity", "status", "terms"}
    expected_fields = (
        common | {"value", "normalized_residual_f64", "weighted_cost_f64"}
        if expected_status == "success"
        else common | {"failure"}
    )
    if (
        set(component) != expected_fields
        or component.get("objective_id") != objective.get("id")
        or component.get("quantity") != quantity
        or component.get("status") != expected_status
        or not isinstance(terms, list)
        or len(terms) != len(expected_terms)
    ):
        raise _integrity("Optimization objective component is open or inconsistent.")
    term_statuses: list[str] = []
    for ordinal, (term, selector) in enumerate(zip(terms, expected_terms), 1):
        if not isinstance(term, dict) or term.get("term_ordinal") != ordinal or term.get("selector") != selector:
            raise _integrity("Optimization term evidence is out of order or names another selector.")
        status = term.get("status")
        term_statuses.append(str(status))
        if status == "success":
            successful_fields = {"term_ordinal", "selector", "status", "ref_lineage", "value"}
            if selector.get("type") == "residue_coupling_projection":
                successful_fields.add("residue_coupling_evidence")
            if set(term) != successful_fields:
                raise _integrity("Successful optimization term is open or malformed.")
            _verify_selector_lineage(selector, term.get("ref_lineage"), plan)
            role = _verify_selector(selector, plan)
            _verify_quantity_role(term.get("value"), complex_value=False, unit=role[0], dimensionality=role[1])
            if selector.get("type") == "residue_coupling_projection":
                selector_spec = selector.get("spec")
                if not isinstance(selector_spec, Mapping):
                    raise _integrity("Residue coupling selector Spec is malformed.")
                _verify_residue_coupling_evidence(
                    term.get("residue_coupling_evidence"),
                    selector_spec,
                    expected_projection=str(selector.get("projection")),
                    projected_value=term.get("value"),
                )
        elif status == "failure":
            if set(term) != {"term_ordinal", "selector", "status", "ref_lineage", "failure"}:
                raise _integrity("Failed optimization term is open or malformed.")
            _verify_selector_lineage(selector, term.get("ref_lineage"), plan)
            _verify_failure_document(term.get("failure"), "optimize_direct")
        elif status == "not_evaluated":
            if set(term) != {"term_ordinal", "selector", "status", "failure"}:
                raise _integrity("Unevaluated optimization term is open or malformed.")
            _verify_failure_document(term.get("failure"), "optimize_direct")
        else:
            raise _integrity("Optimization term status is unknown.")
    if expected_status == "success":
        if any(status != "success" for status in term_statuses):
            raise _integrity("Successful objective contains an unevaluated term.")
        _verify_quantity_role(
            component.get("value"), complex_value=False,
            unit=objective["target"]["si_unit"], dimensionality=objective["target"]["dimensionality"],
        )
        if (
            not _finite_f64(component.get("normalized_residual_f64"))
            or not _finite_f64(component.get("weighted_cost_f64"))
            or _f64_value(component["weighted_cost_f64"]) < 0.0
        ):
            raise _integrity("Optimization objective numerical evidence is malformed.")
    else:
        _verify_failure_document(component.get("failure"), "optimize_direct")
        if any(
            term.get("status") in {"failure", "not_evaluated"}
            and term.get("failure") != component.get("failure")
            for term in terms
            if isinstance(term, Mapping)
        ):
            raise _integrity("Optimization term failure disagrees with its objective failure.")
        if expected_status == "failure":
            failure_indexes = [
                index for index, status in enumerate(term_statuses)
                if status == "failure"
            ]
            ordinary_failure = (
                len(failure_indexes) == 1
                and term_statuses[:failure_indexes[0]] == ["success"] * failure_indexes[0]
                and term_statuses[failure_indexes[0] + 1:]
                == ["not_evaluated"] * (len(term_statuses) - failure_indexes[0] - 1)
            )
            if not ordinary_failure and any(status != "success" for status in term_statuses):
                raise _integrity("Failed objective term status order is inconsistent.")
        elif any(status != "not_evaluated" for status in term_statuses):
            raise _integrity("Unevaluated objective contains evaluated terms.")

def _verify_candidate_outcome(
    value: object,
    *,
    variables: list[object],
    objectives: list[object],
    plan: Mapping[str, object],
    optimization_authorizations: object,
    generation: int,
    column: int | None,
    evaluation_ordinal: int,
    baseline: bool,
) -> None:
    if not isinstance(value, dict):
        raise _integrity("Optimization candidate is not an object.")
    expected = {
        "evaluation_ordinal", "origin", "generation", "population_column",
        "optimizer_coordinates_f64", "parameters", "cache_hit",
        "extrapolation_evidence", "outcome",
    }
    if "discretization" in value:
        expected.add("discretization")
        _verify_discretization(value["discretization"], plan)
    if not baseline:
        expected.add("optimizer_latent_coordinates_f64")
    coordinates = value.get("optimizer_coordinates_f64")
    latent = value.get("optimizer_latent_coordinates_f64")
    if (
        set(value) != expected
        or value.get("evaluation_ordinal") != evaluation_ordinal
        or value.get("origin") != ("baseline" if baseline else "population")
        or value.get("generation") != generation
        or value.get("population_column") != column
        or not isinstance(value.get("cache_hit"), bool)
        or not isinstance(coordinates, list)
        or len(coordinates) != len(variables)
        or any(not _finite_f64(item) or ("domain" not in variable and not 0.0 <= _f64_value(item) <= 1.0)
               for item, variable in zip(coordinates, variables))
        or (not baseline and (not isinstance(latent, list) or len(latent) != len(variables) or any(not _finite_f64(item) for item in latent)))
    ):
        raise _integrity("Optimization candidate envelope is open or malformed.")
    _verify_parameter_set_document(value.get("parameters"), require_empty_authorization=True)
    _verify_extrapolation_evidence(
        value.get("extrapolation_evidence"),
        allowed_sources={"none", "optimization_spec"},
        required_rows=_required_extrapolation_rows(
            plan,
            value["parameters"],
            authorization_source="optimization_spec",
            optimization_authorizations=optimization_authorizations,
        ),
    )
    outcome = value.get("outcome")
    if not isinstance(outcome, dict):
        raise _integrity("Optimization candidate outcome is malformed.")
    if outcome.get("status") == "success":
        components = outcome.get("objective_components")
        if (
            set(outcome) != {"status", "cost_f64", "objective_components"}
            or not _finite_f64(outcome.get("cost_f64"))
            or _f64_value(outcome["cost_f64"]) < 0.0
            or not isinstance(components, list)
            or len(components) != len(objectives)
        ):
            raise _integrity("Successful optimization candidate is malformed.")
        for objective, component in zip(objectives, components):
            _verify_objective_component(component, objective, plan, expected_status="success")
    elif outcome.get("status") == "failure":
        components = outcome.get("objective_components")
        if (
            set(outcome) != {"status", "penalty", "failure", "objective_components"}
            or outcome.get("penalty") != "positive_infinity"
            or not isinstance(components, list)
            or len(components) != len(objectives)
        ):
            raise _integrity("Failed optimization candidate is malformed.")
        _verify_failure_document(outcome.get("failure"), "optimize_direct")
        if outcome["failure"].get("kind") not in {
            "invalid_candidate_physical_parameter", "eliminated_block_solve_failure",
            "root_slope_unresolved", "numerical_resolution_unresolved",
        }:
            raise _integrity("Optimization candidate uses a request-level failure kind.")
        context = _optimization_failure_context(outcome["failure"])
        phase = context["phase"]
        statuses: list[str] = []
        for objective, component in zip(objectives, components):
            status = component.get("status") if isinstance(component, Mapping) else None
            statuses.append(str(status))
            if status == "success":
                _verify_objective_component(component, objective, plan, expected_status="success")
            elif status == "failure":
                _verify_objective_component(component, objective, plan, expected_status="failure")
                if component.get("failure") != outcome["failure"]:
                    raise _integrity("Failed objective does not bind the candidate failure.")
            elif status == "not_evaluated":
                _verify_objective_component(component, objective, plan, expected_status="not_evaluated")
                if component.get("failure") != outcome["failure"]:
                    raise _integrity("Unevaluated objective does not bind the candidate failure.")
            else:
                raise _integrity("Failed candidate objective status order is inconsistent.")
        catalog = _optimization_leaf_catalog(objectives)
        all_leaves = [locator for locator, _ in catalog]
        if phase in {"candidate_prepare", "candidate_compile"}:
            if any(status != "not_evaluated" for status in statuses):
                raise _integrity("Candidate preparation failure contains evaluated objectives.")
            _verify_candidate_failure_context(
                outcome["failure"], objectives=objectives, candidate=value,
                phase=phase, owner={"kind": "candidate"}, affected=all_leaves,
            )
        elif phase == "view_realization":
            if any(status != "not_evaluated" for status in statuses):
                raise _integrity("View-realization failure contains evaluated objectives.")
            dependency = context.get("dependency")
            matches = [
                (locator, selector) for locator, selector in catalog
                if _optimization_dependency(selector, kind="view") == dependency
            ]
            if not matches:
                raise _integrity("View-realization failure dependency is absent from the request.")
            _verify_candidate_failure_context(
                outcome["failure"], objectives=objectives, candidate=value,
                phase=phase, owner={"kind": "dependency"},
                affected=[locator for locator, _ in matches], dependency=dependency,
            )
        elif phase == "quantity_evaluation":
            failed_objectives = [index for index, status in enumerate(statuses) if status == "failure"]
            if len(failed_objectives) != 1:
                raise _integrity("Quantity failure has no unique failed objective.")
            failed_objective = failed_objectives[0]
            if statuses[:failed_objective] != ["success"] * failed_objective or statuses[failed_objective + 1:] != ["not_evaluated"] * (len(statuses) - failed_objective - 1):
                raise _integrity("Quantity failure objective status order is inconsistent.")
            failed_terms = [
                ({"objective_id": objective["id"], "term_ordinal": term["term_ordinal"]}, selector)
                for objective, component in zip(objectives, components)
                if isinstance(objective, Mapping) and isinstance(component, Mapping)
                for term, selector in zip(component.get("terms", []), _selector_terms(objective.get("quantity")))
                if isinstance(term, Mapping) and term.get("status") == "failure"
            ]
            if len(failed_terms) != 1:
                raise _integrity("Quantity failure does not identify exactly one failed leaf.")
            locator, selector = failed_terms[0]
            start = all_leaves.index(locator)
            dependency = context.get("dependency")
            if dependency is None:
                if not _is_projection_only_optimization_failure(outcome["failure"]):
                    raise _integrity("Shared quantity failure omits its dependency identity.")
                affected = [locator]
            else:
                if _is_projection_only_optimization_failure(outcome["failure"]):
                    raise _integrity("Projection-only failure claims a shared dependency.")
                affected = _optimization_quantity_failure_consumers(
                    catalog, start, selector, dependency, outcome["failure"],
                )
            _verify_candidate_failure_context(
                outcome["failure"], objectives=objectives, candidate=value,
                phase=phase, owner={"kind": "leaf", "leaf": locator},
                affected=affected, dependency=dependency,
            )
        elif phase == "objective_aggregation":
            failed_indexes = [index for index, status in enumerate(statuses) if status == "failure"]
            if len(failed_indexes) != 1:
                raise _integrity("Objective aggregation failure has no unique owner.")
            failed_index = failed_indexes[0]
            if statuses[:failed_index] != ["success"] * failed_index or statuses[failed_index + 1:] != ["not_evaluated"] * (len(statuses) - failed_index - 1):
                raise _integrity("Objective aggregation failure status order is inconsistent.")
            objective = objectives[failed_index]
            if not isinstance(objective, Mapping):
                raise _integrity("Objective aggregation owner is malformed.")
            terms = components[failed_index].get("terms") if isinstance(components[failed_index], Mapping) else None
            if (
                not isinstance(terms, list)
                or any(
                    not isinstance(term, Mapping) or term.get("status") != "success"
                    for term in terms
                )
            ):
                raise _integrity("Objective aggregation failure does not preserve successful terms.")
            _verify_candidate_failure_context(
                outcome["failure"], objectives=objectives, candidate=value,
                phase=phase, owner={"kind": "objective", "objective_id": objective["id"]},
                affected=[],
            )
        elif phase == "total_aggregation":
            if any(status != "success" for status in statuses):
                raise _integrity("Total aggregation failure does not preserve successful objectives.")
            _verify_candidate_failure_context(
                outcome["failure"], objectives=objectives, candidate=value,
                phase=phase, owner={"kind": "candidate"}, affected=[],
            )
        else:
            raise _integrity("Population candidate uses a baseline-only or unknown failure phase.")
    else:
        raise _integrity("Optimization candidate outcome discriminator is unknown.")

def _verify_optimization_winner(
    result: Mapping[str, object],
    spec: Mapping[str, object],
    plan: Mapping[str, object],
    ledgers: list[Mapping[str, object]],
) -> None:
    variables = spec.get("variables")
    objectives = spec.get("objectives")
    optimizer = spec.get("optimizer")
    if not isinstance(variables, list) or not isinstance(objectives, list) or not isinstance(optimizer, dict):
        raise _integrity("Optimization Result request spec is malformed.")
    baseline = result.get("baseline")
    _verify_candidate_outcome(
        baseline,
        variables=variables,
        objectives=objectives,
        plan=plan,
        optimization_authorizations=spec.get("allow_extrapolation", []),
        generation=0,
        column=None,
        evaluation_ordinal=0,
        baseline=True,
    )
    if (
        result.get("completed_generations") != len(ledgers)
        or result.get("completed_generations") != optimizer.get("complete_generations")
        or result.get("unused_evaluations") != optimizer.get("unused_evaluations")
        or not ledgers
        or ledgers[-1]["continuation_certificate"].get("boundary") != "terminal_post_update"
    ):
        raise _integrity("Optimization Result does not close its requested complete generations.")
    records = [baseline, *(candidate for ledger in ledgers for candidate in ledger["candidates"])]
    seen: dict[bytes, object] = {}
    winners: list[tuple[float, int, Mapping[str, object]]] = []
    for record in records:
        parameters = _canonical_bytes(record["parameters"])
        cached = record.get("cache_hit")
        comparable_outcome = _optimization_outcome_without_candidate_position(record.get("outcome"))
        if cached is True and (parameters not in seen or seen[parameters] != comparable_outcome):
            raise _integrity("Optimization cache hit does not match its earlier candidate.")
        if cached is False and parameters in seen:
            raise _integrity("Repeated optimization parameters were not marked as a cache hit.")
        seen.setdefault(parameters, comparable_outcome)
        outcome = record["outcome"]
        if outcome.get("status") == "success":
            winners.append((_f64_value(outcome["cost_f64"]), record["evaluation_ordinal"], record))
    winner = min(winners, key=lambda item: (item[0], item[1]))[2]
    best = result.get("best")
    if (
        not isinstance(best, dict)
        or best.get("evaluation_ordinal") != winner.get("evaluation_ordinal")
        or best.get("cost_f64") != winner["outcome"].get("cost_f64")
        or best.get("parameters") != winner.get("parameters")
    ):
        raise _integrity("Optimization winner does not match the earliest lowest finite candidate.")

def _optimization_outcome_without_candidate_position(value: object) -> object:
    if isinstance(value, Mapping):
        if value.get("schema") == "scnsim.optimization_failure_context":
            return {key: ("<candidate-position>" if key == "candidate" else _optimization_outcome_without_candidate_position(item)) for key, item in value.items()}
        return {key: _optimization_outcome_without_candidate_position(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_optimization_outcome_without_candidate_position(item) for item in value]
    return value

def _prior_ledger_is_receipt_backed(
    directory: Path,
    *,
    request_sha256: str,
    attempt_sha256: object,
    artifact_id: str,
    digest: object,
) -> bool:
    if not isinstance(attempt_sha256, str) or _SHA256.fullmatch(attempt_sha256) is None:
        return False
    current_match = _ATTEMPT.fullmatch(directory.name)
    staging_match = _STAGING.fullmatch(directory.name)
    if current_match is not None:
        current_ordinal = int(directory.name)
    elif staging_match is not None:
        current_ordinal = int(staging_match.group(1))
    else:
        return False
    for sibling in directory.parent.iterdir():
        if (
            sibling.is_symlink()
            or not sibling.is_dir()
            or _ATTEMPT.fullmatch(sibling.name) is None
            or int(sibling.name) >= current_ordinal
        ):
            continue
        attempt_path = sibling / "attempt.json"
        receipt_path = sibling / "receipt.json"
        if attempt_path.is_symlink() or receipt_path.is_symlink():
            raise _integrity("Prior attempt evidence is symlinked.", path=str(sibling))
        if not attempt_path.is_file() or not receipt_path.is_file():
            continue
        prior_attempt = _load_canonical(attempt_path)
        if _sha256(_canonical_bytes(prior_attempt)) != attempt_sha256:
            continue
        receipt = _load_canonical(receipt_path)
        if (
            receipt.get("schema") != "scnsim.receipt"
            or receipt.get("schema_version") != 1
            or receipt.get("request_sha256") != request_sha256
            or receipt.get("attempt_sha256") != attempt_sha256
            or receipt.get("outcome") not in {"success", "failure", "interrupted"}
            or not isinstance(receipt.get("artifacts"), list)
        ):
            continue
        links = receipt.get("artifacts")
        if isinstance(links, list) and any(
            isinstance(link, dict)
            and link.get("id") == artifact_id
            and link.get("sha256") == digest
            for link in links
        ):
            return True
    return False

def verified_generation_links(
    directory: Path,
    *,
    request_sha256: str,
    attempt_sha256: str,
    allow_other_artifacts: bool = False,
) -> list[dict[str, str]]:
    """Build and verify the receipt links for completed staged generations."""

    root = _inside(directory, "artifacts/generations")
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise _integrity("Generation artifact directory is unsafe.")
    children = sorted(root.iterdir()) if root.exists() else []
    if any(
        path.is_symlink()
        or not path.is_file()
        or re.fullmatch(r"[0-9]{6,}\.json", path.name) is None
        for path in children
    ):
        raise _integrity("Generation artifact directory contains an unsafe entry.")
    links = [
        {
            "id": f"generation_{path.stem}",
            "sha256": _sha256(path.read_bytes()),
        }
        for path in children
    ]
    _verify_generation_artifacts(
        directory,
        links,
        request_sha256=request_sha256,
        attempt_sha256=attempt_sha256,
        allow_other_artifacts=allow_other_artifacts,
    )
    return links
