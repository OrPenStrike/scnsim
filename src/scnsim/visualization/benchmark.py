"""Pure HTML presentation for one immutable benchmark observation record."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from html import escape
import json
from typing import Any

from ..benchmark.models import BenchmarkResult
from ..benchmark.prepared import array_from_record, record_document
from .plots.common import _plotly, _style, figure_fragment
from .presentation import Theme, _report_html


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"benchmark record {label} must be an object")
    return value


def _required_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"benchmark record {label} must be an array")
    return value


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _recorded(value: object) -> object:
    return "not recorded" if value is None else value


def _objective_units(
    declaration: Mapping[str, object] | None,
) -> dict[str, str]:
    """Read objective display units from the immutable bound declaration."""

    if declaration is None:
        return {}
    analysis_value = declaration.get("analysis")
    if not isinstance(analysis_value, Mapping):
        return {}
    spec_value = analysis_value.get("spec")
    if not isinstance(spec_value, Mapping):
        return {}
    objectives_value = spec_value.get("objectives")
    if not isinstance(objectives_value, Sequence) or isinstance(objectives_value, (str, bytes)):
        return {}

    units_by_id: dict[str, str] = {}
    for raw_objective in objectives_value:
        if not isinstance(raw_objective, Mapping):
            continue
        objective_id = raw_objective.get("id")
        target_value = raw_objective.get("target")
        if not isinstance(objective_id, str) or not isinstance(target_value, Mapping):
            continue
        unit = target_value.get("si_unit")
        if isinstance(unit, str):
            units_by_id[objective_id] = unit
    return units_by_id


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{escape(label)}</th>" for label in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _details(title: str, value: object) -> str:
    return (
        f"<details><summary>{escape(title)}</summary>"
        f"<pre style=\"white-space:pre-wrap;overflow-wrap:anywhere\">"
        f"{escape(_json(value))}</pre></details>"
    )


def _f64_token(value: object) -> str:
    """Decode one event Float64 through the benchmark's tagged record codec."""

    decoded = record_document(json.dumps({"value": {"f64": value}}).encode("utf-8"))
    return f"{decoded['value']:.12g}"


def _f64_roundtrip_token(value: object) -> str:
    """Render one tagged Float64 with a shortest round-trip decimal."""

    decoded = record_document(json.dumps({"value": {"f64": value}}).encode("utf-8"))
    return repr(decoded["value"])


def _complex_token(value: object) -> str:
    complex_record = _required_mapping(value, "complex value")
    return (
        f"{_f64_token(complex_record['real_f64'])}"
        f" {float(_f64_token(complex_record['imag_f64'])):+.12g}j"
    )


def _native_readable(value: object, *, field: str | None = None) -> object:
    """Format recorded numeric tokens for reading without changing their authority."""

    if isinstance(value, Mapping):
        if set(value) == {"f64"}:
            return _f64_token(value["f64"])
        if set(value) == {"real_f64", "imag_f64"}:
            return _complex_token(value)
        if {"dtype", "shape", "data_hex"}.issubset(value):
            return {
                "dtype": value["dtype"],
                "shape": value["shape"],
                "values": _native_readable(array_from_record(dict(value)).tolist()),
            }
        return {
            str(key): _native_readable(item, field=str(key))
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_native_readable(item) for item in value]
    if field is not None and field.endswith("_f64") and isinstance(value, str):
        return _f64_token(value)
    if isinstance(value, complex):
        return f"{value.real:.12g} {value.imag:+.12g}j"
    return value


def _native_json(value: object) -> str:
    return json.dumps(
        _native_readable(value), ensure_ascii=False, sort_keys=False, indent=2,
    )


def _native_compact(value: object, *, field: str | None = None) -> str:
    readable = _native_readable(value, field=field)
    if isinstance(readable, (Mapping, list)):
        return json.dumps(readable, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
    return str(readable)


def _native_value_text(value: object, *, field: str | None = None) -> str:
    """Summarize a recorded scalar value without changing its units or tags."""

    if isinstance(value, Mapping):
        encoded = value.get("si_value_f64")
        unit = value.get("si_unit")
        if isinstance(encoded, str) and isinstance(unit, str):
            return f"{_f64_roundtrip_token(encoded)} {unit}"
        if set(value) == {"f64"}:
            return _f64_roundtrip_token(value["f64"])
        if set(value) == {"real_f64", "imag_f64"}:
            real = _f64_roundtrip_token(value["real_f64"])
            imaginary = float(_f64_roundtrip_token(value["imag_f64"]))
            return f"{real} {imaginary:+.17g}j"
    if field is not None and field.endswith("_f64") and isinstance(value, str):
        return _f64_roundtrip_token(value)
    if field in {"root_omega_rad_s", "root_slope"} and isinstance(value, str):
        return _f64_roundtrip_token(value)
    return _native_compact(value, field=field)


def _native_parameters(value: object) -> str:
    if not isinstance(value, Mapping):
        return _native_compact(value)
    parameters = value
    bindings_value = parameters.get("bindings")
    if not isinstance(bindings_value, Sequence) or isinstance(bindings_value, (str, bytes)):
        return _native_compact(parameters)

    bindings: list[str] = []
    for raw_binding in bindings_value:
        if not isinstance(raw_binding, Mapping):
            bindings.append(_native_compact(raw_binding))
            continue
        binding = raw_binding
        parameter_value = binding.get("parameter")
        if not isinstance(parameter_value, Mapping):
            bindings.append(_native_compact(binding))
            continue
        parameter = parameter_value
        name = parameter.get("parameter_id")
        if name is None:
            name = parameter.get("definitions_id", "parameter")
        parameter_value = binding.get("value")
        bindings.append(
            f"{name}={_native_value_text(parameter_value) if parameter_value is not None else 'not recorded'}"
        )
    return "; ".join(bindings)


def _native_scalar_quantities(value: object) -> str | None:
    """Render scalar-catalog quantities while leaving the catalog in disclosure."""

    found: list[str] = []

    def visit(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            encoded = item.get("si_value_f64")
            unit = item.get("si_unit")
            if isinstance(encoded, str) and isinstance(unit, str):
                found.append(f"{path}={_f64_roundtrip_token(encoded)} {unit}")
                return
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    return "; ".join(found) if found else None


def _native_artifact_label(value: object) -> str:
    artifact = _required_mapping(value, "native artifact reference")
    role = artifact["role"]
    path = artifact["path"]
    digest = artifact["sha256"]
    return f"{role}: {path} (SHA-256 {digest})"


def _python_artifact_label(value: object) -> str:
    artifact = _required_mapping(value, "Python observation artifact reference")
    role = artifact["role"]
    path = artifact["path"]
    digest = artifact["sha256"]
    return f"{role}: {path} (SHA-256 {digest})"


def _python_observation_summary(
    value: object,
    *,
    objective_units: Mapping[str, str],
) -> str:
    """Summarize a verified Python observation without expanding its history."""

    record = _required_mapping(value, "Python numerical observation")
    parts: list[str] = []
    for key in ("status", "failure"):
        if record.get(key) is not None:
            parts.append(f"{key} {_native_compact(record[key], field=key)}")
    if record.get("parameters") is not None:
        parts.append(f"parameters {_native_parameters(record['parameters'])}")
    for key in (
        "cost_f64", "value_f64", "value", "response_value",
        "root_omega_rad_s", "root_slope", "frequency_hz_f64",
        "root_frequency_hz_f64", "linewidth_hz_f64", "cache_hit",
    ):
        if record.get(key) is not None:
            text = _native_value_text(record[key], field=key)
            if key in {"root_omega_rad_s", "frequency_hz_f64", "root_frequency_hz_f64", "linewidth_hz_f64"}:
                unit = "rad/s" if key == "root_omega_rad_s" else "Hz"
                text = f"{text} {unit}"
            parts.append(f"{key} {text}")
    objectives = record.get("objectives", record.get("objective_components"))
    if objectives is not None:
        parts.append(f"objectives {_native_objectives(objectives, objective_units)}")
    arrays = record.get("arrays")
    if isinstance(arrays, Mapping):
        for name, array_value in arrays.items():
            array_item = _required_mapping(array_value, f"Python {name} array")
            catalog = _required_mapping(array_item["catalog"], f"Python {name} array catalog")
            packed = _required_mapping(array_item["values"], f"Python {name} array values")
            shape = " × ".join(str(dimension) for dimension in packed["shape"])
            parts.append(f"{name} shape {shape}, unit {catalog['unit']}")
    return "; ".join(parts) if parts else "No scalar summary fields are recorded."


def _python_observation_sections(
    tasks: Sequence[Mapping[str, object]],
    objective_units: Mapping[str, str],
) -> list[str]:
    """Render only the baseline/winner summary from verified Python projections."""

    sections: list[str] = []
    for task in tasks:
        if task.get("arm") == "original_julia":
            continue
        task_id = task["task_id"]
        events = _required_sequence(task["events"], f"task {task_id}.events")
        for raw_event in events:
            event = _required_mapping(raw_event, f"task {task_id} event")
            if event.get("kind") not in {"completed", "failed", "interrupted"}:
                continue
            payload = _required_mapping(event["payload"], f"task {task_id} event payload")
            observations_value = payload.get("numerical_observations")
            if observations_value is None:
                continue
            observations = _required_mapping(
                observations_value,
                f"task {task_id} numerical observations",
            )
            if observations.get("result_kind") != "optimization":
                continue

            records = _required_sequence(
                observations["records"],
                f"task {task_id} numerical observation records",
            )
            rows: list[tuple[object, ...]] = []
            role_counts: dict[str, int] = {}
            for record_index, raw_record in enumerate(records):
                record = _required_mapping(
                    raw_record,
                    f"task {task_id} numerical observation {record_index}",
                )
                role = str(record["role"])
                role_counts[role] = role_counts.get(role, 0) + 1
                if role != "baseline":
                    continue
                artifact = _required_mapping(
                    record["source_artifact"],
                    f"task {task_id} baseline source artifact",
                )
                rows.append((
                    role,
                    _recorded(record.get("generation")),
                    _recorded(record.get("native_ordinal")),
                    _python_observation_summary(
                        record["value"], objective_units=objective_units,
                    ),
                    _python_artifact_label(artifact),
                ))

            winner_value = observations.get("winner")
            if winner_value is not None:
                winner = _required_mapping(winner_value, f"task {task_id} winner")
                winner_artifact = _required_mapping(
                    winner["source_artifact"],
                    f"task {task_id} winner source artifact",
                )
                rows.append((
                    "winner",
                    _recorded(winner.get("generation")),
                    _recorded(winner.get("native_ordinal")),
                    _python_observation_summary(
                        winner["value"], objective_units=objective_units,
                    ),
                    _python_artifact_label(winner_artifact),
                ))

            terminal_summary_value = observations.get("terminal_summary")
            terminal_summary = (
                {} if terminal_summary_value is None else _required_mapping(
                    terminal_summary_value,
                    f"task {task_id} compact terminal summary",
                )
            )
            result_artifact_value = observations.get("result_artifact")
            result_artifact = (
                None if result_artifact_value is None else _required_mapping(
                    result_artifact_value,
                    f"task {task_id} compact result artifact",
                )
            )
            terminal_field_names = ["type", "algorithm_id", "unused_evaluations"]
            if result_artifact is not None:
                # Task events may span several attempts; only the terminal
                # projection owns this result's completed-generation count.
                terminal_field_names.append("completed_generations")
            terminal_fields = {
                key: terminal_summary[key]
                for key in terminal_field_names
                if key in terminal_summary
            }
            count_text = ", ".join(
                f"{role}: {count}" for role, count in role_counts.items()
            ) or "no numerical rows"
            artifact_text = (
                "No terminal result artifact is recorded."
                if result_artifact is None
                else f"Compact terminal result: {_python_artifact_label(result_artifact)}"
            )
            if winner_value is None:
                winner_text = (
                    "No winner is present in this verified observation projection."
                )
            else:
                winner_text = "Winner is the recorded projection row; no candidate ranking was recomputed."

            sections.append(
                f"<h3>Task {escape(str(task_id))}: Python Optimization summary "
                f"({escape(str(event['kind']))} event)</h3>"
                "<p>This compact summary uses the verified numerical-observation projection. "
                "The ordered evaluation table and event disclosures remain the candidate history; "
                "this section does not decode the compact terminal artifact as a full result.</p>"
                + _native_table(
                    ("record", "generation", "evaluation ordinal", "recorded values", "source artifact"),
                    rows,
                    ("8%", "8%", "10%", "42%", "32%"),
                )
                + ("" if rows else "<p>No baseline or winner rows are present in this verified projection.</p>")
                + f"<p>Projection rows: {escape(count_text)}. {escape(winner_text)}</p>"
                + ("<p>Compact terminal summary fields: "
                   f"{escape(_native_compact(terminal_fields))}</p>" if terminal_fields else "")
                + f"<p>{escape(artifact_text)}</p>"
            )
    return sections


def _diagnostic_persistence_section(
    declaration: Mapping[str, object] | None,
    tasks: Sequence[Mapping[str, object]],
    execution_failures: object,
) -> str:
    """Explain only recorded policies and lifecycle evidence for diagnostic durability."""

    benchmark_value = None if declaration is None else declaration.get("benchmark")
    benchmark = benchmark_value if isinstance(benchmark_value, Mapping) else {}
    checkpoint = benchmark.get("checkpoint")
    diagnostics = benchmark.get("diagnostics")
    checkpoint_text = str(_recorded(checkpoint))
    per_arm_diagnostics = isinstance(diagnostics, Mapping)
    diagnostics_text = (
        _native_compact(diagnostics)
        if per_arm_diagnostics else str(_recorded(diagnostics))
    )

    if checkpoint == "generation":
        checkpoint_description = (
            "Generation checkpointing retains CMA resume snapshots at committed generations."
        )
    elif checkpoint == "off":
        checkpoint_description = (
            "CMA resume snapshots are disabled; baseline/root anchors, committed numerical "
            "observations, and terminal results retain their separate durability."
        )
    else:
        checkpoint_description = "The checkpoint policy is not recorded as a recognized value."

    def describe_diagnostics_policy(policy: object) -> str:
        if policy == "immediate":
            return "diagnostic events are persisted as they are received"
        if policy == "boundary":
            return (
                "diagnostic events are buffered until generation commit, terminal outcome, "
                "or a handleable interruption/failure"
            )
        return "the diagnostic persistence policy is not a recognized value"

    if per_arm_diagnostics:
        diagnostics_description = (
            "Each task selects the concrete policy from this map using its recorded arm. "
            + "; ".join(
                f"{arm}: {describe_diagnostics_policy(policy)}"
                for arm, policy in diagnostics.items()
            )
        )
        boundary_tasks: list[Mapping[str, object]] = []
        unresolved_policy_tasks: list[tuple[str, object]] = []
        policy_by_task_id: dict[str, object] = {}
        for task in tasks:
            task_id = str(task.get("task_id", "not recorded"))
            arm = task.get("arm")
            policy = diagnostics.get(arm) if isinstance(arm, str) else None
            policy_by_task_id[task_id] = policy
            if policy == "boundary":
                boundary_tasks.append(task)
            elif policy != "immediate":
                unresolved_policy_tasks.append((task_id, arm))
    else:
        if diagnostics == "immediate":
            diagnostics_description = "Diagnostic events are persisted as they are received."
        elif diagnostics == "boundary":
            diagnostics_description = (
                "Diagnostic events are buffered until generation commit, terminal outcome, or a "
                "handleable interruption/failure."
            )
        else:
            diagnostics_description = "The diagnostic persistence policy is not recorded as a recognized value."
        boundary_tasks = list(tasks) if diagnostics == "boundary" else []
        unresolved_policy_tasks = []

    uncertain_tasks: list[tuple[str, str]] = []
    if boundary_tasks:
        for task in boundary_tasks:
            task_id = str(task.get("task_id", "not recorded"))
            events = _required_sequence(task.get("events", []), f"task {task_id}.events")
            terminal_attempts: set[str] = set()
            process_exit_attempts: set[str] = set()
            for raw_event in events:
                event = _required_mapping(raw_event, f"task {task_id} event")
                kind = event.get("kind")
                payload = _required_mapping(event.get("payload", {}), f"task {task_id} event payload")
                attempt_id = payload.get("attempt_id")
                if kind in {"completed", "failed", "interrupted"} and isinstance(attempt_id, str):
                    terminal_attempts.add(attempt_id)
                if kind == "failed":
                    failure = payload.get("failure")
                    if isinstance(failure, Mapping) and failure.get("stage") == "process_exit":
                        if isinstance(attempt_id, str):
                            process_exit_attempts.add(attempt_id)
                        else:
                            uncertain_tasks.append((task_id, "process_exit failure without attempt binding"))

            attempts = _required_sequence(task.get("attempts", []), f"task {task_id}.attempts")
            for raw_attempt in attempts:
                attempt = _required_mapping(raw_attempt, f"task {task_id} attempt")
                attempt_id = str(attempt.get("attempt_id", "not recorded"))
                status = attempt.get("status")
                attempt_failure = attempt.get("failure")
                attempt_process_exit = (
                    isinstance(attempt_failure, Mapping)
                    and attempt_failure.get("stage") == "process_exit"
                )
                if attempt_id in process_exit_attempts or attempt_process_exit:
                    uncertain_tasks.append((task_id, f"attempt {attempt_id} ended with process_exit"))
                elif status in {"allocated", "launched"} and attempt_id not in terminal_attempts:
                    uncertain_tasks.append((task_id, f"attempt {attempt_id} remains {status}"))

    global_failures = (
        [] if execution_failures is None
        else _required_sequence(execution_failures, "execution_failures")
    )
    global_process_exit = False
    global_process_exit_for_boundary_task = False
    global_unbound_process_exit = False
    boundary_task_ids = {
        task_id for task_id, policy in policy_by_task_id.items() if policy == "boundary"
    } if per_arm_diagnostics else set()
    immediate_task_ids = {
        task_id for task_id, policy in policy_by_task_id.items() if policy == "immediate"
    } if per_arm_diagnostics else set()
    for raw_failure in global_failures:
        failure = _required_mapping(raw_failure, "execution failure")
        error = failure.get("error")
        if isinstance(error, Mapping) and error.get("stage") == "process_exit":
            global_process_exit = True
            if not per_arm_diagnostics:
                break
            failure_task_id = failure.get("task_id")
            if isinstance(failure_task_id, str) and failure_task_id in boundary_task_ids:
                global_process_exit_for_boundary_task = True
            elif isinstance(failure_task_id, str) and failure_task_id in immediate_task_ids:
                continue
            else:
                failure_arm = failure.get("arm")
                failure_arm_policy = (
                    diagnostics.get(failure_arm)
                    if isinstance(failure_arm, str) else None
                )
                if failure_arm_policy == "boundary":
                    global_process_exit_for_boundary_task = True
                elif failure_arm_policy != "immediate":
                    global_unbound_process_exit = True

    if not per_arm_diagnostics and diagnostics == "boundary" and global_process_exit and not uncertain_tasks:
        uncertain_tasks.append(("benchmark", "recorded process_exit without task-level completeness evidence"))
    elif per_arm_diagnostics and boundary_tasks and not uncertain_tasks:
        if global_process_exit_for_boundary_task:
            uncertain_tasks.append(("benchmark", "recorded process_exit without task-level completeness evidence"))

    if per_arm_diagnostics:
        if uncertain_tasks:
            lifecycle = (
                "The record shows an abnormal or nonterminal process lifecycle for a task using "
                "boundary diagnostics. Any diagnostic tail still buffered in the child is unknown; "
                "this report does not infer a lost-event count. Affected records: "
                + "; ".join(f"{task_id} ({reason})" for task_id, reason in uncertain_tasks)
                + "."
            )
        elif boundary_tasks:
            lifecycle = (
                "No abnormal or nonterminal process lifecycle is recorded for tasks whose selected "
                "policy is boundary. A recorded terminal outcome flushes their pending diagnostic "
                "stream; the report adds no separate completeness marker or event count."
            )
        else:
            lifecycle = (
                "No task selected boundary diagnostics from the recorded per-arm map; no "
                "boundary-buffer tail inference is applied."
            )
        if global_unbound_process_exit:
            lifecycle += (
                " A process_exit without a task or arm binding is also recorded; its policy and any "
                "diagnostic tail cannot be attributed, so no arm-specific tail conclusion is made."
            )
        if unresolved_policy_tasks:
            lifecycle += (
                " A policy could not be selected for these task arms, so no tail inference is made "
                "for them: "
                + "; ".join(
                    f"{task_id} (arm {arm if arm is not None else 'not recorded'})"
                    for task_id, arm in unresolved_policy_tasks
                )
                + "."
            )
    elif diagnostics != "boundary":
        lifecycle = "No boundary-buffer tail inference is applied to the recorded policy."
    elif uncertain_tasks:
        lifecycle = (
            "The record shows an abnormal or nonterminal process lifecycle. Any diagnostic "
            "tail still buffered in the child is unknown; this report does not infer a lost-event count. "
            + "Affected records: "
            + "; ".join(f"{task_id} ({reason})" for task_id, reason in uncertain_tasks)
            + "."
        )
    else:
        lifecycle = (
            "No abnormal or nonterminal process lifecycle is recorded for these tasks. A recorded "
            "terminal outcome flushes its pending diagnostic stream; this report does not add a "
            "separate completeness marker or infer an event count."
        )

    return (
        "<h2>Recorded persistence policies</h2>"
        + _table(
            ("policy", "recorded value", "meaning"),
            (
                ("CMA checkpoint", checkpoint_text, checkpoint_description),
                ("diagnostics", diagnostics_text, diagnostics_description),
            ),
        )
        + f"<p>{escape(lifecycle)}</p>"
    )


def _native_array_summary(name: str, value: object) -> tuple[str, str]:
    item = _required_mapping(value, f"native {name} array")
    catalog = _required_mapping(item["catalog"], f"native {name} catalog")
    packed = _required_mapping(item["values"], f"native {name} values")
    array = array_from_record(dict(packed))
    dimensions = " × ".join(str(dimension) for dimension in array.shape)
    unit = catalog["unit"]
    axes = catalog["axes"]
    coordinates = catalog.get("coordinate_ids")
    display_name = name.upper() if name in {"s", "y", "z"} else name
    labels = (
        f"{display_name}: shape {dimensions}, unit {unit}, axes "
        f"{json.dumps(_native_readable(axes), ensure_ascii=False)}"
    )
    if coordinates is not None:
        labels += f", coordinates {json.dumps(coordinates, ensure_ascii=False)}"
    detail = _details(
        f"Decoded {name} values in recorded units",
        _native_readable(array.tolist()),
    )
    detail += _details(f"{name} source catalog", catalog)
    return labels, detail


def _native_observation_summary(
    result_kind: object,
    role: object,
    value: object,
    objective_units: Mapping[str, str],
) -> tuple[str, list[str]]:
    record = _required_mapping(value, "native numerical value")
    summaries: list[str] = [f"{result_kind} {role}"]
    details: list[str] = []

    parameters = record.get("parameters")
    if parameters is not None:
        summaries.append(f"parameters {_native_parameters(parameters)}")
    for key in ("coordinates", "latent_coordinates"):
        if record.get(key) is not None:
            summaries.append(f"{key} {_native_compact(record[key])}")

    outcome_value = record.get("outcome")
    outcome = outcome_value if isinstance(outcome_value, Mapping) else {}
    status = record.get("status")
    if status is None:
        status = outcome.get("status")
    if status is not None:
        summaries.append(f"status {status}")
    failure = record.get("failure")
    if failure is None:
        failure = outcome.get("failure")
    if failure is not None:
        summaries.append(f"failure {_native_compact(failure)}")

    for key in (
        "cost_f64", "value_f64", "value",
        "response_value", "root_omega_rad_s", "root_slope",
        "frequency_hz_f64", "root_frequency_hz_f64", "linewidth_hz_f64",
        "family", "magnitude", "magnitude_f64", "real", "real_f64",
        "imag", "imag_f64", "cache_hit",
    ):
        field_value = record.get(key)
        if field_value is None and key in {"cost_f64", "value_f64"}:
            field_value = outcome.get(key)
        if field_value is not None:
            value_text = _native_value_text(field_value, field=key)
            if key in {"root_omega_rad_s", "frequency_hz_f64", "root_frequency_hz_f64", "linewidth_hz_f64"}:
                unit = "rad/s" if key == "root_omega_rad_s" else "Hz"
                value_text = f"{value_text} {unit}"
            summaries.append(f"{key} {value_text}")

    objectives = record.get("objectives", record.get("objective_components"))
    if objectives is None:
        objectives = outcome.get("objectives", outcome.get("objective_components"))
    if objectives is not None:
        summaries.append(f"objectives {_native_objectives(objectives, objective_units)}")

    scalar_catalog = record.get("scalar_catalog")
    if scalar_catalog is not None:
        scalar_summary = _native_scalar_quantities(scalar_catalog)
        if scalar_summary is not None:
            summaries.append(f"scalar values {scalar_summary}")

    arrays = record.get("arrays")
    if arrays is not None:
        array_map = _required_mapping(arrays, "native direct arrays")
        for name, array_value in array_map.items():
            label, detail = _native_array_summary(str(name), array_value)
            summaries.append(label)
            details.append(detail)

    if len(summaries) == 1:
        summaries.append("additional values are available in the exact tagged record")
    details.append(_details("Exact tagged native value", value))
    return "; ".join(summaries), details


def _native_objectives(
    value: object,
    objective_units: Mapping[str, str],
) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return _native_value_text(value)
    compact: list[dict[str, object]] = []
    for raw_objective in value:
        if not isinstance(raw_objective, Mapping):
            compact.append({"record": _native_value_text(raw_objective)})
            continue
        objective: dict[str, object] = {}
        objective_id = raw_objective.get("id", raw_objective.get("objective_id"))
        if objective_id is not None:
            objective["id"] = objective_id
        for key in ("status", "failure"):
            if key in raw_objective:
                objective[key] = _native_value_text(raw_objective[key], field=key)
        objective_value = raw_objective.get("value_f64")
        if objective_value is None:
            objective_value = raw_objective.get("value")
        if objective_value is not None:
            value_text = _native_value_text(
                objective_value,
                field="value_f64" if "value_f64" in raw_objective else "value",
            )
            unit = objective_units.get(str(objective_id)) if objective_id is not None else None
            has_recorded_unit = (
                isinstance(objective_value, Mapping)
                and isinstance(objective_value.get("si_unit"), str)
            )
            if unit is not None and not has_recorded_unit:
                value_text = f"{value_text} {unit}"
            objective["value"] = value_text
        for key in ("cost_f64", "weighted_cost_f64", "normalized_residual_f64"):
            if key in raw_objective:
                objective[key.removesuffix("_f64")] = _native_value_text(
                    raw_objective[key], field=key,
                )
        if "terms" in raw_objective:
            terms = raw_objective["terms"]
            if isinstance(terms, Sequence) and not isinstance(terms, (str, bytes)):
                term_summaries: list[object] = []
                for raw_term in terms:
                    if not isinstance(raw_term, Mapping):
                        term_summaries.append(_native_value_text(raw_term))
                        continue
                    term: dict[str, object] = {
                        key: _native_value_text(raw_term[key], field=key)
                        for key in ("term_ordinal", "status", "failure")
                        if key in raw_term
                    }
                    term_value = raw_term.get("value_f64")
                    if term_value is None:
                        term_value = raw_term.get("value")
                    if term_value is not None:
                        term["value"] = _native_value_text(term_value)
                    if "actual_complex" in raw_term:
                        term["actual_complex"] = _native_value_text(
                            raw_term["actual_complex"]
                        )
                    term_summaries.append(term)
                objective["terms"] = term_summaries
        compact.append(objective)
    return json.dumps(compact, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def _native_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    widths: Sequence[str],
) -> str:
    """Keep long native references wrapped inside a locally scrollable table."""

    head = "".join(
        f'<th style="vertical-align:top;overflow-wrap:anywhere;word-break:break-word">'
        f"{escape(label)}</th>"
        for label in headers
    )
    body = "".join(
        "<tr>" + "".join(
            '<td style="vertical-align:top;overflow-wrap:anywhere;word-break:break-word">'
            f"{escape(str(value))}</td>"
            for value in row
        ) + "</tr>"
        for row in rows
    )
    columns = "".join(f'<col style="width:{width}">' for width in widths)
    return (
        '<div style="max-width:100%;overflow-x:auto">'
        '<table style="width:100%;table-layout:fixed"><colgroup>'
        f"{columns}</colgroup><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        "</div>"
    )


def _native_winner_observation_summary(
    winner: Mapping[str, object],
    records: Sequence[object],
    objective_units: Mapping[str, str],
) -> str:
    """Join the reported winner to its recorded candidate row without ranking."""

    winner_artifact = _required_mapping(
        winner["source_artifact"], "native winner source artifact",
    )
    for raw_record in records:
        if not isinstance(raw_record, Mapping) or raw_record.get("role") != "candidate":
            continue
        if raw_record.get("generation") != winner.get("generation"):
            continue
        if raw_record.get("native_ordinal") != winner.get("native_ordinal"):
            continue
        source_value = raw_record.get("source_artifact")
        if not isinstance(source_value, Mapping):
            continue
        if source_value != winner_artifact:
            continue

        candidate_value = raw_record.get("value")
        if not isinstance(candidate_value, Mapping):
            return "matching objective unavailable"
        outcome_value = candidate_value.get("outcome")
        outcome = outcome_value if isinstance(outcome_value, Mapping) else {}
        status = candidate_value.get("status")
        if status is None:
            status = outcome.get("status")
        objectives = candidate_value.get("objectives")
        if objectives is None:
            objectives = candidate_value.get("objective_components")
        if objectives is None:
            objectives = outcome.get("objectives", outcome.get("objective_components"))

        summary: list[str] = []
        if status is not None:
            summary.append(f"matched candidate status {status}")
        if objectives is None:
            summary.append("matching objective unavailable")
        else:
            summary.append(
                f"matching objective {_native_objectives(objectives, objective_units)}"
            )
        return "; ".join(summary)
    return "matching candidate objective unavailable"


def _evaluation_rows(
    tasks: Sequence[Mapping[str, object]],
    objective_units: Mapping[str, str],
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for task in tasks:
        task_id = task["task_id"]
        events = _required_sequence(task["events"], f"task {task_id}.events")
        for raw_event in events:
            event = _required_mapping(raw_event, f"task {task_id} event")
            if event["kind"] != "evaluation":
                continue
            payload = _required_mapping(event["payload"], f"task {task_id} evaluation")
            ordinal = payload.get("evaluation_ordinal", payload.get("source_index", "not recorded"))
            if payload.get("generation") is not None:
                point = f"generation {payload['generation']}, evaluation {ordinal}"
            elif payload.get("origin") is not None:
                point = f"{payload['origin']} ({ordinal})"
            else:
                point = str(ordinal)
            status = payload.get("status")
            if status is None:
                status = "failure" if payload.get("failure") else "recorded"

            values: list[str] = []
            for family in ("S", "Y", "Z"):
                array_record = payload.get(family)
                if array_record is not None:
                    array = _required_mapping(array_record, f"evaluation {family}")
                    shape = tuple(array["shape"])
                    values.append(f"{family} matrix shape {shape}")
            response = payload.get("response_value")
            if response is not None:
                values.append(f"response { _complex_token(response) }")
            frequency = payload.get("frequency_hz_f64")
            if frequency is not None:
                values.append(f"root frequency {_f64_token(frequency)} Hz")
            linewidth = payload.get("linewidth_hz_f64")
            if linewidth is not None:
                values.append(f"linewidth {_f64_token(linewidth)} Hz")
            objectives = payload.get("objectives")
            if isinstance(objectives, Sequence) and not isinstance(objectives, (str, bytes)):
                for objective_value in objectives:
                    objective = _required_mapping(objective_value, "evaluation objective")
                    if objective.get("value_f64") is not None:
                        objective_id = objective.get("id", objective.get("objective_id"))
                        unit = (
                            objective_units.get(objective_id)
                            if isinstance(objective_id, str) else None
                        )
                        unit_text = f" {unit}" if unit is not None else ""
                        values.append(
                            f"{objective.get('id', 'objective')} value "
                            f"{_f64_token(objective['value_f64'])}{unit_text}"
                        )
            cost = payload.get("cost_f64")
            if cost is not None:
                values.append(f"cost {_f64_token(cost)}")
            if payload.get("failure") is not None:
                failure = _required_mapping(payload["failure"], "evaluation failure")
                values.append(f"failure {failure.get('kind', 'recorded')}")
            if not values:
                values.append("No numerical value fields recorded")
            rows.append((task_id, point, status, "; ".join(values)))
    return rows


def _native_observation_sections(
    tasks: Sequence[Mapping[str, object]],
    objective_units: Mapping[str, str],
) -> list[str]:
    """Render terminal native observations without inventing task events."""

    sections: list[str] = []
    for task in tasks:
        if task.get("arm") != "original_julia":
            continue
        task_id = task["task_id"]
        events = _required_sequence(task["events"], f"task {task_id}.events")
        for raw_event in events:
            event = _required_mapping(raw_event, f"task {task_id} event")
            payload = _required_mapping(
                event["payload"], f"task {task_id} terminal payload"
            )
            observations_value = payload.get("numerical_observations")
            if observations_value is None:
                continue
            observations = _required_mapping(
                observations_value,
                f"task {task_id} numerical observations",
            )
            result_kind = observations["result_kind"]
            result_artifact_value = observations["result_artifact"]
            result_artifact = (
                None if result_artifact_value is None else _required_mapping(
                    result_artifact_value, f"task {task_id} result artifact",
                )
            )
            records = _required_sequence(
                observations["records"],
                f"task {task_id} numerical observation records",
            )
            rows: list[tuple[object, ...]] = []
            details: list[str] = []
            for record_index, raw_record in enumerate(records):
                record = _required_mapping(
                    raw_record,
                    f"task {task_id} numerical observation {record_index}",
                )
                source_artifact = _required_mapping(
                    record["source_artifact"],
                    f"task {task_id} observation source artifact {record_index}",
                )
                role = record["role"]
                generation = _recorded(record.get("generation"))
                native_ordinal = _recorded(record.get("native_ordinal"))
                summary, value_details = _native_observation_summary(
                    result_kind, role, record["value"], objective_units,
                )
                rows.append((
                    role,
                    generation,
                    native_ordinal,
                    summary,
                    _native_artifact_label(source_artifact),
                ))
                details.extend(value_details)
                details.append(_details(
                    f"Native observation {record_index} source artifact reference",
                    source_artifact,
                ))

            winner_value = observations.get("winner")
            winner_details = ""
            if winner_value is not None:
                winner = _required_mapping(
                    winner_value,
                    f"task {task_id} numerical observation winner",
                )
                winner_artifact = _required_mapping(
                    winner["source_artifact"],
                    f"task {task_id} winner source artifact",
                )
                winner_summary, winner_value_details = _native_observation_summary(
                    result_kind, "winner", winner["value"], objective_units,
                )
                winner_summary += "; " + _native_winner_observation_summary(
                    winner, records, objective_units,
                )
                winner_details = (
                    "<h4>Reported native winner</h4>"
                    + _native_table(
                        ("generation", "native ordinal", "recorded values", "source artifact"),
                        [(
                            _recorded(winner.get("generation")),
                            _recorded(winner.get("native_ordinal")),
                            winner_summary,
                            _native_artifact_label(winner_artifact),
                        )],
                        ("9%", "10%", "43%", "38%"),
                    )
                    + "".join(winner_value_details)
                    + _details("Native winner source artifact reference", winner_artifact)
                )

            sections.append(
                f"<h3>Task {escape(str(task_id))}: "
                f"{escape(str(result_kind))} observations ({escape(str(event['kind']))} event)</h3>"
                "<p>Post hoc observations from the recorded native artifacts, in "
                "their recorded native order. Parameter and quantity summaries use "
                "round-trip Float64 text with recorded units; expandable tagged "
                "records retain exact values, source catalogs, and lineage. These "
                "records have no per-evaluation timestamps or task durations.</p>"
                + ("<p>No terminal native result artifact is recorded.</p>" if result_artifact is None else "")
                + _details("Native result artifact", result_artifact)
                + _native_table(
                    (
                        "native record role", "generation", "native ordinal",
                        "recorded values", "source artifact",
                    ),
                    rows,
                    ("9%", "7%", "9%", "43%", "32%"),
                )
                + ("" if rows else "<p>No decoded native numerical records are present.</p>")
                + "".join(details)
                + winner_details
            )
    return sections


def _measurement_rows(
    owner: Mapping[str, object],
    *,
    scope: str,
    benchmark_clock: Mapping[str, object] | None,
) -> tuple[list[tuple[Mapping[str, object], Mapping[str, object]]], list[tuple[object, ...]]]:
    task_id = owner.get("task_id", scope)
    arm = owner.get("arm", scope)
    sample = owner.get("sample", scope)
    measurements = _required_sequence(owner["measurements"], f"{scope}.measurements")
    intervals: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    rows: list[tuple[object, ...]] = []
    for raw in measurements:
        measurement = _required_mapping(raw, f"{scope} measurement")
        measurement_clock_value = measurement.get("clock")
        if measurement_clock_value is None:
            measurement_clock = benchmark_clock
            clock_binding = (
                "per-row clock unavailable; legacy root binding"
                if benchmark_clock is not None
                else "per-row clock unavailable; root binding not recorded"
            )
        else:
            measurement_clock = _required_mapping(
                measurement_clock_value, f"{scope} measurement clock"
            )
            clock_binding = "measurement clock"
        clock_description = _clock_description(measurement_clock, clock_binding)
        intervals.append(({
            "task_id": task_id,
            "arm": arm,
            "sample": sample,
            "clock_description": clock_description,
        }, measurement))
        rows.append((
            task_id,
            arm,
            sample,
            scope,
            measurement["stage"],
            clock_description,
            measurement["start_ns"],
            measurement["duration_ns"],
            _json(measurement["counts"]),
            _json(measurement["memory_bytes"]),
            _json(measurement["details"]),
        ))
    return intervals, rows


def _clock_description(
    clock: Mapping[str, object] | None,
    binding: str,
) -> str:
    if clock is None:
        return binding
    return "; ".join((
        binding,
        f"id={_recorded(clock.get('id'))}",
        f"source={_recorded(clock.get('source'))}",
        f"unit={_recorded(clock.get('unit'))}",
        f"monotonic_origin_ns={_recorded(clock.get('monotonic_origin_ns'))}",
    ))


def _timing_figure(
    intervals: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
) -> Any:
    go, _, _ = _plotly()
    stages: list[str] = []
    for _, measurement in intervals:
        stage = str(measurement["stage"])
        if stage not in stages:
            stages.append(stage)
    stage_index = {stage: index for index, stage in enumerate(stages)}
    x: list[int] = []
    y_ms: list[float] = []
    hover: list[str] = []
    for task, measurement in intervals:
        stage = str(measurement["stage"])
        details = measurement["details"]
        details_record = details if isinstance(details, Mapping) else {}
        x.append(stage_index[stage])
        y_ms.append(int(measurement["duration_ns"]) / 1_000_000.0)
        hover.append(
            "<br>".join((
                f"task: {escape(str(task['task_id']))}",
                f"arm: {escape(str(task['arm']))}",
                f"sample: {escape(str(task['sample']))}",
                f"stage: {escape(stage)}",
                f"clock: {escape(str(task['clock_description']))}",
                f"start offset from listed origin (ns): {escape(str(measurement['start_ns']))}",
                f"duration (ms): {y_ms[-1]:.6g}",
                f"attempt: {escape(str(details_record.get('attempt_id', 'not recorded')))}",
            ))
        )
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=x,
        y=y_ms,
        mode="markers",
        name="observed interval",
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        marker={"size": 9},
    ))
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(stages))),
        ticktext=stages,
        title_text="Recorded stage",
    )
    figure.update_yaxes(title_text="Observed duration (ms)")
    _style(figure, Theme.AUTO, title="Recorded timing intervals")
    figure.update_layout(hovermode="closest")
    return figure


def render_benchmark(result: BenchmarkResult) -> str:
    """Render one exact manifest as standalone HTML without running or mutating it."""

    if not isinstance(result, BenchmarkResult):
        raise TypeError("render_benchmark requires BenchmarkResult")
    document = record_document(result.manifest_bytes)
    if document["schema"] != "scnsim.benchmark_record":
        raise ValueError("unsupported benchmark record schema")
    if document["schema_version"] not in (1, 2):
        raise ValueError("unsupported benchmark record version")

    declaration_value = document.get("declaration")
    declaration = (
        _required_mapping(declaration_value, "declaration")
        if declaration_value is not None else None
    )
    clock_value = document.get("clock")
    clock = _required_mapping(clock_value, "clock") if clock_value is not None else None
    tasks = _required_sequence(document.get("tasks", []), "tasks")
    reports = document.get("reports", [])
    task_records = [_required_mapping(task, f"tasks[{index}]") for index, task in enumerate(tasks)]

    body: list[str] = [
        "<h1>SCNSim benchmark observations</h1>",
        "<p>Observed runs and saved evidence only. The global "
        "<code>benchmark_end_to_end</code> interval includes shared preparation "
        "and serial arms; task <code>end_to_end</code> intervals begin before "
        "arm-specific preparation. These scopes and nested stage intervals "
        "remain separate; the chart does not add durations. Start values are "
        "offsets from each row's listed clock origin. Rows with different clock "
        "IDs have separate origins; the chart compares durations by stage and "
        "does not align starts on a shared time axis.</p>",
    ]
    body.append(_table(
        ("record", "schema version", "benchmark request SHA-256", "Plan SHA-256", "source analysis SHA-256", "manifest SHA-256"),
        ((
            document["schema"],
            document["schema_version"],
            _recorded(document.get("benchmark_sha256")),
            _recorded(document.get("plan_sha256")),
            _recorded(document.get("source_analysis_sha256")),
            sha256(result.manifest_bytes).hexdigest(),
        ),),
    ))
    if declaration is None:
        body.append("<p>No immutable benchmark declaration was recorded.</p>")
    else:
        body.append(_details("Benchmark declaration", declaration))
    body.append(_diagnostic_persistence_section(
        declaration,
        task_records,
        document.get("execution_failures"),
    ))
    if clock is None:
        body.append("<p>No benchmark clock record was recorded.</p>")
    else:
        body.append(_details("Clock source and origin", clock))
    preparation_failure = document.get("preparation_failure")
    if preparation_failure is not None:
        body.append("<h2>Preparation failure</h2>")
        body.append(_details("Recorded preparation failure", preparation_failure))
    preparation_failures = document.get("preparation_failures")
    if preparation_failures:
        body.append("<h2>Unbound preparation failures</h2>")
        body.append(_details("Recorded invocation failures", preparation_failures))

    task_rows: list[tuple[object, ...]] = []
    intervals: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    measurement_rows: list[tuple[object, ...]] = []
    for task in task_records:
        task_id = task["task_id"]
        environment = _required_mapping(task["environment"], f"task {task_id}.environment")
        attempts = _required_sequence(task["attempts"], f"task {task_id}.attempts")
        artifacts = _required_sequence(task["artifacts"], f"task {task_id}.artifacts")
        task_rows.append((
            task_id,
            task["arm"],
            task["sample"],
            _recorded(environment.get("device_requested")),
            _recorded(environment.get("cpu_threads_requested")),
            task["request_sha256"],
            len(attempts),
            len(artifacts),
        ))
        task_intervals, task_measurements = _measurement_rows(
            task, scope="task", benchmark_clock=clock,
        )
        intervals.extend(task_intervals)
        measurement_rows.extend(task_measurements)

    benchmark_measurements = {
        "task_id": "benchmark",
        "measurements": _required_sequence(
            document.get("measurements", []), "benchmark-wide measurements"
        ),
    }
    benchmark_intervals, benchmark_rows = _measurement_rows(
        benchmark_measurements, scope="benchmark-wide", benchmark_clock=clock,
    )
    intervals = benchmark_intervals + intervals
    measurement_rows = benchmark_rows + measurement_rows

    if task_rows:
        body.append("<h2>Declared task samples</h2>" + _table(
            (
                "task", "arm", "sample", "requested device", "requested CPU threads",
                "request SHA-256", "attempt records", "artifact records",
            ),
            task_rows,
        ))
    else:
        body.append("<h2>Declared task samples</h2><p>No task records are present.</p>")

    objective_units = _objective_units(declaration)
    evaluation_rows = _evaluation_rows(task_records, objective_units)
    if evaluation_rows:
        body.append(
            "<h2>Recorded numerical evaluations</h2>"
            "<p>Summary numbers are displayed to 12 significant digits; the canonical JSON "
            "manifest remains the exact tagged-value authority.</p>"
            + _table(
                ("task", "point", "recorded status", "observed values"),
                evaluation_rows,
            )
        )
    else:
        body.append(
            "<h2>Recorded numerical evaluations</h2>"
            "<p>No task <code>evaluation</code> events are recorded.</p>"
        )

    python_observations = _python_observation_sections(task_records, objective_units)
    if python_observations:
        body.append("<h2>Recorded Python numerical result summaries</h2>")
        body.extend(python_observations)

    native_observations = _native_observation_sections(task_records, objective_units)
    if native_observations:
        body.append("<h2>Recorded native numerical observations</h2>")
        body.extend(native_observations)

    if intervals:
        figure = _timing_figure(intervals)
        body.append(figure_fragment(
            "Observed durations by stage (start offsets remain clock scoped)",
            figure,
            include_plotlyjs=True,
        ))
        body.append(_table(
            ("record", "arm", "sample", "measurement scope", "stage", "clock binding and origin", "start offset ns", "duration ns", "work counts", "memory bytes", "details"),
            measurement_rows,
        ))
    else:
        body.append("<h2>Timing observations</h2><p>No measurement intervals are recorded.</p>")

    for task in task_records:
        task_id = task["task_id"]
        body.append(f"<h2>Task {escape(str(task_id))}</h2>")
        body.append(_details("Actual environment snapshot", task["environment"]))

        attempts = _required_sequence(task["attempts"], f"task {task_id}.attempts")
        attempt_rows: list[tuple[object, ...]] = []
        for index, raw_attempt in enumerate(attempts):
            attempt = _required_mapping(raw_attempt, f"task {task_id}.attempts[{index}]")
            body.append(_details(
                f"Attempt {attempt['attempt_id']} record",
                attempt,
            ))
            attempt_rows.append((attempt["attempt_id"], attempt["status"]))
        if attempt_rows:
            body.append(_table(
                ("attempt", "status"),
                attempt_rows,
            ))
        else:
            body.append("<p>No attempt records are present for this task.</p>")

        events = _required_sequence(task["events"], f"task {task_id}.events")
        event_rows: list[tuple[object, ...]] = []
        for index, raw_event in enumerate(events):
            event = _required_mapping(raw_event, f"task {task_id}.events[{index}]")
            event_rows.append((event["sequence"], event["kind"]))
            body.append(_details(
                f"Event {event['sequence']}: {event['kind']} payload",
                event["payload"],
            ))
        if event_rows:
            body.append(_table(("sequence", "kind"), event_rows))
        else:
            body.append("<p>No task events are present.</p>")

        body.append(_details("Task artifacts", task["artifacts"]))

    body.append("<h2>Recorded report outcomes</h2>")
    body.append(_details("Report artifacts and report errors", reports))
    rendered_keys = {
        "schema", "schema_version", "benchmark_sha256", "declaration",
        "plan_sha256", "source_analysis_sha256", "clock", "tasks",
        "measurements", "reports", "preparation_failure", "preparation_failures",
    }
    additional = {
        key: value for key, value in document.items() if key not in rendered_keys
    }
    if additional:
        body.append(_details("Additional recorded root fields", additional))
    return _report_html("".join(body), Theme.AUTO)


__all__ = ["render_benchmark"]
