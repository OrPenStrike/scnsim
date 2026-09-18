"""Generate and verify source-bound Chapter 7 engineer artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import xml.etree.ElementTree as ET

import numpy as np

from engineer_publication_binding import check_publication_binding


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "examples" / "engineer" / "chapter-07"
FIGURES = ROOT / "examples" / "engineer" / "figures"
MANIFEST = FIGURES / "chapter-07-artifacts.json"
SOURCES = tuple(CHAPTER / name for name in ("_01-rlgc.qmd", "_02-ports.qmd", "_03-diagrams.qmd", "_04-parameter-points.qmd"))
WRAPPERS = tuple(CHAPTER / name for name in ("01-rlgc.qmd", "02-ports.qmd", "03-diagrams.qmd", "04-parameter-points.qmd"))
EXPECTED_CELL_IDS = (
    "ch7-define-parameters-and-rlgc", "ch7-add-native-line", "ch7-bind-signal-pins",
    "ch7-add-four-signal-ports", "ch7-render-authoring", "ch7-show-authoring-audit",
    "ch7-render-compiled", "ch7-define-three-points", "ch7-prepare-complete-s-request",
    "ch7-solve-three-points", "ch7-show-named-reflection-examples",
)
BASE_ARTIFACT_NAMES = ("07-n2-authoring.md", "07-n2-compiled.md", "07-response-points.csv", "07-response-points.md")
DIAGRAM_ARTIFACT_NAMES = {
    "authoring": "07-n2-authoring.svg",
    "compiled": "07-n2-compiled.svg",
}
REFLECTION_ARTIFACT_NAMES = (
    "07-baseline-reflection.svg",
    "07-length-1p8-mm-reflection.svg", "07-capacitance-times-170-over-175-reflection.svg",
)
CONDITIONAL_ARTIFACT_NAMES = (*DIAGRAM_ARTIFACT_NAMES.values(), *REFLECTION_ARTIFACT_NAMES)
OWNED_ARTIFACT_NAMES = (*BASE_ARTIFACT_NAMES, *CONDITIONAL_ARTIFACT_NAMES)
_DC = "{http://purl.org/dc/elements/1.1/}description"


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _inventory(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("07-*") if path.is_file()}


def _parse_cells() -> tuple[tuple[str, str], ...]:
    cells = []
    for source in SOURCES:
        lines = source.read_text(encoding="utf-8").splitlines(keepends=True); index = 0
        while index < len(lines):
            if lines[index].strip() != "```{python}": index += 1; continue
            index += 1; body = []
            while index < len(lines) and lines[index].strip() != "```": body.append(lines[index]); index += 1
            if index == len(lines): raise ValueError(f"unterminated Python cell in {source.relative_to(ROOT)}")
            cell_id = next((line.split(":", 1)[1].strip() for line in body if line.startswith("#| id:")), None)
            if cell_id is None: raise ValueError(f"missing Python cell id in {source.relative_to(ROOT)}")
            cells.append((cell_id, "".join(line for line in body if not line.startswith("#|")))); index += 1
    if tuple(cell_id for cell_id, _ in cells) != EXPECTED_CELL_IDS: raise ValueError("Chapter 7 cell order changed")
    return tuple(cells)


def _source_tree() -> dict[str, object]:
    paused = ROOT / "src" / "scnsim" / "_agent_knowledge"
    rows = {str(path.relative_to(ROOT)): _hash(path) for path in sorted((ROOT / "src" / "scnsim").rglob("*")) if path.is_file() and not path.is_relative_to(paused) and path.suffix in {".py", ".jl", ".json", ".toml"}}
    return {"sha256": _hash_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()), "files": rows}


def _binding() -> dict[str, object]:
    cells = _parse_cells(); sources = (CHAPTER / "chapter.qmd", *SOURCES, *WRAPPERS)
    return {"generator_sha256": _hash(Path(__file__)), "sources": {str(path.relative_to(ROOT)): _hash(path) for path in sources}, "source_tree": _source_tree(), "cell_ids": list(EXPECTED_CELL_IDS), "cells": {cell_id: _hash_bytes(code.encode()) for cell_id, code in cells}}


def _execute(cells: tuple[tuple[str, str], ...], workspace: Path) -> dict[str, object]:
    execution = workspace / "execution"; execution.mkdir(); path = execution / "chapter-07.py"
    code = "\n".join(f"# CELL {cell_id}\n{body}" for cell_id, body in cells); path.write_text(code, encoding="utf-8")
    name = "_scnsim_engineer_chapter7_generation"; module = types.ModuleType(name); module.__file__ = str(path); sys.modules[name] = module
    previous = Path.cwd()
    try:
        os.chdir(execution); exec(compile(code, str(path), "exec"), module.__dict__); return module.__dict__
    finally:
        os.chdir(previous); sys.modules.pop(name, None)


def _certificate(path: Path) -> dict[str, str]:
    description = ET.fromstring(path.read_bytes()).find(f".//{_DC}")
    if description is None or not isinstance(description.text, str): raise ValueError(f"{path.name} has no SCNSim certificate")
    record = json.loads(description.text); identity = record.get("identity", {})
    required = ("representation", "plan_id", "plan_sha256", "parameters_sha256", "connectivity_sha256", "semantic_sha256", "presentation_sha256")
    if record.get("kind") != "scnsim_circuit_diagram_certificate" or any(not isinstance(identity.get(field), str) for field in required): raise ValueError(f"{path.name} certificate is malformed")
    return {field: identity[field] for field in required}


def _environment() -> dict[str, str]:
    import scnsim
    if Path(scnsim.__file__).resolve() != (ROOT / "src" / "scnsim" / "__init__.py").resolve(): raise RuntimeError("generation imported scnsim from outside this checkout")
    values = {"python": sys.version.split()[0], "scnsim": scnsim.__version__}
    for name in ("kaleido", "matplotlib", "numpy", "pint", "plotly", "schemdraw"): values[name] = importlib.metadata.version(name)
    return values


def _workspace(requested: Path | None) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="scnsim-engineer-chapter7-")) if requested is None else requested.resolve()
    if ROOT == workspace or ROOT in workspace.parents: raise ValueError("execution workspace must be outside the repository")
    if workspace.exists() and any(workspace.iterdir()): raise RuntimeError("execution workspace must be new or empty")
    workspace.mkdir(parents=True, exist_ok=True); return workspace


def _point_identity(point: object) -> dict[str, object]:
    identity = point.identity
    return {"source_index": identity.source_index, "parameters_sha256": identity.parameters_sha256, "batch_result_sha256": identity.batch.result_sha256}


def _write_diagram_outcome(namespace: dict[str, object], output: Path, variable: str, stem: str, label: str) -> tuple[dict[str, object], tuple[str, ...], dict[str, str] | None]:
    result = namespace[variable]; markdown = output / f"{stem}.md"
    if result is None:
        markdown.write_text(f"### {label} outcome\n\nThe existing deterministic renderer reported a typed `schematic_layout` failure. No fallback layout was substituted.\n", encoding="utf-8")
        return {"status": "failure", "stage": "schematic_layout"}, (), None
    svg_name = f"{stem}.svg"; result.drawing.save(output / svg_name)
    markdown.write_text(f"![{label}.](../figures/{svg_name})\n\n[Open the SVG](../figures/{svg_name})\n", encoding="utf-8")
    return {"status": "success"}, (svg_name,), _certificate(output / svg_name)


def _write_points(namespace: dict[str, object], output: Path) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    labels = tuple(namespace["point_labels"]); points = tuple(namespace["direct_points"].points)
    trace_ids = tuple(trace.id for trace in namespace["channel_traces"])
    csv_path = output / "07-response-points.csv"; observations = []; plots = []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(("point", "source_index", "status", "trace", "frequency_GHz", "real", "imag", "magnitude"))
        for label, point in zip(labels, points, strict=True):
            identity = _point_identity(point)
            if not point.succeeded:
                failure = point.failure; writer.writerow((label, point.source_index, "failure", "", "", "", "", ""))
                observations.append({"label": label, "status": "failure", "kind": failure.kind, "stage": failure.stage, "message": str(failure), "identity": identity}); continue
            result = point.result; coordinates = list(result.s.view.coordinates)
            for trace_id in trace_ids:
                trace = result.traces[trace_id]; frequencies = np.asarray(trace.frequencies.to("gigahertz").magnitude); values = np.asarray(trace.value.to("dimensionless").magnitude)
                for frequency, value in zip(frequencies, values, strict=True): writer.writerow((label, point.source_index, "success", trace_id, float(frequency), float(value.real), float(value.imag), float(abs(value))))
            plot_name = {"baseline": "07-baseline-reflection.svg", "length_1p8_mm": "07-length-1p8-mm-reflection.svg", "capacitance_times_170_over_175": "07-capacitance-times-170-over-175-reflection.svg"}[label]
            result.traces["s_readout_head_from_readout_head"].plot(component="magnitude", magnitude="db").write_image(output / plot_name, format="svg"); plots.append(plot_name)
            observations.append({"label": label, "status": "success", "coordinates": coordinates, "trace_ids": list(trace_ids), "identity": identity})
    lines = ["### Three exact Direct points", "", "| point | status | complete named S channels |", "|---|---|---:|"]
    for row in observations: lines.append(f"| `{row['label']}` | {row['status']} | {len(row.get('trace_ids', ())) if row['status'] == 'success' else '—'} |")
    lines += ["", "Every successful point retains all 16 output-from-input channels in the CSV. The plots below show only the explicitly named `s_readout_head_from_readout_head` reflection example.", ""]
    for row in observations:
        if row["status"] == "success":
            name = {"baseline": "07-baseline-reflection.svg", "length_1p8_mm": "07-length-1p8-mm-reflection.svg", "capacitance_times_170_over_175": "07-capacitance-times-170-over-175-reflection.svg"}[row["label"]]
            lines += [f"![{row['label']} readout-head reflection.](../figures/{name})", ""]
    (output / "07-response-points.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return observations, tuple(plots)


def generate(workspace: Path) -> None:
    namespace = _execute(_parse_cells(), workspace); output = workspace / "artifacts"; output.mkdir(); certificates = {}; observations = {}
    for variable, stem, label in (("authoring", "07-n2-authoring", "N=2 authoring diagram"), ("compiled", "07-n2-compiled", "N=2 compiled diagram")):
        observation, names, certificate = _write_diagram_outcome(namespace, output, variable, stem, label); observations[variable] = observation
        if certificate is not None: certificates[names[0]] = certificate
    point_observations, point_names = _write_points(namespace, output); observations["parameter_points"] = point_observations
    if tuple(row.get("status") for row in point_observations) != ("success", "success", "success") or set(point_names) != set(REFLECTION_ARTIFACT_NAMES):
        raise RuntimeError("Chapter 7 requires three successful named reflection artifacts")
    names = (*BASE_ARTIFACT_NAMES, *(name for name in CONDITIONAL_ARTIFACT_NAMES if (output / name).exists()))
    if _inventory(output) != set(names): raise RuntimeError("Chapter 7 generated artifact inventory is not exact")
    artifacts = {name: _hash(output / name) for name in names}; binding = _binding(); batch_identity = namespace["direct_points"].identity
    manifest = {"schema": "scnsim.engineer_chapter7_artifacts.v1", "binding": binding, "execution": {"generator_sha256": binding["generator_sha256"], "source_tree_sha256": binding["source_tree"]["sha256"], "cells": binding["cells"]}, "environment": _environment(), "results": {"batch": {field: getattr(batch_identity, field) for field in ("plan_sha256", "request_sha256", "attempt_sha256", "result_sha256")}}, "observations": observations, "certificates": certificates, "artifacts": artifacts}
    FIGURES.mkdir(parents=True, exist_ok=True); unknown = _inventory(FIGURES) - set(OWNED_ARTIFACT_NAMES)
    if unknown: raise RuntimeError(f"unknown Chapter 7 artifacts require review: {sorted(unknown)!r}")
    for stale in CONDITIONAL_ARTIFACT_NAMES:
        if stale not in names and (FIGURES / stale).exists(): (FIGURES / stale).unlink()
    for name in names: shutil.copy2(output / name, FIGURES / name)
    if _inventory(FIGURES) != set(names): raise RuntimeError("Chapter 7 published artifact inventory is not exact")
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"; MANIFEST.write_text(text, encoding="utf-8"); receipt = workspace / "generation-receipt.json"; receipt.write_text(text, encoding="utf-8"); print(receipt)


def check() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    check_publication_binding(manifest, _binding())
    observations = manifest.get("observations", {})
    diagram_names = []
    for key, name in DIAGRAM_ARTIFACT_NAMES.items():
        observation = observations.get(key, {})
        status = observation.get("status")
        if status == "success":
            diagram_names.append(name)
        elif status != "failure" or observation.get("stage") != "schematic_layout":
            raise RuntimeError(f"Chapter 7 {key} diagram outcome is malformed")
    point_observations = observations.get("parameter_points")
    expected_labels = ("baseline", "length_1p8_mm", "capacitance_times_170_over_175")
    if not isinstance(point_observations, list) or tuple(row.get("label") for row in point_observations) != expected_labels or any(row.get("status") != "success" for row in point_observations):
        raise RuntimeError("Chapter 7 successful reflection outcomes are malformed")
    expected_names = (*BASE_ARTIFACT_NAMES, *diagram_names, *REFLECTION_ARTIFACT_NAMES)
    expected = manifest.get("artifacts")
    if not isinstance(expected, dict) or set(expected) != set(expected_names): raise RuntimeError("Chapter 7 artifact inventory is malformed")
    if _inventory(FIGURES) != set(expected_names): raise RuntimeError("Chapter 7 published artifact inventory is not exact")
    if expected != {name: _hash(FIGURES / name) for name in expected_names}: raise RuntimeError("Chapter 7 artifacts are stale")
    certificates = manifest.get("certificates")
    if not isinstance(certificates, dict) or set(certificates) != set(diagram_names): raise RuntimeError("Chapter 7 certificate inventory is malformed")
    if certificates != {name: _certificate(FIGURES / name) for name in diagram_names}: raise RuntimeError("Chapter 7 certificates are stale")
    print("engineer Chapter 7 artifacts are current")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--check", action="store_true"); parser.add_argument("--workspace", type=Path); args = parser.parse_args()
    if args.check:
        if args.workspace is not None: parser.error("--workspace cannot be used with --check")
        check()
    else: generate(_workspace(args.workspace))


if __name__ == "__main__": main()
