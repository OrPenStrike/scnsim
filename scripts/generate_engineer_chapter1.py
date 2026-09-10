"""Generate and verify the source-bound Chapter 1 engineer artifacts.

The five QMD fragments are the executable authority.  Quarto never calls this
tool: artifact generation is an explicit development operation in a private
workspace, while ``--check`` is source-only and launches no solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import linecache
import os
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from engineer_publication_binding import check_publication_binding


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "examples" / "engineer" / "chapter-01"
FIGURES = ROOT / "examples" / "engineer" / "figures"
MANIFEST = FIGURES / "chapter-01-artifacts.json"
SOURCES = tuple(CHAPTER / f"_{index:02d}-{name}.qmd" for index, name in (
    (1, "build"),
    (2, "diagram"),
    (3, "s11"),
    (4, "root"),
    (5, "change-capacitance"),
))
EXPECTED_CELL_IDS = (
    "ch1-import-and-define-capacitance",
    "ch1-build-grounded-lc-child",
    "ch1-assemble-coupler-and-port",
    "ch1-render-authoring-diagram",
    "ch1-show-authoring-diagram",
    "ch1-show-authoring-audit",
    "ch1-prepare-direct-request",
    "ch1-solve-baseline-s11",
    "ch1-show-baseline-s11",
    "ch1-prepare-loaded-root-request",
    "ch1-evaluate-baseline-root",
    "ch1-show-baseline-root",
    "ch1-select-capacitance",
    "ch1-solve-selected-s11",
    "ch1-show-selected-s11",
    "ch1-evaluate-selected-root",
    "ch1-show-selected-root",
)
ARTIFACT_NAMES = (
    "01-authoring.svg",
    "01-s11-baseline.svg",
    "01-s11-selected.svg",
    "01-s11-data.csv",
    "01-baseline-root.md",
    "01-quantities.csv",
    "01-quantities.md",
)
_DC = "{http://purl.org/dc/elements/1.1/}description"


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _parse_cells() -> tuple[tuple[str, str], ...]:
    cells: list[tuple[str, str]] = []
    for source in SOURCES:
        lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
        index = 0
        while index < len(lines):
            if lines[index].strip() != "```{python}":
                index += 1
                continue
            index += 1
            body: list[str] = []
            while index < len(lines) and lines[index].strip() != "```":
                body.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError(f"unterminated Python cell in {source.relative_to(ROOT)}")
            index += 1
            cell_id = next(
                (line.split(":", 1)[1].strip() for line in body if line.startswith("#| id:")),
                None,
            )
            if cell_id is None:
                raise ValueError(f"missing Python cell id in {source.relative_to(ROOT)}")
            code = "".join(line for line in body if not line.startswith("#|"))
            cells.append((cell_id, code))
    ids = tuple(cell_id for cell_id, _ in cells)
    if ids != EXPECTED_CELL_IDS:
        raise ValueError(f"Chapter 1 cell order changed: {ids!r}")
    return tuple(cells)


def _source_tree() -> dict[str, object]:
    paused = ROOT / "src" / "scnsim" / "_agent_knowledge"
    rows = {
        str(path.relative_to(ROOT)): _hash(path)
        for path in sorted((ROOT / "src" / "scnsim").rglob("*"))
        if path.is_file()
        and not path.is_relative_to(paused)
        and path.suffix in {".py", ".jl", ".json", ".toml"}
    }
    return {
        "sha256": _hash_bytes(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ),
        "files": rows,
    }


def _binding() -> dict[str, object]:
    aggregate = CHAPTER / "chapter.qmd"
    wrappers = tuple(CHAPTER / f"0{index}-{name}.qmd" for index, name in (
        (1, "build"),
        (2, "diagram"),
        (3, "s11"),
        (4, "root"),
        (5, "change-capacitance"),
    ))
    sources = (aggregate, *SOURCES, *wrappers)
    return {
        "generator_sha256": _hash(Path(__file__)),
        "sources": {
            str(path.relative_to(ROOT)): _hash(path)
            for path in sources
        },
        "source_tree": _source_tree(),
        "cell_ids": list(EXPECTED_CELL_IDS),
        "cells": {
            cell_id: _hash_bytes(code.encode())
            for cell_id, code in _parse_cells()
        },
    }


def _environment() -> dict[str, str]:
    import scnsim

    expected = (ROOT / "src" / "scnsim" / "__init__.py").resolve()
    if Path(scnsim.__file__).resolve() != expected:
        raise RuntimeError("generation imported scnsim from outside this checkout")
    versions = {"python": sys.version.split()[0], "scnsim": scnsim.__version__}
    for name in ("kaleido", "matplotlib", "numpy", "pint", "plotly", "schemdraw"):
        versions[name] = importlib.metadata.version(name)
    return versions


def _certificate(path: Path) -> dict[str, str]:
    root = ET.fromstring(path.read_bytes())
    description = root.find(f".//{_DC}")
    if description is None or not isinstance(description.text, str):
        raise ValueError("authoring SVG has no SCNSim certificate")
    value = json.loads(description.text)
    identity = value.get("identity") if isinstance(value, dict) else None
    if value.get("kind") != "scnsim_circuit_diagram_certificate" or not isinstance(identity, dict):
        raise ValueError("authoring SVG certificate is malformed")
    required = (
        "representation", "plan_id", "plan_sha256", "parameters_sha256",
        "connectivity_sha256", "semantic_sha256", "presentation_sha256",
    )
    if any(not isinstance(identity.get(field), str) for field in required):
        raise ValueError("authoring SVG certificate is incomplete")
    if identity["representation"] != "authoring":
        raise ValueError("Chapter 1 SVG is not an authoring diagram")
    return {field: identity[field] for field in required}


def _identity(result: object) -> dict[str, object]:
    identity = getattr(result, "identity")
    if hasattr(identity, "batch"):
        return {
            "batch": _identity(type("Bound", (), {"identity": identity.batch})()),
            "source_index": identity.source_index,
            "parameters_sha256": identity.parameters_sha256,
        }
    return {
        field: getattr(identity, field)
        for field in ("plan_sha256", "request_sha256", "attempt_sha256", "result_sha256")
    }


def _workspace(requested: Path | None) -> Path:
    if requested is None:
        return Path(tempfile.mkdtemp(prefix="scnsim-engineer-chapter1-"))
    workspace = requested.resolve()
    if ROOT == workspace or ROOT in workspace.parents:
        raise ValueError("execution workspace must be outside the repository")
    if workspace.exists() and any(workspace.iterdir()):
        raise RuntimeError("execution workspace must be new or empty")
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _execute(cells: tuple[tuple[str, str], ...], workspace: Path) -> dict[str, object]:
    execution = workspace / "execution"
    execution.mkdir()
    namespace: dict[str, object] = {"__name__": "__main__"}
    previous = Path.cwd()
    try:
        os.chdir(execution)
        for ordinal, (cell_id, code) in enumerate(cells, start=1):
            path = execution / f"{ordinal:02d}-{cell_id}.py"
            path.write_text(code, encoding="utf-8")
            linecache.checkcache(str(path))
            namespace["__file__"] = str(path)
            exec(compile(code, str(path), "exec"), namespace)
    finally:
        os.chdir(previous)
    return namespace


def _response_rows(response: object) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.asarray(response.s.view.frequencies.to("gigahertz").magnitude)
    matrix = np.asarray(response.s.view.matrix.magnitude)
    if matrix.shape != (len(frequencies), 1, 1):
        raise RuntimeError(f"Chapter 1 did not return a one-port response: {matrix.shape}")
    values = matrix[:, 0, 0]
    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(values)):
        raise RuntimeError("Chapter 1 response contains non-finite values")
    return frequencies, values


def _quantity(value: object, unit: str) -> float:
    return float(value.to(unit).magnitude)


def _write_markdown_tables(observations: dict[str, object], directory: Path) -> None:
    rows = observations["quantity_rows"]
    if not isinstance(rows, list) or len(rows) != 2:
        raise RuntimeError("Chapter 1 quantity observations are malformed")
    baseline = rows[0]
    if not isinstance(baseline, dict) or baseline.get("point") != "baseline":
        raise RuntimeError("Chapter 1 baseline quantity observation is missing")
    (directory / "01-baseline-root.md").write_text(
        "\n".join((
            "| Baseline loaded Result | Value |",
            "|---|---:|",
            f"| frequency | {baseline['loaded_root_frequency_GHz']:.9f} GHz |",
            f"| linewidth | {baseline['loaded_root_linewidth_MHz']:.9f} MHz |",
            "",
        )),
        encoding="utf-8",
    )
    lines = [
        "| Point | C | Unloaded `1 / (2π√(LC))` | Loaded root | Loaded linewidth |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("Chapter 1 quantity row is malformed")
        lines.append(
            f"| {row['point']} | {row['capacitance_fF']:.9g} fF | "
            f"{row['unloaded_lc_frequency_GHz']:.9f} GHz | "
            f"{row['loaded_root_frequency_GHz']:.9f} GHz | "
            f"{row['loaded_root_linewidth_MHz']:.9f} MHz |"
        )
    lines.append("")
    (directory / "01-quantities.md").write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(namespace: dict[str, object], exports: Path) -> dict[str, object]:
    exports.mkdir()
    diagram = namespace["diagram"]
    authoring = exports / "01-authoring.svg"
    diagram.show().save(authoring)
    certificate = _certificate(authoring)

    for variable, name in (
        ("baseline_s11_figure", "01-s11-baseline.svg"),
        ("selected_s11_figure", "01-s11-selected.svg"),
    ):
        figure = namespace[variable]
        figure.write_image(exports / name, format="svg")

    baseline_response = namespace["baseline_response"]
    selected_response = namespace["selected_response"]
    baseline_frequency, baseline_values = _response_rows(baseline_response)
    selected_frequency, selected_values = _response_rows(selected_response)
    if not np.array_equal(baseline_frequency, selected_frequency):
        raise RuntimeError("baseline and selected responses use different frequency grids")

    data_path = exports / "01-s11-data.csv"
    with data_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "frequency_GHz",
            "baseline_real", "baseline_imag", "baseline_magnitude_dB", "baseline_phase_deg",
            "selected_real", "selected_imag", "selected_magnitude_dB", "selected_phase_deg",
        ))
        for frequency, baseline, selected in zip(
            baseline_frequency, baseline_values, selected_values, strict=True
        ):
            writer.writerow((
                repr(float(frequency)),
                repr(float(baseline.real)), repr(float(baseline.imag)),
                repr(float(20.0 * np.log10(abs(baseline)))), repr(float(np.angle(baseline, deg=True))),
                repr(float(selected.real)), repr(float(selected.imag)),
                repr(float(20.0 * np.log10(abs(selected)))), repr(float(np.angle(selected, deg=True))),
            ))

    capacitance = namespace["capacitance"]
    inductance = 5.8e-9
    baseline_capacitance = _quantity(capacitance.baseline, "farad")
    selected_capacitance = _quantity(
        namespace["selected_parameters"].values[capacitance], "farad"
    )
    baseline_root = namespace["baseline_root"]
    selected_root = namespace["selected_root"]
    quantity_rows = (
        (
            "baseline", baseline_capacitance / 1e-15,
            1.0 / (2.0 * np.pi * np.sqrt(inductance * baseline_capacitance)) / 1e9,
            _quantity(baseline_root.frequency, "gigahertz"),
            _quantity(baseline_root.linewidth, "megahertz"),
        ),
        (
            "selected", selected_capacitance / 1e-15,
            1.0 / (2.0 * np.pi * np.sqrt(inductance * selected_capacitance)) / 1e9,
            _quantity(selected_root.frequency, "gigahertz"),
            _quantity(selected_root.linewidth, "megahertz"),
        ),
    )
    with (exports / "01-quantities.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("point", "capacitance_fF", "unloaded_lc_frequency_GHz", "loaded_root_frequency_GHz", "loaded_root_linewidth_MHz"))
        for row in quantity_rows:
            writer.writerow((row[0], *(repr(float(value)) for value in row[1:])))

    observations: dict[str, object] = {}
    for name, values in (("baseline", baseline_values), ("selected", selected_values)):
        magnitude_db = 20.0 * np.log10(np.abs(values))
        phase = np.angle(values, deg=True)
        observations[name] = {
            "magnitude_db_min": float(np.min(magnitude_db)),
            "magnitude_db_max": float(np.max(magnitude_db)),
            "phase_deg_min": float(np.min(phase)),
            "phase_deg_max": float(np.max(phase)),
            "phase_deg_span": float(np.max(phase) - np.min(phase)),
        }
    observations["quantity_rows"] = [
        {
            "point": row[0],
            "capacitance_fF": row[1],
            "unloaded_lc_frequency_GHz": row[2],
            "loaded_root_frequency_GHz": row[3],
            "loaded_root_linewidth_MHz": row[4],
            "loaded_minus_unloaded_GHz": row[3] - row[2],
        }
        for row in quantity_rows
    ]
    records = {
        "certificate": certificate,
        "results": {
            "baseline_direct": _identity(baseline_response),
            "selected_direct": _identity(selected_response),
            "baseline_root": _identity(baseline_root),
            "selected_root": _identity(selected_root),
        },
        "observations": observations,
    }
    _write_markdown_tables(observations, exports)
    return records


def _check(manifest: dict[str, object]) -> None:
    if manifest.get("schema") != "scnsim.engineer_chapter1_artifacts.v1":
        raise RuntimeError("unsupported Chapter 1 artifact manifest")
    check_publication_binding(manifest, _binding())
    execution = manifest.get("execution")
    binding = manifest["binding"]
    if (
        not isinstance(execution, dict)
        or execution.get("cells") != binding["cells"]
        or execution.get("source_tree_sha256") != binding["source_tree"]["sha256"]
    ):
        raise RuntimeError("Chapter 1 execution binding is stale")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_NAMES):
        raise RuntimeError("Chapter 1 artifact inventory changed")
    for name in ARTIFACT_NAMES:
        path = FIGURES / name
        if not path.is_file() or artifacts[name] != _hash(path):
            raise RuntimeError(f"Chapter 1 artifact is missing or stale: {name}")
    if manifest.get("certificate") != _certificate(FIGURES / "01-authoring.svg"):
        raise RuntimeError("Chapter 1 authoring certificate is stale")
    print("engineer Chapter 1 artifacts are current")


def _rebind(workspace: Path, cells: tuple[tuple[str, str], ...]) -> None:
    receipt_path = workspace.resolve() / "generation-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "scnsim.engineer_chapter1_artifacts.v1":
        raise RuntimeError("prior Chapter 1 generation receipt has the wrong schema")
    prior_binding = receipt.get("binding")
    if (
        not isinstance(prior_binding, dict)
        or prior_binding.get("source_tree") != _source_tree()
    ):
        raise RuntimeError("package source changed since Chapter 1 execution")
    executed_hashes: dict[str, str] = {}
    for ordinal, (cell_id, code) in enumerate(cells, start=1):
        executed = workspace / "execution" / f"{ordinal:02d}-{cell_id}.py"
        expected = _hash_bytes(code.encode())
        if not executed.is_file() or _hash(executed) != expected:
            raise RuntimeError(f"executed Chapter 1 cell differs from current source: {cell_id}")
        executed_hashes[cell_id] = expected

    prior_artifacts = receipt.get("artifacts")
    old_names = set(ARTIFACT_NAMES) - {"01-baseline-root.md", "01-quantities.md"}
    if not isinstance(prior_artifacts, dict) or set(prior_artifacts) != old_names:
        raise RuntimeError("prior Chapter 1 artifact inventory is not the expected first export")
    for name in old_names:
        path = FIGURES / name
        if not path.is_file() or _hash(path) != prior_artifacts[name]:
            raise RuntimeError(f"prior Chapter 1 artifact changed: {name}")

    observations = receipt.get("observations")
    if not isinstance(observations, dict):
        raise RuntimeError("prior Chapter 1 observations are missing")
    _write_markdown_tables(observations, FIGURES)
    binding = _binding()
    manifest = {
        key: receipt[key]
        for key in ("schema", "environment", "certificate", "results", "observations")
    }
    manifest.update({
        "binding": binding,
        "execution": {
            "cells": executed_hashes,
            "generator_sha256": prior_binding["generator_sha256"],
            "generation_receipt_sha256": _hash(receipt_path),
            "source_tree_sha256": prior_binding["source_tree"]["sha256"],
            "rebound_without_execution": True,
        },
        "artifacts": {name: _hash(FIGURES / name) for name in ARTIFACT_NAMES},
    })
    temporary = FIGURES / ".chapter-01-artifacts.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST)
    print(MANIFEST)
    print("rebound from unchanged executed cells without numerical execution")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="new private workspace outside the repository")
    parser.add_argument("--check", action="store_true", help="verify source and artifact hashes without execution")
    parser.add_argument("--rebind-from", type=Path, help="rebind unchanged executed cells after documentation-only edits")
    args = parser.parse_args()
    cells = _parse_cells()
    if args.check and args.rebind_from is not None:
        parser.error("--check and --rebind-from are mutually exclusive")
    if args.check:
        _check(json.loads(MANIFEST.read_text(encoding="utf-8")))
        return 0
    if args.rebind_from is not None:
        if args.workspace is not None:
            parser.error("--workspace and --rebind-from are mutually exclusive")
        _rebind(args.rebind_from, cells)
        return 0

    workspace = _workspace(args.workspace)
    before = _binding()
    namespace = _execute(cells, workspace)
    exports = workspace / "exports"
    records = _write_outputs(namespace, exports)
    after = _binding()
    if before != after:
        raise RuntimeError("Chapter 1 or package sources changed during generation")

    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        shutil.copyfile(exports / name, FIGURES / name)
    manifest = {
        "schema": "scnsim.engineer_chapter1_artifacts.v1",
        "binding": before,
        "execution": {
            "cells": before["cells"],
            "generator_sha256": before["generator_sha256"],
            "source_tree_sha256": before["source_tree"]["sha256"],
            "rebound_without_execution": False,
        },
        "environment": _environment(),
        "artifacts": {name: _hash(FIGURES / name) for name in ARTIFACT_NAMES},
        **records,
    }
    temporary = FIGURES / ".chapter-01-artifacts.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST)
    (workspace / "generation-receipt.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(MANIFEST)
    print(workspace / "generation-receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
