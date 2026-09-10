"""Generate and verify source-bound Chapter 3 engineer artifacts.

The QMD fragments are the executable authority. Quarto never calls this tool:
generation is an explicit operation in a fresh workspace, while ``--check`` is
source-only and launches no solver.
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


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "examples" / "engineer" / "chapter-03"
FIGURES = ROOT / "examples" / "engineer" / "figures"
MANIFEST = FIGURES / "chapter-03-artifacts.json"
SOURCES = tuple(
    CHAPTER / name
    for name in ("_01-build.qmd", "_02-diagram.qmd", "_03-s21.qmd")
)
WRAPPERS = tuple(
    CHAPTER / name
    for name in ("01-build.qmd", "02-diagram.qmd", "03-s21.qmd")
)
EXPECTED_CELL_IDS = (
    "ch3-build-feedline",
    "ch3-build-readout",
    "ch3-assemble-main-plan",
    "ch3-attempt-complete-diagram",
    "ch3-build-feedline-illustration",
    "ch3-build-readout-illustration",
    "ch3-prepare-direct-s21",
    "ch3-solve-direct-s21",
    "ch3-show-direct-s21",
)
ARTIFACT_NAMES = (
    "03-feedline-illustration.svg",
    "03-readout-illustration.svg",
    "03-direct-s21.svg",
    "03-direct-s21.csv",
    "03-direct-s21.md",
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
        raise ValueError(f"Chapter 3 cell order changed: {ids!r}")
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
    sources = (CHAPTER / "chapter.qmd", *SOURCES, *WRAPPERS)
    cells = _parse_cells()
    return {
        "generator_sha256": _hash(Path(__file__)),
        "sources": {str(path.relative_to(ROOT)): _hash(path) for path in sources},
        "source_tree": _source_tree(),
        "cell_ids": list(EXPECTED_CELL_IDS),
        "cells": {
            cell_id: _hash_bytes(code.encode()) for cell_id, code in cells
        },
    }


def _environment() -> dict[str, str]:
    import scnsim

    expected = (ROOT / "src" / "scnsim" / "__init__.py").resolve()
    if Path(scnsim.__file__).resolve() != expected:
        raise RuntimeError("generation imported scnsim from outside this checkout")
    versions = {"python": sys.version.split()[0], "scnsim": scnsim.__version__}
    for name in ("matplotlib", "numpy", "pint", "schemdraw"):
        versions[name] = importlib.metadata.version(name)
    return versions


def _certificate(path: Path) -> dict[str, str]:
    root = ET.fromstring(path.read_bytes())
    description = root.find(f".//{_DC}")
    if description is None or not isinstance(description.text, str):
        raise ValueError(f"{path.name} has no SCNSim certificate")
    value = json.loads(description.text)
    identity = value.get("identity") if isinstance(value, dict) else None
    required = (
        "representation",
        "plan_id",
        "plan_sha256",
        "parameters_sha256",
        "connectivity_sha256",
        "semantic_sha256",
        "presentation_sha256",
    )
    if (
        value.get("kind") != "scnsim_circuit_diagram_certificate"
        or not isinstance(identity, dict)
        or any(not isinstance(identity.get(field), str) for field in required)
    ):
        raise ValueError(f"{path.name} certificate is malformed")
    return {field: identity[field] for field in required}


def _identity(result: object) -> dict[str, str]:
    identity = getattr(result, "identity")
    return {
        field: getattr(identity, field)
        for field in ("plan_sha256", "request_sha256", "attempt_sha256", "result_sha256")
    }


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "x") and hasattr(value, "y"):
        return {"x": float(value.x), "y": float(value.y)}
    if all(hasattr(value, field) for field in ("xmin", "ymin", "xmax", "ymax")):
        return {
            field: float(getattr(value, field))
            for field in ("xmin", "ymin", "xmax", "ymax")
        }
    raise TypeError(f"unsupported Chapter 3 receipt value: {type(value).__name__}")


def _workspace(requested: Path | None) -> Path:
    if requested is None:
        return Path(tempfile.mkdtemp(prefix="scnsim-engineer-chapter3-"))
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
            print(f"completed {ordinal}/{len(cells)} {cell_id}", flush=True)
    finally:
        os.chdir(previous)
    return namespace


def _write_trace(namespace: dict[str, object], output: Path) -> dict[str, object]:
    trace = namespace["transmission"]
    frequencies = np.asarray(trace.frequencies.to("gigahertz").magnitude)
    values = np.asarray(trace.value.to("dimensionless").magnitude)
    if frequencies.shape != (3,) or values.shape != (3,):
        raise RuntimeError("Chapter 3 transmission is not the declared three samples")
    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(values)):
        raise RuntimeError("Chapter 3 transmission contains non-finite values")
    figure = trace.show(magnitude="db")
    figure.savefig(output / "03-direct-s21.svg", bbox_inches="tight")
    with (output / "03-direct-s21.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frequency_GHz", "real", "imag", "magnitude", "magnitude_dB"))
        for frequency, value in zip(frequencies, values, strict=True):
            writer.writerow(
                (
                    float(frequency),
                    float(value.real),
                    float(value.imag),
                    float(abs(value)),
                    float(20.0 * np.log10(abs(value))),
                )
            )
    lines = [
        "| frequency (GHz) | Re(S21) | Im(S21) | |S21| | |S21| (dB) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for frequency, value in zip(frequencies, values, strict=True):
        lines.append(
            f"| {frequency:.6g} | {value.real:.12g} | {value.imag:.12g} | "
            f"{abs(value):.12g} | {20.0 * np.log10(abs(value)):.12g} |"
        )
    (output / "03-direct-s21.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "samples": [
            {
                "frequency_GHz": float(frequency),
                "real": float(value.real),
                "imag": float(value.imag),
                "magnitude": float(abs(value)),
            }
            for frequency, value in zip(frequencies, values, strict=True)
        ]
    }


def generate(workspace: Path) -> None:
    cells = _parse_cells()
    namespace = _execute(cells, workspace)
    error = namespace.get("complete_diagram_error")
    if namespace.get("complete_diagram") is not None or error is None:
        raise RuntimeError("Chapter 3 complete Default layout did not retain its typed failure")
    if getattr(error, "stage", None) != "schematic_layout":
        raise RuntimeError("Chapter 3 complete diagram failed at the wrong stage")

    output = workspace / "artifacts"
    output.mkdir()
    illustration_certificates: dict[str, object] = {}
    for variable, name in (
        ("feedline_illustration", "03-feedline-illustration.svg"),
        ("readout_illustration", "03-readout-illustration.svg"),
    ):
        namespace[variable].show().save(output / name)
        illustration_certificates[name] = _certificate(output / name)
    observations = _write_trace(namespace, output)
    artifacts = {name: _hash(output / name) for name in ARTIFACT_NAMES}
    binding = _binding()
    manifest = {
        "schema": "scnsim.engineer_chapter3_artifacts.v1",
        "binding": binding,
        "execution": {
            "generator_sha256": binding["generator_sha256"],
            "source_tree_sha256": binding["source_tree"]["sha256"],
            "cells": binding["cells"],
        },
        "environment": _environment(),
        "complete_diagram": {
            "status": "unavailable",
            "stage": error.stage,
            "message": str(error),
            "evidence": _jsonable(dict(error.evidence)),
            "plan_id": "two_port_feedline_readout",
        },
        "illustration_certificates": illustration_certificates,
        "result": _identity(namespace["direct"]),
        "observations": observations,
        "artifacts": artifacts,
    }
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        shutil.copy2(output / name, FIGURES / name)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = workspace / "generation-receipt.json"
    receipt.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(receipt)


def check() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("binding") != _binding():
        raise RuntimeError("Chapter 3 source binding is stale")
    expected = manifest.get("artifacts")
    actual = {name: _hash(FIGURES / name) for name in ARTIFACT_NAMES}
    if expected != actual:
        raise RuntimeError("Chapter 3 artifact hashes are stale")
    print("engineer Chapter 3 artifacts are current")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    if args.check:
        if args.workspace is not None:
            parser.error("--workspace cannot be used with --check")
        check()
        return
    generate(_workspace(args.workspace))


if __name__ == "__main__":
    main()
