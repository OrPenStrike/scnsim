"""Generate and verify source-bound Chapter 4 engineer artifacts.

The QMD fragments are the executable authority. Quarto never calls this tool:
generation is an explicit operation in a fresh workspace, while ``--check`` is
source-only and launches no solver.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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

from engineer_publication_binding import check_publication_binding, installed_course_runtime


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "examples" / "engineer" / "chapter-04"
FIGURES = ROOT / "examples" / "engineer" / "figures"
MANIFEST = FIGURES / "chapter-04-artifacts.json"
SOURCES = tuple(
    CHAPTER / name
    for name in (
        "_01-physical-plan.qmd",
        "_02-explicit-capstone.qmd",
        "_03-ptc-core.qmd",
        "_04-optional-transform.qmd",
        "_05-optional-quantity.qmd",
    )
)
WRAPPERS = tuple(
    CHAPTER / name
    for name in (
        "01-physical-plan.qmd",
        "02-explicit-capstone.qmd",
        "03-ptc-core.qmd",
        "04-optional-transform.qmd",
        "05-optional-quantity.qmd",
    )
)
EXPECTED_CELL_IDS = (
    "ch4-build-feedline",
    "ch4-build-readout-and-floating",
    "ch4-assemble-four-port-root",
    "ch4-configure-explicit-composition",
    "ch4-render-explicit-capstone",
    "ch4-derive-ptc-view",
    "ch4-solve-ptc-direct",
    "ch4-derive-transformed-view",
    "ch4-evaluate-response-element",
)
BASE_ARTIFACT_NAMES = (
    "04-explicit-capstone.svg",
    "04-raw-direct-s21.svg",
    "04-raw-direct-s21.csv",
    "04-raw-direct-s21.md",
    "04-ptc-direct-s21.svg",
    "04-ptc-direct-s21.csv",
    "04-ptc-direct-s21.md",
    "04-response-element.csv",
    "04-response-element.md",
)
RETIRED_COMPARISON_ARTIFACT_NAMES = (
    "04-optional-direct-s21.svg", "04-optional-direct-s21.csv", "04-optional-direct-s21.md",
    "04-independent-direct-hb.md", "04-pump-off-hb.svg", "04-pump-off-hb.csv",
)
OWNED_ARTIFACT_NAMES = BASE_ARTIFACT_NAMES
_DC = "{http://purl.org/dc/elements/1.1/}description"


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _chapter_artifact_inventory(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("04-*") if path.is_file()}


def _publish_artifacts(output: Path, destination: Path) -> None:
    if _chapter_artifact_inventory(output) != set(OWNED_ARTIFACT_NAMES):
        raise RuntimeError("Chapter 4 generated artifact inventory is not exact")
    destination.mkdir(parents=True, exist_ok=True)
    existing = _chapter_artifact_inventory(destination)
    if existing - set((*OWNED_ARTIFACT_NAMES, *RETIRED_COMPARISON_ARTIFACT_NAMES)):
        raise RuntimeError("unknown Chapter 4 artifacts require review")
    # Only owning generation removes the explicitly retired comparison outputs.
    for name in RETIRED_COMPARISON_ARTIFACT_NAMES:
        path = destination / name
        if path.exists():
            path.unlink()
    for name in OWNED_ARTIFACT_NAMES:
        shutil.copy2(output / name, destination / name)
    if _chapter_artifact_inventory(destination) != set(OWNED_ARTIFACT_NAMES):
        raise RuntimeError("Chapter 4 published artifact inventory is not exact")


def _verify_artifact_inventory(manifest: dict[str, object], directory: Path) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(OWNED_ARTIFACT_NAMES):
        raise RuntimeError("Chapter 4 artifact inventory is malformed")
    if _chapter_artifact_inventory(directory) != set(OWNED_ARTIFACT_NAMES):
        raise RuntimeError("Chapter 4 published artifact inventory is not exact")
    if artifacts != {name: _hash(directory / name) for name in OWNED_ARTIFACT_NAMES}:
        raise RuntimeError("Chapter 4 artifact hashes are stale")


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
        raise ValueError(f"Chapter 4 cell order changed: {ids!r}")
    return tuple(cells)


def _source_tree() -> dict[str, object]:
    paused = ROOT / "src" / "scnsim" / "_agent_knowledge"
    rows = {
        str(path.relative_to(ROOT)): _hash(path)
        for path in sorted((ROOT / "src" / "scnsim").rglob("*"))
        if path.is_file()
        and not path.is_relative_to(paused)
        and path.suffix in {".py", ".jl", ".json", ".toml", ".cpp"}
    }
    return {
        "sha256": _hash_bytes(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ),
        "files": rows,
    }


def _binding() -> dict[str, object]:
    sources = (CHAPTER / "chapter.qmd", *SOURCES, *WRAPPERS, ROOT / "examples" / "engineer" / "diagram_layouts.py")
    cells = _parse_cells()
    return {
        "generator_sha256": _hash(Path(__file__)),
        "sources": {str(path.relative_to(ROOT)): _hash(path) for path in (*sources, ROOT / "scripts" / "engineer_publication_binding.py")},
        "source_tree": _source_tree(),
        "cell_ids": list(EXPECTED_CELL_IDS),
        "cells": {
            cell_id: _hash_bytes(code.encode()) for cell_id, code in cells
        },
    }


def _environment() -> dict[str, object]:
    runtime_binding = installed_course_runtime(ROOT)
    import scnsim
    scnsim.configure_runtime(cpu_threads=10)
    versions = {"python": sys.version.split()[0], "scnsim": runtime_binding["scnsim"]}
    for name in ("kaleido", "matplotlib", "numpy", "pint", "plotly", "schemdraw"):
        versions[name] = importlib.metadata.version(name)
    return versions | {"runtime_binding": runtime_binding}


def _certificate(path: Path) -> dict[str, str]:
    root = ET.fromstring(path.read_bytes())
    description = root.find(f".//{_DC}")
    if description is None or not isinstance(description.text, str):
        raise ValueError("Chapter 4 capstone SVG has no SCNSim certificate")
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
        raise ValueError("Chapter 4 capstone SVG certificate is malformed")
    return {field: identity[field] for field in required}


def _memory_call(result: object) -> dict[str, str]:
    return {"storage": result.persistence_kind, "execution_id": result.execution_id}


def _workspace(requested: Path | None) -> Path:
    if requested is None:
        return Path(tempfile.mkdtemp(prefix="scnsim-engineer-chapter4-"))
    workspace = requested.resolve()
    if ROOT == workspace or ROOT in workspace.parents:
        raise ValueError("execution workspace must be outside the repository")
    if workspace.exists() and any(workspace.iterdir()):
        raise RuntimeError("execution workspace must be new or empty")
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@contextmanager
def _execute(cells: tuple[tuple[str, str], ...], workspace: Path):
    execution = workspace / "execution"
    execution.mkdir()
    namespace: dict[str, object] = {"__name__": "__main__"}
    previous = Path.cwd()
    primary = None
    try:
        try:
            os.chdir(execution)
            for ordinal, (cell_id, code) in enumerate(cells, start=1):
                path = execution / f"{ordinal:02d}-{cell_id}.py"
                path.write_text(code, encoding="utf-8")
                linecache.checkcache(str(path))
                namespace["__file__"] = str(path)
                exec(compile(code, str(path), "exec"), namespace)
                print(f"completed {ordinal}/{len(cells)} {cell_id}", flush=True)
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                os.chdir(previous)
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note("course working-directory cleanup: " + repr(error))
        yield namespace
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup_error = None
        for name in ("run",):
            run = namespace.get(name)
            if run is None:
                continue
            try:
                run.close()
            except BaseException as error:
                original = primary if primary is not None else cleanup_error
                if original is None:
                    cleanup_error = error
                else:
                    original.add_note("course Run.close: " + repr(error))
        if primary is None and cleanup_error is not None:
            raise cleanup_error


def _write_trace(trace: object, stem: str, output: Path) -> dict[str, object]:
    frequencies = np.asarray(trace.frequencies.to("gigahertz").magnitude)
    values = np.asarray(trace.value.to("dimensionless").magnitude)
    if frequencies.ndim != 1 or values.shape != frequencies.shape:
        raise RuntimeError(f"{stem} trace shape is invalid")
    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"{stem} trace contains non-finite values")
    trace.plot(component="magnitude", magnitude="db").write_image(
        output / f"{stem}.svg", format="svg"
    )
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
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
        f"![{stem} magnitude at its declared samples.](../figures/{stem}.svg)",
        "",
        "| frequency (GHz) | Re(S21) | Im(S21) | |S21| | |S21| (dB) |",
        "|---:|---:|---:|---:|---:|",
    ]
    samples = []
    for frequency, value in zip(frequencies, values, strict=True):
        magnitude_db = 20.0 * np.log10(abs(value))
        lines.append(
            f"| {frequency:.6g} | {value.real:.12g} | {value.imag:.12g} | "
            f"{abs(value):.12g} | {magnitude_db:.12g} |"
        )
        samples.append(
            {
                "frequency_GHz": float(frequency),
                "real": float(value.real),
                "imag": float(value.imag),
                "magnitude": float(abs(value)),
            }
        )
    lines.extend(("", f"[Download exact complex samples](../figures/{stem}.csv)."))
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"samples": samples}


def _write_response_element(result: object, output: Path) -> dict[str, object]:
    value = complex(result.value.to("dimensionless").magnitude)
    magnitude = float(result.magnitude.to("dimensionless").magnitude)
    real = float(result.real.to("dimensionless").magnitude)
    imag = float(result.imag.to("dimensionless").magnitude)
    with (output / "04-response-element.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frequency_GHz", "real", "imag", "magnitude"))
        writer.writerow((6.2, real, imag, magnitude))
    lines = [
        "| request | frequency (GHz) | Re(S21) | Im(S21) | |S21| |",
        "|---|---:|---:|---:|---:|",
        f"| Direct response element | 6.2 | {real:.12g} | {imag:.12g} | {magnitude:.12g} |",
    ]
    (output / "04-response-element.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not np.isclose(value.real, real) or not np.isclose(value.imag, imag):
        raise RuntimeError("Chapter 4 response-element projections disagree")
    return {"frequency_GHz": 6.2, "real": real, "imag": imag, "magnitude": magnitude}




def _write_capstone(namespace: dict[str, object], output: Path) -> dict[str, str]:
    path = output / "04-explicit-capstone.svg"
    namespace["capstone"].show().save(path)
    return _certificate(path)


def generate(workspace: Path) -> None:
    cells = _parse_cells()
    runtime_environment = _environment()
    with _execute(cells, workspace) as namespace:
        output = workspace / "artifacts"
        output.mkdir()

        certificate = _write_capstone(namespace, output)
        observations = {
            "raw_direct": _write_trace(namespace["raw_transmission"], "04-raw-direct-s21", output),
            "ptc_direct": _write_trace(namespace["ptc_transmission"], "04-ptc-direct-s21", output),
            "response_element": _write_response_element(namespace["transmission_at_6_2"], output),
        }
        artifact_names = OWNED_ARTIFACT_NAMES
        artifacts = {name: _hash(output / name) for name in artifact_names}
        binding = _binding()
        manifest = {
            "schema": "scnsim.engineer_chapter4_artifacts.v1",
            "binding": binding,
            "execution": {
                "generator_sha256": binding["generator_sha256"],
                "source_tree_sha256": binding["source_tree"]["sha256"],
                "cells": binding["cells"],
            },
            "environment": runtime_environment,
            "certificate": certificate,
            "results": {
                "raw_direct": _memory_call(namespace["raw_direct"]),
                "ptc_direct": _memory_call(namespace["ptc_direct"]),
                "response_element": _memory_call(namespace["transmission_at_6_2"]),
            },
            "observations": observations,
            "artifacts": artifacts,
        }
        _publish_artifacts(output, FIGURES)
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt = workspace / "generation-receipt.json"
        receipt.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(receipt)


def check() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    check_publication_binding(manifest, _binding())
    _verify_artifact_inventory(manifest, FIGURES)
    print("engineer Chapter 4 artifacts are current")


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
