"""Generate and verify the source-bound Chapter 2 engineer artifacts.

The four QMD fragments are the executable authority. Quarto never calls this
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

from engineer_publication_binding import check_publication_binding


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "examples" / "engineer" / "chapter-02"
FIGURES = ROOT / "examples" / "engineer" / "figures"
MANIFEST = FIGURES / "chapter-02-artifacts.json"
SOURCES = tuple(
    CHAPTER / name
    for name in (
        "_01-build.qmd",
        "_02-sweep.qmd",
        "_03-optimize.qmd",
        "_04-optional-spaces.qmd",
    )
)
WRAPPERS = tuple(
    CHAPTER / name
    for name in (
        "01-build.qmd",
        "02-sweep.qmd",
        "03-optimize.qmd",
        "04-optional-spaces.qmd",
    )
)
EXPECTED_CELL_IDS = (
    "ch2-import-and-define-parameters",
    "ch2-build-grounded-lc-child",
    "ch2-assemble-coupler-and-port",
    "ch2-prepare-root-request",
    "ch2-define-capacitance-sweep",
    "ch2-run-capacitance-sweep",
    "ch2-show-capacitance-sweep",
    "ch2-define-optimization",
    "ch2-show-optimization-spec",
    "ch2-run-optimization",
    "ch2-evaluate-winner",
    "ch2-render-winner",
    "ch2-show-winner",
    "ch2-define-optional-spaces",
    "ch2-run-optional-grid",
    "ch2-show-optional-grid",
    "ch2-run-and-show-listed-points",
)
ARTIFACT_NAMES = (
    "02-capacitance-sweep.svg",
    "02-capacitance-sweep.csv",
    "02-capacitance-sweep.md",
    "02-optimized-winner.svg",
    "02-optimized-winner.csv",
    "02-optimized-winner.md",
    "02-optional-grid.svg",
    "02-optional-spaces.csv",
    "02-optional-spaces.md",
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
                raise ValueError(
                    f"unterminated Python cell in {source.relative_to(ROOT)}"
                )
            index += 1
            cell_id = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in body
                    if line.startswith("#| id:")
                ),
                None,
            )
            if cell_id is None:
                raise ValueError(
                    f"missing Python cell id in {source.relative_to(ROOT)}"
                )
            code = "".join(line for line in body if not line.startswith("#|"))
            cells.append((cell_id, code))
    ids = tuple(cell_id for cell_id, _ in cells)
    if ids != EXPECTED_CELL_IDS:
        raise ValueError(f"Chapter 2 cell order changed: {ids!r}")
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
        "sources": {
            str(path.relative_to(ROOT)): _hash(path) for path in sources
        },
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
        raise ValueError("winner SVG has no SCNSim certificate")
    value = json.loads(description.text)
    identity = value.get("identity") if isinstance(value, dict) else None
    if (
        value.get("kind") != "scnsim_circuit_diagram_certificate"
        or not isinstance(identity, dict)
    ):
        raise ValueError("winner SVG certificate is malformed")
    required = (
        "representation",
        "plan_id",
        "plan_sha256",
        "parameters_sha256",
        "connectivity_sha256",
        "semantic_sha256",
        "presentation_sha256",
    )
    if any(not isinstance(identity.get(field), str) for field in required):
        raise ValueError("winner SVG certificate is incomplete")
    if identity["representation"] != "authoring":
        raise ValueError("Chapter 2 winner SVG is not an authoring diagram")
    return {field: identity[field] for field in required}


def _identity(result: object) -> dict[str, object]:
    identity = getattr(result, "identity")
    if hasattr(identity, "batch"):
        batch = identity.batch
        return {
            "batch": {
                field: getattr(batch, field)
                for field in (
                    "plan_sha256",
                    "request_sha256",
                    "attempt_sha256",
                    "result_sha256",
                )
            },
            "source_index": identity.source_index,
            "parameters_sha256": identity.parameters_sha256,
        }
    return {
        field: getattr(identity, field)
        for field in (
            "plan_sha256",
            "request_sha256",
            "attempt_sha256",
            "result_sha256",
        )
    }


def _workspace(requested: Path | None) -> Path:
    if requested is None:
        return Path(tempfile.mkdtemp(prefix="scnsim-engineer-chapter2-"))
    workspace = requested.resolve()
    if ROOT == workspace or ROOT in workspace.parents:
        raise ValueError("execution workspace must be outside the repository")
    if workspace.exists() and any(workspace.iterdir()):
        raise RuntimeError("execution workspace must be new or empty")
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _execute(
    cells: tuple[tuple[str, str], ...], workspace: Path
) -> dict[str, object]:
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
            print(f"completed Chapter 2 cell {ordinal:02d}: {cell_id}", flush=True)
    finally:
        os.chdir(previous)
    return namespace


def _quantity(value: object, unit: str) -> float:
    return float(value.to(unit).magnitude)


def _source_index(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _point_rows(
    sweep: object,
    *,
    space: str,
    capacitance: object,
    inductance: object,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in sweep.points:
        values = point.parameters.values
        row: dict[str, object] = {
            "space": space,
            "source_index": _source_index(point.source_index),
            "capacitance_fF": _quantity(values[capacitance], "femtofarad"),
            "inductance_nH": _quantity(values[inductance], "nanohenry"),
            "status": "success" if point.succeeded else "failure",
            "frequency_GHz": None,
            "linewidth_MHz": None,
            "failure_kind": None,
            "failure_stage": None,
            "failure_message": None,
            "parameters_sha256": point.identity.parameters_sha256,
        }
        if point.succeeded:
            result = point.result
            row["frequency_GHz"] = _quantity(result.frequency, "gigahertz")
            row["linewidth_MHz"] = _quantity(result.linewidth, "megahertz")
        else:
            failure = point.failure
            row["failure_kind"] = failure.kind
            row["failure_stage"] = failure.stage
            row["failure_message"] = str(failure)
        rows.append(row)
    return rows


def _csv_value(value: object) -> object:
    return "" if value is None else repr(value) if isinstance(value, float) else value


def _write_point_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _markdown_cell(value: object, *, digits: int = 9) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value).replace("|", "\\|")


def _write_sweep_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| Point | C | Fixed L | Status | Loaded root | Linewidth / failure |",
        "|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        final = (
            f"{_markdown_cell(row['linewidth_MHz'])} MHz"
            if row["status"] == "success"
            else f"{_markdown_cell(row['failure_kind'])} at {_markdown_cell(row['failure_stage'])}"
        )
        lines.append(
            f"| {_markdown_cell(row['source_index'])} | "
            f"{_markdown_cell(row['capacitance_fF'])} fF | "
            f"{_markdown_cell(row['inductance_nH'])} nH | "
            f"{row['status']} | "
            f"{_markdown_cell(row['frequency_GHz'])} GHz | {final} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_optional_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| Space | Index | C | L | Status | Loaded root | Linewidth / failure |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        final = (
            f"{_markdown_cell(row['linewidth_MHz'])} MHz"
            if row["status"] == "success"
            else f"{_markdown_cell(row['failure_kind'])} at {_markdown_cell(row['failure_stage'])}"
        )
        lines.append(
            f"| {row['space']} | {_markdown_cell(row['source_index'])} | "
            f"{_markdown_cell(row['capacitance_fF'])} fF | "
            f"{_markdown_cell(row['inductance_nH'])} nH | "
            f"{row['status']} | {_markdown_cell(row['frequency_GHz'])} GHz | "
            f"{final} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_winner(
    path_csv: Path,
    path_markdown: Path,
    *,
    optimization: object,
    winner_root: object,
    capacitance: object,
    inductance: object,
) -> dict[str, object]:
    values = optimization.best.parameters.values
    row = {
        "capacitance_fF": _quantity(values[capacitance], "femtofarad"),
        "inductance_nH": _quantity(values[inductance], "nanohenry"),
        "target_frequency_GHz": 6.2,
        "loaded_root_frequency_GHz": _quantity(
            winner_root.frequency, "gigahertz"
        ),
        "loaded_root_linewidth_MHz": _quantity(
            winner_root.linewidth, "megahertz"
        ),
        "cost": float(optimization.best.cost),
        "lower_bound_fF": 80.0,
        "upper_bound_fF": 140.0,
        "seed": 17,
        "max_evaluations": 200,
        "completed_generation_ledgers": len(optimization.ledger),
    }
    with path_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow({key: _csv_value(value) for key, value in row.items()})
    path_markdown.write_text(
        "\n".join(
            (
                "| Returned winner | Value |",
                "|---|---:|",
                f"| capacitance | {row['capacitance_fF']:.12g} fF |",
                f"| fixed inductance | {row['inductance_nH']:.12g} nH |",
                f"| independent loaded root | {row['loaded_root_frequency_GHz']:.12g} GHz |",
                f"| independent loaded linewidth | {row['loaded_root_linewidth_MHz']:.12g} MHz |",
                f"| declared target | {row['target_frequency_GHz']:.12g} GHz |",
                f"| optimizer cost | {row['cost']:.12g} |",
                "",
            )
        ),
        encoding="utf-8",
    )
    return row


def _write_outputs(
    namespace: dict[str, object], exports: Path
) -> dict[str, object]:
    exports.mkdir()
    capacitance = namespace["capacitance"]
    inductance = namespace["inductance"]

    capacitance_rows = _point_rows(
        namespace["capacitance_sweep"],
        space="capacitance_grid",
        capacitance=capacitance,
        inductance=inductance,
    )
    _write_point_csv(exports / "02-capacitance-sweep.csv", capacitance_rows)
    _write_sweep_markdown(exports / "02-capacitance-sweep.md", capacitance_rows)
    namespace["capacitance_sweep_figure"].savefig(
        exports / "02-capacitance-sweep.svg",
        format="svg",
        metadata={"Date": None},
    )

    optimization = namespace["optimization"]
    winner_root = namespace["winner_root"]
    winner = _write_winner(
        exports / "02-optimized-winner.csv",
        exports / "02-optimized-winner.md",
        optimization=optimization,
        winner_root=winner_root,
        capacitance=capacitance,
        inductance=inductance,
    )
    winner_svg = exports / "02-optimized-winner.svg"
    namespace["winner_diagram"].show().save(winner_svg)
    certificate = _certificate(winner_svg)

    grid_rows = _point_rows(
        namespace["optional_grid_sweep"],
        space="cartesian_grid",
        capacitance=capacitance,
        inductance=inductance,
    )
    listed_rows = _point_rows(
        namespace["listed_pair_sweep"],
        space="listed_pairs",
        capacitance=capacitance,
        inductance=inductance,
    )
    optional_rows = [*grid_rows, *listed_rows]
    _write_point_csv(exports / "02-optional-spaces.csv", optional_rows)
    _write_optional_markdown(exports / "02-optional-spaces.md", optional_rows)
    namespace["optional_grid_figure"].savefig(
        exports / "02-optional-grid.svg",
        format="svg",
        metadata={"Date": None},
    )

    return {
        "certificate": certificate,
        "results": {
            "capacitance_sweep": _identity(namespace["capacitance_sweep"]),
            "optimization": _identity(optimization),
            "winner_root": _identity(winner_root),
            "optional_grid": _identity(namespace["optional_grid_sweep"]),
            "listed_pairs": _identity(namespace["listed_pair_sweep"]),
        },
        "observations": {
            "capacitance_sweep": capacitance_rows,
            "winner": winner,
            "optional_spaces": optional_rows,
        },
    }


def _check(manifest: dict[str, object]) -> None:
    if manifest.get("schema") != "scnsim.engineer_chapter2_artifacts.v1":
        raise RuntimeError("unsupported Chapter 2 artifact manifest")
    check_publication_binding(manifest, _binding())
    execution = manifest.get("execution")
    binding = manifest["binding"]
    if (
        not isinstance(execution, dict)
        or execution.get("cells") != binding["cells"]
        or execution.get("source_tree_sha256")
        != binding["source_tree"]["sha256"]
    ):
        raise RuntimeError("Chapter 2 execution binding is stale")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_NAMES):
        raise RuntimeError("Chapter 2 artifact inventory changed")
    for name in ARTIFACT_NAMES:
        path = FIGURES / name
        if not path.is_file() or artifacts[name] != _hash(path):
            raise RuntimeError(f"Chapter 2 artifact is missing or stale: {name}")
    if manifest.get("certificate") != _certificate(
        FIGURES / "02-optimized-winner.svg"
    ):
        raise RuntimeError("Chapter 2 winner certificate is stale")
    print("engineer Chapter 2 artifacts are current")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, help="new private workspace outside the repository"
    )
    parser.add_argument(
        "--check", action="store_true", help="verify source and artifact hashes without execution"
    )
    args = parser.parse_args()
    cells = _parse_cells()
    if args.check:
        _check(json.loads(MANIFEST.read_text(encoding="utf-8")))
        return 0

    workspace = _workspace(args.workspace)
    before = _binding()
    namespace = _execute(cells, workspace)
    exports = workspace / "exports"
    records = _write_outputs(namespace, exports)
    after = _binding()
    if before != after:
        raise RuntimeError("Chapter 2 or package sources changed during generation")

    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        shutil.copyfile(exports / name, FIGURES / name)
    manifest = {
        "schema": "scnsim.engineer_chapter2_artifacts.v1",
        "binding": before,
        "execution": {
            "cells": before["cells"],
            "generator_sha256": before["generator_sha256"],
            "source_tree_sha256": before["source_tree"]["sha256"],
        },
        "environment": _environment(),
        "artifacts": {name: _hash(FIGURES / name) for name in ARTIFACT_NAMES},
        **records,
    }
    temporary = FIGURES / ".chapter-02-artifacts.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
