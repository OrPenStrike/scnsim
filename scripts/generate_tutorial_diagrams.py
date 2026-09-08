"""Export the explicitly selected real Tutorial authoring diagrams.

The QMD cells remain the model authority.  Recipes name whole original cells
in execution order; this tool never rebuilds a Plan or derives dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import linecache
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "examples" / "tutorials"
FIGURES = TUTORIALS / "figures"
RECIPES = Path(__file__).with_name("tutorial_diagram_recipes.json")
MANIFEST = FIGURES / "tutorial-diagrams.json"
_DC = "{http://purl.org/dc/elements/1.1/}description"


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _renderer() -> dict[str, object]:
    paused_producer = ROOT / "src" / "scnsim" / "_agent_knowledge"
    rows = {
        str(path.relative_to(ROOT)): _hash(path)
        for path in sorted((ROOT / "src" / "scnsim").rglob("*"))
        if path.is_file()
        and not path.is_relative_to(paused_producer)
        and path.suffix in {".py", ".jl", ".json", ".toml"}
    }
    return {"sha256": _hash_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()), "files": rows}


def _parse_cells(source: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
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
        cell_id = next((line.split(":", 1)[1].strip() for line in body if line.startswith("#| id:")), None)
        if cell_id is None:
            continue
        if cell_id in result:
            raise ValueError(f"duplicate Tutorial cell ID {cell_id!r}")
        result[cell_id] = {
            "raw_sha256": _hash_bytes("".join(body).encode()),
            "source": "".join(line for line in body if not line.startswith("#|")),
        }
    return result


def _load() -> tuple[dict[str, object], list[dict[str, object]], dict[str, dict[str, dict[str, str]]]]:
    recipe = json.loads(RECIPES.read_text(encoding="utf-8"))
    if recipe.get("schema") != "scnsim.tutorial_diagram_recipes.v1":
        raise ValueError("unsupported tutorial diagram recipe schema")
    checkpoints = recipe.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 21:
        raise ValueError("tutorial recipe must contain exactly 21 checkpoints")
    if sum(item.get("asset") is not None for item in checkpoints if isinstance(item, dict)) != 19:
        raise ValueError("tutorial recipe must contain exactly 19 assets")
    chapters: dict[str, dict[str, dict[str, str]]] = {}
    checkpoint_ids: set[str] = set()
    assets: set[str] = set()
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise TypeError("tutorial diagram checkpoint must be an object")
        source, target, cells = checkpoint.get("source"), checkpoint.get("cell_id"), checkpoint.get("cells")
        if not isinstance(source, str) or not isinstance(target, str) or not isinstance(cells, list):
            raise TypeError("tutorial diagram checkpoint has invalid fields")
        if "layout_error" in checkpoint and not isinstance(checkpoint["layout_error"], str):
            raise TypeError(f"checkpoint {target!r} has an invalid layout_error variable")
        if Path(source).name != source or not source.endswith(".qmd"):
            raise ValueError(f"checkpoint {target!r} must name one Tutorial QMD basename")
        if target in checkpoint_ids:
            raise ValueError(f"duplicate tutorial checkpoint {target!r}")
        checkpoint_ids.add(target)
        asset = checkpoint.get("asset")
        if asset is not None:
            if not isinstance(asset, str) or Path(asset).name != asset or not asset.endswith(".svg"):
                raise ValueError(f"checkpoint {target!r} must name one SVG basename")
            if asset in assets:
                raise ValueError(f"duplicate tutorial SVG asset {asset!r}")
            assets.add(asset)
        if not cells or cells[-1] != target or not all(isinstance(cell, str) for cell in cells):
            raise ValueError(f"checkpoint {target!r} must end its explicit cell list with itself")
        if len(cells) != len(set(cells)):
            raise ValueError(f"checkpoint {target!r} repeats a cell")
        chapters.setdefault(source, _parse_cells(TUTORIALS / source))
        missing = [cell for cell in cells if cell not in chapters[source]]
        if missing:
            raise ValueError(f"checkpoint {target!r} names missing cells: {missing}")
        positions = [list(chapters[source]).index(cell) for cell in cells]
        if positions != sorted(positions):
            raise ValueError(f"checkpoint {target!r} reorders QMD cells")
        if checkpoint.get("asset") is not None and any(
            'representation="compiled"' in chapters[source][cell]["source"] for cell in cells
        ):
            raise ValueError(f"authoring asset checkpoint {target!r} includes a compiled render")
    return recipe, checkpoints, chapters


def _binding(chapters: dict[str, dict[str, dict[str, str]]]) -> dict[str, object]:
    return {
        "generator_sha256": _hash(Path(__file__)),
        "recipe_sha256": _hash(RECIPES),
        "renderer": _renderer(),
        "sources": {
            source: {
                "sha256": _hash(TUTORIALS / source),
                "cells": {cell_id: cell["raw_sha256"] for cell_id, cell in cells.items()},
            }
            for source, cells in sorted(chapters.items())
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
        raise ValueError(f"SVG has no SCNSim certificate: {path.name}")
    value = json.loads(description.text)
    identity = value.get("identity") if isinstance(value, dict) else None
    if value.get("kind") != "scnsim_circuit_diagram_certificate" or not isinstance(identity, dict):
        raise ValueError(f"SVG has an invalid SCNSim certificate: {path.name}")
    required = ("representation", "plan_id", "plan_sha256", "parameters_sha256", "connectivity_sha256", "semantic_sha256", "presentation_sha256")
    if any(not isinstance(identity.get(field), str) for field in required):
        raise ValueError(f"SVG has incomplete certificate identity: {path.name}")
    if identity["representation"] != "authoring":
        raise ValueError(f"SVG is not an authoring diagram: {path.name}")
    return {field: identity[field] for field in required}


def _result_identity(diagram: object, certificate: dict[str, str]) -> dict[str, str]:
    from scnsim import CircuitDiagramResult

    if not isinstance(diagram, CircuitDiagramResult):
        raise TypeError("Tutorial checkpoint did not return CircuitDiagramResult")
    if diagram.composition is None:
        raise RuntimeError("Tutorial diagram has no captured composition")
    audit = diagram.audit
    for field in ("representation", "plan_id", "plan_sha256", "connectivity_sha256", "semantic_sha256", "presentation_sha256"):
        if getattr(audit, field) != certificate[field]:
            raise RuntimeError(f"audit and SVG certificate disagree on {field}")
    return certificate


def _execute(checkpoint: dict[str, object], chapter: dict[str, dict[str, str]], workspace: Path) -> tuple[object | None, object | None]:
    scope = workspace / str(checkpoint["source"]).removesuffix(".qmd") / str(checkpoint["cell_id"])
    scope.mkdir(parents=True, exist_ok=False)
    namespace = {"__name__": "__main__"}
    previous = Path.cwd()
    try:
        os.chdir(scope)
        for ordinal, cell_id in enumerate(checkpoint["cells"], start=1):
            cell = chapter[str(cell_id)]
            cell_path = scope / f"{ordinal:02d}-{cell_id}.py"
            cell_path.write_text(cell["source"], encoding="utf-8")
            linecache.checkcache(str(cell_path))
            namespace["__file__"] = str(cell_path)
            exec(compile(cell["source"], str(cell_path), "exec"), namespace)
    finally:
        os.chdir(previous)
    diagram = namespace.get(str(checkpoint["diagram"]))
    layout_error = namespace.get(str(checkpoint["layout_error"])) if "layout_error" in checkpoint else None
    if layout_error is not None:
        from scnsim.errors import SCNSimValidationError

        if not isinstance(layout_error, SCNSimValidationError) or layout_error.stage != "schematic_layout":
            raise RuntimeError(f"{checkpoint['cell_id']} retained an invalid layout failure")
        if diagram is not None:
            raise RuntimeError(f"{checkpoint['cell_id']} retained a layout failure and a diagram")
        return None, layout_error
    if diagram is None:
        raise RuntimeError(f"{checkpoint['cell_id']} did not produce {checkpoint['diagram']}")
    return diagram, None


def _expected_mapping(checkpoints: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{key: item.get(key) for key in ("source", "cell_id", "asset")} for item in checkpoints]


def _check(manifest: dict[str, object], checkpoints: list[dict[str, object]], chapters: dict[str, dict[str, dict[str, str]]]) -> None:
    binding = _binding(chapters)
    if manifest.get("schema") != "scnsim.tutorial_diagrams.v1" or manifest.get("binding") != binding:
        raise RuntimeError("tutorial diagram manifest source binding is stale")
    if manifest.get("mapping") != _expected_mapping(checkpoints):
        raise RuntimeError("tutorial diagram manifest does not match its recipe mapping")
    records = manifest.get("checkpoints")
    if not isinstance(records, list) or len(records) != len(checkpoints):
        raise RuntimeError("tutorial diagram manifest checkpoint inventory changed")
    for expected, record in zip(_expected_mapping(checkpoints), records, strict=True):
        if not isinstance(record, dict) or {key: record.get(key) for key in ("source", "cell_id")} != {key: expected[key] for key in ("source", "cell_id")}:
            raise RuntimeError("tutorial diagram manifest checkpoint status or mapping changed")
        if record.get("status") == "unavailable":
            if expected["asset"] is None or record.get("asset") is not None or record.get("stage") != "schematic_layout" or not isinstance(record.get("reason"), str) or "certificate" in record:
                raise RuntimeError("tutorial diagram unavailable checkpoint is malformed")
            if (FIGURES / str(expected["asset"])).exists():
                raise RuntimeError(f"unavailable tutorial diagram has a stale asset: {expected['asset']}")
            continue
        if record.get("status") != "ok" or record.get("asset") != expected["asset"]:
            raise RuntimeError("tutorial diagram manifest checkpoint status or mapping changed")
        certificate = record.get("certificate")
        if not isinstance(certificate, dict) or certificate.get("representation") != "authoring":
            raise RuntimeError("tutorial diagram manifest lacks certificate identity")
        asset = expected["asset"]
        if asset is None:
            continue
        path = FIGURES / str(asset)
        if not path.is_file() or record.get("svg_sha256") != _hash(path) or _certificate(path) != certificate:
            raise RuntimeError(f"tutorial diagram asset is missing or stale: {asset}")
    print("tutorial diagram manifest is current")


def _workspace(requested: Path | None) -> Path:
    if requested is None:
        return Path(tempfile.mkdtemp(prefix="scnsim-tutorial-diagrams-"))
    workspace = requested.resolve()
    if ROOT == workspace or ROOT in workspace.parents:
        raise ValueError("execution workspace must be outside the repository")
    if workspace.exists() and any(workspace.iterdir()):
        raise RuntimeError(f"workspace must be new or empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="new private execution workspace outside the repository")
    parser.add_argument("--only", action="append", default=[], help="explicit checkpoint cell ID to generate")
    parser.add_argument("--check", action="store_true", help="verify public provenance and SVGs without imports or execution")
    parser.add_argument("--validate-recipes", action="store_true", help="validate recipe cell IDs only")
    args = parser.parse_args()
    _, checkpoints, chapters = _load()
    if args.validate_recipes:
        print("tutorial diagram recipes valid")
        return 0
    if args.check:
        _check(json.loads(MANIFEST.read_text(encoding="utf-8")), checkpoints, chapters)
        return 0
    selected = [item for item in checkpoints if not args.only or item["cell_id"] in args.only]
    if args.only and len(selected) != len(set(args.only)):
        raise ValueError("--only names an unknown or duplicate checkpoint")
    if not selected:
        raise ValueError("no tutorial diagram checkpoints selected")
    workspace = _workspace(args.workspace)
    before = _binding(chapters)
    environment = _environment()
    records: list[dict[str, object]] = []
    private: list[dict[str, object]] = []
    exports = workspace / "exports"
    exports.mkdir()
    for checkpoint in selected:
        try:
            diagram, layout_error = _execute(checkpoint, chapters[str(checkpoint["source"])], workspace)
            if layout_error is not None:
                records.append({
                    "source": checkpoint["source"], "cell_id": checkpoint["cell_id"],
                    "asset": None, "status": "unavailable", "stage": layout_error.stage,
                    "reason": str(layout_error),
                })
                private.append({"checkpoint": checkpoint["cell_id"], "status": "unavailable", "stage": layout_error.stage, "message": str(layout_error)})
                continue
            assert diagram is not None
            exported = exports / f"{checkpoint['cell_id']}.svg"
            drawing = diagram.show()
            if not hasattr(drawing, "save"):
                raise TypeError("CircuitDiagramResult.show() did not return FrozenDrawing")
            drawing.save(exported)
            certificate = _certificate(exported)
            _result_identity(diagram, certificate)
            records.append({
                **{key: checkpoint.get(key) for key in ("source", "cell_id", "asset")},
                "status": "ok", "certificate": certificate,
                "svg_sha256": _hash(exported),
                "validation": {"result": "CircuitDiagramResult", "composition": "captured", "audit": "certificate_matched", "export": "FrozenDrawing.save"},
            })
            private.append({"checkpoint": checkpoint["cell_id"], "status": "ok", "export": str(exported)})
        except Exception as error:
            private.append({"checkpoint": checkpoint["cell_id"], "status": "error", "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc().splitlines()})
            (workspace / "raw-records.json").write_text(json.dumps(private, indent=2) + "\n")
            raise
    (workspace / "raw-records.json").write_text(json.dumps(private, indent=2) + "\n")
    _, after_checkpoints, after_chapters = _load()
    after = _binding(after_chapters)
    if before != after or _expected_mapping(checkpoints) != _expected_mapping(after_checkpoints):
        raise RuntimeError("Tutorial or renderer sources changed during generation; exports remain private")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for record in records:
        if record["asset"] is not None:
            shutil.copyfile(exports / f"{record['cell_id']}.svg", FIGURES / str(record["asset"]))
    unavailable = {item["cell_id"] for item in records if item["status"] == "unavailable"}
    for checkpoint in checkpoints:
        if checkpoint["cell_id"] in unavailable and (FIGURES / str(checkpoint["asset"])).exists():
            raise RuntimeError(f"unavailable tutorial diagram has a stale asset: {checkpoint['asset']}")
    if args.only:
        print("partial tutorial diagram export completed; public manifest is reserved for the full inventory")
        return 0
    manifest = {
        "schema": "scnsim.tutorial_diagrams.v1", "binding": before,
        "environment": environment, "mapping": _expected_mapping(checkpoints), "checkpoints": records,
    }
    temporary = MANIFEST.with_name(f".{MANIFEST.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST)
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
