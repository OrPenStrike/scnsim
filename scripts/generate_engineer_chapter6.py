"""Generate and verify source-bound Chapter 6 engineer artifacts."""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "examples" / "engineer" / "chapter-06"
FIGURES = ROOT / "examples" / "engineer" / "figures"
MANIFEST = FIGURES / "chapter-06-artifacts.json"
SOURCES = tuple(CHAPTER / name for name in ("_01-default.qmd", "_02-port-poses.qmd", "_03-junctions.qmd", "_04-reuse.qmd"))
WRAPPERS = tuple(CHAPTER / name for name in ("01-default.qmd", "02-port-poses.qmd", "03-junctions.qmd", "04-reuse.qmd"))
EXPECTED_CELL_IDS = (
    "ch6-build-four-arm-plan", "ch6-render-default-composition", "ch6-show-default-audit",
    "ch6-set-axes-and-port-poses", "ch6-build-tapped-feedline", "ch6-request-and-render-t",
    "ch6-request-cross", "ch6-show-cross-audit", "ch6-reuse-completed-composition", "ch6-show-fixed-audit",
)
BASE_ARTIFACT_NAMES = ("06-four-arm-default.svg", "06-four-arm-cross.svg", "06-four-arm-fixed.svg", "06-tapped-feedline-t.md")
T_ARTIFACT_NAMES = ("06-tapped-feedline-t.svg",)
OWNED_ARTIFACT_NAMES = (*BASE_ARTIFACT_NAMES, *T_ARTIFACT_NAMES)
_DC = "{http://purl.org/dc/elements/1.1/}description"


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _inventory(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("06-*") if path.is_file()}


def _parse_cells() -> tuple[tuple[str, str], ...]:
    cells = []
    for source in SOURCES:
        lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
        index = 0
        while index < len(lines):
            if lines[index].strip() != "```{python}": index += 1; continue
            index += 1; body = []
            while index < len(lines) and lines[index].strip() != "```": body.append(lines[index]); index += 1
            if index == len(lines): raise ValueError(f"unterminated Python cell in {source.relative_to(ROOT)}")
            cell_id = next((line.split(":", 1)[1].strip() for line in body if line.startswith("#| id:")), None)
            if cell_id is None: raise ValueError(f"missing Python cell id in {source.relative_to(ROOT)}")
            cells.append((cell_id, "".join(line for line in body if not line.startswith("#|")))); index += 1
    if tuple(cell_id for cell_id, _ in cells) != EXPECTED_CELL_IDS: raise ValueError("Chapter 6 cell order changed")
    return tuple(cells)


def _source_tree() -> dict[str, object]:
    paused = ROOT / "src" / "scnsim" / "_agent_knowledge"
    rows = {str(path.relative_to(ROOT)): _hash(path) for path in sorted((ROOT / "src" / "scnsim").rglob("*")) if path.is_file() and not path.is_relative_to(paused) and path.suffix in {".py", ".jl", ".json", ".toml"}}
    return {"sha256": _hash_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()), "files": rows}


def _binding() -> dict[str, object]:
    cells = _parse_cells(); sources = (CHAPTER / "chapter.qmd", *SOURCES, *WRAPPERS)
    return {"generator_sha256": _hash(Path(__file__)), "sources": {str(path.relative_to(ROOT)): _hash(path) for path in sources}, "source_tree": _source_tree(), "cell_ids": list(EXPECTED_CELL_IDS), "cells": {cell_id: _hash_bytes(code.encode()) for cell_id, code in cells}}


def _execute(cells: tuple[tuple[str, str], ...], workspace: Path) -> dict[str, object]:
    execution = workspace / "execution"; execution.mkdir(); path = execution / "chapter-06.py"
    code = "\n".join(f"# CELL {cell_id}\n{body}" for cell_id, body in cells); path.write_text(code, encoding="utf-8")
    name = "_scnsim_engineer_chapter6_generation"; module = types.ModuleType(name); module.__file__ = str(path); sys.modules[name] = module
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
    for name in ("matplotlib", "numpy", "pint", "schemdraw"): values[name] = importlib.metadata.version(name)
    return values


def _workspace(requested: Path | None) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="scnsim-engineer-chapter6-")) if requested is None else requested.resolve()
    if ROOT == workspace or ROOT in workspace.parents: raise ValueError("execution workspace must be outside the repository")
    if workspace.exists() and any(workspace.iterdir()): raise RuntimeError("execution workspace must be new or empty")
    workspace.mkdir(parents=True, exist_ok=True); return workspace


def generate(workspace: Path) -> None:
    namespace = _execute(_parse_cells(), workspace); output = workspace / "artifacts"; output.mkdir()
    drawings = {"06-four-arm-default.svg": namespace["four_arm_default"], "06-four-arm-cross.svg": namespace["four_arm_cross"], "06-four-arm-fixed.svg": namespace["four_arm_fixed"]}
    certificates = {}
    for name, result in drawings.items(): result.drawing.save(output / name); certificates[name] = _certificate(output / name)
    t_result = namespace["tapped_t_diagram"]
    if t_result is None:
        t_status = "failure"
        text = "### Explicit T layout outcome\n\nThe existing fixed layout reported a typed `schematic_layout` failure. No alternative geometry or renderer fallback was used.\n"
        conditional = ()
    else:
        t_status = "success"; t_result.drawing.save(output / T_ARTIFACT_NAMES[0]); certificates[T_ARTIFACT_NAMES[0]] = _certificate(output / T_ARTIFACT_NAMES[0])
        text = "### Explicit T layout outcome\n\n![Three owner-local contacts joined by one explicitly requested T.](../figures/06-tapped-feedline-t.svg)\n\n[Open the SVG](../figures/06-tapped-feedline-t.svg)\n"
        conditional = T_ARTIFACT_NAMES
    (output / "06-tapped-feedline-t.md").write_text(text, encoding="utf-8")
    names = (*BASE_ARTIFACT_NAMES, *conditional); artifacts = {name: _hash(output / name) for name in names}; binding = _binding()
    manifest = {"schema": "scnsim.engineer_chapter6_artifacts.v1", "binding": binding, "execution": {"generator_sha256": binding["generator_sha256"], "source_tree_sha256": binding["source_tree"]["sha256"], "cells": binding["cells"]}, "environment": _environment(), "observations": {"explicit_t": {"status": t_status}}, "certificates": certificates, "artifacts": artifacts}
    if _inventory(output) != set(names): raise RuntimeError("Chapter 6 generated artifact inventory is not exact")
    FIGURES.mkdir(parents=True, exist_ok=True)
    unknown = _inventory(FIGURES) - set(OWNED_ARTIFACT_NAMES)
    if unknown: raise RuntimeError(f"unknown Chapter 6 artifacts require review: {sorted(unknown)!r}")
    for stale in T_ARTIFACT_NAMES:
        if stale not in names and (FIGURES / stale).exists(): (FIGURES / stale).unlink()
    for name in names: shutil.copy2(output / name, FIGURES / name)
    if _inventory(FIGURES) != set(names): raise RuntimeError("Chapter 6 published artifact inventory is not exact")
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"; MANIFEST.write_text(text, encoding="utf-8"); receipt = workspace / "generation-receipt.json"; receipt.write_text(text, encoding="utf-8"); print(receipt)


def check() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("binding") != _binding(): raise RuntimeError("Chapter 6 source binding is stale")
    status = manifest.get("observations", {}).get("explicit_t", {}).get("status")
    if status not in {"failure", "success"}: raise RuntimeError("Chapter 6 explicit T status is malformed")
    names = (*BASE_ARTIFACT_NAMES, *(T_ARTIFACT_NAMES if status == "success" else ()))
    if set(manifest.get("artifacts", {})) != set(names): raise RuntimeError("Chapter 6 artifact inventory is malformed")
    if _inventory(FIGURES) != set(names): raise RuntimeError("Chapter 6 published artifact inventory is not exact")
    if manifest["artifacts"] != {name: _hash(FIGURES / name) for name in names}: raise RuntimeError("Chapter 6 artifacts are stale")
    print("engineer Chapter 6 artifacts are current")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--check", action="store_true"); parser.add_argument("--workspace", type=Path); args = parser.parse_args()
    if args.check:
        if args.workspace is not None: parser.error("--workspace cannot be used with --check")
        check()
    else: generate(_workspace(args.workspace))


if __name__ == "__main__": main()
