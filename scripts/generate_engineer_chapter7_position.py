"""Generate only the standalone Chapter 7 position lesson's source-bound outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import plotly.graph_objects as go
from scnsim import ParameterSweepResult


ROOT = Path(__file__).resolve().parents[1]
LESSON = ROOT / "examples/engineer/chapter-07/05-position-electrical-resolution.qmd"
OUTPUT = ROOT / "examples/engineer/chapter-07/position-figures"
MANIFEST = OUTPUT / "07-position-artifacts.json"
CELL_IDS = (
    "ch7-position-resolution", "ch7-position-build", "ch7-position-explain",
    "ch7-position-solve", "ch7-position-outcomes", "ch7-position-refinement",
)
POSITIONS = (2, 4, 6)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cells() -> tuple[tuple[str, str], ...]:
    lines = LESSON.read_text(encoding="utf-8").splitlines(keepends=True)
    found = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != "```{python}":
            index += 1
            continue
        index += 1
        body = []
        while index < len(lines) and lines[index].strip() != "```":
            body.append(lines[index])
            index += 1
        if index == len(lines):
            raise ValueError("unterminated lesson Python cell")
        ids = [line.split(":", 1)[1].strip() for line in body if line.startswith("#| id:")]
        if len(ids) != 1:
            raise ValueError("lesson Python cell needs exactly one stable ID")
        found.append((ids[0], "".join(line for line in body if not line.startswith("#|"))))
        index += 1
    if tuple(identifier for identifier, _ in found) != CELL_IDS:
        raise ValueError("standalone position lesson cell order changed")
    return tuple(found)


def binding() -> dict[str, object]:
    source_files = sorted(path for path in (ROOT / "src/scnsim").rglob("*") if path.is_file() and path.suffix in {".py", ".jl", ".json", ".toml"})
    tree = {str(path.relative_to(ROOT)): digest(path) for path in source_files if "_agent_knowledge" not in path.parts}
    source_tree_sha = hashlib.sha256(json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "lesson_sha256": digest(LESSON),
        "generator_sha256": digest(Path(__file__)),
        "source_tree_sha256": source_tree_sha,
        "cell_ids": list(CELL_IDS),
    }


def execute(workspace: Path) -> dict[str, object]:
    execution = workspace / "execution"
    execution.mkdir(parents=True, exist_ok=True)
    code = "\n".join(f"# CELL {identifier}\n{body}" for identifier, body in cells())
    compile(code, str(LESSON), "exec")
    module = types.ModuleType("_scnsim_position_generation")
    module.__file__ = str(LESSON)
    previous = Path.cwd()
    try:
        os.chdir(execution)
        exec(compile(code, str(LESSON), "exec"), module.__dict__)
        return module.__dict__
    finally:
        os.chdir(previous)


def generate(workspace: Path) -> None:
    workspace = workspace.resolve()
    if workspace == ROOT or ROOT in workspace.parents:
        raise ValueError("execution workspace must be outside the repository")
    workspace.mkdir(parents=True, exist_ok=True)
    namespace = execute(workspace)
    coarse_result = namespace.get("position_points")
    fine_result = namespace.get("refined_points")
    if not isinstance(coarse_result, ParameterSweepResult) or not isinstance(fine_result, ParameterSweepResult):
        raise RuntimeError("position lesson did not produce both verified sweeps")
    coarse = tuple(coarse_result.points)
    fine = tuple(fine_result.points)
    if len(coarse) != 3 or len(fine) != 3 or not all(point.succeeded for point in (*coarse, *fine)):
        raise RuntimeError("all six declared position outcomes must succeed before publication")
    staged = workspace / "published"
    staged.mkdir(exist_ok=True)
    baseline = ["### Baseline realized grid (M40, x = 4 mm)", "",
                "Read from the verified saved x = 4 mm M40 point.", "",
                "| body | length (mm) | modal velocities (m/s) | hmax (mm) | N | dx (mm) |",
                "|---|---:|---|---:|---:|---:|"]
    for line in coarse[1].result.discretization:
        modal = ", ".join(f"{velocity.to('m/s').magnitude:.6g}" for velocity in line.modal_velocities)
        baseline.append(
            f"| {'/'.join(line.component_path)} | {line.length.to('mm').magnitude:.6g} | "
            f"{modal} | {line.hmax.to('mm').magnitude:.6g} | {line.n_sections} | "
            f"{line.dx.to('mm').magnitude:.6g} |"
        )
    (staged / "07-position-baseline-grid.md").write_text("\n".join(baseline) + "\n", encoding="utf-8")
    point_grids = ["### Three-position realized grids (M40)", "",
                   "Read from the verified saved M40 point results.", "",
                   "| x (mm) | status | left length / N / dx (mm) | right length / N / dx (mm) |",
                   "|---:|---|---|---|"]
    for x, point in zip(POSITIONS, coarse, strict=True):
        left, right = point.result.discretization
        point_grids.append(
            f"| {x} | success | {left.length.to('mm').magnitude:.6g} / {left.n_sections} / "
            f"{left.dx.to('mm').magnitude:.6g} | {right.length.to('mm').magnitude:.6g} / "
            f"{right.n_sections} / {right.dx.to('mm').magnitude:.6g} |"
        )
    (staged / "07-position-point-grids.md").write_text("\n".join(point_grids) + "\n", encoding="utf-8")
    results = ["### Named through-channel response", "",
               "Each figure is the exact saved `right_a ← left_a` channel at one position; phase is wrapped.", ""]
    refinement = ["### Same-point complex refinement", "",
                  "The plotted absolute complex difference is diagnostic sensitivity, not an error bound.", ""]
    difference_figure = go.Figure()
    for x, low, high in zip(POSITIONS, coarse, fine, strict=True):
        trace = low.result.traces["through_a"]
        for component in ("magnitude", "phase"):
            name = f"07-position-{x}mm-{component}.svg"
            args = {"component": component}
            if component == "magnitude":
                args["magnitude"] = "db"
            trace.plot(**args).write_image(staged / name, format="svg")
            results.extend((f"![x={x} mm, {component}.](position-figures/{name})", ""))
        low_grid = low.result.discretization
        high_grid = high.result.discretization
        frequencies = np.asarray(low.result.frequencies.to("GHz").magnitude)
        delta = np.asarray(high.result.traces["through_a"].value.magnitude) - np.asarray(trace.value.magnitude)
        difference_figure.add_trace(go.Scatter(x=frequencies, y=np.abs(delta), mode="lines", name=f"x={x} mm"))
        refinement.append(
            f"- x={x} mm: M40 N=({low_grid[0].n_sections}, {low_grid[1].n_sections}); "
            f"M80 N=({high_grid[0].n_sections}, {high_grid[1].n_sections}); "
            f"maximum sampled |ΔS|={np.max(np.abs(delta)):.6g}."
        )
    difference_figure.update_layout(title="Same-frequency complex response difference", xaxis_title="Frequency (GHz)", yaxis_title="|S80 − S40|", template="plotly_white")
    difference_figure.write_image(staged / "07-position-complex-difference.svg", format="svg")
    refinement.extend(("", "![Three-position complex refinement difference.](position-figures/07-position-complex-difference.svg)", ""))
    (staged / "07-position-results.md").write_text("\n".join(results), encoding="utf-8")
    (staged / "07-position-refinement.md").write_text("\n".join(refinement), encoding="utf-8")
    shutil.copy2(workspace / "execution/position-output/07-position-complex.csv", staged / "07-position-complex.csv")
    artifacts = {path.name: digest(path) for path in sorted(staged.iterdir()) if path.is_file()}
    metadata = {
        "schema": "scnsim.engineer_chapter7_position_artifacts.v1",
        "binding": binding(),
        "publication_binding": binding(),
        "environment": {
            "python": sys.version.split()[0], "scnsim": importlib.metadata.version("scnsim"),
            "numpy": importlib.metadata.version("numpy"), "plotly": importlib.metadata.version("plotly"),
            "kaleido": importlib.metadata.version("kaleido"),
        },
        "results": {
            label: {key: getattr(result.identity, key) for key in ("plan_sha256", "request_sha256", "attempt_sha256", "result_sha256")}
            for label, result in (("m40", coarse_result), ("m80", fine_result))
        },
        "artifacts": artifacts,
    }
    OUTPUT.mkdir(exist_ok=True)
    existing = {path.name for path in OUTPUT.iterdir() if path.is_file()} - {MANIFEST.name}
    previous = set(artifacts) - {"07-position-baseline-grid.md", "07-position-point-grids.md"}
    if existing and existing not in (previous, set(artifacts)):
        raise RuntimeError("position lesson artifact inventory changed; inspect before replacement")
    for path in staged.iterdir():
        shutil.copy2(path, OUTPUT / path.name)
    MANIFEST.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(MANIFEST)


def check() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("publication_binding") != binding():
        raise RuntimeError("position lesson source binding changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("position lesson artifact inventory is malformed")
    found = {path.name for path in OUTPUT.iterdir() if path.is_file()} - {MANIFEST.name}
    if found != set(artifacts) or artifacts != {name: digest(OUTPUT / name) for name in found}:
        raise RuntimeError("position lesson artifacts are stale")
    print("engineer position lesson artifacts are current")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    if args.check:
        if args.workspace is not None:
            parser.error("--check cannot use --workspace")
        check()
    elif args.workspace is None:
        parser.error("generation requires an explicit external --workspace")
    else:
        generate(args.workspace)


if __name__ == "__main__":
    main()
