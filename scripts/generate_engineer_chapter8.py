"""Generate and verify source-bound Chapter 8 two-kernel evidence."""

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
from typing import Any

from engineer_publication_binding import check_publication_binding

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "examples" / "engineer" / "chapter-08"
FIGURES = ROOT / "examples" / "engineer" / "figures"
MANIFEST = FIGURES / "chapter-08-artifacts.json"
SOURCES = tuple(CHAPTER / name for name in ("_01-persist.qmd", "_02-resolve.qmd"))
WRAPPERS = tuple(CHAPTER / name for name in ("01-persist.qmd", "02-resolve.qmd"))
EXPECTED_CELL_IDS = (
    "ch8-build-persisted-plan", "ch8-prepare-persisted-requests",
    "ch8-execute-persisted-results", "ch8-build-result-report",
    "ch8-rebuild-plan-after-restart", "ch8-rebuild-request-after-restart",
    "ch8-resolve-only-after-restart",
)
FIRST_KERNEL_IDS = EXPECTED_CELL_IDS[:4]
SECOND_KERNEL_IDS = EXPECTED_CELL_IDS[4:]
ARTIFACT_NAMES = (
    "08-direct-s11.svg", "08-persisted-results.json", "08-persisted-results.md",
    "08-report.html", "08-resolve-only.md",
)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _inventory(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("08-*") if path.is_file()}


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
    if tuple(cell_id for cell_id, _ in cells) != EXPECTED_CELL_IDS: raise ValueError("Chapter 8 cell order changed")
    return tuple(cells)


def _source_tree() -> dict[str, object]:
    paused = ROOT / "src" / "scnsim" / "_agent_knowledge"
    rows = {str(path.relative_to(ROOT)): _hash(path) for path in sorted((ROOT / "src" / "scnsim").rglob("*")) if path.is_file() and not path.is_relative_to(paused) and path.suffix in {".py", ".jl", ".json", ".toml"}}
    return {"sha256": _hash_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()), "files": rows}


def _binding() -> dict[str, object]:
    cells = _parse_cells(); sources = (CHAPTER / "chapter.qmd", *SOURCES, *WRAPPERS)
    return {"generator_sha256": _hash(Path(__file__)), "sources": {str(path.relative_to(ROOT)): _hash(path) for path in sources}, "source_tree": _source_tree(), "cell_ids": list(EXPECTED_CELL_IDS), "cells": {cell_id: _hash_bytes(code.encode()) for cell_id, code in cells}}


def _environment() -> dict[str, str]:
    import scnsim
    if Path(scnsim.__file__).resolve() != (ROOT / "src" / "scnsim" / "__init__.py").resolve(): raise RuntimeError("generation imported scnsim from outside this checkout")
    values = {"python": sys.version.split()[0], "scnsim": scnsim.__version__}
    for name in ("jupyter_client", "kaleido", "matplotlib", "numpy", "pint", "plotly", "schemdraw"): values[name] = importlib.metadata.version(name)
    return values


def _workspace(requested: Path | None) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="scnsim-engineer-chapter8-")) if requested is None else requested.resolve()
    if ROOT == workspace or ROOT in workspace.parents: raise ValueError("execution workspace must be outside the repository")
    if workspace.exists() and any(workspace.iterdir()): raise RuntimeError("execution workspace must be new or empty")
    workspace.mkdir(parents=True, exist_ok=True); return workspace


def _tree_hashes(directory: Path) -> dict[str, str]:
    if not directory.exists(): return {}
    return {str(path.relative_to(directory)): _hash(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def _execute_message(client: Any, code: str, label: str) -> str:
    message_id = client.execute(code, store_history=True, stop_on_error=True)
    streams = []
    while True:
        message = client.get_iopub_msg(timeout=900)
        if message.get("parent_header", {}).get("msg_id") != message_id: continue
        message_type = message["header"]["msg_type"]
        content = message["content"]
        if message_type == "stream": streams.append(content.get("text", ""))
        elif message_type == "error": raise RuntimeError(f"{label}: {content.get('ename')}: {content.get('evalue')}")
        elif message_type == "status" and content.get("execution_state") == "idle": break
    return "".join(streams)


def _kernel_identity(manager: Any) -> dict[str, object]:
    provisioner = getattr(manager, "provisioner", None); process = getattr(provisioner, "process", None)
    return {"kernel_id": manager.kernel_id, "pid": getattr(process, "pid", None), "connection_file": Path(manager.connection_file).name}


def _payload(stream: str, marker: str) -> dict[str, object]:
    lines = [line for line in stream.splitlines() if line.startswith(marker)]
    if len(lines) != 1: raise RuntimeError(f"missing unique {marker} payload")
    value = json.loads(lines[0][len(marker):])
    if not isinstance(value, dict): raise RuntimeError(f"{marker} payload is malformed")
    return value


def _run_kernel(cells: tuple[tuple[str, str], ...], cwd: Path, final_code: str, marker: str) -> tuple[dict[str, object], dict[str, object]]:
    from jupyter_client import KernelManager

    manager = KernelManager(kernel_name="python3"); manager.start_kernel(cwd=str(cwd)); client = manager.client(); client.start_channels()
    try:
        client.wait_for_ready(timeout=120); identity = _kernel_identity(manager)
        for cell_id, code in cells: _execute_message(client, code, cell_id)
        payload = _payload(_execute_message(client, final_code, marker), marker)
        return identity, payload
    finally:
        client.stop_channels(); manager.shutdown_kernel(now=True)


def _identity_code(name: str) -> str:
    return "{field: getattr(" + name + ".identity, field) for field in ('plan_sha256','request_sha256','attempt_sha256','result_sha256')}"


def generate(workspace: Path) -> None:
    cells = dict(_parse_cells()); execution = workspace / "execution"; execution.mkdir(); output = workspace / "artifacts"; output.mkdir()
    report_path = output / "08-report.html"; figure_path = output / "08-direct-s11.svg"
    first_code = f'''\nimport json as _receipt_json, os as _receipt_os\nfrom pathlib import Path as _ReceiptPath\nreport.save(_ReceiptPath({str(report_path)!r}))\ndirect.s.plot(magnitude="db", theme=Theme.AUTO).write_image(_ReceiptPath({str(figure_path)!r}), format="svg")\n_receipt_payload = {{\n "pid": _receipt_os.getpid(),\n "direct_identity": {_identity_code("direct")},\n "root_identity": {_identity_code("root")},\n "root_frequency_GHz": float(root.frequency.to("gigahertz").magnitude),\n "root_linewidth_MHz": float(root.linewidth.to("megahertz").magnitude),\n "matrix_coordinates": list(direct.s.view.coordinates),\n}}\nprint("__SCNSIM_FIRST__" + _receipt_json.dumps(_receipt_payload, sort_keys=True))\n'''
    first_cells = tuple((cell_id, cells[cell_id]) for cell_id in FIRST_KERNEL_IDS)
    first_kernel, first = _run_kernel(first_cells, execution, first_code, "__SCNSIM_FIRST__")
    shared = execution / "workspaces" / "engineer-chapter-08"; before = _tree_hashes(shared)
    second_code = f'''\nimport json as _receipt_json, os as _receipt_os\n_receipt_payload = {{\n "pid": _receipt_os.getpid(),\n "resolved_identity": {_identity_code("resolved_root")},\n "root_frequency_GHz": float(resolved_root.frequency.to("gigahertz").magnitude),\n "root_linewidth_MHz": float(resolved_root.linewidth.to("megahertz").magnitude),\n}}\nprint("__SCNSIM_SECOND__" + _receipt_json.dumps(_receipt_payload, sort_keys=True))\n'''
    second_cells = tuple((cell_id, cells[cell_id]) for cell_id in SECOND_KERNEL_IDS)
    second_kernel, second = _run_kernel(second_cells, execution, second_code, "__SCNSIM_SECOND__"); after = _tree_hashes(shared)
    if first_kernel["kernel_id"] == second_kernel["kernel_id"]: raise RuntimeError("Chapter 8 did not use two independent Jupyter kernels")
    if first["root_identity"] != second["resolved_identity"]: raise RuntimeError("resolved root identity differs from persisted root")
    if before != after: raise RuntimeError("resolve-only kernel changed the shared workspace")
    public_results = {"direct_identity": first["direct_identity"], "root_identity": first["root_identity"], "root_frequency_GHz": first["root_frequency_GHz"], "root_linewidth_MHz": first["root_linewidth_MHz"], "matrix_coordinates": first["matrix_coordinates"]}
    (output / "08-persisted-results.json").write_text(json.dumps(public_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "08-persisted-results.md").write_text(f"### Persisted Result summary\n\n| Result | value |\n|---|---:|\n| loaded root frequency | {first['root_frequency_GHz']:.12g} GHz |\n| loaded root linewidth | {first['root_linewidth_MHz']:.12g} MHz |\n\n![The declared Direct S11 presentation.](../figures/08-direct-s11.svg)\n", encoding="utf-8")
    (output / "08-resolve-only.md").write_text(f"### Resolve-only restart outcome\n\nA second independent Jupyter kernel resolved the exact stored root: **{second['root_frequency_GHz']:.12g} GHz**, linewidth **{second['root_linewidth_MHz']:.12g} MHz**. Its Result identity matches Lesson 1, and the shared workspace bytes were unchanged.\n", encoding="utf-8")
    artifacts = {name: _hash(output / name) for name in ARTIFACT_NAMES}; binding = _binding()
    if _inventory(output) != set(ARTIFACT_NAMES): raise RuntimeError("Chapter 8 generated artifact inventory is not exact")
    manifest = {"schema": "scnsim.engineer_chapter8_artifacts.v1", "binding": binding, "execution": {"generator_sha256": binding["generator_sha256"], "source_tree_sha256": binding["source_tree"]["sha256"], "cells": binding["cells"], "kernels": {"first": first_kernel, "second": second_kernel}, "first_kernel_cell_ids": list(FIRST_KERNEL_IDS), "second_kernel_cell_ids": list(SECOND_KERNEL_IDS), "second_kernel_operations": ["resolve"], "shared_workspace_unchanged_during_resolve": True}, "environment": _environment(), "results": public_results | {"resolved_identity": second["resolved_identity"]}, "artifacts": artifacts}
    FIGURES.mkdir(parents=True, exist_ok=True)
    unknown = _inventory(FIGURES) - set(ARTIFACT_NAMES)
    if unknown: raise RuntimeError(f"unknown Chapter 8 artifacts require review: {sorted(unknown)!r}")
    for name in ARTIFACT_NAMES: shutil.copy2(output / name, FIGURES / name)
    if _inventory(FIGURES) != set(ARTIFACT_NAMES): raise RuntimeError("Chapter 8 published artifact inventory is not exact")
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"; MANIFEST.write_text(text, encoding="utf-8"); receipt = workspace / "generation-receipt.json"; receipt.write_text(text, encoding="utf-8"); print(receipt)


def check() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    check_publication_binding(manifest, _binding())
    if set(manifest.get("artifacts", {})) != set(ARTIFACT_NAMES): raise RuntimeError("Chapter 8 artifact inventory is malformed")
    if _inventory(FIGURES) != set(ARTIFACT_NAMES): raise RuntimeError("Chapter 8 published artifact inventory is not exact")
    if manifest["artifacts"] != {name: _hash(FIGURES / name) for name in ARTIFACT_NAMES}: raise RuntimeError("Chapter 8 artifacts are stale")
    execution = manifest.get("execution", {}); kernels = execution.get("kernels", {})
    if kernels.get("first", {}).get("kernel_id") == kernels.get("second", {}).get("kernel_id"): raise RuntimeError("Chapter 8 kernel identities are not distinct")
    if execution.get("second_kernel_cell_ids") != list(SECOND_KERNEL_IDS) or execution.get("second_kernel_operations") != ["resolve"] or execution.get("shared_workspace_unchanged_during_resolve") is not True: raise RuntimeError("Chapter 8 restart evidence is malformed")
    results = manifest.get("results", {})
    if results.get("root_identity") != results.get("resolved_identity"): raise RuntimeError("Chapter 8 persisted and resolved root identities differ")
    print("engineer Chapter 8 artifacts are current")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--check", action="store_true"); parser.add_argument("--workspace", type=Path); args = parser.parse_args()
    if args.check:
        if args.workspace is not None: parser.error("--workspace cannot be used with --check")
        check()
    else: generate(_workspace(args.workspace))


if __name__ == "__main__": main()
