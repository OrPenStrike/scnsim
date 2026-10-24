#!/usr/bin/env python3
"""Render a canonical QMD without execution and export portable kernel metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def portable_notebook_bytes(raw: bytes) -> bytes:
    """Omit Quarto's machine-local kernel location; retain all other notebook data."""
    notebook = json.loads(raw)
    kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
    if "path" not in kernelspec:
        return raw
    del kernelspec["path"]
    return (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def render_notebook(source: Path, quarto: str) -> bytes:
    # A distinct basename protects a distributed same-basename notebook from cleanup.
    source_raw = source.read_bytes()
    with tempfile.NamedTemporaryFile(
        dir=source.parent, prefix=f"{source.stem}.reader-ipynb-", suffix=".qmd", delete=False
    ) as staged:
        staged.write(source_raw)
        staged_path = Path(staged.name)
    primary = None
    try:
        rendered = subprocess.run(
            [quarto, "render", str(staged_path), "--to", "ipynb",
             "--no-execute", "--no-clean", "--output", "-"],
            check=True, capture_output=True,
        ).stdout
        return portable_notebook_bytes(rendered)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            staged_path.unlink()
        except OSError as error:
            if primary is None:
                raise
            primary.add_note(f"temporary QMD cleanup also failed: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_qmd", type=Path)
    parser.add_argument("generated_ipynb", type=Path)
    parser.add_argument("--quarto", default="quarto")
    args = parser.parse_args()
    rendered = render_notebook(args.source_qmd.resolve(), args.quarto)
    destination = args.generated_ipynb.resolve()
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as staged:
        staged.write(rendered)
        staged_path = Path(staged.name)
    primary = None
    try:
        staged_path.replace(destination)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            staged_path.unlink(missing_ok=True)
        except OSError as error:
            if primary is None:
                raise
            primary.add_note(f"temporary notebook cleanup also failed: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
