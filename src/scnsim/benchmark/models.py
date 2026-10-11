"""Read-only value model for current Workspace operation diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Read-only current operation report; numerical results live in Results."""

    workspace: Path
    manifest_bytes: bytes

    def document(self) -> dict[str, object]:
        return json.loads(self.manifest_bytes)

    def to_json(self) -> str:
        return self.manifest_bytes.decode("utf-8")

    def to_html(self) -> str:
        from ..visualization.benchmark import render_benchmark

        return render_benchmark(self)

    def show(self):
        """Display the stored report lazily, without initializing a runtime."""
        from ..results.base import HtmlPresentation

        return HtmlPresentation(self.to_html())

    def write_html(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(self.to_html(), encoding="utf-8")
        return target

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_bytes(self.manifest_bytes)
        return target

    @classmethod
    def open(cls, workspace: str | Path) -> BenchmarkResult:
        from ..workspace.evidence import open_record

        root = Path(workspace).expanduser().resolve(strict=False)
        document = open_record(root)
        if not isinstance(document, Mapping):
            raise TypeError("Workspace operation evidence must be a document mapping")
        return cls.from_document(root, dict(document))

    @classmethod
    def from_document(
        cls, workspace: Path, document: dict[str, object]
    ) -> BenchmarkResult:
        from ..numeric_encoding import record_bytes

        if (
            document.get("schema") != "scnsim.operation_benchmark"
            or document.get("schema_version") != 2
        ):
            raise ValueError("unsupported current Workspace operation evidence")
        return cls(workspace, record_bytes(document))
