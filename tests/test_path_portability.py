"""Native paths, editable-source URIs and public Workspace handle lifetime.

One portable set runs against the installed package in each runtime CI job.
Distribution fixtures are real on-disk metadata, not editable installations.
The Windows drive/UNC cases are lexical conversions only, never share access.
No request evaluates a physical model or starts a numerical backend.
"""
from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
import platform
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from urllib.parse import urlsplit, urlunsplit

import scnsim
from scnsim import CircuitRun, SCNSimValidationError
from scnsim.authoring.provenance import _editable_distribution_root, _file_uri_path


_NAMES = ("ordinary", "with spaces", "元件路徑", "percent%name", "literal%20")
_FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "tutorials" / "fixtures" / "primitive_resonator.py"


def _plan():
    # Reuse the published construction-only LC fixture, without solving it.
    return runpy.run_path(str(_FIXTURE))["build_primitive_resonator"]().plan


def _distribution(directory: Path, source: Path, *, authority: str = ""):
    info = directory / "path_catalog-0.0.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: path-catalog\nVersion: 0.0.0\n", encoding="utf-8"
    )
    (info / "top_level.txt").write_text("path_catalog\n", encoding="utf-8")
    parsed = urlsplit(source.as_uri())
    url = urlunsplit((parsed.scheme, authority, parsed.path, "", ""))
    (info / "direct_url.json").write_text(
        json.dumps({"url": url, "dir_info": {"editable": True}}), encoding="utf-8"
    )
    return metadata.Distribution.at(info)


class NativePathTests(unittest.TestCase):
    def test_real_native_paths_and_file_uri_decode_once(self):
        with TemporaryDirectory(prefix="scnsim paths ") as temporary:
            root = Path(temporary)
            for name in _NAMES:
                with self.subTest(name=name):
                    directory = root / name
                    directory.mkdir()
                    file = directory / "內容 %20.txt"
                    file.write_text(name, encoding="utf-8")
                    self.assertEqual(file.read_text(encoding="utf-8"), name)
                    for authority in ("", "localhost"):
                        with self.subTest(authority=authority):
                            parsed = urlsplit(file.as_uri())
                            decoded = _file_uri_path(parsed.path, authority=authority)
                            self.assertEqual(decoded.resolve(strict=True), file.resolve(strict=True))
                            self.assertEqual(decoded.read_text(encoding="utf-8"), name)
                            self.assertEqual(decoded.name, "內容 %20.txt")

    @unittest.skipUnless(os.name == "nt", "Windows-native drive and UNC conversion only")
    def test_windows_drive_and_unc_conversion_without_access(self):
        self.assertEqual(
            _file_uri_path("/C:/with%20spaces/literal%2520", authority=""),
            Path("C:/with spaces/literal%20"),
        )
        self.assertEqual(
            _file_uri_path("/share/with%20spaces/literal%2520", authority="server"),
            Path("//server/share/with spaces/literal%20"),
        )


class SourceProvenancePathTests(unittest.TestCase):
    def test_real_distribution_metadata_resolves_native_editable_root(self):
        with TemporaryDirectory(prefix="scnsim metadata ") as temporary:
            root = Path(temporary)
            for ordinal, name in enumerate(_NAMES):
                source = root / name
                source.mkdir()
                for authority in ("", "localhost"):
                    with self.subTest(name=name, authority=authority):
                        package = _distribution(root / f"metadata-{ordinal}-{authority or 'local'}", source,
                                                authority=authority)
                        self.assertEqual(package.metadata["Name"], "path-catalog")
                        self.assertEqual(_editable_distribution_root(package, "path_catalog.component"),
                                         source.resolve(strict=True))

    def test_unreadable_missing_root_preserves_native_strict_resolution_cause(self):
        with TemporaryDirectory(prefix="scnsim missing root ") as temporary:
            root = Path(temporary).resolve(strict=True)
            missing = root / "missing 元件 literal%20"
            package = _distribution(root / "metadata", missing)
            with self.assertRaises(SCNSimValidationError) as captured:
                _editable_distribution_root(package, "path_catalog.component")
            cause = captured.exception.__cause__
            self.assertIsInstance(cause, FileNotFoundError)
            self.assertEqual(Path(cause.filename), missing)
            self.assertIn("editable catalog source root is unreadable", str(captured.exception))


class WorkspacePathLifetimeTests(unittest.TestCase):
    def test_public_workspace_create_reopen_close_and_release_path_and_string(self):
        with TemporaryDirectory(prefix="scnsim workspace paths ") as temporary:
            root = Path(temporary)
            for as_string in (False, True):
                with self.subTest(workspace_type="str" if as_string else "Path"):
                    path = root / ("Workspace 元件 literal%20 str" if as_string else "Workspace 元件 literal%20 Path")
                    plan = _plan()
                    for _ in range(2):
                        run = CircuitRun(plan=plan, workspace=str(path) if as_string else path,
                                          precision="float64")
                        try:
                            self.assertIsNotNone(run.original)
                            self.assertTrue(path.is_dir())
                        finally:
                            run.close()
                    # On Windows an open SQLite/file handle prevents directory
                    # relocation/removal. This observes release after both closes.
                    moved = path.with_name(path.name + " moved")
                    path.rename(moved)
                    self.assertTrue(moved.is_dir())
                    moved.rename(path)

    def test_public_memory_run_without_workspace(self):
        run = CircuitRun(plan=_plan(), storage="memory", precision="float64")
        try:
            self.assertIsNotNone(run.original)
        finally:
            run.close()


if __name__ == "__main__":
    print(json.dumps({"platform": platform.platform(), "os_name": os.name,
                      "python": sys.executable, "package_source": scnsim.__file__,
                      "installed_metadata_version": metadata.version("scnsim"),
                      "execution_context": os.environ.get("SCNSIM_PATH_TEST_CONTEXT", "unspecified")}), flush=True)
    unittest.main(verbosity=2)
