"""Chapter 4 publication uses the actual explicit complete schematic."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Chapter4GenerationTests(unittest.TestCase):
    def test_export_uses_canonical_capstone_result(self):
        helper_name = "engineer_publication_binding"
        previous = sys.modules.get(helper_name)
        try:
            spec = importlib.util.spec_from_file_location(
                helper_name, ROOT / "scripts/engineer_publication_binding.py")
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            sys.modules[helper_name] = helper
            spec = importlib.util.spec_from_file_location(
                "_chapter4_publication_test", ROOT / "scripts/generate_engineer_chapter4.py")
            chapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(chapter)
            # These exact existing authoring/layout cells create no CircuitRun or solve.
            cells = chapter._parse_cells()[:5]
            self.assertEqual(cells[-1][0], "ch4-render-explicit-capstone")
            namespace = {"__name__": "__main__"}
            for cell_id, code in cells:
                exec(compile(code, str(chapter.CHAPTER) + ":" + cell_id, "exec"), namespace)
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                record = chapter._write_capstone(namespace, output)
                self.assertEqual(record,
                                 chapter._certificate(output / "04-explicit-capstone.svg"))
                self.assertEqual(record["plan_id"], "floating_probe_course")
                self.assertIn("04-explicit-capstone.svg", chapter.OWNED_ARTIFACT_NAMES)
        finally:
            if previous is None:
                sys.modules.pop(helper_name, None)
            else:
                sys.modules[helper_name] = previous
