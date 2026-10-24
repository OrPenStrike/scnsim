"""Chapter 3 publication uses the actual explicit complete schematic."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Chapter3GenerationTests(unittest.TestCase):
    def test_memory_direct_preparation_uses_public_declarations_without_solve(self):
        helper_name = "engineer_publication_binding"
        previous = sys.modules.get(helper_name)
        namespace = {"__name__": "__main__"}
        try:
            spec = importlib.util.spec_from_file_location(
                helper_name, ROOT / "scripts/engineer_publication_binding.py")
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            sys.modules[helper_name] = helper
            spec = importlib.util.spec_from_file_location(
                "_chapter3_preparation_test", ROOT / "scripts/generate_engineer_chapter3.py")
            chapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(chapter)
            cells = chapter._parse_cells()
            # Execute the real Plan declarations and preparation, stopping before solve.
            for cell_id, code in (*cells[:3], cells[6]):
                exec(compile(code, str(chapter.CHAPTER) + ":" + cell_id, "exec"), namespace)
            self.assertEqual(cells[6][0], "ch3-prepare-direct-s21")
            request = namespace["direct_spec"]
            self.assertEqual(request.frequencies.to("GHz").magnitude.tolist(), [5.5, 6.0, 6.5])
            self.assertEqual(request.traces, (namespace["transmission_trace"],))
            trace = request.traces[0]
            self.assertEqual((trace.id, trace.input_port, trace.input_mode,
                              trace.output_port, trace.output_mode),
                             ("transmission", "feedline_in", (), "feedline_out", ()))
            self.assertNotIn("direct", namespace)
        finally:
            try:
                if "run" in namespace:
                    namespace["run"].close()
            finally:
                if previous is None:
                    sys.modules.pop(helper_name, None)
                else:
                    sys.modules[helper_name] = previous

    def test_complete_explicit_layout_exports_its_actual_certificate(self):
        helper_name = "engineer_publication_binding"
        previous = sys.modules.get(helper_name)
        try:
            spec = importlib.util.spec_from_file_location(
                helper_name, ROOT / "scripts/engineer_publication_binding.py")
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            sys.modules[helper_name] = helper
            spec = importlib.util.spec_from_file_location(
                "_chapter3_publication_test", ROOT / "scripts/generate_engineer_chapter3.py")
            chapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(chapter)
            # These exact existing authoring/layout cells create no CircuitRun or solve.
            cells = chapter._parse_cells()[:4]
            self.assertEqual(cells[-1][0], "ch3-attempt-complete-diagram")
            namespace = {"__name__": "__main__"}
            for cell_id, code in cells:
                exec(compile(code, str(chapter.CHAPTER) + ":" + cell_id, "exec"), namespace)
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                record = chapter._write_complete_diagram(namespace, output)
                self.assertEqual(record["status"], "success")
                self.assertEqual(record["certificate"],
                                 chapter._certificate(output / "03-complete-plan.svg"))
                self.assertEqual(record["certificate"]["plan_id"],
                                 "two_port_feedline_readout")
                self.assertIn("03-complete-plan.svg", chapter.ARTIFACT_NAMES)
        finally:
            if previous is None:
                sys.modules.pop(helper_name, None)
            else:
                sys.modules[helper_name] = previous
