"""Portable notebook export changes only machine-local kernel metadata."""

import copy
import json
from pathlib import Path
import runpy
import unittest

portable_notebook_bytes = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/generate_portable_notebook.py")
)["portable_notebook_bytes"]


class PortableNotebookTests(unittest.TestCase):
    def test_machine_local_path_is_removed_with_cells_and_kernel_identity_preserved(self):
        notebook = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {
                "name": "python3", "language": "python", "display_name": "Python 3",
                "path": "machine-local-kernel-directory",
            }, "language_info": {"name": "python"}, "custom": {"keep": True}},
            "cells": [
                {"cell_type": "markdown", "id": "title", "metadata": {},
                 "source": ["# 可攜式 Notebook\n"]},
                {"cell_type": "code", "id": "declaration", "metadata": {"tags": ["example"]},
                 "source": ["value = 1\n"], "execution_count": None, "outputs": []},
            ],
        }
        expected = copy.deepcopy(notebook)
        del expected["metadata"]["kernelspec"]["path"]
        actual = json.loads(portable_notebook_bytes(json.dumps(notebook).encode()))
        self.assertEqual(actual, expected)
        self.assertEqual(actual["cells"], notebook["cells"])
        self.assertIn("path", notebook["metadata"]["kernelspec"])

    def test_path_free_notebook_is_byte_identical(self):
        raw = b'{ "metadata": { "kernelspec": { "name": "python3", "language": "python" } }, "cells": [] }\n'
        self.assertEqual(portable_notebook_bytes(raw), raw)

    def test_other_path_fields_are_preserved(self):
        raw = b'{"metadata":{"path":"relative-resource","kernelspec":{"name":"python3"}},"cells":[]}'
        self.assertEqual(portable_notebook_bytes(raw), raw)


if __name__ == "__main__":
    unittest.main()
