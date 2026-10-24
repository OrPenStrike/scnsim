"""Non-native regressions for the engineer course's companion/kernel ownership."""
import ast
import hashlib
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/engineer_publication_binding.py'


def helper_function(name):
    tree = ast.parse(HELPER.read_text(), filename=str(HELPER))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    namespace = dict(Path=Path)
    exec(compile(module, str(HELPER) + ':' + name, 'exec'), namespace)
    return namespace[name]


class EngineerPublicationBindingTests(unittest.TestCase):
    def test_companion_spec_and_same_source_binding(self):
        present = 'examples' in sys.modules
        original = sys.modules.pop('examples', None)
        path_before = list(sys.path)
        producer_before = {k: v for k, v in sys.modules.items()
                           if k == 'scnsim' or k.startswith('scnsim.')}
        try:
            bind = helper_function('_bind_course_examples')
            bind(ROOT)
            namespace = sys.modules['examples']
            bind(ROOT)
            self.assertIs(sys.modules['examples'], namespace)
            engineer = PathFinder.find_spec('examples.engineer', namespace.__path__)
            diagram = PathFinder.find_spec('examples.engineer.diagram_layouts',
                                          engineer.submodule_search_locations)
            self.assertEqual(Path(diagram.origin).resolve(),
                             (ROOT / 'examples/engineer/diagram_layouts.py').resolve())
            self.assertEqual(sys.path, path_before)
            self.assertEqual({k: v for k, v in sys.modules.items()
                              if k == 'scnsim' or k.startswith('scnsim.')}, producer_before)
        finally:
            sys.modules.pop('examples', None)
            if present:
                sys.modules['examples'] = original

    def test_foreign_companion_namespace_is_preserved(self):
        present = 'examples' in sys.modules
        original = sys.modules.pop('examples', None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                foreign = ModuleType('examples')
                foreign.__path__ = [directory]
                sys.modules['examples'] = foreign
                with self.assertRaisesRegex(RuntimeError, 'existing examples namespace differs'):
                    helper_function('_bind_course_examples')(ROOT)
                self.assertIs(sys.modules['examples'], foreign)
        finally:
            sys.modules.pop('examples', None)
            if present:
                sys.modules['examples'] = original

    def test_kernel_temp_uses_inherited_root_and_native_cleanup(self):
        source = ROOT / "scripts/generate_engineer_chapter8.py"
        tree = ast.parse(source.read_text(), filename=str(source))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "_kernel_temporary_directory")
        namespace = {"tempfile": tempfile}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(source) + ":exact-temp-owner", "exec"), namespace)
        create = namespace["_kernel_temporary_directory"]
        with tempfile.TemporaryDirectory() as inherited:
            with create({"TMPDIR": inherited}) as first, create({"TMPDIR": inherited}) as second:
                self.assertEqual(Path(first).parent, Path(inherited))
                self.assertEqual(Path(second).parent, Path(inherited))
                self.assertNotEqual(first, second)
                self.assertTrue(Path(first).is_dir())
            self.assertFalse(Path(first).exists())
            self.assertFalse(Path(second).exists())
        with create({}) as default:
            self.assertEqual(Path(default).parent, Path(tempfile.gettempdir()))
        self.assertFalse(Path(default).exists())

    def test_kernel_poll_error_preserves_primary_and_later_cleanup(self):
        source = ROOT / 'scripts/generate_engineer_chapter8.py'
        tree = ast.parse(source.read_text(), filename=str(source))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_kernel')
        owned_try = next(n for n in function.body if isinstance(n, ast.Try) and n.finalbody)
        skeleton = ast.parse('''def exercise(client, process, manager, cwd, primary):
    private_facts = {}
    identity = {}
    cleanup_error = None
    kernel_tmp = None
    try:
        raise primary
    finally:
        pass
''')
        skeleton.body[0].body[-1].finalbody = owned_try.finalbody
        ast.fix_missing_locations(skeleton)
        namespace = dict(os=os, Path=Path, json=json,
                         _hash=lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
        exec(compile(skeleton, str(source) + ':exact-finally', 'exec'), namespace)
        events = []
        primary = RuntimeError('original kernel operation')
        child = subprocess.Popen([sys.executable, '-c', 'pass'], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        class OwnedProcess:
            def poll(self):
                events.append('poll')
                raise OSError('liveness readback')
            def wait(self):
                events.append('wait')
                return child.wait()
        client = SimpleNamespace(stop_channels=lambda: events.append('channels'))
        manager = SimpleNamespace(kernel_id='regression',
                                  shutdown_kernel=lambda **kw: events.append('shutdown'))
        try:
            with tempfile.TemporaryDirectory() as directory:
                cwd = Path(directory) / 'execution'
                cwd.mkdir()
                with self.assertRaises(RuntimeError) as caught:
                    namespace['exercise'](client, OwnedProcess(), manager, cwd, primary)
                self.assertIs(caught.exception, primary)
                self.assertEqual(events, ['poll', 'channels', 'shutdown', 'wait'])
                facts = json.loads((cwd.parent / 'kernel-regression-runtime.json').read_text())
                self.assertTrue(facts['waited'])
                self.assertEqual(facts['primary_error'], repr(primary))
        finally:
            child.wait()


if __name__ == '__main__':
    unittest.main(verbosity=2)
