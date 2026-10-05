"""Source recovery for live Notebook catalog declarations.

Python inspection and IPython history bind the actual class and factory
definitions; a matching display name alone cannot establish provenance."""

from __future__ import annotations

from collections.abc import Mapping
import ast
from pathlib import Path
import inspect
import re
import unicodedata
from .provenance import _catalog_lineage, _catalog_qualified_class
from ..canonical import _identifier, _validation, canonical_json_bytes


def _notebook_source_bytes(subject: type[object], _factory: object | None = None) -> bytes:
    """Close the complete custom Library declaration visible in a notebook."""

    classes: list[dict[str, str]] = []
    factories: list[dict[str, str]] = []
    try:
        for candidate in _catalog_lineage(subject):
            class_source, factory_sources = _notebook_declaration_sources(candidate)
            classes.append({
                "qualified_class": _catalog_qualified_class(candidate),
                "source": _normalized_notebook_source(class_source),
            })
            for name, descriptor in candidate.__dict__.items():
                if name.startswith("_"):
                    continue
                function = descriptor.__func__ if isinstance(descriptor, (classmethod, staticmethod)) else descriptor
                if inspect.isfunction(function):
                    factories.append({
                        "qualified_class": _catalog_qualified_class(candidate),
                        "name": _identifier(name, field="factory"),
                        "source": _normalized_notebook_source(factory_sources[name]),
                    })
    except (OSError, TypeError) as error:
        raise _validation("notebook catalog source is unavailable") from error
    return canonical_json_bytes({
        "schema": "scnsim.notebook_catalog_source",
        "schema_version": 1,
        "classes": classes,
        "factories": factories,
    })


def _notebook_declaration_sources(candidate: type[object]) -> tuple[str, dict[str, str]]:
    """Return one class declaration and its factories from source or IPython history.

    Notebook cells deliberately have no importable module file.  When Python
    cannot recover their source through ``inspect``, the current IPython
    kernel's raw input history is the only accepted alternate evidence.  The
    history match is bound to every unwrapped factory's execution location;
    a namesake in another cell is never provenance for the live catalog.
    """

    functions = _catalog_factory_functions(candidate)
    try:
        return (
            inspect.getsource(candidate),
            {name: inspect.getsource(function) for name, function in functions.items()},
        )
    except (OSError, TypeError):
        return _ipython_notebook_declaration_sources(candidate, functions)


def _catalog_factory_functions(candidate: type[object]) -> dict[str, object]:
    """Return the unwrapped public factory functions declared by one catalog."""

    functions: dict[str, object] = {}
    for name, descriptor in candidate.__dict__.items():
        if name.startswith("_"):
            continue
        function = (
            descriptor.__func__
            if isinstance(descriptor, (classmethod, staticmethod))
            else descriptor
        )
        if inspect.isfunction(function):
            functions[name] = inspect.unwrap(function)
    return functions


def _ipython_notebook_declaration_sources(
    candidate: type[object], functions: Mapping[str, object]
) -> tuple[str, dict[str, str]]:
    """Recover one exact notebook declaration from live IPython cell history.

    A Library with no local factory code has no executable location by which a
    history declaration could be proven current, so it remains fail-closed.
    """

    if not functions:
        raise OSError("notebook catalog has no local factory source location")
    locations = {
        name: _ipython_function_location(function)
        for name, function in functions.items()
    }
    execution_counts = {
        execution_count
        for execution_count, _, _ in locations.values()
        if execution_count is not None
    }
    if len(execution_counts) > 1 or (
        execution_counts
        and any(
            execution_count is None
            for execution_count, _, _ in locations.values()
        )
    ):
        raise OSError("notebook catalog factories do not share one declaration cell")
    execution_count = next(iter(execution_counts), None)
    matches: list[tuple[str, ast.ClassDef, dict[str, str]]] = []
    for source in _ipython_history_sources(execution_count):
        try:
            parsed = ast.parse(source)
        except SyntaxError:
            continue
        class_node = _ipython_class_node(parsed, candidate.__qualname__)
        if class_node is None:
            continue
        members = {
            node.name: node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        factory_sources: dict[str, str] = {}
        for name, (_, line_number, _) in locations.items():
            node = members.get(name)
            if node is None or not _node_covers_line(node, line_number):
                break
            factory_sources[name] = _ast_source_segment(source, node)
        else:
            if _history_source_matches_functions(source, functions, locations):
                matches.append((source, class_node, factory_sources))
    if len(matches) != 1:
        identities = {
            _notebook_declaration_identity(source, class_node, factory_sources)
            for source, class_node, factory_sources in matches
        }
        if not matches or len(identities) != 1:
            raise OSError("notebook execution source is missing or ambiguous")
    source, class_node, factory_sources = matches[0]
    return _ast_source_segment(source, class_node), factory_sources


def _ipython_function_location(function: object) -> tuple[int | None, int, str]:
    """Return the trusted IPython execution and line for one live function."""

    code = getattr(function, "__code__", None)
    filename = getattr(code, "co_filename", "")
    match = re.fullmatch(r"<ipython-input-([0-9]+)-[0-9a-f]+>", filename)
    line_number = getattr(code, "co_firstlineno", None)
    temporary_kernel_file = re.fullmatch(
        r"(?:.*/)?ipykernel_[^/]+/[0-9]+\.py", filename
    )
    if (
        not isinstance(line_number, int)
        or line_number < 1
        or (match is None and temporary_kernel_file is None)
    ):
        raise OSError("notebook factory has no IPython execution location")
    return (
        int(match.group(1)) if match is not None else None,
        line_number,
        filename,
    )


def _ipython_history_sources(execution_count: int | None) -> list[str]:
    """Return raw current-session IPython cells for one execution or all cells."""

    try:
        from IPython import get_ipython
    except ImportError as error:
        raise OSError("IPython history is unavailable") from error
    shell = get_ipython()
    history = getattr(shell, "history_manager", None)
    if history is None:
        raise OSError("IPython history is unavailable")
    matches = [
        source
        for _, line, source in history.get_range(raw=True)
        if (execution_count is None or line == execution_count)
        and isinstance(source, str)
    ]
    if not matches:
        raise OSError("notebook execution source is missing or ambiguous")
    return matches


def _history_source_matches_functions(
    source: str,
    functions: Mapping[str, object],
    locations: Mapping[str, tuple[int | None, int, str]],
) -> bool:
    """Bind raw history to a current code object's trusted execution identity."""

    for name, function in functions.items():
        execution_count, _, filename = locations[name]
        if execution_count is not None:
            continue
        if not _ipykernel_filename_matches_source(filename, source):
            return False
        code = getattr(function, "__code__", None)
        if code is None or getattr(code, "co_filename", None) != filename:
            return False
    return True


def _ipykernel_filename_matches_source(filename: str, source: str) -> bool:
    """Verify IPykernel's source-derived code-object filename without I/O."""

    try:
        from ipykernel.compiler import get_tmp_hash_seed, murmur2_x86
    except ImportError:
        return False
    expected = f"{murmur2_x86(source, get_tmp_hash_seed())}.py"
    return Path(filename).name == expected


def _notebook_declaration_identity(
    source: str, class_node: ast.ClassDef, factory_sources: Mapping[str, str]
) -> bytes:
    """Return the exact normalized declaration identity used to collapse reruns."""

    return canonical_json_bytes({
        "class": _normalized_notebook_source(_ast_source_segment(source, class_node)),
        "factories": [
            {"name": name, "source": _normalized_notebook_source(factory_sources[name])}
            for name in sorted(factory_sources)
        ],
    })


def _ipython_class_node(tree: ast.AST, qualified_name: str) -> ast.ClassDef | None:
    """Find one non-local class declaration by its exact qualified path."""

    parts = qualified_name.split(".")
    if not parts or any(not part or part == "<locals>" for part in parts):
        return None
    nodes: list[ast.AST] = [tree]
    for part in parts:
        matches = [
            child
            for node in nodes
            for child in getattr(node, "body", ())
            if isinstance(child, ast.ClassDef) and child.name == part
        ]
        if len(matches) != 1:
            return None
        nodes = matches
    return nodes[0] if isinstance(nodes[0], ast.ClassDef) else None


def _node_covers_line(node: ast.AST, line_number: int) -> bool:
    """Accept a function's definition or decorator line, but no other source."""

    decorator_lines = [decorator.lineno for decorator in getattr(node, "decorator_list", ())]
    first_line = min([getattr(node, "lineno", 0), *decorator_lines])
    last_line = getattr(node, "end_lineno", 0)
    return first_line <= line_number <= last_line


def _ast_source_segment(source: str, node: ast.AST) -> str:
    """Extract a declaration including decorators from the authoritative cell."""

    lines = source.splitlines(keepends=True)
    decorator_lines = [decorator.lineno for decorator in getattr(node, "decorator_list", ())]
    first_line = min([getattr(node, "lineno", 0), *decorator_lines])
    last_line = getattr(node, "end_lineno", 0)
    if first_line < 1 or last_line < first_line or last_line > len(lines):
        raise OSError("notebook declaration has invalid source coordinates")
    return "".join(lines[first_line - 1:last_line])


def _normalized_notebook_source(source: str) -> str:
    return unicodedata.normalize("NFC", inspect.cleandoc(source).replace("\r\n", "\n").replace("\r", "\n"))
