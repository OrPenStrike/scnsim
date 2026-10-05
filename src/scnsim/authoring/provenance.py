"""Portable source provenance for built-in and custom component catalogs.

Wheel records, editable Git trees, and defining source modules bind catalog
creation to source bytes. Notebook declaration recovery is delegated lazily."""

from __future__ import annotations

from collections.abc import Mapping
import base64
import csv
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution, distributions
from io import StringIO
from pathlib import Path
import json
import re
import tokenize
import subprocess
import unicodedata
from urllib.parse import unquote, urlparse
from ..canonical import _nfc, _nonempty, _validation, relative_path, sha256_hex


def catalog_source_record(
    obj_or_class: object, *, factory: object | None = None
) -> dict[str, object]:
    """Capture one catalog-wide portable source identity.

    ``factory`` remains an authoring call-site compatibility input; it never
    enters the returned record because the component snapshot owns the invoked
    factory name.
    """

    subject = obj_or_class if isinstance(obj_or_class, type) else type(obj_or_class)
    # Public catalog identities survive relocation of their defining modules.
    from . import components

    if subject is type(components):
        return _builtin_catalog_source()
    records = [_custom_catalog_source(candidate, factory=factory) for candidate in _catalog_lineage(subject)]
    record = records[-1]
    record["source_sha256"] = sha256_hex({
        "schema": "scnsim.catalog_lineage_source",
        "schema_version": 1,
        "classes": records,
    })
    return record


def _custom_catalog_source(subject: type[object], *, factory: object | None) -> dict[str, object]:
    module_name = _catalog_module_name(subject)
    qualified_class = _catalog_qualified_class(subject)
    identity = {
        "catalog_id": f"{module_name}:{qualified_class}",
        "catalog_kind": "custom",
        "module": module_name,
        "qualified_class": qualified_class,
    }
    module = import_module(module_name)
    source_path = _module_source_path(module)
    if source_path is not None:
        package = _distribution_owning(source_path, module_name)
        if package is not None:
            if _editable_distribution(package):
                package_root = _package_root(module_name, source_path)
                if package_root is None:
                    raise _validation("editable custom catalog must be inside a Python package")
                return _editable_catalog(identity, package_root)
            return _wheel_catalog(identity, package)
        return {
            **identity,
            "source_kind": "module_source",
            "source_sha256": sha256(_normalized_source_bytes(source_path)).hexdigest(),
        }
    from .notebook_source import _notebook_source_bytes

    return {
        **identity,
        "source_kind": "notebook_source",
        "source_sha256": sha256(_notebook_source_bytes(subject, factory)).hexdigest(),
    }


def _catalog_lineage(subject: type[object]) -> list[type[object]]:
    from . import Library

    lineage = [
        candidate for candidate in reversed(subject.__mro__)
        if candidate is not object and candidate is not Library
    ]
    if not lineage or lineage[-1] is not subject:
        raise _validation("catalog class does not have a closed Library lineage")
    return lineage


def _builtin_catalog_source() -> dict[str, object]:
    """Return the reserved provenance record for the public singleton."""

    try:
        package = distribution("scnsim")
    except PackageNotFoundError as error:
        raise _validation("installed SCNSim distribution metadata is unavailable") from error
    identity = {
        "catalog_id": "scnsim.components",
        "catalog_kind": "builtin",
        "module": "scnsim",
        "public_symbol": "components",
    }
    if _editable_distribution(package):
        package_root = Path(import_module("scnsim").__file__).resolve().parent
        return _editable_catalog(identity, package_root)
    return _wheel_catalog(identity, package)


def _wheel_catalog(identity: Mapping[str, object], package: object) -> dict[str, object]:
    record = package.read_text("RECORD")  # type: ignore[union-attr]
    if record is None:
        raise _validation("installed wheel lacks RECORD provenance")
    rows: list[dict[str, object]] = []
    record_self_rows = 0
    for row in csv.reader(StringIO(record)):
        if len(row) != 3:
            raise _validation("wheel RECORD row has invalid field count")
        path, encoded_hash, size_text = row
        normalized_path = relative_path(path)
        parts = normalized_path.split("/")
        if (
            "__pycache__" in parts
            or normalized_path.endswith(".pyc")
            or (
                len(parts) >= 2
                and parts[-2].endswith(".dist-info")
                and parts[-1]
                in {"INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json", "uv_build.json"}
            )
        ):
            continue
        is_record_self = normalized_path.endswith(".dist-info/RECORD")
        if is_record_self:
            if encoded_hash or size_text:
                raise _validation("wheel RECORD self row must have empty hash and size")
            record_self_rows += 1
        elif not encoded_hash or not size_text:
            raise _validation(
                "wheel RECORD rows must bind every included file by hash and size",
                path=normalized_path,
            )
        if encoded_hash:
            algorithm, separator, digest = encoded_hash.partition("=")
            if algorithm != "sha256" or not separator or not digest:
                raise _validation("wheel RECORD uses a non-SHA-256 hash", path=normalized_path)
            try:
                expected = base64.urlsafe_b64decode(digest + "=" * (-len(digest) % 4))
            except Exception as error:
                raise _validation("wheel RECORD hash is malformed", path=normalized_path) from error
            installed = package.locate_file(path)  # type: ignore[union-attr]
            if installed.is_symlink() or not installed.is_file() or sha256(installed.read_bytes()).digest() != expected:
                raise _validation("wheel RECORD content hash does not match", path=normalized_path)
        if size_text:
            installed = package.locate_file(path)  # type: ignore[union-attr]
            if installed.is_symlink() or not size_text.isdecimal() or installed.stat().st_size != int(size_text):
                raise _validation("wheel RECORD size does not match", path=normalized_path)
        rows.append({"path": normalized_path, "hash": encoded_hash, "size": size_text})
    rows.sort(key=lambda row: row["path"])
    if len({row["path"] for row in rows}) != len(rows):
        raise _validation("wheel RECORD contains duplicate paths")
    if record_self_rows != 1:
        raise _validation("wheel RECORD must contain exactly one empty self row")
    return {
        **identity,
        "source_kind": "wheel_record",
        "source_sha256": sha256_hex({"schema": "scnsim.wheel_record", "schema_version": 2, "rows": rows}),
        "distribution": _normalize_distribution_name(package.metadata["Name"]),  # type: ignore[union-attr]
        "version": package.version,  # type: ignore[union-attr]
    }


def _editable_catalog(identity: Mapping[str, object], package_root: Path) -> dict[str, object]:
    git_root = _git_output(package_root, "rev-parse", "--show-toplevel")
    commit = _git_output(package_root, "rev-parse", "HEAD")
    source_rows = _source_tree_manifest(package_root, Path(git_root))
    status = _git_output_bytes(package_root, "status", "--porcelain=v1", "-z", "--no-renames", "--untracked-files=all")
    overlay: list[dict[str, object]] = []
    for record in status.split(b"\0"):
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise _validation("Git status output is malformed")
        raw_path = record[3:]
        try:
            changed_path = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _validation("Git status path is not UTF-8") from error
        candidate = Path(git_root, changed_path)
        try:
            relative = candidate.absolute().relative_to(package_root).as_posix()
        except ValueError:
            continue
        if _excluded_source_path(relative):
            continue
        entry: dict[str, object] = {"status": record[:2].decode("ascii"), "path": relative_path(relative)}
        if candidate.is_file() and not candidate.is_symlink():
            entry["sha256"] = sha256(candidate.read_bytes()).hexdigest()
        overlay.append(entry)
    overlay.sort(key=lambda entry: (entry["path"], entry["status"]))
    return {
        **identity,
        "source_kind": "editable_git",
        "source_sha256": sha256_hex({"schema": "scnsim.package_source", "schema_version": 1, "files": source_rows}),
        "git_commit": _sha256_or_git(commit),
        "dirty_overlay_sha256": sha256_hex({"schema": "scnsim.git_overlay", "schema_version": 1, "entries": overlay}),
    }


def _git_output(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(cwd), *args], text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise _validation("editable SCNSim catalog requires a readable Git repository", stderr=completed.stderr.strip())
    return completed.stdout.rstrip("\n")


def _git_output_bytes(cwd: Path, *args: str) -> bytes:
    completed = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, check=False)
    if completed.returncode != 0:
        raise _validation("editable catalog requires a readable Git repository", stderr=completed.stderr.decode(errors="replace").strip())
    return completed.stdout


def _source_tree_manifest(package_root: Path, git_root: Path) -> list[dict[str, object]]:
    modes = _git_file_modes(git_root)
    rows: list[dict[str, object]] = []
    for path in package_root.rglob("*"):
        if path.is_symlink():
            raise _validation("editable package source contains a symlink")
        relative = path.relative_to(package_root).as_posix()
        if path.is_file() and not _excluded_source_path(relative):
            git_relative = path.relative_to(git_root).as_posix()
            rows.append({
                "path": relative_path(relative),
                "mode": modes.get(git_relative, _filesystem_git_mode(path)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            })
    rows.sort(key=lambda row: row["path"])
    return rows


def _git_file_modes(git_root: Path) -> dict[str, str]:
    output = _git_output_bytes(git_root, "ls-files", "-s", "-z")
    modes: dict[str, str] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            prefix, raw_path = record.split(b"\t", 1)
            mode, _object_id, stage = prefix.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise _validation("Git index output is malformed") from error
        if stage != "0" or mode not in {"100644", "100755"}:
            raise _validation("catalog source has an unsupported Git file mode", mode=mode)
        modes[path] = mode
    return modes


def _filesystem_git_mode(path: Path) -> str:
    return "100755" if path.stat().st_mode & 0o111 else "100644"


def _excluded_source_path(value: str) -> bool:
    parts = value.split("/")
    return "__pycache__" in parts or any(part in {".git", ".hg", ".svn", ".pytest_cache", ".mypy_cache", "build", "dist"} or part.endswith(".egg-info") for part in parts) or value.endswith((".pyc", ".pyo"))


def _catalog_module_name(subject: type[object]) -> str:
    module = getattr(subject, "__module__", None)
    if not isinstance(module, str) or not module or module == "__main__":
        return "__main__" if module == "__main__" else _raise_catalog_identity("catalog class has no portable module name")
    return _nfc(module, field="catalog_module")


def _catalog_qualified_class(subject: type[object]) -> str:
    qualified = getattr(subject, "__qualname__", None)
    if not isinstance(qualified, str) or not qualified or "<locals>" in qualified:
        raise _validation("catalog class has no portable qualified name")
    return _nfc(qualified, field="qualified_class")


def _raise_catalog_identity(message: str) -> object:
    raise _validation(message)


def _module_source_path(module: object) -> Path | None:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or raw_path.startswith("<"):
        return None
    path = Path(raw_path)
    if path.suffix != ".py" or path.is_symlink() or not path.is_file():
        raise _validation("custom catalog module must have a readable regular Python source file")
    return path.resolve()


def _distribution_owning(source_path: Path, module_name: str) -> object | None:
    matches: list[object] = []
    for candidate in distributions():
        files = candidate.files
        if files is not None:
            for file in files:
                located = Path(candidate.locate_file(file))
                if located.exists() and located.resolve() == source_path:
                    matches.append(candidate)
                    break
            else:
                root = _editable_distribution_root(candidate, module_name)
                if root is not None:
                    try:
                        source_path.relative_to(root)
                    except ValueError:
                        continue
                    matches.append(candidate)
        else:
            root = _editable_distribution_root(candidate, module_name)
            if root is not None:
                try:
                    source_path.relative_to(root)
                except ValueError:
                    continue
                matches.append(candidate)
    if len(matches) > 1:
        raise _validation("custom catalog source belongs to multiple installed distributions")
    return matches[0] if matches else None


def _editable_distribution(package: object) -> bool:
    direct_url = package.read_text("direct_url.json")  # type: ignore[union-attr]
    if not direct_url:
        return False
    try:
        direct = json.loads(direct_url)
    except json.JSONDecodeError as error:
        raise _validation("catalog direct_url metadata is invalid") from error
    info = direct.get("dir_info") if isinstance(direct, Mapping) else None
    return isinstance(info, Mapping) and bool(info.get("editable", False))


def _editable_distribution_root(package: object, module_name: str) -> Path | None:
    if not _editable_distribution(package):
        return None
    top_level = package.read_text("top_level.txt")  # type: ignore[union-attr]
    if top_level is None or module_name.split(".", 1)[0] not in {line.strip() for line in top_level.splitlines()}:
        return None
    direct_url = package.read_text("direct_url.json")  # type: ignore[union-attr]
    direct = json.loads(direct_url)
    url = direct.get("url") if isinstance(direct, Mapping) else None
    parsed = urlparse(url) if isinstance(url, str) else None
    if parsed is None or parsed.scheme != "file":
        return None
    try:
        return Path(unquote(parsed.path)).resolve(strict=True)
    except OSError as error:
        raise _validation("editable catalog source root is unreadable") from error


def _package_root(module_name: str, source_path: Path) -> Path | None:
    parts = module_name.split(".")
    if module_name == "__main__" or not parts:
        return None
    directory = source_path.parent
    for _ in range(len(parts) - (1 if source_path.name == "__init__.py" else 2)):
        directory = directory.parent
    return directory if (directory / "__init__.py").is_file() else None


def _normalized_source_bytes(path: Path) -> bytes:
    try:
        with tokenize.open(path) as source:
            text = source.read()
    except (OSError, SyntaxError, UnicodeError) as error:
        raise _validation("custom catalog source cannot be decoded", path=str(path)) from error
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).encode("utf-8")


def _normalize_distribution_name(name: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", _nonempty(name, "distribution")).lower()
    return normalized


def _sha256_or_git(value: str) -> str:
    # The full V1 schema allows Git SHA-1 or SHA-256 object IDs.  Do not force
    # the current checkout's Git object format into the evidence protocol.
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise _validation("Git commit is not a lowercase object ID")
    return value
