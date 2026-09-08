"""Generate/verify SCNSim's transport-1.0 public knowledge package data.

The explicit catalog owns selection and stable IDs; canonical sources own text.
Run with --check to diagnose drift without writes. PEP 517 hooks generate from
a checkout or, only for a standard PKG-INFO-bearing sdist, verify embedded data.
Missing sources in a checkout never select the sdist path. No runtime imports,
network access, source execution, or shared bundle dependency are involved.
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re


BUNDLE = "src/scnsim/_agent_knowledge"
CATALOG = "scripts/agent_knowledge_sources.json"
METADATA_FIELDS = {"id", "kind", "title", "summary", "status", "path", "source_ref"}
RESOURCE_FIELDS = METADATA_FIELDS - {"source_ref"} | {"content_sha256", "provenance"}
MANIFEST_FIELDS = {
    "schema_version", "distribution", "distribution_version", "bundle_id", "resources"
}
ID = re.compile(r"[a-z0-9][a-z0-9._/-]*\Z")
SEGMENT = re.compile(r"[A-Za-z0-9._-]+\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _json(data: bytes):
    return json.loads(data.decode("utf-8"), object_pairs_hook=_object)


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"invalid {label} fields")


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("metadata must contain nonempty strings")


def _relative(value: str) -> str:
    _text(value)
    if any(part in {"", ".", ".."} or not SEGMENT.fullmatch(part)
           for part in value.split("/")):
        raise ValueError("unsafe relative path")
    return value


def _path(root: Path, relative: str) -> Path:
    """Reject symlinks in every segment, including absent output ancestors."""
    current = root
    if current.is_symlink():
        raise ValueError("unsafe symlink root")
    for part in _relative(relative).split("/"):
        current = current / part
        if current.is_symlink():
            raise ValueError("unsafe symlink path")
    return current


def _source_ref(value: str) -> str:
    # These whole-source resources deliberately have no fragment projections.
    return _relative(value)


def _bytes(path: Path) -> bytes:
    if not path.is_file():
        raise ValueError("required knowledge input or resource is missing")
    content = path.read_bytes()
    content.decode("utf-8")
    return content


def _version(root: Path) -> str:
    # Setuptools already supplies its Python 3.10 TOML parser as a build input.
    # Keep that compatibility detail at build time; never depend on it at runtime.
    from setuptools.config.pyprojecttoml import load_file

    project = load_file(_path(root, "pyproject.toml"))["project"]
    if project.get("name") != "scnsim":
        raise ValueError("distribution mismatch")
    version = project.get("version")
    _text(version)
    return version


def _catalog(root: Path) -> list[dict]:
    entries = _json(_bytes(_path(root, CATALOG)))
    if not isinstance(entries, list) or not entries:
        raise ValueError("source catalog must be a nonempty array")
    ids, paths, sources = set(), set(), set()
    for entry in entries:
        _fields(entry, METADATA_FIELDS, "catalog entry")
        for value in entry.values():
            _text(value)
        resource_id = entry["id"]
        if not ID.fullmatch(resource_id):
            raise ValueError("invalid resource ID")
        _relative(resource_id)
        path = _relative(entry["path"])
        if not path.startswith("resources/") or len(path.split("/")) != 2:
            raise ValueError("resource path must be one file under resources")
        source = _source_ref(entry["source_ref"])
        if resource_id in ids or path in paths or source in sources:
            raise ValueError("duplicate resource ID, path, or canonical source")
        ids.add(resource_id)
        paths.add(path)
        sources.add(source)
    return sorted(entries, key=lambda entry: entry["id"])


def _manifest(version: str, resources: list[dict]) -> dict:
    manifest = {
        "schema_version": "1.0",
        "distribution": "scnsim",
        "distribution_version": version,
        "resources": resources,
    }
    manifest["bundle_id"] = "sha256:" + _digest(_canonical(manifest))
    return manifest


def _files(root: Path) -> set[str]:
    """Inspect complete closure; never follow an undeclared symlink."""
    if not root.exists():
        return set()
    files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("unsafe symlink in bundle")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError("unsupported bundle filesystem entry")
    return files


def _verify(root: Path, catalog: list[dict], version: str) -> dict:
    bundle = _path(root, BUNDLE)
    manifest = _json(_bytes(_path(bundle, "manifest.json")))
    _fields(manifest, MANIFEST_FIELDS, "manifest")
    if manifest["schema_version"] != "1.0":
        raise ValueError("unsupported bundle schema")
    if (manifest["distribution"] != "scnsim"
            or manifest["distribution_version"] != version):
        raise ValueError("bundle distribution/version mismatch")
    resources = manifest["resources"]
    if not isinstance(resources, list) or len(resources) != len(catalog):
        raise ValueError("bundle/catalog resource mismatch")
    expected_files = {"manifest.json"}
    for resource, entry in zip(resources, catalog):
        _fields(resource, RESOURCE_FIELDS, "resource")
        for key in METADATA_FIELDS - {"source_ref"}:
            if resource[key] != entry[key]:
                raise ValueError("bundle/catalog metadata drift")
        provenance = resource["provenance"]
        _fields(provenance, {"source_ref", "source_sha256"}, "provenance")
        if provenance["source_ref"] != entry["source_ref"]:
            raise ValueError("bundle/catalog provenance drift")
        for digest in (resource["content_sha256"], provenance["source_sha256"]):
            if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                raise ValueError("invalid resource digest")
        content = _bytes(_path(bundle, resource["path"]))
        if (_digest(content) != resource["content_sha256"]
                or resource["content_sha256"] != provenance["source_sha256"]):
            raise ValueError("resource content/provenance drift")
        source = _path(root, entry["source_ref"])
        if source.exists() and _bytes(source) != content:
            raise ValueError("canonical source drift")
        expected_files.add(resource["path"])
    if _files(bundle) != expected_files:
        raise ValueError("bundle file closure mismatch")
    if manifest != _manifest(version, resources):
        raise ValueError("bundle identity drift")
    return manifest


def prepare_bundle(root: Path, *, check: bool = False) -> dict:
    """Prepare exactly one source-checkout or verified sdist knowledge bundle."""
    root = root.absolute()
    version = _version(root)
    catalog = _catalog(root)
    pkg_info = _path(root, "PKG-INFO")
    if pkg_info.exists():
        metadata = BytesParser().parsebytes(_bytes(pkg_info))
        if metadata.get_all("Name") != ["scnsim"] or metadata.get_all("Version") != [version]:
            raise ValueError("sdist metadata version/distribution mismatch")
        # The sdist intentionally retains only the packaging README source.
        # A partial or accidentally included documentation tree is not a new mode.
        if any(_path(root, name).exists() for name in ("docs", "examples")):
            raise ValueError("sdist unexpectedly contains canonical documentation trees")
        return _verify(root, catalog, version)

    contents = {}
    resources = []
    for entry in catalog:
        content = _bytes(_path(root, entry["source_ref"]))
        digest = _digest(content)
        resource = {key: value for key, value in entry.items() if key != "source_ref"}
        resource.update(content_sha256=digest, provenance={
            "source_ref": entry["source_ref"], "source_sha256": digest
        })
        resources.append(resource)
        contents[entry["path"]] = content
    manifest = _manifest(version, resources)
    contents["manifest.json"] = (
        json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    bundle = _path(root, BUNDLE)
    existing = _files(bundle)
    if existing - contents.keys():
        raise ValueError("undeclared bundle files; remove superseded generated resources explicitly")
    # Validate all source and destination paths before the first write.
    destinations = {name: _path(bundle, name) for name in contents}
    if check:
        if existing != contents.keys() or any(
                _bytes(destinations[name]) != content for name, content in contents.items()):
            raise ValueError("generated knowledge is absent or stale")
    else:
        for name, content in contents.items():
            destination = destinations[name]
            if not destination.exists() or destination.read_bytes() != content:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
    return _verify(root, catalog, version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without changing files")
    args = parser.parse_args()
    try:
        manifest = prepare_bundle(Path(__file__).absolute().parents[1], check=args.check)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f"agent knowledge: {error}") from None
    print(f"scnsim {manifest['distribution_version']}: "
          f"{len(manifest['resources'])} resources; {manifest['bundle_id']}")


if __name__ == "__main__":
    main()
