"""Verify or rebind engineer artifacts after this runtime-protocol checkpoint.

The original ``binding`` and ``execution`` records remain the authority for
the source that produced numerical evidence.  ``publication_binding`` names
the current package/docs bytes checked for publication; it never relabels an
old Result as a current-source execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Mapping


RUNTIME_SOURCE_DIFF = (
    "src/scnsim/_canonical.py",
    "src/scnsim/_julia/runtime.json",
    "src/scnsim/_julia/src/SCNSimBackend.jl",
    "src/scnsim/_schemas/identity-common.schema.json",
    "src/scnsim/_schemas/identity-v2.schema.json",
    "src/scnsim/_workspace.py",
    "src/scnsim/results.py",
    "src/scnsim/runtime.py",
    "src/scnsim/specs.py",
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _payload(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"publication_binding", "publication_rebind"}
    }


def _source_files(binding: object) -> Mapping[str, str]:
    if not isinstance(binding, Mapping):
        raise RuntimeError("engineer artifact binding is malformed")
    source_tree = binding.get("source_tree")
    files = source_tree.get("files") if isinstance(source_tree, Mapping) else None
    if not isinstance(files, Mapping) or any(
        not isinstance(path, str) or not isinstance(digest, str)
        for path, digest in files.items()
    ):
        raise RuntimeError("engineer source-tree binding is malformed")
    return files


def _source_tree_sha256(binding: object) -> object:
    return binding.get("source_tree", {}).get("sha256") if isinstance(binding, Mapping) else None


def _publication_record(binding: Mapping[str, object]) -> dict[str, object]:
    return {
        "generator_sha256": binding.get("generator_sha256"),
        "source_tree_sha256": _source_tree_sha256(binding),
    }


def _changed_sources(executed: object, publication: object) -> dict[str, dict[str, str]]:
    old = _source_files(executed)
    current = _source_files(publication)
    if set(old) != set(current):
        raise RuntimeError("engineer source-tree file inventory changed during runtime-only rebind")
    changed = {
        path: {"executed_sha256": old[path], "publication_sha256": current[path]}
        for path in sorted(old)
        if old[path] != current[path]
    }
    if tuple(changed) != RUNTIME_SOURCE_DIFF:
        raise RuntimeError(f"engineer runtime source delta is outside the authorized set: {tuple(changed)!r}")
    return changed


def check_publication_binding(manifest: Mapping[str, object], current: Mapping[str, object]) -> None:
    """Verify current publication bytes separately from execution provenance."""

    executed = manifest.get("binding")
    publication = manifest.get("publication_binding")
    rebind = manifest.get("publication_rebind")
    if executed == current:
        if publication is not None or rebind is not None:
            raise RuntimeError("current-source engineer artifacts have unexpected rebind metadata")
        return
    if publication != _publication_record(current) or not isinstance(executed, Mapping) or not isinstance(rebind, Mapping):
        raise RuntimeError("engineer artifact publication binding is stale")
    for key in set(executed) | set(current):
        if key not in {"generator_sha256", "source_tree"} and executed.get(key) != current.get(key):
            raise RuntimeError(f"engineer executable teaching binding changed during source-only rebind: {key}")
    changed = _changed_sources(executed, current)
    execution = manifest.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("source_tree_sha256") != _source_tree_sha256(executed)
        or execution.get("cells") != executed.get("cells")
        or not isinstance(execution.get("generator_sha256"), str)
    ):
        raise RuntimeError("engineer historical execution binding changed during publication rebind")
    expected = {
        "schema": "scnsim.engineer_publication_rebind.v1",
        "kind": "multiview_runtime_protocol_source_only",
        "prior_manifest_sha256": rebind.get("prior_manifest_sha256"),
        "execution_payload_sha256": _canonical_sha256(_payload(manifest)),
        "execution_generator_sha256": execution.get("generator_sha256"),
        "publication_generator_sha256": current.get("generator_sha256"),
        "execution_source_tree_sha256": _source_tree_sha256(executed),
        "publication_source_tree_sha256": _source_tree_sha256(current),
        "changed_source_files": changed,
        "numerical_artifacts_unchanged": True,
        "execution_identity_retained": True,
    }
    if not isinstance(expected["prior_manifest_sha256"], str) or dict(rebind) != expected:
        raise RuntimeError("engineer publication rebind evidence is malformed")


def _rebind(path: Path, current: Mapping[str, object]) -> None:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or "publication_binding" in manifest or "publication_rebind" in manifest:
        raise RuntimeError(f"{path} is not an unrebound engineer manifest")
    executed = manifest.get("binding")
    if not isinstance(executed, Mapping):
        raise RuntimeError(f"{path} has no execution binding")
    for key in set(executed) | set(current):
        if key not in {"generator_sha256", "source_tree"} and executed.get(key) != current.get(key):
            raise RuntimeError(f"{path} changed executable teaching source: {key}")
    changed = _changed_sources(executed, current)
    execution = manifest.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("source_tree_sha256") != _source_tree_sha256(executed)
        or execution.get("cells") != executed.get("cells")
        or not isinstance(execution.get("generator_sha256"), str)
    ):
        raise RuntimeError(f"{path} has inconsistent historical execution evidence")
    manifest["publication_binding"] = _publication_record(current)
    manifest["publication_rebind"] = {
        "schema": "scnsim.engineer_publication_rebind.v1",
        "kind": "multiview_runtime_protocol_source_only",
        "prior_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "execution_payload_sha256": _canonical_sha256(_payload(manifest)),
        "execution_generator_sha256": execution.get("generator_sha256"),
        "publication_generator_sha256": current.get("generator_sha256"),
        "execution_source_tree_sha256": _source_tree_sha256(executed),
        "publication_source_tree_sha256": _source_tree_sha256(current),
        "changed_source_files": changed,
        "numerical_artifacts_unchanged": True,
        "execution_identity_retained": True,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    check_publication_binding(manifest, current)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebind", action="store_true", help="add the exact checkpoint publication binding")
    args = parser.parse_args()
    if not args.rebind:
        parser.error("--rebind is required")
    root = Path(__file__).resolve().parents[1]
    for chapter in range(1, 9):
        module = importlib.import_module(f"generate_engineer_chapter{chapter}")
        path = root / "examples" / "engineer" / "figures" / f"chapter-{chapter:02d}-artifacts.json"
        current = module._binding()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(manifest, Mapping) and "publication_binding" in manifest:
            manifest = dict(manifest)
            manifest["publication_binding"] = _publication_record(current)
            rebind = dict(manifest["publication_rebind"])
            rebind["publication_generator_sha256"] = current.get("generator_sha256")
            rebind["publication_source_tree_sha256"] = _source_tree_sha256(current)
            rebind["changed_source_files"] = _changed_sources(manifest["binding"], current)
            manifest["publication_rebind"] = rebind
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, path)
            check_publication_binding(manifest, current)
        else:
            _rebind(path, current)
        print(path)


if __name__ == "__main__":
    main()
