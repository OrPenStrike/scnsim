"""Verify or rebind engineer artifacts after runtime or presentation changes.

The original ``binding`` and ``execution`` records remain the authority for
the source that produced numerical evidence. ``publication_binding`` names
the current package, teaching-cell, and generator bytes checked for
publication. The separately reviewed transition artifact binds one exact
previously published manifest to one exact source-only successor. It preserves
historical redraw identities and never relabels an old Result as a
current-source execution.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Mapping


TRANSITION_ID = "scnsim.root_usability.20260921"
PRIOR_PUBLISHED_GIT_COMMIT = "a61a766dd1cbe429f23a239cced38c1df25e9d45"
TRANSITION_FILE = Path(__file__).with_name("engineer_publication_transitions.json")
_TRANSITION_NAMES = {
    f"chapter-{chapter:02d}-artifacts.json" for chapter in range(1, 9)
}
_TRANSITION_FIELDS = {
    "artifacts",
    "deltas",
    "historical_execution",
    "prior_manifest_sha256",
    "prior_publication_binding_sha256",
    "prior_publication_rebind_sha256",
    "publication_binding_sha256",
}
_DELTA_FIELDS = {
    "generator",
    "source_files",
    "teaching_sources",
    "cell_ids",
    "cells",
}
_HISTORICAL_FIELDS = {
    "artifacts",
    "binding_sha256",
    "generator_sha256",
    "payload_sha256",
    "prior_publication_artifacts",
    "prior_publication_binding_sha256",
    "prior_redrawn_artifacts",
    "source_tree_sha256",
}
_REBIND_V3_FIELDS = {
    "schema",
    "kind",
    "transition_id",
    "prior_manifest_sha256",
    "prior_publication_binding_sha256",
    "publication_binding_sha256",
    "deltas",
    "historical_execution",
    "prior_publication_rebind_sha256",
    "publication_artifacts",
    "redrawn_artifacts",
    "numerical_evidence_reused",
    "prior_published_artifacts_retained",
    "execution_identity_retained",
}


def _reviewed_transitions() -> Mapping[str, Mapping[str, object]]:
    """Load the explicitly reviewed transition artifact; never derive authority."""

    value = json.loads(TRANSITION_FILE.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {
            "schema", "transition_id", "prior_published_git_commit", "transitions"
        }
        or value.get("schema") != "scnsim.engineer_publication_transitions.v1"
        or value.get("transition_id") != TRANSITION_ID
        or value.get("prior_published_git_commit") != PRIOR_PUBLISHED_GIT_COMMIT
        or not isinstance(value.get("transitions"), dict)
    ):
        raise RuntimeError("engineer publication transition authority is malformed")
    transitions = value["transitions"]
    if set(transitions) != _TRANSITION_NAMES or any(
        not isinstance(row, dict)
        or set(row) != _TRANSITION_FIELDS
        or not isinstance(row.get("deltas"), dict)
        or set(row["deltas"]) != _DELTA_FIELDS
        or not isinstance(row.get("historical_execution"), dict)
        or set(row["historical_execution"]) != _HISTORICAL_FIELDS
        for row in transitions.values()
    ):
        raise RuntimeError("engineer publication transition entries are malformed")
    return transitions


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


def _execution_payload(
    manifest: Mapping[str, object], execution_artifacts: Mapping[str, object]
) -> dict[str, object]:
    payload = _payload(manifest)
    payload["artifacts"] = dict(execution_artifacts)
    return payload


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
        "binding_sha256": _canonical_sha256(binding),
        "generator_sha256": binding.get("generator_sha256"),
        "source_tree_sha256": _source_tree_sha256(binding),
    }


def _execution_generator(
    execution: object, executed: Mapping[str, object]
) -> str:
    """Return the generator bound to one of the two closed execution records."""

    if not isinstance(execution, Mapping):
        raise RuntimeError("engineer historical execution evidence is malformed")
    generator = execution.get("generator_sha256")
    if (
        execution.get("source_tree_sha256") == _source_tree_sha256(executed)
        and execution.get("cells") == executed.get("cells")
        and isinstance(generator, str)
    ):
        return generator
    if execution.get("mode") == "receipt_bound_resume":
        continuation = execution.get("continuation")
        restart = execution.get("restart")
        bound_generator = executed.get("generator_sha256")
        if (
            isinstance(continuation, Mapping)
            and isinstance(restart, Mapping)
            and isinstance(bound_generator, str)
            and continuation.get("generator_sha256") == bound_generator
            and restart.get("generator_sha256") == bound_generator
        ):
            return bound_generator
    raise RuntimeError("engineer historical execution evidence is inconsistent")


def _mapping(value: object, *, role: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise RuntimeError(f"engineer {role} binding is malformed")
    return dict(value)


def _mapping_delta(before: object, after: object, *, role: str) -> dict[str, dict[str, str]]:
    old = _mapping(before, role=role)
    new = _mapping(after, role=role)
    return {
        key: {
            "before_sha256": old.get(key, "absent"),
            "after_sha256": new.get(key, "absent"),
        }
        for key in sorted(set(old) | set(new))
        if old.get(key) != new.get(key)
    }


def _apply_v2_delta(values: dict[str, str], delta: object, *, role: str) -> None:
    if not isinstance(delta, Mapping):
        raise RuntimeError(f"historical engineer {role} delta is malformed")
    for key, row in delta.items():
        if (
            not isinstance(key, str)
            or not isinstance(row, Mapping)
            or set(row) != {"executed_sha256", "publication_sha256"}
            or values.get(key, "absent") != row.get("executed_sha256")
            or not isinstance(row.get("publication_sha256"), str)
        ):
            raise RuntimeError(f"historical engineer {role} delta is inconsistent")
        values[key] = str(row["publication_sha256"])


def _prior_publication_binding(
    manifest: Mapping[str, object],
    *,
    reviewed_prior: Mapping[str, object] | None = None,
) -> dict[str, object]:
    executed = manifest.get("binding")
    if not isinstance(executed, Mapping):
        raise RuntimeError("engineer historical execution binding is malformed")
    publication = manifest.get("publication_binding")
    rebind = manifest.get("publication_rebind")
    if publication is None and rebind is None:
        return deepcopy(dict(executed))
    if isinstance(rebind, Mapping) and rebind.get("schema") == "scnsim.engineer_publication_rebind.v3":
        historical = rebind.get("historical_execution")
        execution_artifacts = historical.get("artifacts") if isinstance(historical, Mapping) else None
        if (
            reviewed_prior is None
            or set(rebind) != _REBIND_V3_FIELDS
            or rebind.get("kind") != "reviewed_source_only_transition"
            or not isinstance(historical, Mapping)
            or set(historical) != _HISTORICAL_FIELDS
            or not isinstance(execution_artifacts, Mapping)
            or historical.get("payload_sha256")
            != _canonical_sha256(_execution_payload(manifest, execution_artifacts))
            or historical.get("binding_sha256") != _canonical_sha256(executed)
            or historical.get("generator_sha256")
            != _execution_generator(manifest.get("execution"), executed)
            or historical.get("source_tree_sha256") != _source_tree_sha256(executed)
            or publication != _publication_record(reviewed_prior)
            or rebind.get("publication_binding_sha256")
            != _canonical_sha256(reviewed_prior)
            or manifest.get("artifacts") != rebind.get("publication_artifacts")
            or rebind.get("redrawn_artifacts") != {}
            or rebind.get("numerical_evidence_reused") is not True
            or rebind.get("prior_published_artifacts_retained") is not True
            or rebind.get("execution_identity_retained") is not True
        ):
            raise RuntimeError("engineer prior v3 publication binding is inconsistent")
        return deepcopy(dict(reviewed_prior))
    if (
        not isinstance(publication, Mapping)
        or not isinstance(rebind, Mapping)
        or rebind.get("schema") != "scnsim.engineer_publication_rebind.v2"
    ):
        raise RuntimeError("engineer prior publication record is unsupported")
    old_artifacts = rebind.get("execution_artifacts")
    if not isinstance(old_artifacts, Mapping) or rebind.get("execution_payload_sha256") != _canonical_sha256(
        _execution_payload(manifest, old_artifacts)
    ):
        raise RuntimeError("engineer prior execution payload seal is invalid")
    current = deepcopy(dict(executed))
    source_tree = current.get("source_tree")
    if not isinstance(source_tree, dict):
        raise RuntimeError("engineer prior source tree is malformed")
    files = _mapping(source_tree.get("files"), role="prior source-tree")
    _apply_v2_delta(files, rebind.get("changed_source_files"), role="source-tree")
    source_tree["files"] = files
    source_tree["sha256"] = _canonical_sha256(files)
    if source_tree["sha256"] != rebind.get("publication_source_tree_sha256"):
        raise RuntimeError("engineer prior source-tree digest is inconsistent")
    sources = _mapping(current.get("sources"), role="prior teaching-source")
    cells = _mapping(current.get("cells"), role="prior cell")
    _apply_v2_delta(sources, rebind.get("changed_teaching_sources"), role="teaching-source")
    _apply_v2_delta(cells, rebind.get("changed_cells"), role="cell")
    current["sources"] = sources
    current["cells"] = cells
    current["generator_sha256"] = rebind.get("publication_generator_sha256")
    if (
        _canonical_sha256(executed) != rebind.get("execution_binding_sha256")
        or _source_tree_sha256(executed) != rebind.get("execution_source_tree_sha256")
        or _publication_record(current) != dict(publication)
        or _canonical_sha256(current) != rebind.get("publication_binding_sha256")
        or manifest.get("artifacts") != rebind.get("publication_artifacts")
        or set(old_artifacts) != set(manifest.get("artifacts", {}))
        or rebind.get("execution_generator_sha256")
        != _execution_generator(manifest.get("execution"), executed)
        or rebind.get("numerical_evidence_reused") is not True
        or rebind.get("execution_identity_retained") is not True
    ):
        raise RuntimeError("engineer prior publication binding is inconsistent")
    return current


def _historical_execution(
    manifest: Mapping[str, object], prior_binding: Mapping[str, object]
) -> dict[str, object]:
    executed = manifest.get("binding")
    if not isinstance(executed, Mapping):
        raise RuntimeError("engineer execution binding is malformed")
    prior_rebind = manifest.get("publication_rebind")
    if (
        isinstance(prior_rebind, Mapping)
        and prior_rebind.get("schema") == "scnsim.engineer_publication_rebind.v3"
    ):
        prior_historical = prior_rebind.get("historical_execution")
        artifacts = prior_historical.get("artifacts") if isinstance(prior_historical, Mapping) else None
        payload_sha256 = prior_historical.get("payload_sha256") if isinstance(prior_historical, Mapping) else None
    elif isinstance(prior_rebind, Mapping):
        artifacts = prior_rebind.get("execution_artifacts")
        payload_sha256 = prior_rebind.get("execution_payload_sha256")
    else:
        artifacts = manifest.get("artifacts")
        payload_sha256 = _canonical_sha256(_execution_payload(manifest, artifacts)) if isinstance(artifacts, Mapping) else None
    if (
        not isinstance(artifacts, Mapping)
        or not isinstance(payload_sha256, str)
        or payload_sha256 != _canonical_sha256(_execution_payload(manifest, artifacts))
    ):
        raise RuntimeError("engineer historical execution evidence is inconsistent")
    prior_artifacts = manifest.get("artifacts")
    if not isinstance(prior_artifacts, Mapping):
        raise RuntimeError("engineer prior publication artifacts are malformed")
    redrawn = {
        name: {
            "execution_sha256": artifacts[name],
            "prior_publication_sha256": prior_artifacts[name],
        }
        for name in sorted(prior_artifacts)
        if artifacts.get(name) != prior_artifacts[name]
    }
    return {
        "binding_sha256": _canonical_sha256(executed),
        "generator_sha256": _execution_generator(manifest.get("execution"), executed),
        "source_tree_sha256": _source_tree_sha256(executed),
        "payload_sha256": payload_sha256,
        "artifacts": dict(artifacts),
        "prior_redrawn_artifacts": redrawn,
        "prior_publication_binding_sha256": _canonical_sha256(prior_binding),
        "prior_publication_artifacts": dict(prior_artifacts),
    }


def _transition_for(
    current: Mapping[str, object], rebind: Mapping[str, object] | None = None
) -> tuple[str, Mapping[str, object]]:
    matches = [
        (name, transition)
        for name, transition in _reviewed_transitions().items()
        if transition.get("publication_binding_sha256") == _canonical_sha256(current)
        and (rebind is None or transition.get("prior_manifest_sha256") == rebind.get("prior_manifest_sha256"))
    ]
    if len(matches) != 1:
        raise RuntimeError("engineer publication binding has no unique reviewed transition")
    return matches[0]


def _transition_deltas(
    prior: Mapping[str, object], current: Mapping[str, object]
) -> dict[str, object]:
    prior_tree = prior.get("source_tree")
    current_tree = current.get("source_tree")
    if not isinstance(prior_tree, Mapping) or not isinstance(current_tree, Mapping):
        raise RuntimeError("engineer source-tree binding is malformed")
    return {
        "generator": {
            "before_sha256": prior.get("generator_sha256"),
            "after_sha256": current.get("generator_sha256"),
        },
        "source_files": _mapping_delta(
            prior_tree.get("files"), current_tree.get("files"), role="source-tree"
        ),
        "teaching_sources": _mapping_delta(
            prior.get("sources"), current.get("sources"), role="teaching-source"
        ),
        "cell_ids": {
            "before": prior.get("cell_ids"),
            "after": current.get("cell_ids"),
        },
        "cells": _mapping_delta(prior.get("cells"), current.get("cells"), role="cell"),
    }


def _prior_binding_from_transition(
    current: Mapping[str, object], transition: Mapping[str, object]
) -> dict[str, object]:
    """Reverse one exact reviewed delta to recover its trusted prior binding."""

    prior = deepcopy(dict(current))
    deltas = transition.get("deltas")
    if not isinstance(deltas, Mapping) or set(deltas) != _DELTA_FIELDS:
        raise RuntimeError("engineer transition delta is malformed")

    generator = deltas.get("generator")
    if (
        not isinstance(generator, Mapping)
        or set(generator) != {"before_sha256", "after_sha256"}
        or prior.get("generator_sha256") != generator.get("after_sha256")
        or not isinstance(generator.get("before_sha256"), str)
    ):
        raise RuntimeError("engineer transition generator delta is inconsistent")
    prior["generator_sha256"] = generator["before_sha256"]

    def reverse(values: object, rows: object, *, role: str) -> dict[str, str]:
        restored = _mapping(values, role=role)
        if not isinstance(rows, Mapping):
            raise RuntimeError(f"engineer transition {role} delta is malformed")
        for key, row in rows.items():
            if (
                not isinstance(key, str)
                or not isinstance(row, Mapping)
                or set(row) != {"before_sha256", "after_sha256"}
                or not isinstance(row.get("before_sha256"), str)
                or not isinstance(row.get("after_sha256"), str)
                or restored.get(key, "absent") != row["after_sha256"]
            ):
                raise RuntimeError(f"engineer transition {role} delta is inconsistent")
            if row["before_sha256"] == "absent":
                restored.pop(key, None)
            else:
                restored[key] = str(row["before_sha256"])
        return restored

    source_tree = prior.get("source_tree")
    if not isinstance(source_tree, dict):
        raise RuntimeError("engineer transition source tree is malformed")
    files = reverse(
        source_tree.get("files"), deltas.get("source_files"), role="source-tree"
    )
    source_tree["files"] = files
    source_tree["sha256"] = _canonical_sha256(files)
    prior["sources"] = reverse(
        prior.get("sources"), deltas.get("teaching_sources"), role="teaching-source"
    )
    cell_ids = deltas.get("cell_ids")
    if (
        not isinstance(cell_ids, Mapping)
        or set(cell_ids) != {"before", "after"}
        or not isinstance(cell_ids.get("before"), list)
        or not isinstance(cell_ids.get("after"), list)
        or any(
            not isinstance(identifier, str)
            for identifiers in (cell_ids["before"], cell_ids["after"])
            for identifier in identifiers
        )
        or prior.get("cell_ids") != cell_ids["after"]
    ):
        raise RuntimeError("engineer transition cell-ID order is inconsistent")
    prior["cell_ids"] = list(cell_ids["before"])
    prior["cells"] = reverse(
        prior.get("cells"), deltas.get("cells"), role="cell"
    )
    if _canonical_sha256(prior) != transition.get("prior_publication_binding_sha256"):
        raise RuntimeError("engineer transition does not recover its exact prior binding")
    return prior


def _rebind_record(
    *,
    prior_manifest_sha256: str,
    prior_binding: Mapping[str, object],
    current: Mapping[str, object],
    historical: Mapping[str, object],
    artifacts: Mapping[str, object],
    prior_rebind: object,
) -> dict[str, object]:
    _, transition = _transition_for(current)
    deltas = _transition_deltas(prior_binding, current)
    if (
        transition.get("prior_manifest_sha256") != prior_manifest_sha256
        or transition.get("prior_publication_binding_sha256") != _canonical_sha256(prior_binding)
        or transition.get("deltas") != deltas
        or transition.get("artifacts") != dict(artifacts)
        or transition.get("historical_execution") != dict(historical)
        or transition.get("prior_publication_rebind_sha256")
        != (_canonical_sha256(prior_rebind) if isinstance(prior_rebind, Mapping) else None)
    ):
        raise RuntimeError("engineer candidate differs from the exact reviewed transition")
    if historical.get("prior_publication_artifacts") != dict(artifacts):
        raise RuntimeError("source-only publication must retain exact prior published artifacts")
    return {
        "schema": "scnsim.engineer_publication_rebind.v3",
        "kind": "reviewed_source_only_transition",
        "transition_id": TRANSITION_ID,
        "prior_manifest_sha256": prior_manifest_sha256,
        "prior_publication_binding_sha256": _canonical_sha256(prior_binding),
        "publication_binding_sha256": _canonical_sha256(current),
        "deltas": deltas,
        "historical_execution": dict(historical),
        "prior_publication_rebind_sha256": (
            _canonical_sha256(prior_rebind) if isinstance(prior_rebind, Mapping) else None
        ),
        "publication_artifacts": dict(artifacts),
        "redrawn_artifacts": {},
        "numerical_evidence_reused": True,
        "prior_published_artifacts_retained": True,
        "execution_identity_retained": True,
    }


def check_publication_binding(manifest: Mapping[str, object], current: Mapping[str, object]) -> None:
    """Verify one exact reviewed transition from trusted published evidence."""

    publication = manifest.get("publication_binding")
    rebind = manifest.get("publication_rebind")
    if manifest.get("binding") == current:
        if publication is not None or rebind is not None:
            raise RuntimeError("current-source engineer artifacts have unexpected rebind metadata")
        return
    if not isinstance(rebind, Mapping) or publication != _publication_record(current):
        raise RuntimeError("engineer artifact publication binding is stale")
    _, transition = _transition_for(current, rebind)
    artifacts = manifest.get("artifacts")
    historical = rebind.get("historical_execution")
    if not isinstance(artifacts, Mapping) or not isinstance(historical, Mapping):
        raise RuntimeError("engineer publication evidence is malformed")
    execution_artifacts = historical.get("artifacts")
    if (
        not isinstance(execution_artifacts, Mapping)
        or historical.get("payload_sha256")
        != _canonical_sha256(_execution_payload(manifest, execution_artifacts))
        or historical.get("binding_sha256") != _canonical_sha256(manifest.get("binding"))
        or historical.get("generator_sha256")
        != _execution_generator(manifest.get("execution"), manifest.get("binding"))
        or historical.get("source_tree_sha256") != _source_tree_sha256(manifest.get("binding"))
        or dict(historical) != transition.get("historical_execution")
        or transition.get("artifacts") != dict(artifacts)
        or rebind.get("deltas") != transition.get("deltas")
    ):
        raise RuntimeError("engineer historical execution or transition evidence is invalid")
    expected = {
        "schema": "scnsim.engineer_publication_rebind.v3",
        "kind": "reviewed_source_only_transition",
        "transition_id": TRANSITION_ID,
        "prior_manifest_sha256": transition.get("prior_manifest_sha256"),
        "prior_publication_binding_sha256": transition.get("prior_publication_binding_sha256"),
        "publication_binding_sha256": _canonical_sha256(current),
        "deltas": transition.get("deltas"),
        "historical_execution": dict(historical),
        "prior_publication_rebind_sha256": transition.get("prior_publication_rebind_sha256"),
        "publication_artifacts": dict(artifacts),
        "redrawn_artifacts": {},
        "numerical_evidence_reused": True,
        "prior_published_artifacts_retained": True,
        "execution_identity_retained": True,
    }
    if dict(rebind) != expected:
        raise RuntimeError("engineer publication rebind evidence is malformed")


def _prepare_rebind(
    path: Path,
    current: Mapping[str, object],
    *,
    validate_historical: object | None = None,
) -> bytes:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{path} is not an engineer manifest")
    name, transition = _transition_for(current)
    if name != path.name or hashlib.sha256(raw).hexdigest() != transition.get("prior_manifest_sha256"):
        raise RuntimeError(f"{path} is not the exact trusted prior published manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts != transition.get("artifacts"):
        raise RuntimeError(f"{path} prior published artifacts are inconsistent")
    actual_artifacts = {
        artifact: hashlib.sha256((path.parent / artifact).read_bytes()).hexdigest()
        for artifact in artifacts
    }
    if actual_artifacts != dict(artifacts):
        raise RuntimeError(f"{path} prior published artifact bytes changed")
    if validate_historical is not None:
        validate_historical(manifest, artifact_sha256=actual_artifacts)  # type: ignore[operator]
    prior_binding = _prior_publication_binding(
        manifest,
        reviewed_prior=_prior_binding_from_transition(current, transition),
    )
    historical = _historical_execution(manifest, prior_binding)
    previous_rebind = manifest.get("publication_rebind")
    manifest["publication_binding"] = _publication_record(current)
    manifest["publication_rebind"] = _rebind_record(
        prior_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        prior_binding=prior_binding,
        current=current,
        historical=historical,
        artifacts=artifacts,
        prior_rebind=previous_rebind,
    )
    check_publication_binding(manifest, current)
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def _publish_all(candidates: list[tuple[Path, bytes]]) -> None:
    """Stage every validated candidate before per-file atomic replacement."""

    staged: list[tuple[Path, Path]] = []
    try:
        for path, data in candidates:
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((path, temporary))
            if temporary.read_bytes() != data:
                raise RuntimeError(f"staged engineer manifest bytes changed: {path.name}")
        for path, temporary in staged:
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebind", action="store_true", help="publish the exact reviewed source-only transition")
    parser.add_argument("--transition-id", help="select the exact reviewed transition record")
    args = parser.parse_args()
    if not args.rebind or args.transition_id != TRANSITION_ID:
        parser.error(f"--rebind --transition-id {TRANSITION_ID} is required")
    root = Path(__file__).resolve().parents[1]
    candidates: list[tuple[Path, bytes]] = []
    for chapter in range(1, 9):
        module = importlib.import_module(f"generate_engineer_chapter{chapter}")
        path = root / "examples" / "engineer" / "figures" / f"chapter-{chapter:02d}-artifacts.json"
        validator = module.validate_published_manifest if chapter == 8 else None
        candidates.append(
            (path, _prepare_rebind(path, module._binding(), validate_historical=validator))
        )
    _publish_all(candidates)
    for path, _ in candidates:
        print(path)


if __name__ == "__main__":
    main()
