"""Verify or rebind engineer artifacts after runtime or presentation changes.

The original ``binding`` and ``execution`` records remain the authority for
the source that produced numerical evidence. ``publication_binding`` names
the current package, teaching-cell, and generator bytes checked for
publication. Version 2 additionally preserves the original artifact hashes
while binding Plotly redraws from retained exact arrays; neither form relabels
an old Result as a current-source execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Mapping


PUBLICATION_SOURCE_DIFF = (
    "src/scnsim/__init__.py",
    "src/scnsim/_analysis.py",
    "src/scnsim/_backend.py",
    "src/scnsim/_canonical.py",
    "src/scnsim/_evidence.py",
    "src/scnsim/_execution.py",
    "src/scnsim/_immutable_values.py",
    "src/scnsim/_julia/runtime.json",
    "src/scnsim/_julia/src/SCNSimBackend.jl",
    "src/scnsim/_julia/src/compile.jl",
    "src/scnsim/_julia/src/direct_quantities.jl",
    "src/scnsim/_julia/src/optimization.jl",
    "src/scnsim/_julia/src/quantity_core.jl",
    "src/scnsim/_julia/src/result_artifacts.jl",
    "src/scnsim/_julia/src/terminal.jl",
    "src/scnsim/_julia/src/view.jl",
    "src/scnsim/_julia/src/wire.jl",
    "src/scnsim/_numeric_presentation.py",
    "src/scnsim/_physical_values.py",
    "src/scnsim/_report.py",
    "src/scnsim/_result_decode.py",
    "src/scnsim/_schemas/identity-common.schema.json",
    "src/scnsim/_schemas/identity-v2.schema.json",
    "src/scnsim/_workspace.py",
    "src/scnsim/authoring.py",
    "src/scnsim/errors.py",
    "src/scnsim/presentation.py",
    "src/scnsim/results.py",
    "src/scnsim/runtime.py",
    "src/scnsim/specs.py",
)

# These are the exact reviewed execution-to-presentation transitions. A later
# teaching-code change must be classified here deliberately; rebind cannot
# turn an arbitrary changed computation cell into presentation-only evidence.
PRESENTATION_CELL_TRANSITIONS = {
    "ch1-show-baseline-root": ("d601449ff80daf8573e06338d758786ed7195a66a07ed7b07ef9dc2da999952a", "26df437286c6d44080be9de2eb42e83cdcd2b1ada90f7b025830da9d375888f5"),
    "ch1-show-baseline-s11": ("5c98f5ea1d4c7978a4d01664a184edd2ea57574539bb954e4d8dd77dc542082e", "62b087ee60c58bf2e8e8eac1837fe5f4aa1b27faa78a0c32946aa5b7a6bf7513"),
    "ch1-show-selected-root": ("04f52d32c5143afb4b67a18f69f4d39ac854a3f3f7741871bcc1779899ddc280", "48fd0acfa72f1b3aacc94c1204b2d62d10208e66f5ead4613c95b183a5fd2d42"),
    "ch1-show-selected-s11": ("2540c88e3bba52f7db7b3e81a7968f8ee0d6ff0e2085083f04d9b403e7b015ad", "4aa2594f2d57c5b59c2eed6bd607708d84235bca55028e118487b26e00fbe362"),
    "ch2-evaluate-winner": ("509d52dab87f08b001b85b2bd62ca85f4b584b74d1e894b3f6f1cb82ef1b11bc", "c4fdcf94b27353d2e38b8e7144eb95f929cbccca59b7db34bdf1e87676adce0d"),
    "ch2-run-and-show-listed-points": ("2d41893956b59ede861b26d87f3748c017f0a87ee21fbb6b10095ed3f15b50b7", "2ef8d8af15894aadc876cd1b0744c7e0f749ef1f62ce66c92eaf1fff72855e0c"),
    "ch2-show-capacitance-sweep": ("c78015638340211210b761f2a864c8c322c07868dcb746cb48b71dd738551205", "084fd3a70ad64ebf1270f1c4c3e25eeae318f8f9961da601bf9126e21e0854bc"),
    "ch2-show-optional-grid": ("599baa5a2aefc846e166e8bf4f8dc5503aa2c5a9d1f0969432eaee5a76763ff1", "3c5ea27e08b6e2d7f46a188f5f58f4bdf59b8795f2c0c123b902a99420918f87"),
    "ch3-show-direct-s21": ("20add56295f4c2324709d47c4d293fa8feafcf0c44a5432e29f3349b52fb6714", "b57e18884f3adc0e0180fb110f75f22bd02b87bbb9911307ff977acb3007a04b"),
    "ch4-evaluate-response-element": ("9628c6b35300eba482bd09ff7d29127f52b82b8f82d952a36fc3e399f72e3b1d", "0418fa324a978a0136c8554c579e103815e0bc36dc7d702781b3a4a39c7d8b09"),
    "ch4-show-independent-direct-and-hb": ("66499e42dc29c8ef455447f5af475759ad1297f397d03d2ce2b1ff93cdd277d9", "73847cf2d53237c1046d68f16a538e093b61752797384d2b34fe784f3505a3f4"),
    "ch4-solve-ptc-direct": ("d1763605bc5f07ce5424f7d5b82f9eeb46dbe2f61cb4c6a7905255a05dae3b77", "5306c80cfda2aa80213f669800398e2a2c4ce7fc8fa2dde7f7663bb4705dacb2"),
    "ch7-show-named-reflection-examples": ("8476b905460e98f35aed328206edd105caa61e0a960e1bba91cb2cce9434031b", "2355b94eff2976153c2406c8f1905120dcd150f555aa7ce9dcb5ae363adbf068"),
    "ch7-solve-three-points": ("bf9215dc3d92e50fa562fce2c9e94259382376b0db109096bc48819692fa7c11", "c5b7073674116a7b096f28a2b96ebc955f8459a2f01aaced6238209b5e3eb3b2"),
    "ch8-execute-persisted-results": ("30e9dc3dd5794e035b18ea00468d57b02e979faf6ef779236c03637b4350d5f7", "3a37698a0225819aa901b8b175e0dd5379f82b94be1d3833abca8fc29a8676eb"),
    "ch8-resolve-only-after-restart": ("ad1162845ddf397348f3ce256e200b90320e75ba51903fba0af5d4b5bf66c1dc", "99759d0dd820399aec176014e3192431ec44082542795e60450e63bbc4c53a2f"),
}

PRESENTATION_TEACHING_TRANSITIONS = {
    "examples/engineer/chapter-01/_03-s11.qmd": ("00a647d99fb786a4d36d4df38a78f8a64353e7c8515dbd8ffd63d2e243731f6b", "8064aa76398b60e6608ac86b9006a05dfdfa9d05fbeb593332b4d10b96cfb71c"),
    "examples/engineer/chapter-01/_04-root.qmd": ("55d2d2a75184d0a42b9bec951b8a9d84b07c42e0f9231e4e482230aea72f1bcb", "7d6b2c5dca7b5fea676ac7406b77c00354b407c8788c284018b6e095a103ba1d"),
    "examples/engineer/chapter-01/_05-change-capacitance.qmd": ("2f79b5ae89cf2dfd9e0d89c4e534f97eb149be324f0d88c0a5afc0be7ac7e69c", "34a2cb8e72cf1973877e339e1be8c60a2dced4c5bf5838e81ada880bfa8d3a99"),
    "examples/engineer/chapter-02/_02-sweep.qmd": ("f85ebce62ab60d8a464a895bcdb2111e32a356062591862fecb00186d67d8ddf", "0c2085d85ead63ed062ce27c566c1588f76e11092de9fd3b2e7fc8873f47feb6"),
    "examples/engineer/chapter-02/_03-optimize.qmd": ("993d6308cc575cf0917eeb152adaf642f2765b8ce6a01237f988c7525ab15cb5", "ee221421afda1bbbf7964ad051f2e62fbc9c2a31cad7eca4129e9b5675a577c5"),
    "examples/engineer/chapter-02/_04-optional-spaces.qmd": ("2db849772c7880a6ec9802a467882eb663f117b4c5564c1e5c6764f09a65d760", "988be9375e8eebb6baac1e1747ca75c1baa467562e5d888529932e6efe82cd37"),
    "examples/engineer/chapter-03/_03-s21.qmd": ("f66f10171a3539d08b6de62c36c2a29f6df599ad2089a11a0b0cc44ba5b89521", "df0890c91f0a526e7f6081bc9016345abbe5cb89801e0fafd8babda9f8cfe634"),
    "examples/engineer/chapter-04/_03-ptc-core.qmd": ("04abb6f9eba466ab7f67e4454f66628f1196e3f888adfd380839ee19624ea29b", "2bd3d00aab0dce789d528c0f14a6c1bdb9e414de13a55e1ea479f00ec8d4907e"),
    "examples/engineer/chapter-04/_05-optional-quantity.qmd": ("49de9323165c00efd48f7e95b91ffe4921307d14a873ca4f933366ede78a8905", "1f03d2df1b1e0ff95c4fa9082f5ca4090949ce91fe4dac292e812a0297a16719"),
    "examples/engineer/chapter-04/_06-optional-hb.qmd": ("c51dfd0ee42149a3760e44a0bf60119255038647384ffbc2dbf25beca912391b", "b888ecae0eaa3f3f9255732afce00c08095aa597e1c4798b09083732b4e75117"),
    "examples/engineer/chapter-07/_04-parameter-points.qmd": ("115b3f3f4b4c51955041d91fcd83c7d9401cb0950e1c6abbe710c87b605e4000", "3e55017d9f5c09fa3133bfdc1e7ef21b6ef7ca024b364fdca5ebf719d8ca6e85"),
    "examples/engineer/chapter-08/_01-persist.qmd": ("10055ee0163109ac811c700232f10ceb0e8b7e4fd658685aa7464aac03deabb1", "9480dd61a5f96bb010fc57b3e5751e421a5b3c493709acfcc9f519dfd934a642"),
    "examples/engineer/chapter-08/_02-resolve.qmd": ("9afa4ea41a4031b24b1294f7e5e3eda5e7733c79ded4ca69e6bc41051b63b841", "d1de6f99e1628c720bb4ebb6d83347976cc79cdaeee3bdbe356f011c45429d64"),
}


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


def _presentation_environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "plotly": importlib.metadata.version("plotly"),
        "kaleido": importlib.metadata.version("kaleido"),
    }


def _changed_sources(executed: object, publication: object) -> dict[str, dict[str, str]]:
    old = _source_files(executed)
    current = _source_files(publication)
    removed = set(old) - set(current)
    if removed:
        raise RuntimeError(f"engineer source-tree files disappeared during presentation rebind: {sorted(removed)!r}")
    changed = {
        path: {
            "executed_sha256": old.get(path, "absent"),
            "publication_sha256": current[path],
        }
        for path in sorted(current)
        if old.get(path) != current[path]
    }
    if tuple(changed) != PUBLICATION_SOURCE_DIFF:
        raise RuntimeError(f"engineer publication source delta is outside the authorized set: {tuple(changed)!r}")
    return changed


def _mapping_delta(
    executed: object,
    publication: object,
    *,
    allowed: Mapping[str, tuple[str, str]],
    role: str,
) -> dict[str, dict[str, str]]:
    if not isinstance(executed, Mapping) or not isinstance(publication, Mapping):
        raise RuntimeError("engineer teaching binding is malformed")
    removed = set(executed) - set(publication)
    if removed:
        raise RuntimeError(f"engineer teaching binding removed entries: {sorted(removed)!r}")
    changed = {
        key: {
            "executed_sha256": executed.get(key, "absent"),
            "publication_sha256": publication[key],
        }
        for key in sorted(publication)
        if executed.get(key) != publication[key]
    }
    invalid = {
        key: value
        for key, value in changed.items()
        if allowed.get(key)
        != (value["executed_sha256"], value["publication_sha256"])
    }
    if invalid:
        raise RuntimeError(
            f"engineer {role} delta is not an exact reviewed presentation transition: {invalid!r}"
        )
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
    changed = _changed_sources(executed, current)
    execution = manifest.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("source_tree_sha256") != _source_tree_sha256(executed)
        or execution.get("cells") != executed.get("cells")
        or not isinstance(execution.get("generator_sha256"), str)
    ):
        raise RuntimeError("engineer historical execution binding changed during publication rebind")
    execution_artifacts = rebind.get("execution_artifacts")
    publication_artifacts = manifest.get("artifacts")
    if (
        not isinstance(execution_artifacts, Mapping)
        or not isinstance(publication_artifacts, Mapping)
        or set(execution_artifacts) != set(publication_artifacts)
    ):
        raise RuntimeError("engineer presentation artifact binding is malformed")
    execution_payload_sha256 = rebind.get("execution_payload_sha256")
    if (
        not isinstance(execution_payload_sha256, str)
        or execution_payload_sha256
        != _canonical_sha256(_execution_payload(manifest, execution_artifacts))
    ):
        raise RuntimeError("engineer historical execution payload seal is invalid")
    presentation_environment = rebind.get("presentation_environment")
    if (
        not isinstance(presentation_environment, Mapping)
        or set(presentation_environment) != {"python", "plotly", "kaleido"}
        or any(not isinstance(value, str) for value in presentation_environment.values())
    ):
        raise RuntimeError("engineer presentation environment is malformed")
    expected = {
        "schema": "scnsim.engineer_publication_rebind.v2",
        "kind": "plotly_presentation_from_retained_evidence",
        "prior_manifest_sha256": rebind.get("prior_manifest_sha256"),
        "execution_binding_sha256": _canonical_sha256(executed),
        "publication_binding_sha256": _canonical_sha256(current),
        "execution_generator_sha256": execution.get("generator_sha256"),
        "publication_generator_sha256": current.get("generator_sha256"),
        "execution_source_tree_sha256": _source_tree_sha256(executed),
        "publication_source_tree_sha256": _source_tree_sha256(current),
        "changed_source_files": changed,
        "changed_teaching_sources": _mapping_delta(
            executed.get("sources"), current.get("sources"),
            allowed=PRESENTATION_TEACHING_TRANSITIONS, role="teaching-source",
        ),
        "changed_cells": _mapping_delta(
            executed.get("cells"), current.get("cells"),
            allowed=PRESENTATION_CELL_TRANSITIONS, role="cell",
        ),
        "execution_payload_sha256": execution_payload_sha256,
        "execution_artifacts": dict(execution_artifacts),
        "publication_artifacts": dict(publication_artifacts),
        "redrawn_presentation_artifacts": sorted(
            name for name in publication_artifacts
            if execution_artifacts[name] != publication_artifacts[name]
        ),
        "presentation_environment": dict(presentation_environment),
        "numerical_evidence_reused": True,
        "presentation_redrawn_without_execution": True,
        "execution_identity_retained": True,
    }
    if not isinstance(expected["prior_manifest_sha256"], str) or dict(rebind) != expected:
        raise RuntimeError("engineer publication rebind evidence is malformed")


def _rebind(path: Path, current: Mapping[str, object], artifact_names: tuple[str, ...]) -> None:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{path} is not an engineer manifest")
    executed = manifest.get("binding")
    if not isinstance(executed, Mapping):
        raise RuntimeError(f"{path} has no execution binding")
    changed = _changed_sources(executed, current)
    execution = manifest.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("source_tree_sha256") != _source_tree_sha256(executed)
        or execution.get("cells") != executed.get("cells")
        or not isinstance(execution.get("generator_sha256"), str)
    ):
        raise RuntimeError(f"{path} has inconsistent historical execution evidence")
    previous_rebind = manifest.get("publication_rebind")
    if isinstance(previous_rebind, Mapping) and previous_rebind.get("schema") == "scnsim.engineer_publication_rebind.v2":
        old_artifacts = previous_rebind.get("execution_artifacts")
        prior_manifest_sha256 = previous_rebind.get("prior_manifest_sha256")
        presentation_environment = previous_rebind.get("presentation_environment")
        execution_payload_sha256 = previous_rebind.get("execution_payload_sha256")
    else:
        old_artifacts = manifest.get("artifacts")
        prior_manifest_sha256 = hashlib.sha256(raw).hexdigest()
        presentation_environment = _presentation_environment()
        execution_payload_sha256 = None
    if not isinstance(old_artifacts, Mapping) or set(old_artifacts) != set(artifact_names):
        raise RuntimeError(f"{path} has an unexpected historical artifact inventory")
    if not isinstance(prior_manifest_sha256, str):
        raise RuntimeError(f"{path} has no prior manifest identity")
    if not isinstance(presentation_environment, Mapping):
        raise RuntimeError(f"{path} has no presentation environment")
    computed_execution_payload_sha256 = _canonical_sha256(
        _execution_payload(manifest, old_artifacts)
    )
    if execution_payload_sha256 is None:
        execution_payload_sha256 = computed_execution_payload_sha256
    elif execution_payload_sha256 != computed_execution_payload_sha256:
        raise RuntimeError(f"{path} historical execution payload changed")
    current_artifacts = {
        name: hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
        for name in artifact_names
    }
    manifest["artifacts"] = current_artifacts
    manifest["publication_binding"] = _publication_record(current)
    manifest["publication_rebind"] = {
        "schema": "scnsim.engineer_publication_rebind.v2",
        "kind": "plotly_presentation_from_retained_evidence",
        "prior_manifest_sha256": prior_manifest_sha256,
        "execution_binding_sha256": _canonical_sha256(executed),
        "publication_binding_sha256": _canonical_sha256(current),
        "execution_generator_sha256": execution.get("generator_sha256"),
        "publication_generator_sha256": current.get("generator_sha256"),
        "execution_source_tree_sha256": _source_tree_sha256(executed),
        "publication_source_tree_sha256": _source_tree_sha256(current),
        "changed_source_files": changed,
        "changed_teaching_sources": _mapping_delta(
            executed.get("sources"), current.get("sources"),
            allowed=PRESENTATION_TEACHING_TRANSITIONS, role="teaching-source",
        ),
        "changed_cells": _mapping_delta(
            executed.get("cells"), current.get("cells"),
            allowed=PRESENTATION_CELL_TRANSITIONS, role="cell",
        ),
        "execution_payload_sha256": execution_payload_sha256,
        "execution_artifacts": dict(old_artifacts),
        "publication_artifacts": current_artifacts,
        "redrawn_presentation_artifacts": sorted(
            name for name in current_artifacts if old_artifacts[name] != current_artifacts[name]
        ),
        "presentation_environment": dict(presentation_environment),
        "numerical_evidence_reused": True,
        "presentation_redrawn_without_execution": True,
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
        if not isinstance(manifest, Mapping):
            raise RuntimeError(f"{path} is not an engineer manifest")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError(f"{path} has no artifact inventory")
        _rebind(path, current, tuple(artifacts))
        print(path)


if __name__ == "__main__":
    main()
