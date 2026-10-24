"""Plan-bound Workspace roots, leaves, locking and publication lifecycle."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..canonical import canonical_json_bytes as _canonical_bytes, sha256_hex as _sha256
from ..errors import UnsupportedEvidenceVersionError, WorkspaceCommitIndeterminateError, WorkspacePlanReplacedError, WorkspaceVersioningDowngradeForbidden
from .documents import (
    plan_workspace_document,
    replaceable_workspace_document,
    versioned_workspace_document,
)
from ._os_io import lock_fd, unlock_fd, replace_path, require_workspace_platform
from .primitives import _relative_path, _inside
from .storage import _LEAF_STAGING, _WorkspacePublishIndeterminate, _atomic_write, _decode_bytes, _fsync_directory, _fsync_tree, _load_canonical, _publish_workspace_state, _remove_leaf
from .operation_lease import _exclusive_activity_lease
from .validation.common import _UUID4, _integrity, _valid_sha, _valid_uuid


def _old_evidence(path, found=None):
    return UnsupportedEvidenceVersionError(
        "Workspace evidence was written by a pre-1.0 development build that 1.0 does not support; use a new Workspace and recompute.",
        stage="workspace", evidence={"path": str(path), "found_evidence_version": found,
                                     "supported_evidence_version": 3,
                                     "action": "use a new Workspace and recompute"})


def _preflight_workspace(root: Path) -> None:
    """Admit only v3 evidence before creating directories, locks or databases."""
    from .operation_store import preflight_operation_store

    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise _integrity("Workspace root is missing or symlinked.", path=str(root))
    for name in ("requests", ".zgroup", ".zarray", "zarr.json", "benchmark.json"):
        path = root / name
        if path.exists() or path.is_symlink():
            raise _old_evidence(path)
    state_path = root / "workspace.json"
    if not state_path.exists() and not state_path.is_symlink():
        for parent in (root / "leaves", root):
            if parent.is_symlink():
                raise _integrity("Workspace leaf directory must not be symlinked.", path=str(parent))
            if not parent.is_dir():
                continue
            for child in parent.iterdir():
                if ((parent == root / "leaves" or child.name.startswith("iteration"))
                        and child.is_dir()):
                    if child.is_symlink():
                        raise _integrity("Workspace leaf must not be symlinked.", path=str(child))
                    document = _load_canonical(child / "workspace.json")
                    if document.get("evidence_version") != 3:
                        raise _old_evidence(child, document.get("evidence_version"))
                    preflight_operation_store(child / "operations")
        return
    state = _load_canonical(state_path)
    _assert_root_envelope(state)
    if state["kind"] == "replaceable_workspace":
        directories = [state["active_leaf"]["directory"]]
    else:
        iterations = state["iterations"]
        if (not isinstance(iterations, list)
                or any(not isinstance(item, dict) or not isinstance(item.get("directory"), str)
                       for item in iterations)):
            raise _integrity("Versioned workspace iterations are malformed.")
        directories = [item["directory"] for item in iterations]
    for directory in directories:
        leaf = _inside(root, directory)
        if leaf.is_symlink():
            raise _integrity("Workspace leaf must not be symlinked.", path=str(leaf))
        document = _load_canonical(leaf / "workspace.json")
        if document.get("evidence_version") != 3:
            raise _old_evidence(leaf, document.get("evidence_version"))
        for name in ("requests", ".zgroup", ".zarray", "zarr.json", "benchmark.json"):
            legacy = leaf / name
            if legacy.exists() or legacy.is_symlink():
                raise _old_evidence(legacy)
        preflight_operation_store(leaf / "operations")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

def _new_uuid(*, excluding: str | None = None) -> str:
    value = str(uuid.uuid4())
    while value == excluding:
        value = str(uuid.uuid4())
    return value


@dataclass(frozen=True)
class WorkspaceBinding:
    """One Run's concrete, Plan-bound leaf beneath a stable workspace root."""

    root: Path
    leaf: Path
    plan_sha256: str
    workspace_instance_id: str

    @contextmanager
    def writer(self) -> Iterator[WorkspaceBinding]:
        """Hold the root's exclusive lock for one Workspace mutation."""

        with _workspace_lock(self.root, exclusive=True):
            state = _load_canonical(self.root / "workspace.json")
            _assert_root_envelope(state)
            self.assert_current()
            _validate_active_evidence(self.root, state)
            _finish_root_maintenance(self.root, state)
            self.assert_current()
            yield self

    @contextmanager
    def reader(self) -> Iterator[WorkspaceBinding]:
        """Read one stable chain without acquiring an execution slot."""

        with _workspace_lock(self.root, exclusive=False):
            self.assert_current()
            yield self

    def assert_current(self) -> None:
        """Reject a stale Run before it can read another Plan's leaf."""

        root = _load_canonical(self.root / "workspace.json")
        _assert_root_envelope(root)
        kind = root.get("kind")
        if kind == "replaceable_workspace":
            active = root.get("active_leaf")
            if not isinstance(active, dict):
                raise _integrity("Replaceable workspace lacks an active leaf.")
            expected_directory = f"leaves/{self.workspace_instance_id}"
            matches = (
                active.get("workspace_instance_id") == self.workspace_instance_id
                and active.get("plan_sha256") == self.plan_sha256
                and active.get("directory") == expected_directory
                and self.leaf == self.root / expected_directory
            )
        elif kind == "versioned_workspace":
            iterations = root.get("iterations")
            next_iteration = root.get("next_iteration")
            if not isinstance(iterations, list) or not isinstance(next_iteration, int) or next_iteration < 1:
                raise _integrity("Versioned workspace lacks its iteration index.")
            plans: set[str] = set()
            ordinals: set[int] = set()
            leaf_ids: set[str] = set()
            match: Mapping[str, object] | None = None
            for expected_ordinal, entry in enumerate(iterations, 1):
                if not isinstance(entry, dict) or set(entry) != {"ordinal", "directory", "workspace_instance_id", "plan_sha256"}:
                    raise _integrity("Versioned workspace has an open iteration entry.")
                ordinal = entry.get("ordinal")
                identity = _valid_uuid(entry.get("workspace_instance_id"))
                plan = _valid_sha(entry.get("plan_sha256"))
                if (
                    not isinstance(ordinal, int)
                    or ordinal != expected_ordinal
                    or ordinal in ordinals
                    or plan in plans
                    or identity in leaf_ids
                    or identity == root.get("workspace_instance_id")
                    or entry.get("directory") != f"iteration{ordinal:02d}"
                ):
                    raise _integrity("Versioned workspace iteration index is inconsistent.")
                ordinals.add(ordinal)
                plans.add(plan)
                leaf_ids.add(identity)
                # The root index is shared authority, so its own shape and
                # uniqueness are still checked.  A sibling's leaf evidence is
                # not: a Run is bound to exactly one versioned iteration and
                # read-only APIs must not turn unrelated historical damage
                # into a latest/current selector or a failure of this leaf.
                if identity == self.workspace_instance_id and plan == self.plan_sha256:
                    match = entry
            if sorted(ordinals) != list(range(1, len(ordinals) + 1)) or next_iteration != len(ordinals) + 1:
                raise _integrity("Versioned workspace next_iteration is not canonical.")
            matches = match is not None
            if match is not None:
                directory = match.get("directory")
                ordinal = match.get("ordinal")
                if not isinstance(ordinal, int) or not isinstance(directory, str) or directory != f"iteration{ordinal:02d}":
                    raise _integrity("Versioned workspace points outside an iteration.")
                expected_leaf = self.root / directory
                pre_upgrade_leaf = self.root / "leaves" / self.workspace_instance_id
                if self.leaf == pre_upgrade_leaf:
                    object.__setattr__(self, "leaf", expected_leaf)
                elif self.leaf != expected_leaf:
                    raise WorkspacePlanReplacedError(
                        "This Run's versioned workspace leaf no longer matches its bound iteration.",
                        stage="workspace",
                        evidence={"workspace": str(self.root), "workspace_instance_id": self.workspace_instance_id},
                    )
        else:
            raise _integrity("Workspace root has an unknown kind.", kind=kind)
        if not matches:
            raise WorkspacePlanReplacedError(
                "This Run's workspace leaf was replaced by another topology.",
                stage="workspace",
                evidence={
                    "workspace": str(self.root),
                    "plan_sha256": self.plan_sha256,
                    "workspace_instance_id": self.workspace_instance_id,
                },
            )
        self._verify_leaf()

    def _verify_leaf(self) -> None:
        if self.leaf.is_symlink() or not self.leaf.is_dir():
            raise _integrity("Workspace leaf path is missing or symlinked.", leaf=str(self.leaf))
        leaf = _load_canonical(self.leaf / "workspace.json")
        if (
            set(leaf) != {"schema", "schema_version", "evidence_version", "kind", "workspace_instance_id", "plan_sha256"}
            or
            leaf.get("schema") != "scnsim.workspace"
            or leaf.get("schema_version") != 1
            or leaf.get("evidence_version") != 3
            or leaf.get("kind") != "plan_workspace"
            or leaf.get("workspace_instance_id") != self.workspace_instance_id
            or leaf.get("plan_sha256") != self.plan_sha256
        ):
            raise _integrity("Leaf workspace binding disagrees with its Run.", leaf=str(self.leaf))
        plan = self.leaf / "plan.json"
        if plan.is_symlink() or not plan.is_file() or _sha256(plan.read_bytes()) != self.plan_sha256:
            raise _integrity("Leaf plan bytes do not match its sealed identity.", leaf=str(self.leaf))
        requests = self.leaf / "requests"
        if requests.is_symlink() or (requests.exists() and not requests.is_dir()):
            raise _integrity("Workspace requests path is unsafe.", path=str(requests))


@contextmanager
def _workspace_lock(root: Path, *, exclusive: bool) -> Iterator[None]:
    _preflight_workspace(root)
    require_workspace_platform(stage="workspace")
    if root.is_symlink() or not root.is_dir():
        raise _integrity("Workspace root is missing or symlinked.", path=str(root))
    lock_path = root / ".scnsim.lock"
    if lock_path.is_symlink():
        raise _integrity("Workspace lock path must not be a symlink.", path=str(lock_path))
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    acquired = False
    primary: BaseException | None = None
    try:
        lock_fd(descriptor, exclusive=exclusive)
        acquired = True
        _preflight_workspace(root)
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        if acquired:
            try:
                unlock_fd(descriptor)
            except BaseException as error:
                if primary is not None:
                    primary.add_note(f"Workspace lock unlock also failed: {error!r}")
                else:
                    cleanup_error = error
        try:
            os.close(descriptor)
        except BaseException as error:
            if primary is not None:
                primary.add_note(f"Workspace lock close also failed: {error!r}")
            elif cleanup_error is not None:
                cleanup_error.add_note(f"Workspace lock close also failed: {error!r}")
            else:
                cleanup_error = error
        if primary is None and cleanup_error is not None:
            raise cleanup_error

def bind_workspace(
    workspace: str | os.PathLike[str],
    *,
    plan_sha256: str,
    plan_bytes: bytes,
    versioned: bool,
    commit: Callable[[], None],
) -> WorkspaceBinding:
    """Bind prepared Plan evidence, then invoke its non-failing seal commit."""

    require_workspace_platform(stage="workspace")
    _valid_sha(plan_sha256)
    if _sha256(plan_bytes) != plan_sha256:
        raise _integrity("Sealed Plan bytes do not match their identity.")
    plan = _decode_bytes(plan_bytes, "plan")
    if plan.get("schema") != "scnsim.plan" or plan.get("schema_version") != 2:
        raise _integrity("Plan bytes are not a schema-version 2 Plan envelope.")
    if not callable(commit):
        raise TypeError("workspace commit must be callable")
    root = Path(workspace).expanduser().resolve(strict=False)
    _preflight_workspace(root)
    root.mkdir(parents=True, exist_ok=True)
    with _exclusive_activity_lease(root), _workspace_lock(root, exclusive=True):
        _preflight_workspace(root)
        try:
            state_path = root / "workspace.json"
            if not state_path.exists():
                binding = _recover_or_create_root(root, plan_sha256, plan_bytes, versioned)
            else:
                root_state = _load_canonical(state_path)
                _assert_root_envelope(root_state)
                _validate_active_evidence(root, root_state)
                _finish_root_maintenance(root, root_state)
                root_state = _load_canonical(state_path)
                _assert_root_envelope(root_state)
                kind = root_state.get("kind")
                if kind == "replaceable_workspace":
                    current = _binding_from_replaceable(root, root_state)
                    current._verify_leaf()
                    if not versioned and current.plan_sha256 != plan_sha256:
                        recovered = _adopt_replaceable_orphan(root, root_state, current, plan_sha256)
                        if recovered is not None:
                            binding = recovered
                        else:
                            _cleanup_replaceable_staging(root, current)
                            new = _create_leaf(
                                root / "leaves",
                                plan_sha256,
                                plan_bytes,
                                excluding=str(root_state["workspace_instance_id"]),
                            )
                            updated = dict(root_state)
                            updated["active_leaf"] = _leaf_pointer(new, root)
                            updated["maintenance"] = _retired_leaf_maintenance(current, root)
                            _publish_workspace_state(state_path, _canonical_bytes(updated))
                            binding = new
                    elif versioned:
                        _cleanup_replaceable_staging(root, current)
                        binding = _upgrade_to_versioned(root, root_state, current)
                    else:
                        _cleanup_replaceable_staging(root, current)
                        current.assert_current()
                        binding = current
                elif kind == "versioned_workspace":
                    if not versioned:
                        raise WorkspaceVersioningDowngradeForbidden(
                            "This workspace preserves topology history; choose another workspace for replacement mode.",
                            stage="workspace",
                            evidence={"workspace": str(root)},
                        )
                    binding = _bind_versioned(root, root_state, plan_sha256, plan_bytes)
                else:
                    raise _integrity("Workspace root has an unknown kind.", kind=kind)
        except _WorkspacePublishIndeterminate as exc:
            commit()
            raise WorkspaceCommitIndeterminateError(
                "The workspace pointer may be committed, but durable confirmation failed.",
                stage="workspace_commit",
                evidence={
                    "workspace": str(root),
                    "plan_sha256": plan_sha256,
                    "plan_sealed": True,
                    "workspace_may_have_changed": True,
                },
            ) from exc
        commit()
        try:
            state = _load_canonical(root / "workspace.json")
            _assert_root_envelope(state)
            _finish_root_maintenance(root, state)
        except BaseException:
            # Publication and Plan sealing are already committed. Exact pending
            # cleanup remains visible in the root inventory for the next lock.
            pass
        return binding

def _retired_leaf_maintenance(binding: WorkspaceBinding, root: Path) -> dict[str, object]:
    pointer = _leaf_pointer(binding, root)
    if binding.leaf.parent != root / "leaves":
        raise _integrity("Only a replaceable leaf may become cleanup-pending.")
    return {"kind": "retired_leaf_cleanup", **pointer}

def _finish_root_maintenance(root: Path, state: Mapping[str, object]) -> None:
    maintenance = state.get("maintenance")
    if maintenance is None:
        return
    if not isinstance(maintenance, dict) or set(maintenance) != {
        "kind", "directory", "workspace_instance_id", "plan_sha256"
    } or maintenance.get("kind") != "retired_leaf_cleanup":
        raise _integrity("Workspace maintenance record is open or malformed.")
    identity = _valid_uuid(maintenance.get("workspace_instance_id"))
    plan_sha256 = _valid_sha(maintenance.get("plan_sha256"))
    directory = maintenance.get("directory")
    if directory != f"leaves/{identity}":
        raise _integrity("Workspace maintenance target is not a replaceable leaf.")
    active = state.get("active_leaf")
    if isinstance(active, dict) and active.get("workspace_instance_id") == identity:
        raise _integrity("Workspace maintenance cannot retire the active leaf.")
    leaves = root / "leaves"
    if leaves.exists() and (leaves.is_symlink() or not leaves.is_dir()):
        raise _integrity("Workspace maintenance leaf parent is unsafe.", path=str(leaves))
    target = root / _relative_path(directory)
    if target.exists():
        retired = _recover_leaf(root, target, plan_sha256, expected_directory=str(directory))
        _remove_leaf(retired.leaf)
    if leaves.exists() and not any(leaves.iterdir()):
        leaves.rmdir()
        _fsync_directory(root)
    cleared = dict(state)
    cleared.pop("maintenance", None)
    _atomic_write(root / "workspace.json", _canonical_bytes(cleared))

def _create_root(root: Path, plan_sha256: str, plan_bytes: bytes, versioned: bool) -> WorkspaceBinding:
    if versioned:
        leaf = _create_leaf(root, plan_sha256, plan_bytes, directory="iteration01")
        state = versioned_workspace_document(
            workspace_instance_id=_new_uuid(excluding=leaf.workspace_instance_id),
            iterations=[{"ordinal": 1, **_leaf_pointer(leaf, root)}],
        )
    else:
        leaf = _create_leaf(root / "leaves", plan_sha256, plan_bytes)
        state = replaceable_workspace_document(
            workspace_instance_id=_new_uuid(excluding=leaf.workspace_instance_id),
            leaf_instance_id=leaf.workspace_instance_id,
            plan_sha256=leaf.plan_sha256,
        )
    _publish_workspace_state(root / "workspace.json", _canonical_bytes(state))
    return leaf

def _recover_or_create_root(
    root: Path,
    plan_sha256: str,
    plan_bytes: bytes,
    versioned: bool,
) -> WorkspaceBinding:
    """Finish only a recognizable interrupted initial root creation.

    A final leaf without its root pointer is retained and either adopted when it
    exactly matches this bind or rejected untouched.  Hidden leaf staging is
    the sole disposable initial-creation state.
    """

    _cleanup_initial_leaf_staging(root)
    entries = {
        child.name: child for child in root.iterdir()
        if child.name not in {".scnsim.lock", ".scnsim-operation-lease.lock"}
    }
    if not entries:
        return _create_root(root, plan_sha256, plan_bytes, versioned)
    if versioned and set(entries) == {"iteration01"}:
        leaf = _recover_leaf(root, entries["iteration01"], plan_sha256, expected_directory="iteration01")
        state = versioned_workspace_document(
            workspace_instance_id=_new_uuid(excluding=leaf.workspace_instance_id),
            iterations=[{"ordinal": 1, **_leaf_pointer(leaf, root)}],
        )
        _publish_workspace_state(root / "workspace.json", _canonical_bytes(state))
        return leaf
    if not versioned and set(entries) == {"leaves"}:
        leaves = entries["leaves"]
        if leaves.is_symlink() or not leaves.is_dir():
            raise _integrity("Interrupted replacement workspace has an unsafe leaves directory.")
        children = list(leaves.iterdir())
        if len(children) == 1 and _UUID4.fullmatch(children[0].name) is not None:
            leaf = _recover_leaf(root, children[0], plan_sha256, expected_directory=f"leaves/{children[0].name}")
            state = replaceable_workspace_document(
                workspace_instance_id=_new_uuid(excluding=leaf.workspace_instance_id),
                leaf_instance_id=leaf.workspace_instance_id,
                plan_sha256=leaf.plan_sha256,
            )
            _publish_workspace_state(root / "workspace.json", _canonical_bytes(state))
            return leaf
    raise _integrity("Workspace has unbound initial-creation evidence; refusing to delete it.", workspace=str(root))

def _recover_leaf(root: Path, leaf_path: Path, plan_sha256: str, *, expected_directory: str) -> WorkspaceBinding:
    if leaf_path.is_symlink() or not leaf_path.is_dir():
        raise _integrity("Interrupted workspace leaf is unsafe.", leaf=str(leaf_path))
    leaf = _load_canonical(leaf_path / "workspace.json")
    if (
        set(leaf) != {"schema", "schema_version", "evidence_version", "kind", "workspace_instance_id", "plan_sha256"}
        or leaf.get("schema") != "scnsim.workspace"
        or leaf.get("schema_version") != 1
            or leaf.get("evidence_version") != 3
        or leaf.get("kind") != "plan_workspace"
        or leaf.get("plan_sha256") != plan_sha256
    ):
        raise _integrity("Interrupted workspace leaf cannot be safely adopted.", leaf=str(leaf_path))
    identity = _valid_uuid(leaf.get("workspace_instance_id"))
    binding = WorkspaceBinding(root, leaf_path, plan_sha256, identity)
    if _leaf_pointer(binding, root).get("directory") != expected_directory:
        raise _integrity("Interrupted workspace leaf has an unexpected path.", leaf=str(leaf_path))
    binding._verify_leaf()
    return binding

def _cleanup_initial_leaf_staging(root: Path) -> None:
    parents = [root]
    leaves = root / "leaves"
    if leaves.exists():
        if leaves.is_symlink() or not leaves.is_dir():
            raise _integrity("Initial workspace leaves directory is unsafe.")
        parents.append(leaves)
    for parent in parents:
        for child in parent.iterdir():
            if not child.name.startswith(".staging-leaf-"):
                continue
            if child.is_symlink() or not child.is_dir() or _LEAF_STAGING.fullmatch(child.name) is None:
                raise _integrity("Initial workspace staging path is malformed.", path=str(child))
            shutil.rmtree(child)
        _fsync_directory(parent)
    if leaves.exists() and not any(leaves.iterdir()):
        leaves.rmdir()
    _fsync_directory(root)

def _create_leaf(
    parent: Path,
    plan_sha256: str,
    plan_bytes: bytes,
    *,
    directory: str | None = None,
    excluding: str | None = None,
) -> WorkspaceBinding:
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise _integrity("Workspace leaf parent is unsafe.", path=str(parent))
    parent.mkdir(parents=True, exist_ok=True)
    leaf_id = _new_uuid(excluding=excluding)
    target = parent / (directory or leaf_id)
    staging = parent / f".staging-leaf-{leaf_id}"
    if target.exists() or staging.exists():
        raise _integrity("Workspace leaf allocation would overwrite evidence.", target=str(target))
    staging.mkdir()
    leaf_state = plan_workspace_document(
        workspace_instance_id=leaf_id,
        plan_sha256=plan_sha256,
    )
    _atomic_write(staging / "workspace.json", _canonical_bytes(leaf_state))
    _atomic_write(staging / "plan.json", plan_bytes)
    _fsync_tree(staging)
    replace_path(staging, target)
    _fsync_directory(parent)
    return WorkspaceBinding(target.parents[1] if target.parent.name == "leaves" else target.parent, target, plan_sha256, leaf_id)

def _leaf_pointer(binding: WorkspaceBinding, root: Path) -> dict[str, object]:
    return {
        "directory": binding.leaf.relative_to(root).as_posix(),
        "workspace_instance_id": binding.workspace_instance_id,
        "plan_sha256": binding.plan_sha256,
    }

def _binding_from_replaceable(root: Path, state: Mapping[str, object]) -> WorkspaceBinding:
    active = state.get("active_leaf")
    if not isinstance(active, dict):
        raise _integrity("Replaceable workspace has no active leaf.")
    directory = active.get("directory")
    leaf_id = _valid_uuid(active.get("workspace_instance_id"))
    plan_sha256 = _valid_sha(active.get("plan_sha256"))
    if directory != f"leaves/{leaf_id}":
        raise _integrity("Replaceable workspace points outside its leaves directory.", directory=directory)
    return WorkspaceBinding(root, root / _relative_path(directory), plan_sha256, leaf_id)

def _validate_active_evidence(root: Path, state: Mapping[str, object]) -> None:
    """Verify indexed active evidence before any retired-leaf maintenance."""

    if state.get("kind") == "replaceable_workspace":
        _binding_from_replaceable(root, state)._verify_leaf()
        return
    _validated_versioned_index(root, state)

def _cleanup_replaceable_staging(root: Path, current: WorkspaceBinding) -> None:
    leaves = root / "leaves"
    if leaves.is_symlink() or not leaves.is_dir():
        raise _integrity("Replaceable workspace leaves directory is absent.")
    for child in leaves.iterdir():
        name = child.name
        if child == current.leaf:
            continue
        if child.is_symlink() or not child.is_dir():
            raise _integrity("Replaceable workspace contains a malformed leaf path.", path=str(child))
        if _LEAF_STAGING.fullmatch(name) is not None:
            shutil.rmtree(child)
            continue
        if _UUID4.fullmatch(name) is None:
            raise _integrity("Replaceable workspace has a malformed leaf name.", path=str(child))
        _verified_unbound_leaf(root, child, f"leaves/{name}")
        raise _integrity(
            "Replaceable workspace contains preserved unbound leaf evidence.",
            path=str(child),
        )
    _fsync_directory(leaves)

def _adopt_replaceable_orphan(
    root: Path,
    state: Mapping[str, object],
    current: WorkspaceBinding,
    requested_plan_sha256: str,
) -> WorkspaceBinding | None:
    leaves = root / "leaves"
    matches: list[WorkspaceBinding] = []
    unbound: list[WorkspaceBinding] = []
    for child in leaves.iterdir():
        if child == current.leaf or _LEAF_STAGING.fullmatch(child.name) is not None:
            continue
        if child.is_symlink() or not child.is_dir() or _UUID4.fullmatch(child.name) is None:
            raise _integrity("Replaceable workspace contains malformed orphan evidence.", path=str(child))
        orphan = _verified_unbound_leaf(root, child, f"leaves/{child.name}")
        unbound.append(orphan)
        requests = child / "requests"
        if requests.exists() and (requests.is_symlink() or not requests.is_dir() or any(requests.iterdir())):
            continue
        if orphan.plan_sha256 == requested_plan_sha256:
            matches.append(orphan)
    if not matches:
        return None
    if len(matches) != 1 or len(unbound) != 1:
        raise _integrity("Replaceable workspace has competing unbound leaf evidence.")
    recovered = matches[0]
    updated = replaceable_workspace_document(
        workspace_instance_id=str(state.get("workspace_instance_id")),
        leaf_instance_id=recovered.workspace_instance_id,
        plan_sha256=recovered.plan_sha256,
        maintenance=_retired_leaf_maintenance(current, root),
    )
    _publish_workspace_state(root / "workspace.json", _canonical_bytes(updated))
    return recovered

def _verified_unbound_leaf(root: Path, path: Path, expected_directory: str) -> WorkspaceBinding:
    state = _load_canonical(path / "workspace.json")
    return _recover_leaf(
        root,
        path,
        _valid_sha(state.get("plan_sha256")),
        expected_directory=expected_directory,
    )

def _assert_root_envelope(state: Mapping[str, object]) -> None:
    if state.get("evidence_version") != 3:
        raise _old_evidence("workspace.json", state.get("evidence_version"))
    root_id = state.get("workspace_instance_id")
    kind = state.get("kind")
    if (
        state.get("schema") != "scnsim.workspace"
        or state.get("schema_version") != 1
        or kind not in {"replaceable_workspace", "versioned_workspace"}
        or not isinstance(root_id, str)
        or _UUID4.fullmatch(root_id) is None
    ):
        raise _integrity("Workspace root is not a valid V1 workspace envelope.")
    if kind == "replaceable_workspace":
        active = state.get("active_leaf")
        if set(state) not in (
            {"schema", "schema_version", "evidence_version", "kind", "workspace_instance_id", "active_leaf"},
            {"schema", "schema_version", "evidence_version", "kind", "workspace_instance_id", "active_leaf", "maintenance"},
        ) or not isinstance(active, dict) or set(active) != {"directory", "workspace_instance_id", "plan_sha256"}:
            raise _integrity("Replaceable workspace envelope is open or malformed.")
        leaf_id = _valid_uuid(active.get("workspace_instance_id"))
        if leaf_id == root_id or active.get("directory") != f"leaves/{leaf_id}":
            raise _integrity("Replaceable workspace active pointer is not canonical.")
        _valid_sha(active.get("plan_sha256"))
    else:
        if set(state) not in (
            {"schema", "schema_version", "evidence_version", "kind", "workspace_instance_id", "next_iteration", "iterations"},
            {"schema", "schema_version", "evidence_version", "kind", "workspace_instance_id", "next_iteration", "iterations", "maintenance"},
        ):
            raise _integrity("Versioned workspace envelope is open or malformed.")
    maintenance = state.get("maintenance")
    if maintenance is not None and (
        not isinstance(maintenance, dict)
        or set(maintenance) != {"kind", "directory", "workspace_instance_id", "plan_sha256"}
        or maintenance.get("kind") != "retired_leaf_cleanup"
        or maintenance.get("directory") != f"leaves/{_valid_uuid(maintenance.get('workspace_instance_id'))}"
    ):
        raise _integrity("Workspace maintenance record is open or malformed.")
    if isinstance(maintenance, dict):
        _valid_sha(maintenance.get("plan_sha256"))
        active_pointer = state.get("active_leaf")
        if (
            kind == "replaceable_workspace"
            and isinstance(active_pointer, dict)
            and active_pointer.get("workspace_instance_id")
            == maintenance.get("workspace_instance_id")
        ):
            raise _integrity("Workspace maintenance cannot retire the active leaf.")

def _upgrade_to_versioned(root: Path, state: Mapping[str, object], current: WorkspaceBinding) -> WorkspaceBinding:
    current._verify_leaf()
    if any(path.is_symlink() for path in current.leaf.rglob("*")):
        raise _integrity("Workspace conversion refuses symlinked evidence.")
    destination = root / "iteration01"
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir() or any(path.is_symlink() for path in destination.rglob("*")):
            raise _integrity("Interrupted workspace upgrade left unsafe iteration01 evidence.")
        upgraded = _recover_leaf(
            root,
            destination,
            current.plan_sha256,
            expected_directory="iteration01",
        )
        if upgraded.workspace_instance_id != current.workspace_instance_id:
            raise _integrity("Interrupted workspace upgrade copy has the wrong identity.")
    else:
        shutil.copytree(current.leaf, destination)
        _fsync_tree(destination)
        upgraded = WorkspaceBinding(root, destination, current.plan_sha256, current.workspace_instance_id)
        upgraded._verify_leaf()
    index = versioned_workspace_document(
        workspace_instance_id=_new_uuid(excluding=upgraded.workspace_instance_id),
        iterations=[{"ordinal": 1, **_leaf_pointer(upgraded, root)}],
        maintenance=_retired_leaf_maintenance(current, root),
    )
    _publish_workspace_state(root / "workspace.json", _canonical_bytes(index))
    return upgraded

def _validated_versioned_index(
    root: Path,
    state: Mapping[str, object],
) -> tuple[
    list[Mapping[str, object]],
    int,
    str,
    tuple[tuple[str, WorkspaceBinding], ...],
]:
    """Validate a complete versioned index and each indexed leaf."""

    iterations = state.get("iterations")
    next_iteration = state.get("next_iteration")
    if not isinstance(iterations, list) or not isinstance(next_iteration, int) or next_iteration < 1:
        raise _integrity("Versioned workspace index is malformed.")
    checked_iterations: list[Mapping[str, object]] = []
    bindings: list[tuple[str, WorkspaceBinding]] = []
    seen_plans: set[str] = set()
    seen_ordinals: set[int] = set()
    seen_leaf_ids: set[str] = set()
    root_id = _valid_uuid(state.get("workspace_instance_id"))
    for expected_ordinal, entry in enumerate(iterations, 1):
        if not isinstance(entry, dict) or set(entry) != {"ordinal", "directory", "workspace_instance_id", "plan_sha256"}:
            raise _integrity("Versioned workspace contains a malformed iteration entry.")
        ordinal = entry.get("ordinal")
        directory = entry.get("directory")
        identity = _valid_uuid(entry.get("workspace_instance_id"))
        plan = _valid_sha(entry.get("plan_sha256"))
        if not isinstance(ordinal, int) or ordinal != expected_ordinal or ordinal in seen_ordinals or plan in seen_plans or identity in seen_leaf_ids or identity == root_id:
            raise _integrity("Versioned workspace has duplicate iteration identity.")
        if not isinstance(directory, str) or directory != f"iteration{ordinal:02d}":
            raise _integrity("Versioned workspace iteration directory disagrees with its ordinal.")
        seen_ordinals.add(ordinal)
        seen_plans.add(plan)
        seen_leaf_ids.add(identity)
        binding = WorkspaceBinding(root, root / directory, plan, identity)
        binding._verify_leaf()
        checked_iterations.append(entry)
        bindings.append((plan, binding))
    if sorted(seen_ordinals) != list(range(1, len(seen_ordinals) + 1)) or next_iteration != len(seen_ordinals) + 1:
        raise _integrity("Versioned workspace next_iteration is not canonical.")
    return checked_iterations, next_iteration, root_id, tuple(bindings)

def _bind_versioned(root: Path, state: Mapping[str, object], plan_sha256: str, plan_bytes: bytes) -> WorkspaceBinding:
    iterations, next_iteration, root_id, bindings = _validated_versioned_index(root, state)
    existing = next(
        (binding for plan, binding in bindings if plan == plan_sha256),
        None,
    )
    indexed = {binding.leaf.name for _, binding in bindings}
    _cleanup_upgrade_duplicate(root, indexed)
    recovered = _adopt_versioned_orphan(root, state, plan_sha256, next_iteration, indexed)
    if recovered is not None:
        return recovered
    _cleanup_versioned_staging(root, indexed)
    if existing is not None:
        return existing
    directory = f"iteration{next_iteration:02d}"
    leaf = _create_leaf(root, plan_sha256, plan_bytes, directory=directory, excluding=root_id)
    state = versioned_workspace_document(
        workspace_instance_id=root_id,
        iterations=[*iterations, {"ordinal": next_iteration, **_leaf_pointer(leaf, root)}],
        maintenance=state.get("maintenance") if isinstance(state.get("maintenance"), Mapping) else None,
    )
    _publish_workspace_state(root / "workspace.json", _canonical_bytes(state))
    return leaf

def _cleanup_upgrade_duplicate(root: Path, indexed: set[str]) -> None:
    leaves = root / "leaves"
    if leaves.is_symlink():
        raise _integrity("Versioned workspace retains unsafe replacement evidence.", path=str(leaves))
    if not leaves.exists():
        return
    if not leaves.is_dir() or "iteration01" not in indexed:
        raise _integrity("Versioned workspace retains unsafe replacement evidence.", path=str(leaves))
    children = list(leaves.iterdir())
    if len(children) != 1 or children[0].is_symlink() or _UUID4.fullmatch(children[0].name) is None:
        raise _integrity("Versioned workspace upgrade duplicate is malformed.", path=str(leaves))
    old = _verified_unbound_leaf(root, children[0], f"leaves/{children[0].name}")
    upgraded = _verified_unbound_leaf(root, root / "iteration01", "iteration01")
    if old.workspace_instance_id != upgraded.workspace_instance_id or old.plan_sha256 != upgraded.plan_sha256:
        raise _integrity("Versioned workspace upgrade copies disagree.")
    _remove_leaf(old.leaf)
    leaves.rmdir()
    _fsync_directory(root)

def _adopt_versioned_orphan(
    root: Path,
    state: Mapping[str, object],
    requested_plan_sha256: str,
    next_iteration: int,
    indexed: set[str],
) -> WorkspaceBinding | None:
    directory = f"iteration{next_iteration:02d}"
    candidate = root / directory
    if not candidate.exists():
        return None
    if candidate.name in indexed or candidate.is_symlink() or not candidate.is_dir():
        raise _integrity("Versioned workspace has unsafe unindexed iteration evidence.", path=str(candidate))
    leaf = _verified_unbound_leaf(root, candidate, directory)
    requests = candidate / "requests"
    if leaf.plan_sha256 != requested_plan_sha256:
        if requests.exists() and (requests.is_symlink() or not requests.is_dir() or any(requests.iterdir())):
            raise _integrity("Unindexed iteration contains request evidence and cannot be discarded.")
        raise _integrity("Unindexed iteration belongs to another Plan and is preserved.")
    iterations = state.get("iterations")
    if not isinstance(iterations, list):
        raise _integrity("Versioned workspace index is malformed.")
    updated = versioned_workspace_document(
        workspace_instance_id=str(state.get("workspace_instance_id")),
        iterations=[*iterations, {"ordinal": next_iteration, **_leaf_pointer(leaf, root)}],
        maintenance=state.get("maintenance") if isinstance(state.get("maintenance"), Mapping) else None,
    )
    _publish_workspace_state(root / "workspace.json", _canonical_bytes(updated))
    return leaf

def _cleanup_versioned_staging(root: Path, indexed: set[str]) -> None:
    leaves = root / "leaves"
    if leaves.is_symlink() or leaves.exists():
        raise _integrity("Versioned workspace retains unbound replacement leaf evidence.", path=str(leaves))
    for child in root.iterdir():
        staging = _LEAF_STAGING.fullmatch(child.name) is not None
        iteration = re.fullmatch(r"iteration[0-9]{2,}", child.name) is not None
        if staging:
            if child.is_symlink() or not child.is_dir():
                raise _integrity("Versioned workspace contains a malformed staging path.", path=str(child))
            shutil.rmtree(child)
        elif iteration and child.name not in indexed:
            raise _integrity("Versioned workspace has unindexed iteration evidence.", path=str(child))
    _fsync_directory(root)
