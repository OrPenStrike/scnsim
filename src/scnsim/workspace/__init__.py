"""Verified workspace state, evidence, and receipt lifecycle."""

from .primitives import _inside
from .records import (
    AttemptAllocation,
    BaselineCheckpoint,
    PointCheckpoint,
    VerifiedSuccess,
    _IncomingCheckpointEvidenceError,
)
from .store import WorkspaceBinding, bind_workspace
from .validation.common import _required_extrapolation_rows
from .validation.inventory import (
    _compare_artifacts,
    _verify_artifact_inventory,
    _verify_hb_artifact_inventory,
)
from .validation.optimization import (
    _verify_attempt_checkpoint_consumption,
    _verify_baseline_checkpoint_directory,
    _verify_baseline_checkpoint_document,
    _verify_generation_artifacts,
    _verify_terminal_optimization_failure,
    verified_generation_links,
)
from .validation.requests import (
    _plan_coordinates,
    _verify_failure_document,
    _verify_request_document,
)
from .validation.results import _verify_result_document
from .validation.sweeps import (
    _verify_parameter_sweep_artifacts,
    _verify_point_checkpoints,
)

__all__ = [
    "AttemptAllocation",
    "BaselineCheckpoint",
    "PointCheckpoint",
    "VerifiedSuccess",
    "WorkspaceBinding",
    "bind_workspace",
    "verified_generation_links",
    "_IncomingCheckpointEvidenceError",
    "_compare_artifacts",
    "_inside",
    "_plan_coordinates",
    "_required_extrapolation_rows",
    "_verify_artifact_inventory",
    "_verify_attempt_checkpoint_consumption",
    "_verify_baseline_checkpoint_directory",
    "_verify_baseline_checkpoint_document",
    "_verify_failure_document",
    "_verify_generation_artifacts",
    "_verify_hb_artifact_inventory",
    "_verify_parameter_sweep_artifacts",
    "_verify_point_checkpoints",
    "_verify_request_document",
    "_verify_result_document",
    "_verify_terminal_optimization_failure",
]
