"""Encode numerical diagnostics without acquiring persistence or runtime ownership."""
from __future__ import annotations
import math
import numpy as np
from ..numeric_encoding import array_record, record_bytes, record_document
from .. import errors
from .models import NumericalFailure

def evidence_bytes(value: object) -> bytes:
    """Represent numerical diagnostics, including unresolved nonfinite values."""
    def encode(item: object) -> object:
        if isinstance(item, dict):
            return {str(k): encode(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [encode(v) for v in item]
        if isinstance(item, np.ndarray):
            return array_record(item)
        if isinstance(item, np.generic):
            return encode(item.item())
        if isinstance(item, complex):
            return {"real": encode(item.real), "imag": encode(item.imag)}
        if isinstance(item, float):
            if not math.isfinite(item):
                return "nan" if math.isnan(item) else ("+inf" if item > 0 else "-inf")
            return item
        return item
    return record_bytes(encode(value))



def numerical_error(failure: NumericalFailure):
    """Restore the existing public failure type from a numerical handoff."""
    classes = (errors.DirectResponseFormationError, errors.PortRealizabilityError,
               errors.EliminatedBlockSolveFailure, errors.RootSlopeUnresolved,
               errors.NumericalResolutionUnresolved, errors.InvalidCandidatePhysicalParameter,
               errors.CompilerInvariantError, errors.UnsupportedSingularCapacitanceForDiagonalRootV1)
    error_type = {cls.kind: cls for cls in classes}[failure.kind]
    return error_type(failure.detail, stage=failure.stage, evidence=record_document(failure.evidence_bytes))
