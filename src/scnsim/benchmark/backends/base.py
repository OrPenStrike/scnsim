"""Exact descriptor transport and numerical evidence shared by adapters.

Models remains the type authority. Infrastructure failures escape adapters;
only numerical outcomes enter the host's candidate failure/context machinery.
"""

from __future__ import annotations

from dataclasses import fields
import json
import math

import numpy as np

from ..models import EvaluationJob
from ..prepared import array_record, record_bytes


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


def job_record(job: EvaluationJob) -> dict[str, object]:
    """Serialize every numeric axis using shared row-major binary64 transport."""
    view = job.view
    model = view.model
    # Compiler section rows own physical path/section identity. View transforms
    # preserve their order while transforming the corresponding incidences.
    sections = [row for row in json.loads(model.evidence_bytes)["branches"]
                if row["kind"] == "pi_section"]
    series_rl = [
        {"id": "\x1f".join(row["component_path"]) + "\x1esection-" + str(row["section"]),
         "component_path": row["component_path"], "section": row["section"],
         **{name: array_record(getattr(block, name))
            for name in ("incidence", "resistance", "inductance")}}
        for block, row in zip(model.series_rl, sections, strict=True)
    ]
    record = {field.name: getattr(job, field.name) for field in fields(job) if field.name != "view"}
    if job.frequencies_hz is not None:
        record["frequencies_hz"] = array_record(job.frequencies_hz)
    if job.root_hint_hz is not None:
        record["root_hint_hz"] = array_record(np.asarray(job.root_hint_hz, dtype=np.float64))
    if job.omega_start_rad_s is not None:
        record["omega_start_rad_s"] = array_record(np.asarray(job.omega_start_rad_s, dtype=np.complex128))
    record["view"] = {
        "model": {
            "node_ids": list(model.node_ids), "port_ids": list(model.port_ids),
            **{name: array_record(getattr(model, name)) for name in ("C", "K", "G", "B", "R", "M")},
            "series_rl": series_rl,
        },
        "coordinates": list(view.coordinates), "terminal_ids": list(view.terminal_ids),
        "original_node_ids": list(view.original_node_ids),
        "selected_indices": list(view.selected_indices), "port_realizable": view.port_realizable,
        **{name: None if getattr(view, name) is None else array_record(getattr(view, name))
           for name in ("coordinate_port_map", "selected_map", "Bk", "Rk", "Dk", "Go")},
    }
    return record
