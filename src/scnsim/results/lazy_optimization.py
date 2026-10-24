"""Lazy, detached views over a fixed optimization result reader."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from operator import index as integer_index
from typing import Any

from .base import _freeze


def _public_optimization_candidate(record: Mapping[str, object]) -> dict[str, object]:
    """Detach request approval from a public copy, retaining the stored occurrence."""
    return dict(record, parameters=dict(record["parameters"], allow_extrapolation=[]))


def _sequence_index(value: int | slice, length: int) -> int | range:
    if isinstance(value, slice):
        return range(*value.indices(length))
    index = integer_index(value)
    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise IndexError("optimization result index out of range")
    return index


class LazyGenerationSequence(Sequence[Mapping[str, object]]):
    """Read and decode one sealed generation only when it is selected."""

    __slots__ = ("_reader", "_length", "_candidate", "_generation_rows", "_candidate_ordinal",
                 "_generation_projection")

    def __init__(self, reader: object, length: int,
                 candidate: Callable[[Mapping[str, object]], Mapping[str, object]], *,
                 generation_rows: Callable[[Mapping[str, object]], Sequence[Mapping[str, object]]] =
                 lambda block: block["rows"],
                 candidate_ordinal: Callable[[Mapping[str, object]], int] =
                 lambda row: row["occurrence"]["evaluation_ordinal"],
                 generation_projection: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None):
        object.__setattr__(self, "_reader", reader)
        object.__setattr__(self, "_length", length)
        object.__setattr__(self, "_candidate", candidate)
        object.__setattr__(self, "_generation_rows", generation_rows)
        object.__setattr__(self, "_candidate_ordinal", candidate_ordinal)
        object.__setattr__(self, "_generation_projection", generation_projection)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("lazy optimization sequences are immutable")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, selected: int | slice):
        index = _sequence_index(selected, self._length)
        if isinstance(index, range):
            return tuple(self[item] for item in index)
        block = self._reader.read_generation(index)
        if self._generation_projection is not None:
            return _freeze(self._generation_projection(block))
        candidates = []
        for row in self._generation_rows(block):
            ordinal = self._candidate_ordinal(row)
            candidate = (
                self._reader.read_candidate(ordinal)
                if "occurrence" in row else row
            )
            candidates.append(self._candidate(candidate))
        return _freeze({"generation": index + 1, "candidates": candidates})


class LazyCandidateDiscretizationSequence(Sequence[object]):
    """Flat population-only discretization access, excluding the baseline."""

    __slots__ = ("_reader", "_length", "_decode")

    def __init__(self, reader: object, length: int,
                 decode: Callable[[object], object]):
        object.__setattr__(self, "_reader", reader)
        object.__setattr__(self, "_length", length)
        object.__setattr__(self, "_decode", decode)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("lazy optimization sequences are immutable")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, selected: int | slice):
        index = _sequence_index(selected, self._length)
        if isinstance(index, range):
            return tuple(self[item] for item in index)
        record = self._reader.read_candidate_discretization(index)
        return _freeze(self._decode(record))


class LazyOptimizationStageHistory(Sequence[Mapping[str, object]]):
    """Compact history for one fixed stage, projected only when selected."""

    __slots__ = ("_reader", "_stage_id", "_model_identity")

    def __init__(self, reader: object, stage_id: str, model_identity: Mapping[str, object]):
        object.__setattr__(self, "_reader", reader)
        object.__setattr__(self, "_stage_id", stage_id)
        object.__setattr__(self, "_model_identity", _freeze(model_identity))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("optimization stage histories are immutable")

    def _matches_model_identity(self, value: object) -> bool:
        """Compare the same JSON identity after immutable Result freezing."""
        return isinstance(value, Mapping) and _freeze(value) == self._model_identity

    def _rows(self) -> Sequence[Mapping[str, object]]:
        selected = self._reader.project("stage_history", self._stage_id)
        if (not isinstance(selected, Mapping)
                or selected.get("stage_id") != self._stage_id
                or not self._matches_model_identity(selected.get("model_identity"))):
            from ..errors import EvidenceIntegrityError

            raise EvidenceIntegrityError(
                "Optimization stage history differs from its fixed stage identity",
                stage="result_decode",
                evidence={"stage_id": self._stage_id},
            )
        rows = selected.get("rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            from ..errors import EvidenceIntegrityError

            raise EvidenceIntegrityError(
                "Optimization stage history projection is malformed",
                stage="result_decode",
                evidence={"stage_id": self._stage_id},
            )
        return rows

    def __len__(self) -> int:
        return len(self._rows())

    def __iter__(self):
        rows = self._rows()
        detached = []
        for index, row in enumerate(rows):
            if (not isinstance(row, Mapping)
                    or row.get("stage_id") != self._stage_id
                    or not self._matches_model_identity(row.get("model_identity"))):
                from ..errors import EvidenceIntegrityError

                raise EvidenceIntegrityError(
                    "Optimization history row belongs to another stage or model",
                    stage="result_decode",
                    evidence={"stage_id": self._stage_id, "index": index},
                )
            detached.append(_freeze(_public_optimization_candidate(row)))
        return iter(tuple(detached))

    def __getitem__(self, selected: int | slice) -> Mapping[str, object] | tuple[Mapping[str, object], ...]:
        rows = self._rows()
        index = _sequence_index(selected, len(rows))
        if isinstance(index, range):
            selected_rows = []
            for item in index:
                row = rows[item]
                if (not isinstance(row, Mapping)
                        or row.get("stage_id") != self._stage_id
                        or not self._matches_model_identity(row.get("model_identity"))):
                    from ..errors import EvidenceIntegrityError

                    raise EvidenceIntegrityError(
                        "Optimization history row belongs to another stage or model",
                        stage="result_decode",
                        evidence={"stage_id": self._stage_id, "index": item},
                    )
                selected_rows.append(_freeze(_public_optimization_candidate(row)))
            return tuple(selected_rows)
        row = rows[index]
        if (not isinstance(row, Mapping)
                or row.get("stage_id") != self._stage_id
                or not self._matches_model_identity(row.get("model_identity"))):
            from ..errors import EvidenceIntegrityError

            raise EvidenceIntegrityError(
                "Optimization history row belongs to another stage or model",
                stage="result_decode",
                evidence={"stage_id": self._stage_id, "index": index},
            )
        return _freeze(_public_optimization_candidate(row))

