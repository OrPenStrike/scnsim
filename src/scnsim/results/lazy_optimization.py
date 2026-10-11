"""Lazy, detached views over a fixed optimization result reader."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from operator import index as integer_index
from typing import Any

from .base import _freeze


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

    __slots__ = ("_reader", "_length", "_candidate", "_generation_rows", "_candidate_ordinal")

    def __init__(self, reader: object, length: int,
                 candidate: Callable[[Mapping[str, object]], Mapping[str, object]], *,
                 generation_rows: Callable[[Mapping[str, object]], Sequence[Mapping[str, object]]] =
                 lambda block: block["rows"],
                 candidate_ordinal: Callable[[Mapping[str, object]], int] =
                 lambda row: row["occurrence"]["evaluation_ordinal"]):
        object.__setattr__(self, "_reader", reader)
        object.__setattr__(self, "_length", length)
        object.__setattr__(self, "_candidate", candidate)
        object.__setattr__(self, "_generation_rows", generation_rows)
        object.__setattr__(self, "_candidate_ordinal", candidate_ordinal)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("lazy optimization sequences are immutable")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, selected: int | slice):
        index = _sequence_index(selected, self._length)
        if isinstance(index, range):
            return tuple(self[item] for item in index)
        block = self._reader.read_generation(index)
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

