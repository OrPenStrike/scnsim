"""Private immutable storage helpers for retained public numerical values.

NumPy's ordinary read-only flag is reversible when an exposed array owns its
memory.  SCNSim instead retains numerical values on immutable ``bytes``
backing, so normal public access cannot make the shared storage writable.
"""

from __future__ import annotations

import numpy as np
from pint import Quantity

from . import units


def immutable_array(value: object) -> np.ndarray:
    """Return a detached, C-contiguous array with immutable backing."""

    source = np.asarray(value)
    if source.dtype.hasobject:
        raise TypeError("retained numerical arrays cannot use object dtype")
    contiguous = np.ascontiguousarray(source)
    backing = contiguous.tobytes(order="C")
    return np.frombuffer(backing, dtype=contiguous.dtype).reshape(source.shape)


def immutable_quantity(value: Quantity) -> Quantity:
    """Detach one SCNSim Quantity onto immutable numerical backing."""

    if not isinstance(value, Quantity) or value._REGISTRY is not units.registry:
        raise TypeError("retained quantities must use scnsim.units")
    return units.registry.Quantity(immutable_array(value.magnitude), value.units)


def quantity_view(value: Quantity) -> Quantity:
    """Return a cheap public wrapper around already-immutable backing."""

    if not isinstance(value, Quantity) or value._REGISTRY is not units.registry:
        raise TypeError("retained quantities must use scnsim.units")
    return units.registry.Quantity(np.asarray(value.magnitude).view(), value.units)
