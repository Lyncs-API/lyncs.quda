__all__ = [
    "evenodd",
    "continous",
]

from array import array
import numpy as np
from lyncs_utils import prod
from .lib import lib


def _get_params(arr, axes=None):
    if axes is None:
        axes = tuple(range(len(arr.shape)))
    elif isinstance(axes, int):
        axes = (axes,)

    axes = sorted(axes)
    if not axes == list(range(min(axes), max(axes) + 1)):
        raise ValueError(f"Given axes {axes} are not consecutive numbers")

    outer = prod(arr.shape[: min(axes)])
    inner = prod(arr.shape[max(axes) + 1 :]) * arr.dtype.itemsize
    shape = np.array(arr.shape)[axes]

    return shape, inner, outer


def evenodd(arr, axes=None, swap=False, out=None):
    """
    From continous to evenodd ordering.

    Parameters
    ----------
    - axes: axes representing the lattice
    - swap: swap even and odd in the output
    - out: output array
    """
    shape, inner, outer = _get_params(arr, axes)
    if out is None:
        out = np.empty_like(arr)
    lib.evenodd(out, arr, len(shape), array("i", shape), outer, inner, swap=swap)
    return out


def continous(arr, axes=None, swap=False, out=None):
    """
    From evenodd to continous ordering.

    Parameters
    ----------
    - axes: axes representing the lattice
    - swap: swap even and odd in the output
    - out: output array
    """
    shape, inner, outer = _get_params(arr, axes)
    if out is None:
        out = np.empty_like(arr)
    lib.continous(out, arr, len(shape), array("i", shape), outer, inner, swap=swap)
    return out
