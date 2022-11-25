__all__ = [
    "evenodd",
    "continous",
    "to_quda",
    "from_quda",
]

from array import array
import numpy
from lyncs_utils import prod
from .lib import lib
from .lattice_field import backend


def _get_axes(arr, axes=None):
    if axes is None:
        axes = tuple(range(len(arr.shape)))
    elif isinstance(axes, int):
        axes = (axes,)
    axes = sorted(axes)
    if not axes == list(range(min(axes), max(axes) + 1)):
        raise ValueError(f"Given axes {axes} are not consecutive numbers")
    return numpy.array(axes)


def _get_params(arr, axes=None):
    axes = _get_axes(arr, axes)
    outer = prod(arr.shape[: min(axes)])
    inner = prod(arr.shape[max(axes) + 1 :]) * arr.dtype.itemsize
    shape = numpy.array(arr.shape)[axes]
    return shape, inner, outer


def evenodd(arr, axes=None, swap=False, out=None):
    """
    From continous to evenodd ordering (EO,outer dofs,volume/2,inner dofs).

    Parameters
    ----------
    - axes: axes representing the lattice
    - swap: swap even and odd in the output
    - out: output array
    """
    shape, inner, outer = _get_params(arr, axes)
    arr = to_numpy(arr)
    if out is None:
        out = numpy.empty_like(arr)
    lib.evenodd(out, arr, len(shape), array("i", shape), outer, inner, swap=swap)
    return out


def continous(arr, axes=None, swap=False, out=None):
    """
    From evenodd to continous ordering (outer dofs, volume, inner dofs).

    Parameters
    ----------
    - axes: axes representing the lattice
    - swap: swap even and odd in the output
    - out: output array
    """
    shape, inner, outer = _get_params(arr, axes)
    arr = to_numpy(arr)
    if out is None:
        out = numpy.empty_like(arr)
    lib.continous(out, arr, len(shape), array("i", shape), outer, inner, swap=swap)
    return out


def to_numpy(arr):
    "Converts any input to numpy array"
    try:
        arr = arr.get()
    except AttributeError:
        pass
    return numpy.asarray(arr, order="C")


def to_quda(arr, axes=tuple(range(4)), swap=False):
    """
    Converts standard CPU array to QUDA format with QUDA_FLOAT_FIELD_ORDER.
    I.E. (extra, lattice, dofs) on CPU -> (extra, EO, dofs, lattice/2) on GPU
    """
    # ? what is extra?  QUDA distinguishes inner and out dofs
    # extra initially echos with outer but not in the end...

    # CPU: T,Z,Y,X; QUDA: X,Y,Z,T
    axes = _get_axes(arr, axes)
    arr = to_numpy(arr)
    arr = arr.transpose(
        *range(min(axes)), *reversed(axes), *range(max(axes) + 1, len(arr.shape))
    )
    arr = evenodd(arr, axes, swap)
    # Flattening the lattice
    shape = numpy.array(arr.shape)
    arr = arr.reshape(*shape[: min(axes)], 2, -1, *shape[max(axes) + 1 :])
    # Transposing lattice (min(axes)+1) and inner dofs
    arr = arr.transpose(
        *range(min(axes) + 1), *range(min(axes) + 2, len(arr.shape)), min(axes) + 1
    )
    # Reshaping to expected shape
    arr = arr.reshape(*shape[: min(axes)], *shape[max(axes) + 1 :], *shape[axes])
    with backend() as bck:
        return bck.asarray(arr)


def from_quda(arr, axes=tuple(range(4)), swap=False):
    """
    Converts QUDA array to standard CPU format.
    I.E. (extra, EO, dofs, lattice/2) on GPU -> (extra, lattice, dofs) on CPU
    """
    axes = _get_axes(arr, axes)
    arr = to_numpy(arr)
    shape = arr.shape
    # Flattening the lattice
    arr = arr.reshape(*shape[: min(axes)], 2, *shape[min(axes) : -len(axes)], -1)
    # Transposing lattice len(arr.shape) - 1  and inner dofs
    arr = arr.transpose(
        *range(min(axes) + 1),
        len(arr.shape) - 1,
        *range(min(axes) + 1, len(arr.shape) - 1),
    )
    # Reshaping to expected shape
    arr = arr.reshape(
        *shape[: min(axes)], *shape[-len(axes) :], *shape[min(axes) : -len(axes)]
    )
    arr = continous(arr, axes, swap)
    return arr.transpose(
        *range(min(axes)), *reversed(axes), *range(max(axes) + 1, len(arr.shape))
    )
