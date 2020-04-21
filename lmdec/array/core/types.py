from typing import Union, TYPE_CHECKING
from numpy import ndarray
from dask.array.core import Array
from scipy.sparse.csr import csr_matrix
from scipy.sparse.csc import csc_matrix
from scipy.sparse.coo import coo_matrix

from ..sparse.finite.bcsr import bcsr_matrix
from ..sparse.finite.fcsr import fcsr_matrix

if TYPE_CHECKING:
    from lmdec import ScaledArray

LargeArrayType = Union[ndarray, Array, bcsr_matrix, fcsr_matrix, csr_matrix, csc_matrix, coo_matrix, "ScaledArray"]
ArrayType = Union[ndarray, Array]
DaskArrayType = Array

