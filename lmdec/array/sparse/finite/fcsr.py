"""Compressed Finite Sparse Row Matrix"""

import numba as nb
from numba import jitclass
import numpy as np

from .types import MAX_BCSR, POINT_STORAGE_np, INDEX_STORAGE_np
from .bcsr import bcsr_matrix
from .matmul import finite_matmul_2d, finite_matmul_1d


def fcsr_matrix(arg1, shape=None):
    """
    Can be instantiated in 2 ways:

        fcsr_constructor(bcsr_iterable)
            bcsr_iterable is a iterable of bcsr matrices

        fcsr_constructor(fcoo, shape)
            fcoo is a dictionary of binary coo form
            See coo_to_fcoo

    :return: FCSR
    """
    if isinstance(arg1, (tuple, list)):
        if len(arg1) > 10:
            raise Exception("To Large of FCSR. Adjust MAX_BCSR in transform.py")
        for b_matrix in arg1:
            if shape is None:
                shape = b_matrix.shape
            elif shape == b_matrix.shape:
                pass
            else:
                raise Exception("Shapes don't Match")

        b_matrix_decomp = list(arg1)
        for empty_b_matrix in range(10-len(arg1)):
            b_matrix_decomp.append(bcsr_matrix(_row_p=np.array([], dtype=POINT_STORAGE_np),
                                               _col_i=np.array([], dtype=INDEX_STORAGE_np),
                                               _shape=shape))

        b_matrix_decomp = tuple(b_matrix_decomp)
        return fcsr_matrix(_bcsr_decomp=b_matrix_decomp, _shape=shape)

    elif isinstance(arg1, dict):
        if shape is None:
            raise Exception("Shape must be defined")
        b_matrix_decomp = []
        for key in arg1.keys():
            temp_bcsr_matrix = bcsr_matrix(arg1[key][0],
                                           arg1[key][1],
                                           _shape=shape)
            temp_bcsr_matrix.alpha = key
            b_matrix_decomp.append(temp_bcsr_matrix)

        for empty_b_matrix in range(10-len(arg1)):
            b_matrix_decomp.append(bcsr_matrix(_row_p=np.array([], dtype=POINT_STORAGE_np),
                                               _col_i=np.array([], dtype=INDEX_STORAGE_np),
                                               _shape=shape))

        b_matrix_decomp = tuple(b_matrix_decomp)
        return fcsr_matrix(_bcsr_decomp=b_matrix_decomp, _shape=shape)
    else:
        raise NotImplementedError


bcsr_type = nb.deferred_type()
bcsr_type.define(bcsr_matrix.class_type.instance_type)

fcsr_spec = [
    ('shape', nb.types.UniTuple(nb.int64, 2)),
    ('bcsr_decomp', nb.types.UniTuple(bcsr_type, MAX_BCSR)),
]


@jitclass(fcsr_spec)
class fcsr_matrix:
    """
    Finite Compressed Row Matrix
    """
    def __init__(self, _bcsr_decomp, _shape):
        """
        """

        self.shape = _shape
        self.bcsr_decomp = _bcsr_decomp

    @property
    def depth(self):
        return len(self.bcsr_decomp)

    def dot1d(self, other):
        """

        :param other:
        :return:
        """
        if len(other.shape) != 1:
            raise Exception('Must be a 1d Array')

        d1 = other.shape[0]
        m, n = self.shape

        if n != d1:
            raise Exception('Dimension MisMatch')

        return finite_matmul_1d(self.bcsr_decomp, other, m)

    def dot2d(self, other):
        """

        :param other:
        :return:
        """
        if len(other.shape) != 2:
            raise Exception('Must be a 2d Array')

        m, n = self.shape
        d1, k = other.shape

        # if k > 1 and other.flags.c_contiguous:
        #     raise Exception("Use Fortran Array")

        if n != d1:
            raise Exception('Dimension MisMatch')

        return finite_matmul_2d(self.bcsr_decomp, other, m, k)

    def to_array(self):
        array = np.zeros(self.shape)
        for sub_bcsr_matrix in self.bcsr_decomp:
            if not sub_bcsr_matrix.empty:
                array += sub_bcsr_matrix.to_array()
        return array

    @property
    def size(self):
        nnz = 0
        for sub_bcsr_matrix in self.bcsr_decomp:
            if not sub_bcsr_matrix.empty:
                nnz += sub_bcsr_matrix.size
        return nnz

    @property
    def sparsity(self):
        return self.size / (self.shape[0] * self.shape[1])

    @property
    def mem(self):
        return self.__sizeof__()

    def __sizeof__(self):
        """
        returns roughly the memory storage of instance in Bytes

        Storage Includes:
            self.row_p
            self.col_i

        :return:
        """
        mem = 0
        for sub_bcsr_matrix in self.bcsr_decomp:
            if not sub_bcsr_matrix.empty:
                mem += sub_bcsr_matrix.mem
        return mem
