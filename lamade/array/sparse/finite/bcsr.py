"""Compressed Binary Sparse Row Matrix"""

import numba as nb
import numpy as np

from .types import POINT_STORAGE_nb, INDEX_STORAGE_nb, FLOAT_STORAGE_nb, FLOAT_STORAGE_np
from .matmul import binary_matmul_1d, binary_matmul_2d
from .transform import coo_to_bcsr, bcsr_to_bcoo, bcoo_to_dense


bcsr_spec = [
    ('row_p', POINT_STORAGE_nb[::1]),
    ('col_i', INDEX_STORAGE_nb[::1]),
    ('shape', nb.types.UniTuple(nb.int64, 2)),
    ('alpha', FLOAT_STORAGE_nb),
    ('empty', nb.boolean),
]


@nb.jitclass(bcsr_spec)
class bcsr_matrix:
    def __init__(self, _row_p, _col_i, _shape):
        """
        :param _row_p: np.array of integers
        :param _col_i: np.array of integers
        :param _shape: 2-tuple of integers (m, n) representing shape of Matrix

        See Documentation for Format
        """
        self.row_p = _row_p
        self.col_i = _col_i
        self.shape = _shape
        self.alpha = 1.0

        if len(self.row_p) == 0:
            self.empty = True

    def dot1d(self, other):
        """Matrix-Vector Dot Product

        :param other:
        :return:
        """
        if len(other.shape) != 1:
            raise Exception('Must be 1d Array')

        d1 = other.shape[0]
        m, n = self.shape

        if n != d1:
            raise Exception('Dimension MisMatch')

        if self.empty:
            return np.zeros(m, dtype=FLOAT_STORAGE_np)
        else:
            return self.alpha*binary_matmul_1d(self.row_p, self.col_i, other, m)

    def dot2d(self, other):
        """Matrix-Matrix Dot Product
        :param other: Fortran Stored Array
        :return:
        """
        m, n = self.shape

        if len(other.shape) != 2:
            raise Exception("Must be a 2d np.array")

        d1, k = other.shape

        # Somehow this check crashes the Kernel
        # if k > 1 and other.flags.c_contiguous:
        #     raise Exception("Use Fortran Array")

        if n != d1:
            raise Exception('Dimension MisMatch')

        if self.empty:
            return np.zeros((m, k), dtype=FLOAT_STORAGE_np)
        else:
            return self.alpha*binary_matmul_2d(self.row_p, self.col_i, other, m, k)

    def __str__(self):
        return self.to_array()

    def __repr__(self):
        return "bcsr_matrix(%d, %d, %d)".format(self.shape[0], self.shape[1], self.row_p[-1])

    def to_array(self):
        _rows, _cols = bcsr_to_bcoo(self.row_p, self.col_i, self.shape)
        return self.alpha*bcoo_to_dense(_rows, _cols, self.shape)

    @property
    def size(self):
        return len(self.col_i)

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
        return self.col_i.size*self.col_i.itemsize + self.row_p.size*self.row_p.itemsize

    @property
    def T(self):
        m, n = self.shape
        rows, cols = bcsr_to_bcoo(self.row_p, self.col_i, self.shape)
        row_p, col_i = coo_to_bcsr(n, len(rows), cols, rows)

        temp_b_matrix = bcsr_matrix(row_p, col_i, (n, m))
        temp_b_matrix.alpha = self.alpha
        return temp_b_matrix
