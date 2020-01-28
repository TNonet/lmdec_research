import numba as nb
import numpy as np
from .types import FLOAT_STORAGE_np


@nb.jit(nopython=True)
def binary_matmul_1d(row_p, col_i, other, m):
    """Matrix-Vector Dot Product
    A*v
    A -> (row_p, col_i)
        CSR format (documentation in folder)
        shape: (m, n)
    v -> other
        np.array(ndims=1)
        shape: (n,)

    :param out: [ADD HERE]
    :param row_p: np.array(integer type)
        shape: (m+1,)
        note: map between row to col_i
    :param col_i: np.array(integer type)
        shape: (nnz,), nnz -> number of non-zeros in A
    :param other: np.array(numeric type)
        shape: (n,)
    :param m: integer type
        note: number of columns of A == number of rows of v
    :param k: integer type
        note: number of columns of v

    :return: out: np.array(float64)
        shape: (m, 1)
        format: C
    """
    out = np.zeros(m, dtype=FLOAT_STORAGE_np)
    for i in range(m):
        vector_sum = 0
        for r in col_i[row_p[i]:row_p[i + 1]]:
            vector_sum += other[r]
        out[i] = vector_sum
    return out


@nb.jit(nopython=True, parallel=True, nogil=True)
def binary_matmul_2d(row_p, col_i, other, m, k):
    out_2d = np.zeros((m, k), dtype=FLOAT_STORAGE_np)
    out_2d = np.asfortranarray(out_2d)
    for j in nb.prange(k):
        out_2d[:, j] = binary_matmul_1d(row_p, col_i, other[:, j], m)
    return out_2d


@nb.jit(nopython=True)
def finite_matmul_1d(bcsr_decomp, other, m):
    out = np.zeros(m, dtype=FLOAT_STORAGE_np)
    for bcsr_matrix in bcsr_decomp:
        if not bcsr_matrix.empty:
            out += bcsr_matrix.dot1d(other)
    return out


@nb.jit(nopython=True)
def finite_matmul_2d(bcsr_decomp, other, m, k):
    out = np.zeros((m, k), dtype=FLOAT_STORAGE_np)
    out = np.asfortranarray(out)
    for bcsr_matrix in bcsr_decomp:
        if not bcsr_matrix.empty:
            out += bcsr_matrix.dot2d(other)
    return out
