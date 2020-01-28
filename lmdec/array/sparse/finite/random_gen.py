from .bcsr import bcsr_matrix
from .fcsr import fcsr_matrix
from .random import SNP_to_coo
from .transform import coo_to_bcsr, coo_to_fcoo, fcoo_to_fcsr
from .types import FLOAT_STORAGE_np

import numpy as np
from scipy.sparse.csr import csr_matrix


def test_bcsr_matrix(m=None, n=None, density=.1, sym=False, return_np=True):
    while True:
        if sym:
            _m = _n = m or np.random.randint(1, 100)
        else:
            _m = m or np.random.randint(1, 100)
            _n = n or np.random.randint(1, 100)
        rows_array, cols_array, data_array, shape = SNP_to_coo(_m, _n, density, data_n=1, sym=sym)

        row_p, col_i = coo_to_bcsr(_m, len(rows_array), rows_array, cols_array)

        array_sparse = bcsr_matrix(row_p, col_i, (_m, _n))
        array_scipy = csr_matrix((data_array.astype(FLOAT_STORAGE_np), (rows_array, cols_array)), shape)

        if return_np:
            array_np = array_scipy.toarray()
            yield array_np, array_sparse, array_scipy
        else:
            yield array_sparse, array_scipy


def test_fcsr_matrix(m=None, n=None, k=None, data_n=1, density=.1, sym=False, return_np=True):
    while True:
        _k = k or np.random.randint(1, 10)
        if sym:
            _m = _n = m or np.random.randint(1, 100)
        else:
            _m = m or np.random.randint(1, 100)
            _n = n or np.random.randint(1, 100)

        rows_array, cols_array, data_array, shape = SNP_to_coo(_m, _n, density, data_n=_k, sym=sym)
        array_scipy = csr_matrix((data_array.astype(FLOAT_STORAGE_np), (rows_array, cols_array)), shape)

        fcoo = coo_to_fcoo(rows_array, cols_array, data_array, shape)
        fcsr = fcoo_to_fcsr(fcoo, shape)
        array_sparse = fcsr_matrix(fcsr, shape)

        if return_np:
            array_np = array_sparse.to_array()
            yield array_np, array_sparse, array_scipy
        else:
            yield array_sparse, array_scipy
