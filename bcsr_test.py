import unittest
import numpy as np

from fsm.sparse.bcsr import bcsr_matrix
from fsm.sparse.transform import coo_to_bcsr, dense_to_coo
from random_gen import test_bcsr_matrix


test_gen = test_bcsr_matrix()


class BinaryTestCase(unittest.TestCase):
    """Tests
    1. Identity
        a) Variable Shapes (10, 100)
    2. Random
        100x) Variable Shapes (m, n) from SNP
    3. Vector Dot Product
        a) Square (n, n) * (n,)
        b) Non-Square (m, n) * (n,)
        c) [Failure] (n, k) * (n!=k,)
        d) [Failure] (m, k) * (n!=k,)
        e) [Failure] (m, n) * (n, k>1)
        f) [Failure] (m, n) * (n, k1>1, k2>1)
    4. Matrix Dot Product
        a) Square (n, n) * (n, 1)
        b) Square (n, n) * (n, k)
        c) Non-Square (m, n) * (n, 1)
        d) Non-Square (m, n) * (n, k)
        c) [Failure] (m, n) * (k1, k2)
    5. Transpose
        a) One Transpose Check with Numpy
        b) Double Transpose check with self and numpy
    """
    def test_Identity(self):
        for i in range(10):
            n = 2**i
            array_np = np.eye(n)
            rows, cols, data, shape = dense_to_coo(array_np)
            row_p, col_i = coo_to_bcsr(n, len(rows), rows, cols)
            array_sparse = bcsr_matrix(row_p, col_i, (n, n))

            np.testing.assert_array_equal(array_np, array_sparse.to_array())

    def test_Random(self):
        for i in range(10):
            array_np, array_sparse, _ = next(test_gen)
            np.testing.assert_array_equal(array_np, array_sparse.to_array())

    def test_Vector(self):
        for i in range(10):
            array_np, array_sparse, _ = next(test_gen)
            m, n = array_np.shape
            u = np.random.normal(size=n)
            np.testing.assert_array_almost_equal(array_np.dot(u), array_sparse.dot1d(u))

    @unittest.expectedFailure
    def test_Vector_shape_mismatch(self):
        array_np, array_sparse, _ = next(test_gen)
        m, n = array_np.shape
        u = np.random.normal(size=n+1)
        array_sparse.dot1d(u)

    @unittest.expectedFailure
    def test_Vector_shape_overloads(self):
        array_np, array_sparse, _ = next(test_gen)
        m, n = array_np.shape
        u = np.random.rand(n, n, n)
        array_sparse.dot1d(u)

    def test_Matrix(self):
        for i in range(10):
            array_np, array_sparse, _ = next(test_gen)
            k = np.random.randint(2, 100)
            m, n = array_np.shape
            u = np.random.rand(n, k)
            u = np.asfortranarray(u)
            np.testing.assert_array_almost_equal(array_np.dot(u), array_sparse.dot2d(u))

    def test_Matrix_1d(self):
        for i in range(10):
            array_np, array_sparse, _ = next(test_gen)
            m, n = array_np.shape
            u = np.random.rand(n, 1)
            u = np.asfortranarray(u)
            np.testing.assert_array_almost_equal(array_np.dot(u), array_sparse.dot2d(u))

    @unittest.expectedFailure
    def test_Matrix_shape_mistmatch(self):
        for i in range(10):
            array_np, array_sparse, _ = next(test_gen)
            k = np.random.randint(2, 100)
            m, n = array_np.shape
            u = np.random.rand(k, n+1).T
            np.testing.assert_array_almost_equal(array_np.dot(u), array_sparse.dot2d(u))

    @unittest.expectedFailure
    def test_Matrix_shape_overload(self):
        for i in range(10):
            array_np, array_sparse, _ = next(test_gen)
            k = np.random.randint(2, 100)
            m, n = array_np.shape
            u = np.random.rand(k, k, n).T
            np.testing.assert_array_almost_equal(array_np.dot(u), array_sparse.dot2d(u))

    def test_Transpose(self):
        for i in range(10):
            array_np, array_sparse, _ = next(test_gen)
            np.testing.assert_array_equal(array_np.T, array_sparse.T.to_array())

    def test_SNP_sym(self):
        temp_test_gen = test_bcsr_matrix(sym=True)
        for i in range(10):
            array_np, array_sparse, _ = next(temp_test_gen)
            np.testing.assert_array_equal(array_np, array_sparse.T.to_array())
            np.testing.assert_array_equal(array_np.T, array_sparse.to_array())

    def test_Transpose_Transpose(self):
        for i in range(10):
            array_np, array_sparse, _ = next(test_gen)
            np.testing.assert_array_equal(array_sparse.T.T.to_array(), array_sparse.to_array())

    def test_Scipy(self):
        for i in range(10):
            array_np, array_sparse, array_scipy = next(test_gen)
            np.testing.assert_array_equal(array_np, array_scipy.toarray())
            np.testing.assert_array_equal(array_sparse.to_array(), array_scipy.toarray())


if __name__ == '__main__':
    unittest.main()
