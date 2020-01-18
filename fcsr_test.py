import unittest
import numpy as np
from random_gen import test_fcsr_matrix


test_gen = test_fcsr_matrix()


class FiniteTestCase(unittest.TestCase):
    """Tests
    1. Vector Dot Product
        a) Square (n, n) * (n,)
        b) Non-Square (m, n) * (n,)
        c) [Failure] (n, k) * (n!=k,)
        d) [Failure] (m, k) * (n!=k,)
        e) [Failure] (m, n) * (n, k>1)
        f) [Failure] (m, n) * (n, k1>1, k2>1)
    2. Matrix Dot Product
        a) Square (n, n) * (n, 1)
        b) Square (n, n) * (n, k)
        c) Non-Square (m, n) * (n, 1)
        d) Non-Square (m, n) * (n, k)
        c) [Failure] (m, n) * (k1, k2)
    3. Transpose
        a) One Transpose Check with Numpy
        b) Double Transpose check with self and numpy
    """
    def test_Scipy(self):
        for i in range(10):
            array_np, array_sparse, array_scipy = next(test_gen)
            np.testing.assert_array_equal(array_np, array_scipy.toarray())
            np.testing.assert_array_equal(array_sparse.to_array(), array_scipy.toarray())

    def test_Vector(self):
        for i in range(10):
            array_np, array_sparse, _ = next(test_gen)
            m, n = array_np.shape
            u = np.random.randn(n)
            np.testing.assert_array_almost_equal(array_np.dot(u), array_sparse.dot1d(u))

    def test_Matrix(self):
        for i in range(10):
            k = np.random.randint(2, 100)
            array_np, array_sparse, _ = next(test_gen)
            m, n = array_np.shape
            u = np.random.randn(n, k)
            u = np.asfortranarray(u)
            np.testing.assert_array_almost_equal(array_np.dot(u), array_sparse.dot2d(u))


if __name__ == '__main__':
    unittest.main()
