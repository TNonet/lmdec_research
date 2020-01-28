import numpy as np
from numba import jit
from .types import FLOAT_STORAGE_np, INDEX_STORAGE_np


@jit(nopython=True)
def SNP_to_coo(m, n, max_density, min_density=0, data_n=1, sym=False):
    """
    Creates a sparse M by N matrix following the procedure:

    A = [A1, A2, ..., AN] st.

    prob_Aj ~ U[min_density, max_density]
    A[i, j] ~ Binomial(Data, prob_Aj)

    Where data is n and prob_Aj is p in
        X ~ Binomial(n, p)


    However, this is not symmetrical so we adjust to:

    A = [0, A2, A3, ..., AN;
         0,  0, A3, ..., AN;
         0,  0, 0,  ..., AN;
         .
         0,  0, 0,  ..., AN;
         0,  0, 0,  ..., 0]

    A[i, j] ~ Binomial(Data_Range, prob_Aj) if i < j, else 0
    A <- A + A'

    A[i,i] ~ Binomial(Data_Range, prob_Aj)

    :param m: Number of Rows
    :param n: Number of Columns
    :param max_density: Maximum density of a column
    :param min_density: Minimum density of a column
    :param data_n: Generation of
    :param sym: Boolean is matrix symmetric or not
    """
    m = np.int64(m)
    n = np.int64(n)

    if sym and m != n:
        raise Exception('Symmetric Matrix must have n == m')

    binomials = np.random.uniform(min_density, max_density, size=n)
    rows = []
    cols = []
    data = []

    for j in range(n):
        b_temp = binomials[j]
        if sym:
            rng_max = min(j, m)  # Only fill triangle without diagonal
        else:
            rng_max = m
        for i in range(0, rng_max):
            data_temp = np.random.binomial(data_n, b_temp)
            if data_temp:
                # A[i,j] -> A
                rows.append(i)
                cols.append(j)
                data.append(data_temp)
                if sym:
                    # A[j, i] -> A transpose
                    rows.append(j)
                    cols.append(i)
                    data.append(data_temp)
        # A[j, j] -> Diag(A)
        if sym:
            data_temp = np.random.binomial(data_n, b_temp)
            if data_temp:
                rows.append(j)
                cols.append(j)
                data.append(data_temp)

    shape = (m, n)
    data_array = np.array(data).astype(FLOAT_STORAGE_np)
    rows_array = np.array(rows).astype(INDEX_STORAGE_np)
    cols_array = np.array(cols).astype(INDEX_STORAGE_np)

    return rows_array, cols_array, data_array, shape


@jit(nopython=True)
def rand_to_coo(m, n, sparsity, data_n=1, sym=False):
    """
    Creates a sparse M by N matrix following the procedure:

    A = a_{ij} st.


    a_{ij} ~ categorical([0, 1, ..., data_n],
            p=[1-sparsity, sparsity/data_n, ..., sparsity/data_n])


    :param m: Integer Number of Rows
    :param n: Integer Number of Columns
    :param sparsity: Float in (0, 1)
            Probability of a single value being non-zero
            Of the non-zero values they are selected with
            equal values.
    :param data_n: Integer >= 1.
            Allows the possible non-zero values to be selected
            from {1, 2, ..., data_n}
    :param sym: Boolean
            If True and m == n then will generate a symmetric matrix
                a_ij = aji ~ previously specified categorical distribution
    """
    m = np.int64(m)
    n = np.int64(n)

    if sym and m != n:
        raise Exception('Symmetric Matrix must have n == m')

    shape = (m, n)

    rows = []
    cols = []
    data = []

    if sym:
        for i in range(0, m):
            for j in range(i+1, n):
                if np.random.rand() <= sparsity/2:
                    rows.append(i)
                    rows.append(j)
                    cols.append(j)
                    cols.append(i)
                    data_temp = np.random.randint(0, data_n+1)
                    data.append(data_temp)
                    data.append(data_temp)
        for i in range(m):
            if np.random.rand() <= sparsity:
                rows.append(i)
                cols.append(i)
                data.append(np.random.randint(0, data_n + 1))
    else:
        for i in range(m):
            for j in range(n):
                if np.random.rand() <= sparsity:
                    rows.append(i)
                    cols.append(j)
                    data.append(np.random.randint(0, data_n+1))

    data_array = np.array(data).astype(FLOAT_STORAGE_np)
    rows_array = np.array(rows).astype(INDEX_STORAGE_np)
    cols_array = np.array(cols).astype(INDEX_STORAGE_np)

    return rows_array, cols_array, data_array, shape


@jit(nopython=True)
def SNP_rand_to_coo(m, n, max_density, noise, min_density=0, data_n=1, sym=False):
    """
    Creates a sparse M by N matrix following the procedure:

    A = [A1, A2, ..., AN] st.

    prob_Aj ~ U[min_density, max_density]

    Probability noise:
        A[i, j] ~ Categorical([0, 1, ..., data_n])
    Probability 1-noise:
        A[i, j] ~ Binomial(Data, prob_Aj)

    Where data is n and prob_Aj is p in
        X ~ Binomial(n, p)

    However, this is not symmetrical so we adjust to:

    A = [0, A2, A3, ..., AN;
         0,  0, A3, ..., AN;
         0,  0, 0,  ..., AN;
         .
         0,  0, 0,  ..., AN;
         0,  0, 0,  ..., 0]

    A[i, j] ~ drawn from the same distribution as above if i < j, else 0
    A <- A + A'

    A[i,i] ~ drawn from the same distribution as above

    :param m: Number of Rows
    :param n: Number of Columns
    :param max_density: Maximum density of a column
    :param noise: Probability that an element will be random
    :param min_density: Minimum density of a column
    :param data_n: Generation of
    :param sym: Boolean is matrix symmetric or not
    """
    m = np.int64(m)
    n = np.int64(n)

    if sym and m != n:
        raise Exception('Symmetric Matrix must have n == m')

    binomials = np.random.uniform(min_density, max_density, size=n)
    rows = []
    cols = []
    data = []

    for j in range(n):
        b_temp = binomials[j]
        if sym:
            rng_max = min(j, m)  # Only fill triangle without diagonal
        else:
            rng_max = m
        for i in range(0, rng_max):
            if np.random.rand() > noise:
                data_temp = np.random.binomial(data_n, b_temp)
            else:
                data_temp = np.random.randint(data_n+1)
            if data_temp:
                # A[i,j] -> A
                rows.append(i)
                cols.append(j)
                data.append(data_temp)
                if sym:
                    # A[j, i] -> A transpose
                    rows.append(j)
                    cols.append(i)
                    data.append(data_temp)
        # A[j, j] -> Diag(A)
        if sym:
            if np.random.rand() > noise:
                data_temp = np.random.binomial(data_n, b_temp)
            else:
                data_temp = np.random.randint(data_n + 1)
            if data_temp:
                rows.append(j)
                cols.append(j)
                data.append(data_temp)

    shape = (m, n)
    data_array = np.array(data).astype(FLOAT_STORAGE_np)
    rows_array = np.array(rows).astype(INDEX_STORAGE_np)
    cols_array = np.array(cols).astype(INDEX_STORAGE_np)

    return rows_array, cols_array, data_array, shape
