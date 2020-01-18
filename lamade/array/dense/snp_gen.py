import numpy as np
import numba as nb
import dask
from dask import array
from dask.array.linalg import tsqr
import zarr
from numcodecs import Blosc
import h5py
import random 
import os

@nb.jit(nopython=True)
def SNP(m, n, max_density=.5, min_density=0.05, data_n=2, K=5):
    """
    Creates a sparse M by N matrix following the procedure:
    A = [A1, A2, ..., AN] st.
    prob_Aj ~ U[min_density, max_density]
    Probability noise:
        A[i, j] ~ Categorical([0, 1, ..., data_n])
    Probability 1-noise:
        A[i, j] ~ Binomial(data_n, prob_Aj)
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
    :param min_density: Minimum density of a column
    :param data_n: Number of indepdenent bernouli trials 
    """
    m = np.int64(m)
    n = np.int64(n)

    nk = n//K
    
    array = np.zeros((m,n),dtype=np.int8)

    for i in range(m):
        binomials = np.random.uniform(min_density, max_density, size=K)    
        for k in range(0, K):
            array[i,k*nk:(k+1)*nk] = np.random.binomial(data_n, binomials[k], nk)
            
                
    return array

@nb.jit(nopython=True)
def SNP_row(m, n, nk, max_density, min_density, data_n, K):
    binomials = np.random.uniform(min_density, max_density, size=K) 
    array = np.zeros(n, dtype=np.int8)
    for k in range(0, K):
        array[k*nk:(k+1)*nk] = np.random.binomial(data_n, binomials[k], nk)
    return array

@nb.jit(nopython=True)
def SNP_rows(m, n, c, nk, max_density, min_density, data_n, K):
    array = np.zeros((c, n), dtype=np.int8)

    for i in range(c):
        binomials = np.random.uniform(min_density, max_density, size=K) 
        for k in range(0, K):
            array[i, k*nk:(k+1)*nk] = np.random.binomial(data_n, binomials[k], nk)

    return array

def SNP_dask(m, n, max_density=.5, min_density=0.05, data_n=2, K=5, seed = 1, max_size = 1e8):
    #compressor = Blosc(cname='zstd', clevel=1)
    compressor = None
    z = zarr.zeros((m, n), dtype='i1', compressor = compressor)
    random.seed(a=seed)


    # max_size -> memory size of the section of rows to return at 1 time.

    c = max(1, int(max_size//n))
    num_iters = int(m//c)
    c_last = m - num_iters*c
    nk = n//K

    print(data_n)
    print('Creating SNP data with {} sections'.format(num_iters))

    for i in range(num_iters):
        if i % max(1, (num_iters//10)) == 0:
            print('Rows: {} out of: {}'.format(i*c, m))

        if c > 1:
            z[i*c:(i+1)*c, :] = SNP_rows(m, n, c, nk, max_density, min_density, data_n, K)
        else:
            z[i, :] = SNP_row(m, n, nk, max_density, min_density, data_n, K)

    for i in range(int(num_iters*c), m):
        z[i, :] = SNP_row(m, n, nk, max_density, min_density, data_n, K)

    z_dask = dask.array.from_zarr(z)
    del(z)
    z_dask = z_dask.rechunk({0: 'auto', 1: -1})

    return z_dask

def save_dask_array(name, m, n, max_density=.5, min_density=0.05,
                    data_n=2, K=5):
    array = SNP_dask(m, n, max_density, min_density, data_n, K)
    print("Compressing Array and Saving")
    #file_location = '/Users/tnonet/Documents/' + str(name)
    file_location = '/nobackup1/tnonet/' + str(name)
    abs_path = os.path.abspath(file_location)
    # Save to /nobackup1/tnonet
    dask.array.to_hdf5(abs_path, '/array', array, compression="lzf")