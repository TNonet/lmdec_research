import numpy as np
import numba as nb
import dask
from dask import array
from dask.array.linalg import tsqr
import random
import zarr
import time
import matplotlib.pyplot as plt
from numcodecs import Blosc
import h5py

from memory_profiler import profile
from memory_profiler import memory_usage

from .snp_gen import SNP_dask
from .metrics import scaled_svd_acc

import psutil

from .decomp import *

def svd_mem_compare(array_path, num_eigen, method,
                    max_density=.5, min_density=0.05,
                    data_n=2, K=5, logs = {}, **kwargs):
    """Calculates SVD decomposition for top k vectors.
    
    :param array_path: Load string_name as dask array
    :param n_power_iter: Number of Power Iterations to run
    :param method: String {'compressed', 'full'} Method for finding SVD
    :param max_density: Maximum density of a column
    :param min_density: Minimum density of a column
    :param data_n: Number of independent bernouli trials
    :K: Number of singular vectors to include in creating array
    :logs: Used to capture start time and end times for operations
    """
    
    #Generate Dask_Array
    logs['vars'] = locals()
    logs['start'] = time.time()
    logs['start_array'] = logs['start']

    f = h5py.File(array_path, 'r')
    array = dask.array.from_array(f['/array'])
    array = array.rechunk({0: 'auto', 1: -1})


    logs['end_array'] = time.time()
    
    #Run SVD with Memory Track
    svd_logs = {}
    track_svd(array, keys=kwargs, k=num_eigen, method=method, logs=svd_logs)    
    #Return INFO
    
    logs.update(svd_logs)
    logs['end'] = time.time()
        
    
def track_svd(array, keys, k=5, method='compressed', logs={}):
    """
    array: Dask Array
    """
    logs['start_svd'] = time.time()
    logs['k'] = k

    if 'n_power_iter' in keys:
        power_iter = keys['n_power_iter']
    else:
        power_iter = 10

    logs['n_power_iter'] = power_iter

    if method == 'compressed':
        u_g, s_g, _ = dask.array.linalg.svd_compressed(array, k, n_power_iter=power_iter)
        u_v, s_v = dask.persist(u_g, s_g)
        logs['over_column_sampling'] = 10
    elif method == 'block':
        if 'over_column_sampling' in keys:
            over_column_sampling = keys['over_column_sampling']
        else:
            over_column_sampling = 10

        if 'tsqr_period' in keys:
            tsqr_period = keys['tsqr_period']
        else:
            tsqr_period = 5

        if 'max_iter' in keys:
            max_iter = keys['max_iter']
        else:
            max_iter = 20

        logs['tsqr_period'] = tsqr_period
        logs['over_column_sampling'] = over_column_sampling
        logs['max_iter'] = max_iter
        logs['method'] = method
        logs['cpus'] = psutil.cpu_count()
        logs['avail_memory'] = psutil.virtual_memory()._asdict()

        n_blocks = int(np.ceil(max_iter/tsqr_period))
        u_g, s_g, _ = SVD_m1(array, k, n_blocks)
        u_v, s_v = dask.persist(u_g, s_g)


    logs['end_svd'] = time.time()
    logs['start_acc'] = time.time()
    logs['acc'] = scaled_svd_acc(array, u_v, s_v, square=True).compute()
    
    logs['end_acc'] = time.time()
