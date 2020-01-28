import dask.array as da
import numpy as np
from dask.array.linalg import tsqr
import time
from typing import Union, Tuple

from ..array.core.types import LargeArrayType, ArrayType
from ..array.core.matrix_ops import sym_mat_mult, subspace_to_SVD
from ..array.core.metrics import acc_format_svd, approx_array_function, scaled_svd_acc
from .svd_init import rnormal_start, rerand_svd_start
from lmdec.array.core.wrappers.time_logging import tlog


# def SVD_m1(array: LargeArrayType,
#            k: int = 5,
#            max_iter: int = 20,
#            tol: float = 1e-16,
#            over_column_sampling: int = 10,
#            warm_start: bool = False,
#            init_row_sampling_factor: int = 5,
#            tsqr_period: int = 5) -> Tuple[ArrayType, ...]:
#     """
#     SVD_m1 returns an k part approxmation of the SVD of array.
#
#     :param array: dask array of size m by n
#     :param k: number of SVD components to approximate
#     :param max_iter: maximum number of power iterations
#     :param tol: tolerance objective, if found will return before max_iter is reached
#         if |A'Ax - lambda*x| < tol*|A'A|: return current solution
#         else: continue iterations
#     :param over_column_sampling: number of extra columns to include in iterations
#     :param tsqr_period: number of iterations to preform between TSQR
#     """
#
#     if warm_start:
#         x_k = rerand_svd_start(array,
#                                k + over_column_sampling,
#                                init_row_sampling_factor)
#     else:
#         x_k = rnormal_start(array, k + over_column_sampling)
#
#     # # # # Block Power Iteration Start
#     for i in range(max_iter):
#         print(i)
#         x_k = mat_mult(array, x_k, tsqr_period=-1, final_tsqr=True,
#                        n_power_iter=tsqr_period)
#         x_k_t = array.dot(x_k)
#         U_k, S_k, V_k = tsqr(x_k_t, compute_svd=True)
#         U_k, S_k, V_k = dask.persist(U_k, S_k, V_k)
#
#         if scaled_svd_acc(array, U_k[:, 0:k], S_k[:k]) < tol:
#             return full_svd_to_k_svd(U_k, S_k, V_k, k)
#
#     x_k_t = array.dot(x_k)
#     U_k, S_k, V_k = tsqr(x_k_t, compute_svd=True)
#
#     return full_svd_to_k_svd(U_k, S_k, V_k, k)


@tlog
def SVD_m2(array: LargeArrayType,
           k: int = 5,
           max_iter: int = 20,
           rtol: float = 1e-6,
           over_sampling: int = 10,
           warm_start: bool = False,
           warm_start_row_factor: int = 5,
           num_warm_starts: int = 5,
           metric_sample_factor: float = .1,
           error_sample_period: int = 5,
           tsqr_value_threshold: Union[float, int] = 1e12,
           tsqr_period_threshold: int = 5,
           seed: int = 42,
           log: int = 1,
           compute: bool = True) -> Tuple[Union[ArrayType, dict], ...]:
    """
    Finds the top k approximate Singular Values, and Left and Right Vectors of array.

    U, S, V = SVD_m2(array)

    :param array: Dask Array of size (m by n)
    :param k:
    :param max_iter:
    :param rtol:
    :param over_sampling:
    :param warm_start:
    :param num_warm_starts:
    :param warm_start_row_factor:
    :param metric_sample_factor:
    :param error_sample_period:
    :param tsqr_value_threshold:
    :param tsqr_period_threshold:
    :param seed:
    :param compute:
    :param log:
    :return: (U, S, V) SVD decomposition of array
    """

    flog = {}
    sub_log = max(log - 1, 0)

    vec_t = k + over_sampling

    if warm_start:
        rerand_svd_start_return = rerand_svd_start(array,
                                                   k=vec_t,
                                                   num_warm_starts=num_warm_starts,
                                                   warm_start_row_factor=warm_start_row_factor,
                                                   seed=seed,
                                                   log=sub_log)
        if sub_log:
            x, rerand_svd_start_log = rerand_svd_start_return
            flog[rerand_svd_start.__name__] = rerand_svd_start_log
        else:
            x = rerand_svd_start_return
    else:
        rnormal_start_return = rnormal_start(array, vec_t, log=sub_log, seed=seed)

        if sub_log:
            x, rnormal_start_log = rnormal_start_return
            flog[rnormal_start.__name__] = rnormal_start_log
        else:
            x = rnormal_start_return

    non_tsqr_iter_count = 0
    error_sample_count = 0

    amax = approx_array_function(da.max, p=metric_sample_factor)
    amin = approx_array_function(da.min, p=metric_sample_factor)

    sym_mat_mult_logs = []
    tsqr_logs = []
    svd_logs = []
    acc_logs = []

    S_old = 1 / k * np.ones(k)
    for i in range(max_iter):
        print('Iter: {}'.format(i))
        non_tsqr_iter_count += 1
        error_sample_count += 1

        sym_mat_mult_return = sym_mat_mult(array, x, log=sub_log, compute=compute)

        if sub_log:
            x, sym_mat_mult_log = sym_mat_mult_return
            sym_mat_mult_logs.append(sym_mat_mult_log)
        else:
            x = sym_mat_mult_return

        t = max(amax(x, log=0), -amin(x, log=0))  # Place holder for max value!
        t = t.persist()
        print('    Largest absolute value {}'.format(t.compute()))
        if t > tsqr_value_threshold or non_tsqr_iter_count > tsqr_period_threshold:
            print('    TSQR')
            tsqr_start = time.time()
            x, _ = tsqr(x)
            x = x.persist()
            tsqr_end = time.time()
            non_tsqr_iter_count = 0
            if sub_log:
                tsqr_logs.append({'start': tsqr_start,
                                  'end': tsqr_end,
                                  'params': {}})

            if error_sample_count > error_sample_period:
                error_sample_count = 0
                power_to_SVD_return = subspace_to_SVD(array, x, k=k, compute=compute, log=sub_log)
                if sub_log:
                    U_k, S_k, V_k, power_to_SVD_log = power_to_SVD_return
                    svd_logs.append(power_to_SVD_log)
                else:
                    U_k, S_k, V_k = power_to_SVD_return

                U_error, S_error = acc_format_svd(U_k, S_k, array.shape)

                scaled_svd_acc_return = scaled_svd_acc(array,
                                                       U_error,
                                                       S_error,
                                                       p=metric_sample_factor,
                                                       compute=compute,
                                                       log=sub_log)

                if sub_log:
                    temp_acc, scaled_svd_acc_log = scaled_svd_acc_return
                    acc_logs.append(scaled_svd_acc_log)
                else:
                    temp_acc = scaled_svd_acc_return

                print('    Scaled Accuracy: {}'.format(temp_acc.compute()))
                print('    S Accuracy:      {}'.format(da.linalg.norm(S_old - S_k).compute()))
                S_old = S_k

                if temp_acc < rtol:
                    break
        else:
            if sub_log:
                tsqr_logs.append({})

    power_to_SVD_return = subspace_to_SVD(array, x, k=k, compute=compute, log=sub_log)

    if sub_log:
        U_k, S_k, V_k, power_to_SVD_log = power_to_SVD_return
        svd_logs.append(power_to_SVD_log)
    else:
        U_k, S_k, V_k = power_to_SVD_return

    if sub_log:
        flog[sym_mat_mult.__name__] = sym_mat_mult_logs
        flog[tsqr.__name__] = tsqr_logs
        flog[subspace_to_SVD.__name__] = svd_logs
        flog[scaled_svd_acc.__name__] = acc_logs

    if log:
        return U_k, S_k, V_k, flog
    else:
        return U_k, S_k, V_k
