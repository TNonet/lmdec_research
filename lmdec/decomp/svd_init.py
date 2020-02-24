import dask
import dask.array as da
from dask.array.linalg import tsqr
from dask.array.random import RandomState
import numpy as np

from ..array.core.matrix_ops import full_svd_to_k_svd, sym_mat_mult, subspace_to_SVD
from ..array.core.metrics import scaled_svd_acc, relative_converge_acc
from lmdec import acc_format_svd
from ..array.core.random import array_split
from lmdec.array.core.wrappers.time_logging import tlog
from ..array.core.types import LargeArrayType, ArrayType

from typing import Tuple, Union


def v_svd_start(array: LargeArrayType,
                v: ArrayType,
                log: int = 1):
    """
    Given a V matrix from a
    :param array:
    :param v:
    :param log:
    :return:

    Given a matrix A
    """
    x_k = array.dot(v)
    U, _, _ = tsqr(x_k, compute_svd=True)

    U = U.rechunk('auto')

    if log:
        return U,  {}
    else:
        return U

@tlog
def eigengap_svd_start(array: LargeArrayType,
                       k: int,
                       b_max: int,
                       warm_start_row_factor: Union[int, float] = 5,
                       tol: float = 1e-6,
                       seed: int = 42,
                       log: int = 1):
    """Computes intelligent initial guess for:
        1) Active column (k + buffer) subspace of array 
        2) The "locally optimal" size of buffer that balance eigen_gap convergence rates and dot product times.

        
    Examples
    --------
    
    Suppose A has eigenvalues (all m1) of:
        [s1, s2, s2, ...,                         s10, ... ]
        [10,  9,  5, 4.99, 4.98, 4.97, 4.96, 3, 1, .5, ... ]
    
    If one desires the top three most active subspaces of A.
    Using the block power method and a buffer size of 0 the convergence rate would be:
        ~ lambda_3 / lambda_4 = 1.002
        Therefore each iteration only improves the overall solution by .2%.
        
    However, if we were to select a buffer of 5 then the convergence rate would be:
        ~lambda_3 / lambda_9 = 1.667
        Now the convergence rate is much higher and will require many fewer iterations 
        at the cost of each iteration taking more time.
        
    Moreover, if we select a buffer size of 6 then the convergence rate would be:
        ~lambda_3 / lambda_9 = 5.
        Here the marginal increase of compute dot products of an (n, 9) matrix vs an (n, 8) is 
        more than made up for with the massive increase in convergence rate.
        
    Method
    ------
    
    A in R(m by n) s.t. m < n
    
    x in R(n by (k + b')) s.t. (k + b') << n 
    
    Select (k + b_max)*warm_start_row_factor rows from A that form A1
    
    Perform Monitored SVD on A1
        U, S, V, log <- SVD(A1) 
        log contains operation costs of SVD:
            Operational cost of A1.dot(x)
        
    Extrapolate from A1.dot(x) to determine cost of full scale dot product with A.
    
    Using S and k, find eigen_gap for potential buffers from 0 -> b_max
    Using desired_tolerance predict how many iterations it will take to find convergence
    Using cost of each operation find time to finish iterations.

    :param tol:
    :param array:
    :param k:
    :param b_max:
    :param warm_start_row_factor:
    :param seed:
    :param log:
    :return:

    """

    flog = {}
    sub_log = max(0, log - 1)

    m, n = array.shape

    rows = warm_start_row_factor * (k + b_max)
    row_fraction = rows / m

    _sub_svd_start_return = _sub_svd_start(array,
                                           k=rows,
                                           row_sampling_fraction=row_fraction,
                                           seed=seed,
                                           log=sub_log)
    if sub_log:
        U, S, _, _sub_svd_start_log = _sub_svd_start_return
        flog[_sub_svd_start.__name__] = _sub_svd_start_log
    else:
        U, S, _ = _sub_svd_start_return

    U_0, S_0 = full_svd_to_k_svd(u=U, s=S, k=k+b_max)
    S_0_k = full_svd_to_k_svd(s=S_0, k=k)

    S_0_k = S_0_k.persist()

    x = array.T.dot(U_0)
    Q, _ = tsqr(x)
    x = sym_mat_mult(array, Q, p=1, compute=True, log=0)
    Q, _ = tsqr(x)

    U_1, S_1, _ = subspace_to_SVD(array, Q, compute=True, log=0)

    S_1_k = full_svd_to_k_svd(s=S_1, k=k)

    U_1, S_1_k, = dask.persist(U_1, S_1_k)

    relative_converge_acc_return = relative_converge_acc(S_0_k, S_1_k, log=sub_log)

    if sub_log:
        init_acc, relative_converge_acc_log = relative_converge_acc_return
    else:
        init_acc = relative_converge_acc_return

    try:
        S = S_1.compute()
    except AttributeError:
        S = S_1

    S_gap = float('inf') * np.ones_like(S)
    S_gap[k:b_max] = S[k - 1] / S[k:b_max]

    req_iter = _project_accuracy(init_acc, S_gap, tol)/6

    def k_cost(n_dimn):
        coeffs = np.array([0.00995696, 0.94308865])  # Found by brute force atm
        return np.polyval(coeffs, n_dimn)

    cost = _project_cost(k_cost, req_iter)

    buff_opt: int = np.argmin(cost)

    U_k = full_svd_to_k_svd(u=U_1, k=buff_opt)

    x = array.T.dot(U_k)

    if log:
        flog['S'] = S
        flog['req_iter'] = req_iter
        flog['costs'] = cost
        return x, flog
    else:
        return x


def _logbase(x: Union[np.ndarray, int, float],
             b: Union[np.ndarray, int, float]) -> np.ndarray:
    """
    Returns log_{b}{x}

    log_{b}(x) = ln(x)/ln(b)

    :param x: Numeric value or Array
    :param b: Numeric value or Array
    """
    return np.log(x) / np.log(b)


def _project_accuracy(acc_init: Union[int, float],
                      eigen_ratio: np.ndarray,
                      acc_final: Union[int, float]) -> np.ndarray:
    """
    Returns number of iterations to find desired tolerance.

    Suppose:
        Acc(i+c) = Acc(i)(1/eigen_ratio)^c

    If we desire Acc(i+c) <= acc_f

    Then c <= log_{acc_f/acc_0}(1/eigen_ratio)

    :param acc_init:
    :param eigen_ratio:
    :param acc_final:
    :return:
    """
    req_iter = _logbase(acc_final / acc_init, 1 / eigen_ratio)
    req_iter[req_iter == 0] = float('inf')
    return req_iter


def _project_cost(cost_func, req_iter):
    return [num_iter * cost_func(k) for k, num_iter in enumerate(req_iter)]


@tlog
def sample_svd_start(array: da.core.Array,
                     k: int,
                     warm_start_row_factor: int = 5,
                     seed: int = 42,
                     log: int = 1) -> Union[da.core.Array,
                                            Tuple[da.core.Array, dict]]:
    """
    Randomly selects rows of array to find initial SVD start.
    Returns best of k trial

    :param array:
    :param k:
    :param warm_start_row_factor:
    :param num_warm_starts:
    :param seed:
    :param log:
    :return:
    """
    rows = warm_start_row_factor * k
    m, n = array.shape
    row_fraction = rows / m

    sub_log = max(log - 1, 0)

    _sub_svd_start_return = _sub_svd_start(array,
                                           k=k,
                                           row_sampling_fraction=row_fraction,
                                           seed=seed,  # seed + i -> to prevent warm starts from being identical
                                           log=sub_log)
    if sub_log:
        U, S, _, _sub_svd_start_log = _sub_svd_start_return
    else:
        U, S, _ = _sub_svd_start_return

    if sub_log:
        flog = {_sub_svd_start.__name__: _sub_svd_start_log}
    else:
        flog = {}

    x = array.T.dot(U)
    q, _ = tsqr(x)

    if log:
        return q, flog
    else:
        return q


@tlog
def _sub_svd_start(array: da.core.Array,
                   k: int,
                   row_sampling_fraction: float = 1e-4,
                   seed: int = 42,
                   log: int = 1) -> Union[Tuple[da.core.Array, da.core.Array, da.core.Array],
                                          Tuple[da.core.Array, da.core.Array, da.core.Array, dict]]:
    """
    Performs SVD decomposition on a small section of rows to find a warm start for
    U to start block power iterations.

    :param array: Dask array of size m by n
    :param k: number of columns in U to return
    :param row_sampling_fraction: fraction of randomly selected rows of array to base SVD on.
    :param seed: randomization seed

    Shuffle A so that the desired rows are on top

    A.shape = (m, n)

    A = [A_start,
        A_rest]

    A_start.shape = (k, n)
    A_rest.shape = (m-k, n)

    U, S, V' <- SVD(A_start.T)

    U.shape <- (n, m_start)

    return top k columns of U[:, 0:k].
    """
    sub_log = max(log - 1, 0)
    flog = {}

    array_split_return = array_split(array_shape=array.shape,
                                     p=row_sampling_fraction,
                                     axis=1,
                                     seed=seed,
                                     log=sub_log)

    if sub_log:
        sub_array_index, _, array_split_log = array_split_return
        flog[array_split.__name__] = [array_split_log]
    else:
        sub_array_index, _ = array_split_return

    sub_array = array[sub_array_index, :].T
    sub_array = sub_array.rechunk({0: 'auto', 1: -1})

    _sub_svd_return = _sub_svd(array, sub_array, k=k, log=sub_log)
    if sub_log:
        U, S, V, _sub_svd_log = _sub_svd_return
        flog[_sub_svd.__name__] = [_sub_svd_log]
    else:
        U, S, V = _sub_svd_return

    if log:
        return U, S, V, flog
    else:
        return U, S, V


@tlog
def _sub_svd(array: da.core.Array,
             sub_array: da.core.Array,
             k: int = 5,
             log: int = 1) -> Union[Tuple[da.core.Array, da.core.Array, da.core.Array],
                                    Tuple[da.core.Array, da.core.Array, da.core.Array, dict]]:
    """
    Explanation of why

        USV = SVD(A)

        VSU' = SVD(A')

        Returning V here which is U
    :param array:
    :param sub_array:
    :param k:
    :param log:
    :return:
    """
    U, _, _ = tsqr(sub_array, compute_svd=True)

    U = U[:, :k]

    x_k = array.dot(U)
    U, S, V = tsqr(x_k, compute_svd=True)

    #U, S, V = full_svd_to_k_svd(u=U, s=S, v=V, k=k)

    U = U.rechunk('auto')
    S = S.rechunk('auto')
    V = V.rechunk('auto')

    if log:
        return U, S, V, {}
    else:
        return U, S, V


@tlog
def rnormal_start(array: da.core.Array,
                  k: int,
                  seed: int = 42,
                  log: int = 1) -> Union[da.core.Array,
                                         Tuple[da.core.Array, dict]]:
    """
    Returns a random start on U for block power iteration.
    :param array: Dask array of size m by n
    :param k: number of columns in U to return
    :param seed: randomization seed
    :param log:
    """
    m, n = array.shape
    state = RandomState(seed)

    omega = state.standard_normal(
        size=(n, k), chunks=(array.chunks[1], (k,)))

    if log:
        return omega, {}
    else:
        return omega
