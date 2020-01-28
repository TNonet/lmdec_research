import dask
import dask.array as da
from dask.array.linalg import tsqr
from dask.base import wait
import time
import numpy as np

from typing import Union, Optional, Tuple, List

from lmdec.array.core.wrappers.time_logging import tlog
from .random import array_split, array_geometric_partition
from .types import ArrayType, LargeArrayType, DaskArrayType
from lmdec.array.core.wrappers.array_serialization import array_serializer


@tlog
@array_serializer('x')
def subspace_to_SVD(array: Union[LargeArrayType, ArrayType],
                    x: ArrayType,
                    k: Optional[int] = None,
                    compute: bool = False,
                    log: int = 1) -> Union[Tuple[ArrayType, ArrayType, ArrayType],
                                           Tuple[ArrayType, ArrayType, ArrayType, dict]]:
    """
    :param array:
    :param x:
    :param k:
    :param compute:
    :param log:
    :return:
    """
    flog = {}
    sub_log = max(log - 1, 0)

    dot_log = {'start': time.time()}
    x_t = array.dot(x)
    if compute and isinstance(array, DaskArrayType):
        x_t = x_t.persist()

    dot_log['end'] = time.time()

    tsqr_log = {'start': time.time()}

    U, S, V = tsqr(x_t, compute_svd=True)
    if compute:
        U, S, V = dask.persist(U, S, V)
    tsqr_log['end'] = time.time()

    U_k, S_k, V_k = full_svd_to_k_svd(U, S, V, k=k)

    if compute:
        U_k, S_k, V_k = dask.persist(U_k, S_k, V_k)

    if sub_log:
        flog['dot'] = dot_log
        flog['tsqr'] = tsqr_log

    if log:
        return U_k, S_k, V_k, flog
    else:
        return U_k, S_k, V_k


@tlog
@array_serializer('x')
def sym_mat_mult(array: LargeArrayType,
                 x: ArrayType,
                 p: Union[int, float, slice] = 1,
                 seed: int = 42,
                 compute: bool = False,
                 log: int = 1) -> Union[ArrayType, Tuple[ArrayType, dict]]:
    """Peforms one iteration of the block power iteration with momentum (optional)

    x_{k+1} <- A'(Ax) - beta*x_{past}


    :param array: Dask Array of shape (m by n)
    :param x: Dask Array of shape (n by k). k > 1
    :param p: Percentage of Array to compute dot product with:
        p == 1: Full Dot Product
        p < 1: Stochastic Dot Product
        type(p) == List[slice]
            Order sections
    :param seed:
    :param compute:
        We do not pre-compute x. As any function in SVDecomp should return a computed function
        if it is deemed necessary
    :param log:
    :return: Dask Array of shape (n by k)
    """
    sub_log = max(0, log - 1)

    if isinstance(p, slice):
        return _l_approx_sym_mat_mult(array=array, x=x, l=p, compute=compute, log=sub_log)
    elif p < 1:
        return _p_approx_sym_mat_mult(array=array, x=x, p=p, seed=seed, compute=compute, log=sub_log)

    dot_log = {'start': time.time()}
    x = array.dot(x)
    if compute and isinstance(x, DaskArrayType):
        x = x.persist()
        wait(x)
    dot_log['end'] = time.time()

    tdot_log = {'start': time.time()}
    x = array.T.dot(x)
    if compute and isinstance(x, DaskArrayType):
        x = x.persist()
        wait(x)
    tdot_log['end'] = time.time()

    if log:
        flog = {}
        if sub_log:
            flog['dot'] = [dot_log]
            flog['T.dot'] = [tdot_log]
        return x, flog
    else:
        return x


def full_svd_to_k_svd(u: Optional[ArrayType] = None,
                      s: Optional[ArrayType] = None,
                      v: Optional[ArrayType] = None,
                      k: Optional[int] = None,
                      log: int = 0) -> Union[ArrayType, Tuple[ArrayType, ...]]:
    """Removes buffer from SVD decomposition

    :param u:
    :param s:
    :param v:
    :param k:
    :param log:
    :return:
    """
    if log:
        raise Exception('Does not return Logging')

    return_list = []
    if u is not None:
        u = u[:, :k]
        return_list.append(u)
    if s is not None:
        s = s[:k]
        return_list.append(s)
    if v is not None:
        v = v.T
        v = v[:k, :]
        return_list.append(v)

    if len(return_list) == 1:
        return return_list[0]
    else:
        return (*return_list,)


@tlog
@array_serializer('x')
def _time_sym_mat_mult(array: LargeArrayType,
                       x: ArrayType,
                       p: float = 0.5,
                       min_size: int = 5,
                       compute: bool = True,
                       log: int = 1) -> Union[ArrayType, Tuple[ArrayType, dict]]:
    """Splits x up into variable column widths to determine computation of cost of;

        array.dot(x) s.t. x in R(n, k) as k varies.

    Let x = [x_s1, x_s2, x_s3, ..., x_si]

    Where x_si.shape ~= (n, k*p^i)

    A'Ax = [A'Ax_s1, A'Ax_s2, A'Ax_s3, ..., A'Ax_s4]

    Compute must be True to find accurate timings.

    :param array:
    :param x:
    :param p:
    :param compute:
    :param log:
    :return:
    """

    if not log:
        return sym_mat_mult(array=array, x=x, p=p, compute=compute, log=0)
    if not compute:
        raise Exception("Compute must be true to accurately record time of computations.")

    partitions, dims = array_geometric_partition(x.shape, p=p, min_size=min_size, axis=2)

    partition_dot = []
    partition_log = []
    for part in partitions:
        x_i, sym_mat_mult_log = sym_mat_mult(array, x[:, part], p=1, compute=True, log=1)
        partition_dot.append(x_i)
        partition_log.append(sym_mat_mult_log['end'] - sym_mat_mult_log['start'])

    coeffs = np.polyfit(dims, partition_log, deg=1)

    flog = {'run_time_coeffs': coeffs,
            'partition_dot_time': partition_log,
            'partition_dot_size': dims}

    if isinstance(x, DaskArrayType):
        return da.rechunk(da.hstack(partition_dot)), flog
    else:
        return np.hstack(partition_dot), flog


@tlog
@array_serializer('x')
def _p_approx_sym_mat_mult(array: LargeArrayType,
                           x: ArrayType,
                           p: Union[float, int] = .1,
                           seed: int = 42,
                           compute: bool = False,
                           log: int = 1) -> Union[ArrayType, Tuple[ArrayType, dict]]:
    """

    :param array:
    :param x:
    :param p:
    :param seed:
    :param compute:
    :param log:
    :return:
    """
    flog = {}
    sub_log = max(log - 1, 0)

    array_split_return = array_split(array.shape, p=p, seed=seed, axis=1, log=sub_log)

    if sub_log:
        index_sex, _, array_split_log = array_split_return
        flog[array_split.__name__] = array_split_log
    else:
        index_sex, _ = array_split_return

    sub_array = array[index_sex, :]

    sym_mat_mult_return = sym_mat_mult(sub_array, x, p=1, compute=compute, log=sub_log)

    if sub_log:
        x, sym_mat_mult_log = sym_mat_mult_return
        flog[sym_mat_mult.__name__] = sym_mat_mult_log
    else:
        x = sym_mat_mult_return

    if log:
        return x, flog
    else:
        return x


@tlog
@array_serializer('x')
def _l_approx_sym_mat_mult(array: LargeArrayType,
                           x: ArrayType,
                           l: slice,
                           compute: bool = False,
                           log: int = 1) -> Union[ArrayType, Tuple[ArrayType, dict]]:
    """

    :param array:
    :param x:
    :param l:
    :param compute:
    :param log:
    :return:
    """
    flog = {}
    sub_log = max(log - 1, 0)

    sub_array = array[l, :]

    sym_mat_mult_return = sym_mat_mult(sub_array, x, p=1, compute=compute, log=sub_log)

    if sub_log:
        x, sym_mat_mult_log = sym_mat_mult_return
        flog[sym_mat_mult.__name__] = sym_mat_mult_log
    else:
        x = sym_mat_mult_return

    if log:
        return x, flog
    else:
        return x
