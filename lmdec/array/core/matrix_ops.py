import dask.array as da
from dask.array.linalg import tsqr
from dask.array import broadcast_to
from dask.base import wait
import time
import numpy as np

from typing import Union, Optional, Tuple, TYPE_CHECKING

from lmdec.array.core.wrappers.time_logging import time_param_log
from .random import array_split, array_geometric_partition
from .types import ArrayType, LargeArrayType, DaskArrayType
from lmdec.array.core.wrappers.array_serialization import array_serializer

if TYPE_CHECKING:
    from lmdec.array.core.scaled import ScaledArray


@time_param_log
@array_serializer('x')
def subspace_to_SVD(x: ArrayType,
                    a: Optional["ScaledArray"] = None,
                    k: Optional[int] = None,
                    full_v: bool = False,
                    sqrt_s: bool = True,
                    log: int = 0) -> Union[Tuple[ArrayType, ArrayType, ArrayType],
                                           Tuple[ArrayType, ArrayType, ArrayType, dict]]:
    """Computes Truncated SVD of an array using an active subspace.

    Let A be a {N \times P} matrix
    Let x be a subspace of AA'

    U, S, V = subspace_to_SVD(x)
    USV.shape = (N, k)

    U, S, V = subspace_to_SVD(x, A, full_v=True)
    USV.shape == AA'.shape
    USV is low rank approximation to AA'

    Parameters
    ----------
    x : array_like, shape (N, K) or (N, )
        Active subspace of aa'
    a : array_like, shape (N, P), optional
        Array to be factored into USV from active subspace of x
    k : int
        Number of components of SVD to return
        1 <= k <= x.shape[1]
    full_v : bool
        Whether to return:
            V as a {K by K} matrix if full_v is False
            Or;
            V as a {K by P} matrix if full_v is True
                Requires a to be given as matrix
    sqrt_s : bool
        Whether to return the square root of the singular values of x.
    log : int >= 0
        Indicator in how many layers to log

    Returns
    -------
    u : (N, k) dask array
        Unitary array. Top k = min(K, k {if supplied}) left singular vectors of x
    s : (k, ) dask array
        Vector of of Top k = min(K, k {if supplied}) singular values in decreasing order
    v : (k, P) or (k, k) dask array
        Unitary array. Top k = min(K, k {if supplied}) right singular vectors of x.
        If full_v is True right singular vectors will be in (k, P)
        If full_v is false right singular vectors will be in (k, k)
    """
    flog = {}
    sub_log = max(log - 1, 0)

    U, S, V = tsqr(x, compute_svd=True)

    if full_v:
        dot_log = {'start': time.time()}
        x_t = a.T.dot(U)
        V, _, _ = tsqr(x_t, compute_svd=True)
        V = V.T
        dot_log['end'] = time.time()
        if sub_log:
            flog['dot'] = dot_log

    if sqrt_s:
        S = np.sqrt(S)

    if k:
        U, S, V = svd_to_trunc_svd(U, S, V, k=k)

    U = U.rechunk()
    V = V.rechunk()

    if log:
        return U, S, V, flog
    else:
        return U, S, V


def svd_to_trunc_svd(u: Optional[ArrayType] = None,
                     s: Optional[ArrayType] = None,
                     v: Optional[ArrayType] = None,
                     k: Optional[int] = None) -> Union[ArrayType, Tuple[ArrayType, ...]]:
    """Trims a full or partial SVD into a Truncated SVD with K Components

    Let X be an array of shape {n \times p}

    U, S, V = SVD(X)
    U of shape {n \times k}
    S of shape {k}
    V of shape {k \times p} where k = min(n, p)

    U, S, V = TruncSVD(X, k=10)
    U of shape {n \times k}
    S of shape {k}
    V of shape {k \times p}

    Parameters
    ----------
    u : array_like, shape (N, K)
        Left Singular Vectors of a matrix
    s : array_like, shape (K)
        Singular Values of a matrix
    v : array_like, shape (K, P)
        Right Singular Vectors of a matrix
    k : integer > 1
        Number of components in truncated SVD

    Returns
    -------
    uk: array_like, shape (N, k)
        Left Truncated Singular Vectors of a matrix
    s : array_like, shape (k)
        Truncated Singular Values of a matrix
    v : array_like, shape (k, P)
        Right Truncated Singular Vectors of a matrix
    """
    return_list = []

    if u is not None:
        u = u[:, :k]
        return_list.append(u)
    if s is not None:
        s = s[:k]
        return_list.append(s)
    if v is not None:
        v = v[:k, :]
        return_list.append(v)

    if len(return_list) == 1:
        return return_list[0]
    else:
        return (*return_list, )


def diag_dot(diag_array, x, return_diag=False):
    """Computes dot product between diag_array and x

    Parameters
    ----------
    diag_array : array_like, shape (K, ) or (K, 1)
                Diagonal entries of a diagonal maitrx
                [d1, d2, d3, ..., dk] -> [d1*e1, d2*e2, d2*e3, ..., dk*ek], where ei is the ith unit column vector
    x : array_like, shape (K, ...)
    return_diag : boolean
        If return_diag is True, return broadcasted array prepped for operation

    Returns
    -------
    out : array_like, shape of x
    """
    if len(x.shape) not in [1, 2]:
        raise ValueError("x must have (M, K) or (K, ). Current Shape = {}".format(x.shape))
    if diag_array.shape[0] != x.shape[0]:
        raise ValueError('shapes {} and {} not aligned: {} (dim 0 and 1) != {} (dim 0)'.format(diag_array.shape,
                                                                                               x.shape,
                                                                                               diag_array.shape[0],
                                                                                               x.shape[0]))
    if len(diag_array.shape) not in [1, 2]:
        raise ValueError('diag_array must have dimension (K, ) or (K, 1). Current shape = {}'.format(diag_array.shape))

    if len(x.shape) == 1:
        if len(diag_array.shape) == 2:
            d = np.squeeze(diag_array)
        else:
            d = diag_array
    else:
        if len(diag_array.shape) == 1:
            d = diag_array[:, np.newaxis]
        else:
            d = diag_array
        d = broadcast_to(d, x.shape)
    if return_diag:
        return d
    else:
        return np.multiply(d, x)


@time_param_log
@array_serializer('x')
def sym_mat_mult(array: LargeArrayType,
                 x: ArrayType,
                 scale: Union[float, int] = 1,
                 p: Union[int, float, slice] = 1,
                 seed: int = 42,
                 compute: bool = False,
                 log: int = 1) -> Union[ArrayType, Tuple[ArrayType, dict]]:
    """Peforms one iteration of the block power iteration

    x_{k+1} <- scale*A'(Ax) - beta*x_{past}


    :param array: Dask Array of shape (m by n)
    :param x: Dask Array of shape (n by k). k > 1
    :param p: Percentage of Array to compute dot product with:
        p == 1: Full Dot Product
        p < 1: Stochastic Dot Product
        type(p) == List[slice]
            Order sections
    :param scale: Scaling for Covariance/Gram Matrix Standards
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

    if scale != 1:
        x *= scale

    if log:
        flog = {}
        if sub_log:
            flog['dot'] = [dot_log]
            flog['T.dot'] = [tdot_log]
        return x, flog
    else:
        return x


@time_param_log
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

    partitions, dims = array_geometric_partition(x.shape, f=p, min_size=min_size, axis=2)

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


@time_param_log
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

    array_split_return = array_split(array.shape, f=p, axis=1, seed=seed, log=sub_log)

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


@time_param_log
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
