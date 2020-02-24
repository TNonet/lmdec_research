import numpy as np
import dask.array as da
import warnings
from math import sqrt

from .random import array_split
from lmdec.array.core.wrappers.time_logging import tlog
from .types import ArrayType, LargeArrayType, DaskArrayType

from typing import Tuple, Union, Callable


def approx_array_function(array_func: Callable,
                          p: Union[int, float],
                          log: int = 1) -> Callable:
    """
    Goal: Takes in a dask function

    :param array_func: dask or numpy function
    :param p:
    :param log:
    :return:
    """

    @tlog
    def _f(array, log=log):
        flog = {}
        sub_log = max(log - 1, 0)  # Logging reflect number of sub function calls to record

        return_tuple = array_split(array.shape, p=p, axis=1, log=sub_log)
        if sub_log:
            index_set, _, array_split_log = return_tuple
            flog[array_split.__name__] = array_split_log
        else:
            index_set, _ = return_tuple

        array1 = array[index_set, :]

        if log:
            return array_func(array1), flog
        else:
            return array_func(array1)

    return _f


@tlog
def rmse_k(array: LargeArrayType,
           u: ArrayType,
           s: ArrayType,
           p: Union[float, int] = 1,
           log: int = 1) -> Union[Tuple[float, dict], float]:
    """
    Computes RMSE_k Norm

    Assumes data went through acc_format_svd(..., square=True).

    Therefore -> s_i = (s_i)^2

    sqrt((1/nk) * sum(||(1/m)A'Au_i - u_i*s_i|| for i = 1, ..., k))

    Where ||.|| refers to (||.||_2)^2

    This is equivalent to

    sqrt((1/nk)*(||(1/m)*A'Au - us||_{F})^2)

    :param array: Matrix A
    :param u: array_like of singular vectors
    :param s: array_like of singular values
    :param p: numeric between (0, 1] representing percentage of data to use
    :param log: integer representing number of functions and sub_function layers to log.
    :return:
    """

    n, m = array.shape
    _, k = u.shape

    flog = {}

    acc = da.linalg.norm((1/n) * (array.dot(array.T.dot(u)) - u.dot(s)), ord='fro')
    acc = da.sqrt(acc**2/(m*k)).compute()

    if log:
        return acc, flog
    else:
        return acc


@tlog
def scaled_relative_converge_acc(x: ArrayType,
                                 y: ArrayType,
                                 p: Union[int, float] = 1,
                                 norm: Union[str,int] = 2,
                                 log: int = 1) -> Union[float, Tuple[float, dict]]:
    """
    Computes scaled relative change between x and y.

    Let step be an iterative generator that converges to x*.

    {x_0, x_1, ..., x_k-1, x_k, x_k+1, ... x*}

    Thus:
        x_k = step(x_{k-1})

    Quotient Accuracy is:

        2||x_k - x_{k-1}||/(||x_k|| + ||x_{k-1}||) , where ||.|| is the norm specified.

    If 0 < p < 1:

        Where I selects I rows of x_k
        2||x_k[I] - x_{k-1}[I]||/(||x_k[I]|| + ||x_{k-1}[I]||) , where ||.|| is the norm specified.

    :param x:
    :param y:
    :param p:
    :param norm:
    :param log:
    :return:
    """
    sub_log = max(0, log - 1)
    flog = {}

    _relative_converge_acc_return = relative_converge_acc(x=x,
                                                          y=y,
                                                          norm=norm,
                                                          p=p,
                                                          log=sub_log)
    if sub_log:
        acc, _relative_converge_acc_return_log = _relative_converge_acc_return
        flog[relative_converge_acc.__name__] = _relative_converge_acc_return_log
    else:
        acc = _relative_converge_acc_return

    scale = np.linalg.norm(x, norm) + np.linalg.norm(y, norm)
    acc /= scale

    if log:
        return acc, flog
    else:
        return acc


@tlog
def relative_converge_acc(x: ArrayType,
                          y: ArrayType,
                          p: Union[int, float] = 1,
                          norm: Union[str, int] = 2,
                          log: int = 1) -> Union[float, Tuple[float, dict]]:
    """
    Computes relative change between x and y.

    Let step be an iterative generator that converges to x*.

    {x_0, x_1, ..., x_k-1, x_k, x_k+1, ... x*}

    Thus:
        x_k = step(x_{k-1})

    Quotient Accuracy is:

        ||x_k - x_{k-1}|| , where ||.|| is the norm specified.

    If 0 < p < 1:

        Where I selects I rows of x_k
        ||x_k[I] - x_{k-1}[I]|| , where ||.|| is the norm specified.



    :param x:
    :param y:
    :param p:
    :param norm:
    :param log:
    :return:
    """
    flog = {}
    sub_log = max(0, log - 1)
    if p <= 0:
        raise Exception('p must be positive, referring to percentage of data tested in norm.')
    elif p > 1:
        raise Exception('p must be equal to or less than unit, referring to percentage of data tested in norm.')
    elif p < 1:
        raise NotImplemented('Approximate Relative Norm not implemented yet')

    else:
        _successive_norm_return = _successive_norm(x=x,
                                                   y=y,
                                                   norm=norm,
                                                   log=sub_log)
        if sub_log:
            acc, _successive_norm_return_log = _successive_norm_return
            flog[_successive_norm.__name__] = _successive_norm_return_log
        else:
            acc = _successive_norm_return

    if isinstance(acc, DaskArrayType):
        acc = acc.compute()

    if log:
        return acc, flog
    else:
        return acc


@tlog
def scaled_svd_acc(array: LargeArrayType,
                   u: ArrayType,
                   s: ArrayType,
                   scale: Union[float, int] = 1,
                   norm: Union[int, str] = 'fro',
                   p: Union[int, float] = 1,
                   log: int = 1) -> Union[float, Tuple[float, dict]]:
    """
    Computes scaled accuracy of u and s being left eigen pairs of AA'

    ||AA'u - s*u||/||AA'u||

    Where ||.|| is the specified norm.

    :param array: dask array of shape m by n
    :param u:
    :param s:
    :param scale:
    :param norm:
    :param p:
    :param log
    :return:
    """

    flog = {}

    if p <= 0:
        raise Exception('p must be positive, referring to percentage of data tested in norm.')
    elif p > 1:
        raise Exception('p must be equal to or less than unit, referring to percentage of data tested in norm.')
    elif p < 1:
        sub_log = max(0, log - 1)
        _approx_scaled_svd_acc_return = _approx_scaled_svd_acc(array=array,
                                                               u=u,
                                                               s=s,
                                                               scale=scale,
                                                               p=p,
                                                               norm=norm,
                                                               log=sub_log)
        if sub_log:
            acc, _approx_scaled_svd_acc_log = _approx_scaled_svd_acc_return
            flog[_approx_scaled_svd_acc.__name__] = _approx_scaled_svd_acc_log
        else:
            acc = _approx_scaled_svd_acc_return

    else:
        denom = (array.dot(array.T.dot(u)))

        if scale != 1:
            denom *= scale

        acc = _scaled_norm(denom - u.dot(s), denom, norm)

    if isinstance(acc, DaskArrayType):
        acc = acc.compute()

    if log:
        return acc, flog
    else:
        return acc


@tlog
def _approx_scaled_svd_acc(array: LargeArrayType,
                           u: ArrayType,
                           s: ArrayType,
                           scale: Union[float, int] = 1,
                           p: float = 0.1,
                           norm: Union[int, str] = 'fro',
                           log: int = 1) -> Union[float, Tuple[float, dict]]:
    """
    Compute approximate scaled accuracy of u and s being left eigen pairs of AA'.

    :param array: dask array of shape m by n
    :param u:
    :param s:
    :param p:
    :param norm:
    :param log:
    :return:

    Non-Approximation:

        ||AA'u - u*s||/||AA'u||

            Where ||.|| is the specified norm.

    A in R(m, n), m < n
    U in R(m, k), m << k

    Grab rows of A!

    I <- subset of {1, 2, ...., m} of size floor(f*m)

    A1 = A[I, :] -> A1.shape = (f, n)
    A2 = A[~I, :] ->  A2.shape = (m-f, n)

    A is congruent by elementary matrix operations to:
        [A1
         A2]

    U1 = U[I, :] -> U1.shape = (f, k)
    U2 = U[~I, :] -> U2.shape = (m-f, k)

    U is congruent by elementary matrix operations to:
        [U1,
         U2]

    AA'U - US == [A1A1'U1 + A1A2'U2 - U1S
                 [A2A1'U1 + A2A2'U2 - U2S]

    If we only want an approximation we can look at:

        ||A1A1'U1 + A1A2'U2 - U1S|| / ||A1A1'U1 + A1A2'U2||

    If f << 1 then this operation is faster than the full computation.
    If the matrix is believe to be homogeneous the accuracy can be quite comparable
    to that of the full computation.
    """
    flog = {}
    sub_log = max(log - 1, 0)

    return_tuple = array_split(array.shape, p=p, axis=1, log=sub_log)

    if sub_log:
        index_set, not_index_set, array_split_log = return_tuple
        flog[array_split.__name__] = array_split_log
    else:
        index_set, not_index_set = return_tuple

    array1 = array[index_set, :]
    array2 = array[not_index_set, :]
    u1 = u[index_set, :]
    u2 = u[not_index_set, :]

    denom = array1.dot(array1.T.dot(u1) + array2.T.dot(u2))

    if scale != 1:
        denom *= 1

    acc = _scaled_norm(denom - u1.dot(s), denom, norm, log=0)

    if isinstance(acc, DaskArrayType):
        acc = acc.compute()

    if log:
        return acc, flog
    else:
        return acc


@tlog
def _scaled_norm(a: ArrayType,
                 scale: ArrayType,
                 norm: Union[str, int],
                 log: int = 0) -> float:
    """
    Computes ||a||/||scale|| where norm is as specified in parameters

    :param a:
    :param scale:
    :param norm:
    :param log:
    :return:
    """
    if log:
        raise Exception("Function only build computation graph. No log is returned")

    if norm in ['nuc', '2', 2]:
        warnings.warn('Nuclear and 2 norm are costly operations.'
                      + '\n See https://docs.dask.org/en/latest/array-api.html for norm list')

    return np.linalg.norm(a, norm) / np.linalg.norm(scale, norm)


@tlog
def _successive_norm(x: ArrayType,
                     y: ArrayType,
                     norm: Union[str, int, float] = 2,
                     log: int = 1) -> Union[ArrayType, Tuple[ArrayType, dict]]:
    """
    Computes ||x - y|| where norm is as specified in parameters


    :param x:
    :param y:
    :param norm:
    :param log:
    :return:
    """
    acc = np.linalg.norm(x - y, norm)

    if log:
        return acc, {}
    else:
        return acc
