import numpy as np
import warnings

from .random import array_split
from .logging import tlog
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


def acc_format_svd(u: ArrayType,
                   s: ArrayType,
                   array_shape: Tuple[int, int],
                   square: bool = True,
                   log: int = 0) -> Tuple[ArrayType, ArrayType]:
    """
    Handles variable shapes and formats for U, S and returns
    proper formatting U, S for error calculations.

    Singular vectors are assumed ground truth and will adjust singular values

    :param u: Left Singular Vectors
    :param s: Singular Values or square root of Singluar Values
    :param array_shape:
    :param square: Indictor of state of s.
    :param log:

    :return:
    """
    if log:
        raise Exception("Does not return Logging")

    m, n = array_shape

    if len(u.shape) == 1:
        # Single singular vector -> expects 1 singular value
        m_vector = u.shape[1]
        k_vector = 1
    elif len(u.shape) == 2:
        m_vector, k_vector = u.shape
    else:
        raise Exception('u is of the wrong shape, {}. \n Must be of shape: ({}, k) or ({},)'.format(
            u.shape, m, m))

    s_copy = s.copy()
    if square:
        s_copy = np.square(s_copy)

    if len(s_copy.shape) == 1:
        k_values = s_copy.shape[0]
        s_copy: ArrayType = np.diag(s_copy)
    elif len(s_copy.shape) == 2:
        k_values1, k_values2 = s_copy.shape
        if k_values1 != k_values2:
            raise Exception('s is the wrong shape, {}. \n Must be of shape ({}, {}) or ({},)'.format(
                s_copy.shape,
                k_vector,
                k_vector,
                k_vector))
        k_values = k_values1
        s_copy: ArrayType = np.diag(np.diag(s_copy))
    else:
        raise Exception('s is the wrong shape, {}. \n Must be of shape ({}, {}) or ({},)'.format(
            s_copy.shape,
            k_vector,
            k_vector,
            k_vector))

    if m == m_vector:
        pass
    elif m != m_vector and m == k_vector:
        # U might be coming in as Transpose U
        u = u.T
        m_vector, k_vector = k_vector, m_vector
    else:
        raise Exception('u or s is of the wrong shape')

    if k_values != k_vector:
        raise Exception('u or s is of the wrong shape')

    return u, s_copy


@tlog
def relative_converge_acc(x: ArrayType,
                          y: ArrayType,
                          p: Union[int, float] = 1,
                          norm: str = 2,
                          compute: bool = True,
                          log: int = 1) -> Union[ArrayType,
                                                 Tuple[ArrayType, dict]]:
    """
    Computes relative change between x and y.

    Let step be an iterative generator that conveges to x*.

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
    :param compute:
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

    if compute and isinstance(acc, DaskArrayType):
        acc = acc.persist()

    if log:
        return acc, flog
    else:
        return acc


@tlog
def scaled_svd_acc(array: LargeArrayType,
                   u: ArrayType,
                   s: ArrayType,
                   norm: str = 'fro',
                   p: Union[int, float] = 1,
                   compute: bool = True,
                   log: int = 1) -> Union[ArrayType,
                                          Tuple[ArrayType, dict]]:
    """
    Computes scaled accuracy of u and s being left eigen pairs of AA'

    ||AA'u - s*u||/||AA'u||

    Where ||.|| is the specified norm.

    :param array: dask array of shape m by n
    :param u:
    :param s:
    :param norm:
    :param p:
    :param compute:
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
                                                               p=p,
                                                               norm=norm,
                                                               compute=compute,
                                                               log=sub_log)
        if sub_log:
            acc, _approx_scaled_svd_acc_log = _approx_scaled_svd_acc_return
            flog[_approx_scaled_svd_acc.__name__] = _approx_scaled_svd_acc_log
        else:
            acc = _approx_scaled_svd_acc_return

    else:
        denom = array.dot(array.T.dot(u))

        acc = _scaled_norm(denom - u.dot(s), denom, norm)

    if compute and isinstance(acc, DaskArrayType):
        acc = acc.persist()

    if log:
        return acc, flog
    else:
        return acc


@tlog
def _approx_scaled_svd_acc(array: LargeArrayType,
                           u: ArrayType,
                           s: ArrayType,
                           p: float = 0.1,
                           norm: str = 'fro',
                           compute: bool = True,
                           log: int = 1) -> Union[ArrayType,
                                                  Tuple[ArrayType, dict]]:
    """
    Compute approximate scaled accuracy of u and s being left eigen pairs of AA'.

    :param array: dask array of shape m by n
    :param u:
    :param s:
    :param p:
    :param norm:
    :param compute:
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

    acc = _scaled_norm(denom - u1.dot(s), denom, norm, log=0)

    if compute and isinstance(acc, DaskArrayType):
        acc = acc.persist()

    if log:
        return acc, flog
    else:
        return acc


@tlog
def _scaled_norm(a: ArrayType,
                 scale: ArrayType,
                 norm: Union[str, int],
                 log: int = 0) -> ArrayType:
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
