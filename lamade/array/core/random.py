from typing import Tuple, Union, List
import numpy as np
import warnings

from .logging import tlog


@tlog
def array_split(array_shape: Tuple[int, ...],
                p: Union[int, float],
                axis: int = 1,
                seed: int = 42,
                log: int = 1) -> Union[Tuple[np.ndarray, np.ndarray],
                                       Tuple[np.ndarray, np.ndarray, dict]]:
    """

    :param array_shape:
    :param p:
    :param axis:
    :param seed:
    :param log:
    :return:
    """

    sub_log = max(log - 1, 0)

    m, n = _array_shape_check(array_shape)

    if axis == 1:
        pass
    elif axis == 2:
        return array_split((n, m), p=p, axis=1, log=sub_log)
    else:
        raise Exception('Can only split on axis = 1 or 2. axis = {}'.format(axis))

    sample_size = int(p * m)

    if p > 1:
        raise Exception('Cannot split more than unity. p = {}'.format(p))
    elif p < 0:
        raise Exception('Cannot split non-positive fraction. p = {}'.format(p))

    if sample_size == m:
        warnings.warn('Degenerate Split. Decrease p. p = {} ~= 1'.format(p))
    elif sample_size == 0:
        warnings.warn('Degenerate Split. Decrease p. p = {} ~= 1'.format(p))
    else:
        pass

    index = np.arange(0, m)
    np.random.seed(seed=seed)
    np.random.shuffle(index)
    index_set = np.sort(index[0:sample_size])
    not_index_set = np.sort(index[sample_size:])

    if log:
        flog = {}
        return index_set, not_index_set, flog
    else:
        return index_set, not_index_set


def _array_shape_check(array_shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Checks to ensure that array is proper shape
    :param array_shape:
    :return:
    """
    if len(array_shape) == 2:
        m, n = array_shape
    elif len(array_shape) == 1:
        m = array_shape[0]
        n = 1
    else:
        raise Exception('Can only split 2-D arrays. Shape = {}'.format(array_shape))

    return m, n


@tlog
def array_partition(array_shape: Tuple[int, ...],
                    p: Union[float, int],
                    min_size: int = 5,
                    axis: int = 1,
                    log: int = 0) -> Tuple[List[slice], List[int]]:
    """

    :param min_size:
    :param array_shape:
    :param p:
    :param axis:
    :param log:
    :return:
    """
    if log:
        raise Exception('Function is not logged.')

    m, n = _array_shape_check(array_shape)

    if axis == 1:
        pass
    elif axis == 2:
        return array_partition((n, m), p=p, min_size=min_size, axis=1)
    else:
        raise Exception('Can only partition on axis = 1 or 2. axis = {}'.format(axis))

    if p <= 0:
        raise Exception('Cannot partition with non-positive geometric series.')
    elif p >= 1:
        raise Exception('Cannot partition with divergent geometric series.')

    if min_size < 1:
        raise Exception('Min size must be a positive integer.')

    n_left = n
    partition = []
    while n_left*p >= min_size:
        part = int(n_left*p)
        partition.append(part)
        n_left -= part

    if n_left:
        partition.append(n_left)

    if sum(partition) != n:
        raise Exception("This error should not have occurred. Please report")

    partition = sorted(partition, reverse=True)

    slices = []
    offset = 0

    for end in partition:
        slices.append(slice(offset, end+offset))
        offset += end

    return slices, partition
