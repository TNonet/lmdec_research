from typing import Tuple, Union, List
import numpy as np
import warnings

from lamade.array.core.wrappers.time_logging import tlog


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
        warnings.warn('Degenerate Split. Decrease p. p = {} ~= 0'.format(p))
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
def array_geometric_partition(array_shape: Tuple[int, ...],
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
        return array_geometric_partition((n, m), p=p, min_size=min_size, axis=1)
    else:
        raise Exception('Can only partition on axis = 1 or 2. axis = {}'.format(axis))

    if p <= 0:
        raise Exception('Cannot partition with non-positive geometric series.')
    elif p >= 1:
        raise Exception('Cannot partition with divergent geometric series.')

    if min_size < 1:
        raise Exception('Min size must be a positive integer.')

    m_left = m
    partition = []
    while m_left * p >= min_size:
        part = int(m_left * p)
        partition.append(part)
        m_left -= part

    if m_left:
        partition.append(m_left)

    if sum(partition) != m:
        raise Exception("This error should not have occurred. Please report")

    partition = sorted(partition, reverse=True)

    slices = []
    offset = 0

    for end in partition:
        slices.append(slice(offset, end + offset))
        offset += end

    return slices, partition


@tlog
def array_constant_partition(array_shape: Tuple[int, ...],
                             p: Union[float, int],
                             min_size: int = 5,
                             axis: int = 1,
                             log: int = 0) -> List[slice]:
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
        return array_constant_partition((n, m), p=p, min_size=min_size, axis=1)
    else:
        raise Exception('Can only partition on axis = 1 or 2. axis = {}'.format(axis))

    if p <= 0:
        raise Exception('Cannot partition with non-positive fraction.')
    elif p >= 1:
        raise Exception('Cannot partition with fraction more than 1.')

    if min_size < 1:
        raise Exception('Min size must be a positive integer.')

    num_partitions = int(1/p)
    partition_size = m // num_partitions
    if num_partitions == 1:
        raise warnings.warn('Only 1 Partition was created. int(1/p) == 1')

    slices = []
    if m % num_partitions == 0:
        # Partition work perfectly -> Split perfectly
        sections = range(0, m, partition_size)
    else:
        sections = list(range(0, m-partition_size, partition_size))
        sections.append(m-1)

    for start in sections:
        slices.append(slice(start, start + partition_size))

    return slices

# Axis 1 Testing
# for p in [.001, .1, .2, .5]:
#     a_test = np.random.rand(1000, 1000)
#     parts = array_constant_partition(a_test.shape, p=.1)
#     a_copy = np.zeros_like(a_test)
#     for part in parts:
#         a_copy[part, :] = a_test[part, :]
#     np.testing.assert_almost_equal(a_copy, a_test)

# Axis 2 Testing
# for p in [.001, .1, .2, .5]:
#     a_test = np.random.rand(1000, 1000)
#     parts = array_constant_partition(a_test.shape, p=.1, axis=2)
#     a_copy = np.zeros_like(a_test)
#     for part in parts:
#         a_copy[:, part] = a_test[:, part]
#     np.testing.assert_almost_equal(a_copy, a_test)
