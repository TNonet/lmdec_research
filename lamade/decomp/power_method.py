from typing import Union, Tuple
import dask
from dask.array.linalg import tsqr
import numpy as np
import time

from ..array.core.types import LargeArrayType, ArrayType
from ..array.core.matrix_ops import sym_mat_mult, subspace_to_SVD, full_svd_to_k_svd
from .svd_init import rerand_svd_start, rnormal_start, eigengap_svd_start
from ..array.core.metrics import relative_converge_acc


class PowerMethod:
    """

    """

    def __init__(self, max_iter: int = 100,
                 k: int = 5,
                 scoring_method: str = 'q-vals',
                 p: Union[int, float] = 1,
                 scoring_tol: Union[float, int] = .1,
                 buffer: int = 10,
                 seed: int = 42,
                 compute=True,
                 sub_svd_start: Union[bool, str] = True,
                 init_row_sampling_factor: int = 5,
                 num_warm_starts: int = 1):

        """
        """
        self.max_iter = max_iter
        self.k = k
        if scoring_method == 'q-vals':
            self.scoring_method = scoring_method
        else:
            raise NotImplemented('Other scoring methods are not implemented.')

        self.scoring_tol = scoring_tol
        self.p = p
        self.buffer = buffer
        self.sub_svd_start = sub_svd_start
        self.init_row_sampling_factor = init_row_sampling_factor
        self.seed = seed
        self.compute = compute
        self.num_warm_starts = num_warm_starts

        self.logs = {}

    def __start(self, array: ArrayType, seed: int) -> ArrayType:
        """

        :param array:
        :return:
        """
        print('Normal Start Method')
        vec_t = self.k + self.buffer

        if self.sub_svd_start:
            x = rerand_svd_start(array,
                                 k=vec_t,
                                 num_warm_starts=self.num_warm_starts,
                                 warm_start_row_factor=self.init_row_sampling_factor,
                                 seed=seed,
                                 log=0)
        else:
            x = rnormal_start(array, vec_t, log=0, seed=seed)

        U, S, _ = subspace_to_SVD(array, x, k=self.k, compute=False, log=0)
        U_k, S_k = full_svd_to_k_svd(U, S, k=self.k)

        U_k, S_k = dask.persist(U_k, S_k)

        # First item already exists from __start
        self.logs['tol'] = [np.nan]
        self.logs['S'].append(S_k.compute())

        return x

    def svd(self, array: ArrayType, seed: Union[None, int] = None) -> Tuple[ArrayType, ArrayType, ArrayType]:
        """Computes SVD of array

        :return:
        """
        self.logs = {'tol': [], 'S': [], 'start': time.time()}

        if seed is None:
            seed = self.seed

        x = self.__start(array, seed)

        for i in range(self.max_iter):
            x = self.__step(array, x, seed)
            self.__score(array, x)

            if self.logs['tol'][-1] < self.scoring_tol:
                break

        U_k, S_k, V_k = self.__svd(array, x)
        self.logs['end'] = time.time()
        return U_k, S_k, V_k

    def __step(self, array: ArrayType, x: ArrayType, seed: int) -> ArrayType:
        """
        array: da.core.Array,
                 x: da.core.Array,
                 p: Union[int, float],
                 seed: int = 42,
                 compute: bool = False,
                 log: int = 1)
        :param array:
        :param x:
        :param seed:
        :return:
        """
        x = sym_mat_mult(array, x, self.p, seed=seed, compute=self.compute, log=0)

        q, _ = tsqr(x)

        q = q.persist()

        return q

    def __score(self, array: ArrayType, x: ArrayType) -> None:
        """

        :param array:
        :param x:
        :return:
        """
        U, S, _ = subspace_to_SVD(array, x, k=self.k, compute=False, log=0)
        U_k, S_k = full_svd_to_k_svd(U, S, k=self.k)
        U_k, S_k = dask.persist(U_k, S_k)
        # First item already exists from __start
        self.logs['tol'].append(relative_converge_acc(S_k, self.logs['S'][-1], log=0).compute())

        self.logs['S'].append(S_k.compute())


    def __svd(self, array: ArrayType, x: ArrayType) -> Tuple[ArrayType, ArrayType, ArrayType]:
        """

        :param array:
        :param x:
        :param: seed:
        :return:
        """
        U, S, V = subspace_to_SVD(array, x, k=self.k, compute=False, log=0)

        U_k, S_k, V_k = full_svd_to_k_svd(U, S, V, k=self.k)
        U_k, S_k, V_k = dask.persist(U_k, S_k, V_k)

        return U_k, S_k, V_k


class PowerMethodEigenStart(PowerMethod):

    def __init__(self, buffer_max: int = 50, reg: Union[int, float] = .01, *args, **kwargs) -> None:
        """
        Only difference is
        :param buffer_max:
        :param reg:
        """
        self.buffer_max = buffer_max
        self.reg = reg
        self.buff_opt = None
        self.seed = 10
        super().__init__(*args, **kwargs)

    def __start(self, array: ArrayType, seed: int) -> ArrayType:
        """

        :param array:
        :return:
        """
        print('Eigen Start Method')
        x = eigengap_svd_start(array,
                               k=self.k,
                               b_max=self.buffer_max,
                               reg=self.reg,
                               tol=self.scoring_tol,
                               warm_start_row_factor=self.init_row_sampling_factor,
                               seed=seed,
                               log=0)

        m, n = x.shape

        self.buff_opt = n - self.k
        print(n - self.k)

        U, S, _ = subspace_to_SVD(array, x, k=self.k, compute=False, log=0)
        U_k, S_k = full_svd_to_k_svd(U, S, k=self.k)

        U_k, S_k = dask.persist(U_k, S_k)

        # First item already exists from __start
        self.logs['tol'] = [np.nan]
        self.logs['S'].append(S_k.compute())

        return x

    def svd(self, array: ArrayType, seed: Union[None, int] = None) -> Tuple[ArrayType, ArrayType, ArrayType]:
        """Computes SVD of array

        :return:
        """
        self.logs = {'tol': [], 'S': [], 'start': time.time()}

        if seed is None:
            seed = self.seed

        x = self.__start(array, seed)

        for i in range(self.max_iter):
            x = self.__step(array, x, seed)
            self.__score(array, x)

            if self.logs['tol'][-1] < self.scoring_tol:
                break

        U_k, S_k, V_k = self.__svd(array, x)
        self.logs['end'] = time.time()
        return U_k, S_k, V_k

    def __step(self, array: ArrayType, x: ArrayType, seed: int) -> ArrayType:
        """
        array: da.core.Array,
                 x: da.core.Array,
                 p: Union[int, float],
                 seed: int = 42,
                 compute: bool = False,
                 log: int = 1)
        :param array:
        :param x:
        :param seed:
        :return:
        """
        x = sym_mat_mult(array, x, self.p, seed=seed, compute=self.compute, log=0)

        q, _ = tsqr(x)

        q = q.persist()

        return q

    def __score(self, array: ArrayType, x: ArrayType) -> None:
        """

        :param array:
        :param x:
        :return:
        """
        U, S, _ = subspace_to_SVD(array, x, k=self.k, compute=False, log=0)
        U_k, S_k = full_svd_to_k_svd(U, S, k=self.k)
        U_k, S_k = dask.persist(U_k, S_k)
        # First item already exists from __start
        self.logs['tol'].append(relative_converge_acc(S_k, self.logs['S'][-1], log=0).compute())

        self.logs['S'].append(S_k.compute())


    def __svd(self, array: ArrayType, x: ArrayType) -> Tuple[ArrayType, ArrayType, ArrayType]:
        """

        :param array:
        :param x:
        :param: seed:
        :return:
        """
        U, S, V = subspace_to_SVD(array, x, k=self.k, compute=False, log=0)

        U_k, S_k, V_k = full_svd_to_k_svd(U, S, V, k=self.k)
        U_k, S_k, V_k = dask.persist(U_k, S_k, V_k)

        return U_k, S_k, V_k
