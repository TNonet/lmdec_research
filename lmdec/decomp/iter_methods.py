from abc import ABCMeta, abstractmethod

from typing import Union, List

import dask
from dask.array.linalg import tsqr
import numpy as np
import time

from ..array.core.types import LargeArrayType, ArrayType
from ..array.core.matrix_ops import sym_mat_mult, subspace_to_SVD, full_svd_to_k_svd
from .svd_init import rerand_svd_start, rnormal_start, eigengap_svd_start
from ..array.core.metrics import relative_converge_acc


class _IterAlgo(metaclass=ABCMeta):
    """Base Class for All Iteration Based Methods that fit the following form:

    Required Functions:
        Initialization Method, I, with parameters (data, *args, **kwargs)
        Update Solution, u, with parameters (data, x_i, *args, **kwargs)
        Update parameters, p, with parameters (data, x_i, acc_i, *args, **kwargs)
        Calculate Solution Quality, s, with parameters (data, x_i, *args, **kwargs)
        Check Solution, c, with parameters (data, x_i, *args, **kwargs)
        Finalization Method, f, with parameters (data, x_i, *args, **kwargs)

    x_0 <- I(data, *args, **kwargs)

    for i in {0, 1, 2, ..., max_iter}:
        acc_i <- s(data, x_i, *args, **kwargs)

        if c(x_i):
            break out of loop

        args, kwargs <- p(data, x_i, acc_i, *args, *kwargs)
        x_i+1 <- u(data, x_i, *args, *kwargs)

    x_f <- f(data, x_i, *args, **kwargs)
    """

    def __init__(self, max_iter: int = 50, log: Union[str, List] = 'acc'):
        self.max_iter = max_iter

        if isinstance(log, str):
            log = [log]

        self.log = log
        self.logs = {}

    @abstractmethod
    def __initialization(self, data, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def __solution_update(self, data, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def __parameter_update(self, data, x, acc,  *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __solution_accuracy(self, data, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __accuracy_check(self, data, x, acc, *args, **kwargs) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __finilization(self, data, x, *args, **kwargs):
        raise NotImplementedError

    def __fit(self, data, *args, **kwargs):
        x_k = self.__initialization(data, *args, **kwargs)

        for i in range(self.max_iter):
            acc_k = self.__solution_accuracy(data, x_k, *args, **kwargs)
            sol_k = self.__accuracy_check(data, x_k, acc_k, *args, **kwargs)

            if sol_k:
                break

            args, kwargs = self.__parameter_update(data, x_k, *args, **kwargs)

            x_k = self.__solution_accuracy(data, x_k, *args, **kwargs)

        return self.__finilization(data, x_k, *args, **kwargs)


class PowerMethod(_IterAlgo):

    def __init__(self, max_iter: int = 50, k: int = 5, scoring_method: str = 'q-vals', p: Union[int, float] = 1,
                 scoring_tol: Union[float, int] = .1, buffer: int = 10, seed: int = 42, compute=True,
                 sub_svd_start: Union[bool, str] = True, init_row_sampling_factor: int = 5, num_warm_starts: int = 1):

        """
        """
        super().__init__(max_iter)
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

    def __initialization(self, data, *args, **kwargs):
        print('Normal Start Method')
        vec_t = self.k + self.buffer

        if self.sub_svd_start:
            x = rerand_svd_start(data,
                                 k=vec_t,
                                 num_warm_starts=self.num_warm_starts,
                                 warm_start_row_factor=self.init_row_sampling_factor,
                                 log=0)
        else:
            x = rnormal_start(data, vec_t, log=0)

        U, S, _ = subspace_to_SVD(data, x, k=self.k, compute=False, log=0)
        U_k, S_k = full_svd_to_k_svd(U, S, k=self.k)
        U_k, S_k = dask.persist(U_k, S_k)

        # First item already exists from __start
        self.logs['tol'] = [np.nan]
        self.logs['S'].append(S_k.compute())

        return x

    def __solution_update(self, data, x, *args, **kwargs):
        x = sym_mat_mult(data, x, self.p, compute=self.compute, log=0)

        q, _ = tsqr(x)

        q = q.persist()
        return q

    def __parameter_update(self, data, x, acc, *args, **kwargs):
        return args, kwargs

    def __solution_accuracy(self, data, x, *args, **kwargs):
        U, S, _ = subspace_to_SVD(dasta, x, k=self.k, compute=False, log=0)
        U_k, S_k = full_svd_to_k_svd(U, S, k=self.k)
        U_k, S_k = dask.persist(U_k, S_k)
        # First item already exists from __start
        self.logs['tol'].append(relative_converge_acc(S_k, self.logs['S'][-1], log=0).compute())

        self.logs['S'].append(S_k.compute())
        return S_k.compute()

    def __accuracy_check(self, data, x, acc, *args, **kwargs) -> bool:
        return acc <= self.scoring_tol

    def __finilization(self, data, x, *args, **kwargs):
        U, S, V = subspace_to_SVD(data, x, k=self.k, compute=False, log=0)

        U_k, S_k, V_k = full_svd_to_k_svd(U, S, V, k=self.k)
        U_k, S_k, V_k = dask.persist(U_k, S_k, V_k)

        return U_k, S_k, V_k


class EigenPowerMethod(PowerMethod, _IterAlgo):

    def __init__(self, max_iter: int = 50, k: int = 5, max_buffer: int = 50, lift: int = 1,
                 scoring_method: str = 'q-vals', p: Union[int, float] = 1, scoring_tol: Union[float, int] = .1,
                 seed: int = 42, compute=True, init_row_sampling_factor: int = 5):
        """
        Super with key word arguments.

        :param max_iter:
        :param k:
        :param max_buffer:
        :param lift:
        :param scoring_method:
        :param p:
        :param scoring_tol:
        :param seed:
        :param compute:
        :param init_row_sampling_factor:
        """
        super().__init__(max_iter, k, scoring_method, p, scoring_tol, seed, compute, init_row_sampling_factor)
        self.max_buffer = max_buffer
        self.lift = lift


    # def __init__(self, max_iter: int = 50, k: int = 5, scoring_method: str = 'q-vals', p: Union[int, float] = 1,
    #              scoring_tol: Union[float, int] = .1, seed: int = 42, compute=True, init_row_sampling_factor: int = 5):
    #     super().__init__(max_iter=max_iter,
    #                      k=k,
    #                      scoring_method=scoring_method,
    #                      p=p,
    #                      scoring_tol=scoring_tol,
    #                      seed=seed,
    #                      compute=compute,
    #                      init_row_sampling_factor=init_row_sampling_factor)

    def __initialization(self, data, *args, **kwargs):
        print('Eigen Start Method')
        x = eigengap_svd_start(data,
                               k=self.k,
                               b_max=self.max_buffer,
                               reg=self.reg,
                               tol=self.scoring_tol,
                               warm_start_row_factor=self.init_row_sampling_factor,
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

