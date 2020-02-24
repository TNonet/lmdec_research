from abc import ABCMeta, abstractmethod

from typing import Union

import dask
import dask.array as da
from dask.array.linalg import tsqr
import numpy as np
import time

from pandas_plink import read_rel

from ..array.core.matrix_ops import sym_mat_mult, subspace_to_SVD, full_svd_to_k_svd
from .svd_init import sample_svd_start, rnormal_start, eigengap_svd_start, v_svd_start
from ..array.core.metrics import relative_converge_acc, scaled_relative_converge_acc, rmse_k
from ..array.core.transform import acc_format_svd
from ..array.core.random import array_constant_partition
from ..array.core.types import ArrayType


class _IterAlgo(metaclass=ABCMeta):
    """Base Class for All Iteration Based Methods that fit the following form:

    Required Functions:
        Initialization Method, I, with parameters (data, *args, **kwargs)
        Update Solution, u, with parameters (data, x_i, *args, **kwargs)
        Update parameters, p, with parameters (data, x_i, acc_i, *args, **kwargs)
        Calculate Solution Quality, s, with parameters (data, x_i, *args, **kwargs)
        Check Solution, c, with parameters (data, x_i, *args, **kwargs)
        Finalization Method, f, with parameters (data, x_i, *args, **kwargs)


    Algorithm Overview

        x_0 <- I(data, *args, **kwargs)

        for i in {0, 1, 2, ..., max_iter}:
            acc_i <- s(data, x_i, *args, **kwargs)

            if c(x_i):
                break out of loop

            args, kwargs <- p(data, x_i, acc_i, *args, *kwargs)
            x_i+1 <- u(data, x_i, *args, *kwargs)

        x_f <- f(data, x_i, *args, **kwargs)
    """

    def __init__(self, max_iter: int = 50, scoring_method: str = 'q-vals', p: Union[int, float] = 1,
                 tol: Union[float, int] = .1):
        self.max_iter = max_iter
        self.p = p
        self.tol = tol

        self.scoring_method = scoring_method
        if scoring_method not in ['q-vals', 'q-vects', 'rmse']:
            raise Exception('Must use scoring method in {}'.format(['q-vals',
                                                                    'q-vects',
                                                                    'rmse']))

        if scoring_method in ['q-vects', 'rmse']:
            self.is_vector_iterations = True
            self.vector_iterations = []
        else:
            self.is_vector_iterations = False
            self.vector_iterations = None

        self.tol_iterations = []
        self.value_iterations = []

    @abstractmethod
    def _initialization(self, data, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _solution_step(self, data, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _parameter_step(self, data, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _solution_accuracy(self, data, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _finalization(self, data, x, *args, **kwargs):
        raise NotImplementedError

    def q_vals(self, s_k, scale=True, p=1, norm=2):
        try:
            if scale:
                return scaled_relative_converge_acc(s_k,
                                                    self.value_iterations[-1],
                                                    p=p,
                                                    norm=norm,
                                                    log=0)
            else:
                return relative_converge_acc(s_k,
                                             self.value_iterations[-1],
                                             p=p,
                                             norm=norm,
                                             log=0)
        except IndexError:
            return float('inf')

    def q_vects(self, vect_k, scale=True, p=1, norm='fro'):
        """

        :param vect_k:
        :param scale:
        :param p:
        :param norm:
        :return:

        Assume u_k are left singular vectors!
        """
        try:
            if scale:
                return scaled_relative_converge_acc(vect_k,
                                                    self.vector_iterations[-1],
                                                    p=p,
                                                    norm=norm,
                                                    log=0)
            else:
                return relative_converge_acc(vect_k,
                                             self.vector_iterations[-1],
                                             p=p,
                                             norm=norm,
                                             log=0)
        except IndexError:
            return float('INF')

    @staticmethod
    def rmse(array, u_k, s_k, p=1):
        u_k, s_k = acc_format_svd(u_k, s_k**2, array.shape)
        return rmse_k(array=array, u=u_k, s=s_k, p=p, log=0)

    @staticmethod
    def plink_to_dask(file_path):
        # https://pypi.org/project/pandas-plink/
        # https://pythonhosted.org/pandas_plink/

        K = read_rel(file_path)
        return K.values

    def svd(self, data, *args, **kwargs):
        if isinstance(data, str):
            try:
                data = self.plink_to_dask(data)
            except Exception:
                raise Exception('Error loading {}. Should be in .rel.bin format. \n'
                                'See: https://pypi.org/project/pandas-plink/')
        self.time = time.time()
        x_k = self._initialization(data, *args, **kwargs)

        for i in range(self.max_iter):
            acc = self._solution_accuracy(data, x_k, *args, **kwargs)

            if acc <= self.tol:
                break

            self._parameter_step(data, x_k, *args, **kwargs)

            x_k = self._solution_step(data, x_k, *args, **kwargs)

        U, S, V =  self._finalization(data, x_k, *args, **kwargs)
        self.time = time.time() - self.time
        return U, S, V


class PowerMethod(_IterAlgo):

    def __init__(self, k: int = 10,  max_iter: int = 50, scoring_method: str = 'q-vals', p: Union[int, float] = 1,
                 tol: Union[float, int] = .1, buffer: int = 10, seed: int = 42,
                 sub_svd_start: Union[bool, str] = True, init_row_sampling_factor: int = 5):

        """
        """
        super().__init__(max_iter=max_iter,
                         scoring_method=scoring_method,
                         p=p,
                         tol=tol)
        self.k = k
        self.buffer = buffer
        self.sub_svd_start = sub_svd_start
        self.init_row_sampling_factor = init_row_sampling_factor
        self.seed = seed
        self.compute = True

    def _initialization(self, data, *args, **kwargs):
        vec_t = self.k + self.buffer

        if self.sub_svd_start:
            x = sample_svd_start(data,
                                 k=vec_t,
                                 warm_start_row_factor=self.init_row_sampling_factor,
                                 log=0)
        else:
            x = rnormal_start(data, vec_t, log=0)


        return x

    def _solution_step(self, data, x, *args, **kwargs):
        n, p = data.shape
        x = sym_mat_mult(array=data, x=x, p=self.p, scale=1/n, compute=self.compute, log=0)

        q, _ = tsqr(x)

        q = q.persist()
        return q

    def _parameter_step(self, data, x, *args, **kwargs):
        pass

    def _solution_accuracy(self, array, x, *args, **kwargs):
        U, S, _ = subspace_to_SVD(array, x, k=self.k, compute=False, log=0)
        U_k, S_k = full_svd_to_k_svd(U, S, k=self.k)
        U_k, S_k = dask.persist(U_k, S_k)

        # First item already exists from __start
        if self.scoring_method == 'q-vals':
            acc = self.q_vals(S_k)
        else:
            # Need to record vectors!
            if self.scoring_method == 'q-vects':
                acc = self.q_vects(U_k)
            else:
                acc = self.rmse(array, U_k, S_k)
            try:
                _U_k = U_k.compute()
            except AttributeError:
                _U_k = U_k
            self.vector_iterations.append(_U_k)

        try:
            _S_k = S_k.compute()
        except AttributeError:
            _S_k = S_k

        self.value_iterations.append(_S_k)

        try:
            acc = acc.compute()
        except AttributeError:
            pass

        self.tol_iterations.append(acc)

        return acc

    def _finalization(self, data, x, *args, **kwargs):
        return subspace_to_SVD(data, x, k=self.k, compute=False, full_v=True, log=0)


class EigenPowerMethod(PowerMethod):

    def __init__(self, max_iter: int = 50, k: int = 5, max_buffer: int = 50,
                 scoring_method: str = 'q-vals', p: Union[int, float] = 1, tol: Union[float, int] = .1,
                 seed: int = 42, init_row_sampling_factor: int = 5):
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
        super().__init__(max_iter=max_iter,
                         k=k,
                         scoring_method=scoring_method,
                         p=p,
                         tol=tol,
                         seed=seed)
        self.init_row_sampling_factor = self.init_row_sampling_factor
        self.max_buffer = max_buffer

    def _initialization(self, data, *args, **kwargs):
        x = eigengap_svd_start(data,
                               k=self.k,
                               b_max=self.max_buffer,
                               tol=self.tol,
                               warm_start_row_factor=self.init_row_sampling_factor,
                               log=0)

        m, n = x.shape

        self.buff_opt = n - self.k

        return x


class _vPowerMethod(PowerMethod):
    """

    """

    def __init__(self, v_start: ArrayType, k: int = 10, buffer: int = 10, max_iter: int = 50,
                 scoring_method: str = 'q-vals', p: Union[int, float] = 1, tol: Union[float, int] = .1):
        """
        """
        super().__init__(max_iter=max_iter,
                         scoring_method=scoring_method,
                         p=p,
                         tol=tol)
        self.v_start = v_start
        self.k = k
        self.buffer = buffer

    def _initialization(self, data, *args, **kwargs):
        return self.v_start

    def _finalization(self, data, x, *args, **kwargs):
        return subspace_to_SVD(data, x, k=self.k+self.buffer, compute=False, full_v=True, log=0)


class SuccessiveStochasticPowerMethod:
    """ Implementation of Power Method that uses more and more of the matrix

    Suppose A exists in R(n by p)

    If we have reason to believe that that rows of A come from a small number of samples.

    Or, in other words a Truncated SVD with k << n explains much of the variance in the n rows of A.

    Therefore, it would be reasonable to assume that with just n' < n rows of A we can accurately learn the same
    factors that we would learn with all n rows.

    Algorithm:

        A <- Shuffle(A)
        I_rows = [0:n1, 0:n2, 0:n3, ..., 0:n]
        U0, S0, V0 <- Direct Truncated SVD of Small Subset Rows of A'

        convergence <- False
        until convergence:
            i = i + 1
            Ui, Si, Vi <- PowerMethod(A[I_rows[i], :], V_init=V{i-1})

            convergence = ||Si - S{i-1}||_2

        return Ui

    However, since shuffling an array in expensive, we will replace

        A <- Shuffle(A)
        I_rows = [0:n1, 0:n2, 0:n3, ..., 0:n]

    With:

        A <- A
        I_rows = shuffle([0:n1, n1:n2, n2:n3, ..., ni:n])

    However, through testing. It seems that

        Si are depressed from S_full by a factor.
        This factor seems to be of the form

            S_full ~= S_i * (a*sqrt(f_i -1) + 1) where f_i refers to the fraction of the array being used!
    """

    def __init__(self, k: int = 10, tol: Union[int, float] = 1e-6, scoring_method: str = 'q-vals', buffer: int = 10,
                 f: float = .1, p: float = .1, max_sub_iter: int = 50, sub_start: str = 'warm'):
        self.k = k
        self.tol = tol
        self.scoring_method = scoring_method
        self.buffer = buffer
        self.init_row_sampling_factor = 5
        self.f = f
        self.p = p
        self.max_sub_ter = max_sub_iter
        self.value_iterations = []
        self.tol_iterations = []
        self.value_scalers = []
        self.time = None
        if sub_start in ['warm', 'eigen', 'rand']:
            self.sub_start = sub_start
        else:
            raise Exception('Not a Valid Start')

    def svd(self, array):
        self.time = time.time()
        vec_t = self.k + self.buffer

        n, p = array.shape
        if self.sub_start == 'warm':
            x = sample_svd_start(array,
                                 k=vec_t,
                                 warm_start_row_factor=self.init_row_sampling_factor,
                                 log=0)
        else:
            raise NotImplementedError('Only Warm Start is Initialized')

        sub_array = None
        partitions = array_constant_partition((n, p), p=self.f, min_size=vec_t)
        np.random.shuffle(partitions)

        for i, part in enumerate(partitions):
            _PM = _vPowerMethod(v_start=x, k=self.k, buffer=self.buffer, max_iter=self.max_sub_ter,
                                scoring_method=self.scoring_method, p=self.p, tol=self.tol)

            if sub_array is None:
                sub_array = array[part, :]
            else:
                sub_array = da.vstack([sub_array, array[part, :]])

            n_sub, p_sub = sub_array.shape

            self.value_scalers.append(n/n_sub)

            U, S, V = _PM.svd(sub_array)
            x = V.T

            self.value_iterations.append(_PM.value_iterations)
            self.tol_iterations.append(_PM.tol_iterations)

            if i > 1:
                print(da.linalg.norm(self.value_iterations[i][-1]-self.value_iterations[i-1][-1]))

        U, S, V = full_svd_to_k_svd(U, S, V, self.k)
        self.time = time.time() - self.time
        return U, S, V
