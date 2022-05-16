import torch
torch_type = torch.Tensor

# import numpy as np #??

str_type_error = "All array should be from the same type/backend. Current types are : {}"

def sinkhorn1(a, b, M, reg, method='sinkhorn', numItermax=1000,
             stopThr=1e-9, verbose=False, log=False, warn=True,
             **kwargs):

    if method.lower() == 'sinkhorn_stabilized':
        return sinkhorn_stabilized1(a, b, M, reg, numItermax=numItermax,
                                   stopThr=stopThr, verbose=verbose,
                                   log=log, warn=warn,
                                   **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn_stabilized1(a, b, M, reg, numItermax=1000, tau=1e3, stopThr=1e-9,
                        warmstart=None, verbose=False, print_period=20,
                        log=False, warn=True, **kwargs):
    print('sinkhorn_stabilized')
    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.ones((M.shape[0],), type_as=M) / M.shape[0]
    if len(b) == 0:
        b = nx.ones((M.shape[1],), type_as=M) / M.shape[1]

    # test if multiple target
    if len(b.shape) > 1:
        n_hists = b.shape[1]
        a = a[:, None]
    else:
        n_hists = 0

    # init data
    dim_a = len(a)
    dim_b = len(b)

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = nx.zeros(dim_a, type_as=M), nx.zeros(dim_b, type_as=M)
    else:
        alpha, beta = warmstart

    if n_hists:
        u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
        v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
    else:
        u, v = nx.ones(dim_a, type_as=M), nx.ones(dim_b, type_as=M)
        u /= dim_a
        v /= dim_b

    def get_K(alpha, beta):
        """log space computation"""
        return nx.exp(-(M - alpha.reshape((dim_a, 1))
                        - beta.reshape((1, dim_b))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return nx.exp(-(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b)))
                      / reg + nx.log(u.reshape((dim_a, 1))) + nx.log(v.reshape((1, dim_b))))

    K = get_K(alpha, beta)
    transp = K
    err = 1
    for ii in range(numItermax):

        uprev = u
        vprev = v

        # sinkhorn update
        v = b / (nx.dot(K.T, u))
        u = a / (nx.dot(K, v))

        # remove numerical problems and store them in K
        if nx.max(nx.abs(u)) > tau or nx.max(nx.abs(v)) > tau:
            if n_hists:
                alpha, beta = alpha + reg * nx.max(nx.log(u), 1), beta + reg * nx.max(nx.log(v))
            else:
                alpha, beta = alpha + reg * nx.log(u), beta + reg * nx.log(v)
                if n_hists:
                    u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
                    v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
                else:
                    u = nx.ones(dim_a, type_as=M) / dim_a
                    v = nx.ones(dim_b, type_as=M) / dim_b
            K = get_K(alpha, beta)

        if ii % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                err_u = nx.max(nx.abs(u - uprev))
                err_u /= max(nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.0)
                err_v = nx.max(nx.abs(v - vprev))
                err_v /= max(nx.max(nx.abs(v)), nx.max(nx.abs(vprev)), 1.0)
                err = 0.5 * (err_u + err_v)
            else:
                transp = get_Gamma(alpha, beta, u, v)
                err = nx.norm(nx.sum(transp, axis=0) - b)
            if log:
                log['err'].append(err)

            if verbose:
                if ii % (print_period * 20) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))

        if err <= stopThr:
            break

        if nx.any(nx.isnan(u)) or nx.any(nx.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            # warnings.warn('Numerical errors at iteration %d' % ii)
            print('warnings.warn(Numerical errors at iteration %d % ii)')
            u = uprev
            v = vprev
            break
    else:
        if warn:
            # warnings.warn("Sinkhorn did not converge. You might want to "
            #               "increase the number of iterations `numItermax` "
            #               "or the regularization parameter `reg`.")
            print('warnings, Sinkhorn did not converge. You might want to')
    if log:
        if n_hists:
            alpha = alpha[:, None]
            beta = beta[:, None]
        logu = alpha / reg + nx.log(u)
        logv = beta / reg + nx.log(v)
        log["n_iter"] = ii
        log['logu'] = logu
        log['logv'] = logv
        log['alpha'] = alpha + reg * nx.log(u)
        log['beta'] = beta + reg * nx.log(v)
        log['warmstart'] = (log['alpha'], log['beta'])
        if n_hists:
            res = nx.stack([
                nx.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
                for i in range(n_hists)
            ])
            return res, log

        else:
            return get_Gamma(alpha, beta, u, v), log
    else:
        if n_hists:
            res = nx.stack([
                nx.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
                for i in range(n_hists)
            ])
            return res
        else:
            return get_Gamma(alpha, beta, u, v)

# def list_to_array(*lst):
#     r""" Convert a list if in numpy format """
#     if len(lst) > 1:
#         return [np.array(a) if isinstance(a, list) else a for a in lst]
#     else:
#         return np.array(lst[0]) if isinstance(lst[0], list) else lst[0]
def list_to_array(*lst):
    r""" Convert a list if in torch format """
    if len(lst) > 1:
        return [torch.Tensor(a, requires_grad=True) if isinstance(a, list) else a for a in lst]
    else:
        return torch.Tensor(lst[0], requires_grad=True) if isinstance(lst[0], list) else lst[0]

def get_backend(*args):
    """Returns the proper backend for a list of input arrays

        Also raises TypeError if all arrays are not from the same backend
    """
    # check that some arrays given
    print('on get_backend')
    if not len(args) > 0:
        raise ValueError(" The function takes at least one parameter")
    # check all same type
    if not len(set(type(a) for a in args)) == 1:
        raise ValueError(str_type_error.format([type(a) for a in args]))

    # if isinstance(args[0], np.ndarray):
    #     return NumpyBackend()
    print('if isinstance(args[0], torch_type):')
    if isinstance(args[0], torch_type):
        print('is torch_type')
        return TorchBackend()
    # elif isinstance(args[0], jax_type):
    #     return JaxBackend()
    # elif isinstance(args[0], cp_type):  # pragma: no cover
    #     return CupyBackend()
    # elif isinstance(args[0], tf_type):
    #     return TensorflowBackend()
    else:
        raise ValueError("Unknown type of non implemented backend.")


class Backend():
    """
    Backend abstract class.
    Implementations: :py:class:`JaxBackend`, :py:class:`NumpyBackend`, :py:class:`TorchBackend`,
    :py:class:`CupyBackend`, :py:class:`TensorflowBackend`

    - The `__name__` class attribute refers to the name of the backend.
    - The `__type__` class attribute refers to the data structure used by the backend.
    """

    __name__ = None
    __type__ = None
    __type_list__ = None

    rng_ = None

    def __str__(self):
        return self.__name__

    # convert to numpy
    def to_numpy(self, a):
        """Returns the numpy version of a tensor"""
        raise NotImplementedError()

    # convert from numpy
    def from_numpy(self, a, type_as=None):
        """Creates a tensor cloning a numpy array, with the given precision (defaulting to input's precision) and the given device (in case of GPUs)"""
        raise NotImplementedError()

    def set_gradients(self, val, inputs, grads):
        """Define the gradients for the value val wrt the inputs """
        raise NotImplementedError()

    def zeros(self, shape, type_as=None):
        r"""
        Creates a tensor full of zeros.

        This function follows the api from :any:`numpy.zeros`

        See: https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
        """
        raise NotImplementedError()

    def ones(self, shape, type_as=None):
        r"""
        Creates a tensor full of ones.

        This function follows the api from :any:`numpy.ones`

        See: https://numpy.org/doc/stable/reference/generated/numpy.ones.html
        """
        raise NotImplementedError()

    def arange(self, stop, start=0, step=1, type_as=None):
        r"""
        Returns evenly spaced values within a given interval.

        This function follows the api from :any:`numpy.arange`

        See: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
        """
        raise NotImplementedError()

    def full(self, shape, fill_value, type_as=None):
        r"""
        Creates a tensor with given shape, filled with given value.

        This function follows the api from :any:`numpy.full`

        See: https://numpy.org/doc/stable/reference/generated/numpy.full.html
        """
        raise NotImplementedError()

    def eye(self, N, M=None, type_as=None):
        r"""
        Creates the identity matrix of given size.

        This function follows the api from :any:`numpy.eye`

        See: https://numpy.org/doc/stable/reference/generated/numpy.eye.html
        """
        raise NotImplementedError()

    def sum(self, a, axis=None, keepdims=False):
        r"""
        Sums tensor elements over given dimensions.

        This function follows the api from :any:`numpy.sum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        """
        raise NotImplementedError()

    def cumsum(self, a, axis=None):
        r"""
        Returns the cumulative sum of tensor elements over given dimensions.

        This function follows the api from :any:`numpy.cumsum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
        """
        raise NotImplementedError()

    def max(self, a, axis=None, keepdims=False):
        r"""
        Returns the maximum of an array or maximum along given dimensions.

        This function follows the api from :any:`numpy.amax`

        See: https://numpy.org/doc/stable/reference/generated/numpy.amax.html
        """
        raise NotImplementedError()

    def min(self, a, axis=None, keepdims=False):
        r"""
        Returns the maximum of an array or maximum along given dimensions.

        This function follows the api from :any:`numpy.amin`

        See: https://numpy.org/doc/stable/reference/generated/numpy.amin.html
        """
        raise NotImplementedError()

    def maximum(self, a, b):
        r"""
        Returns element-wise maximum of array elements.

        This function follows the api from :any:`numpy.maximum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
        """
        raise NotImplementedError()

    def minimum(self, a, b):
        r"""
        Returns element-wise minimum of array elements.

        This function follows the api from :any:`numpy.minimum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
        """
        raise NotImplementedError()

    def dot(self, a, b):
        r"""
        Returns the dot product of two tensors.

        This function follows the api from :any:`numpy.dot`

        See: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
        """
        raise NotImplementedError()

    def abs(self, a):
        r"""
        Computes the absolute value element-wise.

        This function follows the api from :any:`numpy.absolute`

        See: https://numpy.org/doc/stable/reference/generated/numpy.absolute.html
        """
        raise NotImplementedError()

    def exp(self, a):
        r"""
        Computes the exponential value element-wise.

        This function follows the api from :any:`numpy.exp`

        See: https://numpy.org/doc/stable/reference/generated/numpy.exp.html
        """
        raise NotImplementedError()

    def log(self, a):
        r"""
        Computes the natural logarithm, element-wise.

        This function follows the api from :any:`numpy.log`

        See: https://numpy.org/doc/stable/reference/generated/numpy.log.html
        """
        raise NotImplementedError()

    def sqrt(self, a):
        r"""
        Returns the non-ngeative square root of a tensor, element-wise.

        This function follows the api from :any:`numpy.sqrt`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
        """
        raise NotImplementedError()

    def power(self, a, exponents):
        r"""
        First tensor elements raised to powers from second tensor, element-wise.

        This function follows the api from :any:`numpy.power`

        See: https://numpy.org/doc/stable/reference/generated/numpy.power.html
        """
        raise NotImplementedError()

    def norm(self, a):
        r"""
        Computes the matrix frobenius norm.

        This function follows the api from :any:`numpy.linalg.norm`

        See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
        """
        raise NotImplementedError()

    def any(self, a):
        r"""
        Tests whether any tensor element along given dimensions evaluates to True.

        This function follows the api from :any:`numpy.any`

        See: https://numpy.org/doc/stable/reference/generated/numpy.any.html
        """
        raise NotImplementedError()

    def isnan(self, a):
        r"""
        Tests element-wise for NaN and returns result as a boolean tensor.

        This function follows the api from :any:`numpy.isnan`

        See: https://numpy.org/doc/stable/reference/generated/numpy.isnan.html
        """
        raise NotImplementedError()

    def isinf(self, a):
        r"""
        Tests element-wise for positive or negative infinity and returns result as a boolean tensor.

        This function follows the api from :any:`numpy.isinf`

        See: https://numpy.org/doc/stable/reference/generated/numpy.isinf.html
        """
        raise NotImplementedError()

    def einsum(self, subscripts, *operands):
        r"""
        Evaluates the Einstein summation convention on the operands.

        This function follows the api from :any:`numpy.einsum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        raise NotImplementedError()

    def sort(self, a, axis=-1):
        r"""
        Returns a sorted copy of a tensor.

        This function follows the api from :any:`numpy.sort`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        """
        raise NotImplementedError()

    def argsort(self, a, axis=None):
        r"""
        Returns the indices that would sort a tensor.

        This function follows the api from :any:`numpy.argsort`

        See: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
        """
        raise NotImplementedError()

    def searchsorted(self, a, v, side='left'):
        r"""
        Finds indices where elements should be inserted to maintain order in given tensor.

        This function follows the api from :any:`numpy.searchsorted`

        See: https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        """
        raise NotImplementedError()

    def flip(self, a, axis=None):
        r"""
        Reverses the order of elements in a tensor along given dimensions.

        This function follows the api from :any:`numpy.flip`

        See: https://numpy.org/doc/stable/reference/generated/numpy.flip.html
        """
        raise NotImplementedError()

    def clip(self, a, a_min, a_max):
        """
        Limits the values in a tensor.

        This function follows the api from :any:`numpy.clip`

        See: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
        """
        raise NotImplementedError()

    def repeat(self, a, repeats, axis=None):
        r"""
        Repeats elements of a tensor.

        This function follows the api from :any:`numpy.repeat`

        See: https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        """
        raise NotImplementedError()

    def take_along_axis(self, arr, indices, axis):
        r"""
        Gathers elements of a tensor along given dimensions.

        This function follows the api from :any:`numpy.take_along_axis`

        See: https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html
        """
        raise NotImplementedError()

    def concatenate(self, arrays, axis=0):
        r"""
        Joins a sequence of tensors along an existing dimension.

        This function follows the api from :any:`numpy.concatenate`

        See: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        """
        raise NotImplementedError()

    def zero_pad(self, a, pad_width):
        r"""
        Pads a tensor.

        This function follows the api from :any:`numpy.pad`

        See: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        """
        raise NotImplementedError()

    def argmax(self, a, axis=None):
        r"""
        Returns the indices of the maximum values of a tensor along given dimensions.

        This function follows the api from :any:`numpy.argmax`

        See: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        """
        raise NotImplementedError()

    def mean(self, a, axis=None):
        r"""
        Computes the arithmetic mean of a tensor along given dimensions.

        This function follows the api from :any:`numpy.mean`

        See: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
        """
        raise NotImplementedError()

    def std(self, a, axis=None):
        r"""
        Computes the standard deviation of a tensor along given dimensions.

        This function follows the api from :any:`numpy.std`

        See: https://numpy.org/doc/stable/reference/generated/numpy.std.html
        """
        raise NotImplementedError()

    def linspace(self, start, stop, num):
        r"""
        Returns a specified number of evenly spaced values over a given interval.

        This function follows the api from :any:`numpy.linspace`

        See: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        """
        raise NotImplementedError()

    def meshgrid(self, a, b):
        r"""
        Returns coordinate matrices from coordinate vectors (Numpy convention).

        This function follows the api from :any:`numpy.meshgrid`

        See: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        """
        raise NotImplementedError()

    def diag(self, a, k=0):
        r"""
        Extracts or constructs a diagonal tensor.

        This function follows the api from :any:`numpy.diag`

        See: https://numpy.org/doc/stable/reference/generated/numpy.diag.html
        """
        raise NotImplementedError()

    def unique(self, a):
        r"""
        Finds unique elements of given tensor.

        This function follows the api from :any:`numpy.unique`

        See: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        """
        raise NotImplementedError()

    def logsumexp(self, a, axis=None):
        r"""
        Computes the log of the sum of exponentials of input elements.

        This function follows the api from :any:`scipy.special.logsumexp`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
        """
        raise NotImplementedError()

    def stack(self, arrays, axis=0):
        r"""
        Joins a sequence of tensors along a new dimension.

        This function follows the api from :any:`numpy.stack`

        See: https://numpy.org/doc/stable/reference/generated/numpy.stack.html
        """
        raise NotImplementedError()

    def outer(self, a, b):
        r"""
        Computes the outer product between two vectors.

        This function follows the api from :any:`numpy.outer`

        See: https://numpy.org/doc/stable/reference/generated/numpy.outer.html
        """
        raise NotImplementedError()

    def reshape(self, a, shape):
        r"""
        Gives a new shape to a tensor without changing its data.

        This function follows the api from :any:`numpy.reshape`

        See: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        r"""
        Sets the seed for the random generator.

        This function follows the api from :any:`numpy.random.seed`

        See: https://numpy.org/doc/stable/reference/generated/numpy.random.seed.html
        """
        raise NotImplementedError()

    def rand(self, *size, type_as=None):
        r"""
        Generate uniform random numbers.

        This function follows the api from :any:`numpy.random.rand`

        See: https://numpy.org/doc/stable/reference/generated/numpy.random.rand.html
        """
        raise NotImplementedError()

    def randn(self, *size, type_as=None):
        r"""
        Generate normal Gaussian random numbers.

        This function follows the api from :any:`numpy.random.rand`

        See: https://numpy.org/doc/stable/reference/generated/numpy.random.rand.html
        """
        raise NotImplementedError()

    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        r"""
        Creates a sparse tensor in COOrdinate format.

        This function follows the api from :any:`scipy.sparse.coo_matrix`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
        """
        raise NotImplementedError()

    def issparse(self, a):
        r"""
        Checks whether or not the input tensor is a sparse tensor.

        This function follows the api from :any:`scipy.sparse.issparse`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.issparse.html
        """
        raise NotImplementedError()

    def tocsr(self, a):
        r"""
        Converts this matrix to Compressed Sparse Row format.

        This function follows the api from :any:`scipy.sparse.coo_matrix.tocsr`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tocsr.html
        """
        raise NotImplementedError()

    def eliminate_zeros(self, a, threshold=0.):
        r"""
        Removes entries smaller than the given threshold from the sparse tensor.

        This function follows the api from :any:`scipy.sparse.csr_matrix.eliminate_zeros`

        See: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.eliminate_zeros.html
        """
        raise NotImplementedError()

    def todense(self, a):
        r"""
        Converts a sparse tensor to a dense tensor.

        This function follows the api from :any:`scipy.sparse.csr_matrix.toarray`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.toarray.html
        """
        raise NotImplementedError()

    def where(self, condition, x, y):
        r"""
        Returns elements chosen from x or y depending on condition.

        This function follows the api from :any:`numpy.where`

        See: https://numpy.org/doc/stable/reference/generated/numpy.where.html
        """
        raise NotImplementedError()

    def copy(self, a):
        r"""
        Returns a copy of the given tensor.

        This function follows the api from :any:`numpy.copy`

        See: https://numpy.org/doc/stable/reference/generated/numpy.copy.html
        """
        raise NotImplementedError()

    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        r"""
        Returns True if two arrays are element-wise equal within a tolerance.

        This function follows the api from :any:`numpy.allclose`

        See: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
        """
        raise NotImplementedError()

    def dtype_device(self, a):
        r"""
        Returns the dtype and the device of the given tensor.
        """
        raise NotImplementedError()

    def assert_same_dtype_device(self, a, b):
        r"""
        Checks whether or not the two given inputs have the same dtype as well as the same device
        """
        raise NotImplementedError()

    def squeeze(self, a, axis=None):
        r"""
        Remove axes of length one from a.

        This function follows the api from :any:`numpy.squeeze`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
        """
        raise NotImplementedError()

    def bitsize(self, type_as):
        r"""
        Gives the number of bits used by the data type of the given tensor.
        """
        raise NotImplementedError()

    def device_type(self, type_as):
        r"""
        Returns CPU or GPU depending on the device where the given tensor is located.
        """
        raise NotImplementedError()

    def _bench(self, callable, *args, n_runs=1, warmup_runs=1):
        r"""
        Executes a benchmark of the given callable with the given arguments.
        """
        raise NotImplementedError()

class TorchBackend(Backend):
    """
    PyTorch implementation of the backend

    - `__name__` is "torch"
    - `__type__` is torch.Tensor
    """

    __name__ = 'torch'
    __type__ = torch_type
    __type_list__ = None

    rng_ = None

    def __init__(self):

        self.rng_ = torch.Generator()
        self.rng_.seed()

        self.__type_list__ = [torch.tensor(1, dtype=torch.float32, requires_grad=True),
                              torch.tensor(1, dtype=torch.float64, requires_grad=True)]

        if torch.cuda.is_available():
            self.__type_list__.append(torch.tensor(1, dtype=torch.float32, device='cuda', requires_grad=True))
            self.__type_list__.append(torch.tensor(1, dtype=torch.float64, device='cuda', requires_grad=True))

        from torch.autograd import Function

        # define a function that takes inputs val and grads
        # ad returns a val tensor with proper gradients
        class ValFunction(Function):

            @staticmethod
            def forward(ctx, val, grads, *inputs):
                ctx.grads = grads
                return val

            @staticmethod
            def backward(ctx, grad_output):
                # the gradients are grad
                return (None, None) + tuple(g * grad_output for g in ctx.grads)

        self.ValFunction = ValFunction

    # def to_numpy(self, a):
    #     return a.cpu().detach().numpy()
    #
    # def from_numpy(self, a, type_as=None):
    #     if isinstance(a, float):
    #         a = np.array(a)
    #     if type_as is None:
    #         return torch.from_numpy(a)
    #     else:
    #         return torch.as_tensor(a, dtype=type_as.dtype, device=type_as.device)

    def set_gradients(self, val, inputs, grads):

        Func = self.ValFunction()

        res = Func.apply(val, grads, *inputs)

        return res

    def zeros(self, shape, type_as=None):
        if isinstance(shape, int):
            shape = (shape,)
        if type_as is None:
            return torch.zeros(shape)
        else:
            return torch.zeros(shape, dtype=type_as.dtype, device=type_as.device)

    def ones(self, shape, type_as=None):
        if isinstance(shape, int):
            shape = (shape,)
        if type_as is None:
            return torch.ones(shape)
        else:
            return torch.ones(shape, dtype=type_as.dtype, device=type_as.device)

    def arange(self, stop, start=0, step=1, type_as=None):
        if type_as is None:
            return torch.arange(start, stop, step)
        else:
            return torch.arange(start, stop, step, device=type_as.device)

    def full(self, shape, fill_value, type_as=None):
        if isinstance(shape, int):
            shape = (shape,)
        if type_as is None:
            return torch.full(shape, fill_value)
        else:
            return torch.full(shape, fill_value, dtype=type_as.dtype, device=type_as.device)

    def eye(self, N, M=None, type_as=None):
        if M is None:
            M = N
        if type_as is None:
            return torch.eye(N, m=M)
        else:
            return torch.eye(N, m=M, dtype=type_as.dtype, device=type_as.device)

    def sum(self, a, axis=None, keepdims=False):
        if axis is None:
            return torch.sum(a)
        else:
            return torch.sum(a, axis, keepdim=keepdims)

    def cumsum(self, a, axis=None):
        if axis is None:
            return torch.cumsum(a.flatten(), 0)
        else:
            return torch.cumsum(a, axis)

    def max(self, a, axis=None, keepdims=False):
        if axis is None:
            return torch.max(a)
        else:
            return torch.max(a, axis, keepdim=keepdims)[0]

    def min(self, a, axis=None, keepdims=False):
        if axis is None:
            return torch.min(a)
        else:
            return torch.min(a, axis, keepdim=keepdims)[0]

    def maximum(self, a, b):
        if isinstance(a, int) or isinstance(a, float):
            a = torch.tensor([float(a)], dtype=b.dtype, device=b.device, requires_grad=True)
        if isinstance(b, int) or isinstance(b, float):
            b = torch.tensor([float(b)], dtype=a.dtype, device=a.device, requires_grad=True)
        if hasattr(torch, "maximum"):
            return torch.maximum(a, b)
        else:
            return torch.max(torch.stack(torch.broadcast_tensors(a, b)), axis=0)[0]

    def minimum(self, a, b):
        if isinstance(a, int) or isinstance(a, float):
            a = torch.tensor([float(a)], dtype=b.dtype, device=b.device, requires_grad=True)
        if isinstance(b, int) or isinstance(b, float):
            b = torch.tensor([float(b)], dtype=a.dtype, device=a.device, requires_grad=True)
        if hasattr(torch, "minimum"):
            return torch.minimum(a, b)
        else:
            return torch.min(torch.stack(torch.broadcast_tensors(a, b)), axis=0)[0]

    def dot(self, a, b):
        return torch.matmul(a, b)

    def abs(self, a):
        return torch.abs(a)

    def exp(self, a):
        return torch.exp(a)

    def log(self, a):
        return torch.log(a)

    def sqrt(self, a):
        return torch.sqrt(a)

    def power(self, a, exponents):
        return torch.pow(a, exponents)

    def norm(self, a):
        return torch.sqrt(torch.sum(torch.square(a)))

    def any(self, a):
        return torch.any(a)

    def isnan(self, a):
        return torch.isnan(a)

    def isinf(self, a):
        return torch.isinf(a)

    def einsum(self, subscripts, *operands):
        return torch.einsum(subscripts, *operands)

    def sort(self, a, axis=-1):
        sorted0, indices = torch.sort(a, dim=axis)
        return sorted0

    def argsort(self, a, axis=-1):
        sorted, indices = torch.sort(a, dim=axis)
        return indices

    def searchsorted(self, a, v, side='left'):
        right = (side != 'left')
        return torch.searchsorted(a, v, right=right)

    def flip(self, a, axis=None):
        if axis is None:
            return torch.flip(a, tuple(i for i in range(len(a.shape))))
        if isinstance(axis, int):
            return torch.flip(a, (axis,))
        else:
            return torch.flip(a, dims=axis)

    def outer(self, a, b):
        return torch.outer(a, b)

    def clip(self, a, a_min, a_max):
        return torch.clamp(a, a_min, a_max)

    def repeat(self, a, repeats, axis=None):
        return torch.repeat_interleave(a, repeats, dim=axis)

    def take_along_axis(self, arr, indices, axis):
        return torch.gather(arr, axis, indices)

    def concatenate(self, arrays, axis=0):
        return torch.cat(arrays, dim=axis)

    def zero_pad(self, a, pad_width):
        from torch.nn.functional import pad
        # pad_width is an array of ndim tuples indicating how many 0 before and after
        # we need to add. We first need to make it compliant with torch syntax, that
        # starts with the last dim, then second last, etc.
        how_pad = tuple(element for tupl in pad_width[::-1] for element in tupl)
        return pad(a, how_pad)

    def argmax(self, a, axis=None):
        return torch.argmax(a, dim=axis)

    def mean(self, a, axis=None):
        if axis is not None:
            return torch.mean(a, dim=axis)
        else:
            return torch.mean(a)

    def std(self, a, axis=None):
        if axis is not None:
            return torch.std(a, dim=axis, unbiased=False)
        else:
            return torch.std(a, unbiased=False)

    def linspace(self, start, stop, num):
        return torch.linspace(start, stop, num, dtype=torch.float64)

    def meshgrid(self, a, b):
        X, Y = torch.meshgrid(a, b)
        return X.T, Y.T

    def diag(self, a, k=0):
        return torch.diag(a, diagonal=k)

    def unique(self, a):
        return torch.unique(a)

    def logsumexp(self, a, axis=None):
        if axis is not None:
            return torch.logsumexp(a, dim=axis)
        else:
            return torch.logsumexp(a, dim=tuple(range(len(a.shape))))

    def stack(self, arrays, axis=0):
        return torch.stack(arrays, dim=axis)

    def reshape(self, a, shape):
        return torch.reshape(a, shape)

    def seed(self, seed=None):
        if isinstance(seed, int):
            self.rng_.manual_seed(seed)
        elif isinstance(seed, torch.Generator):
            self.rng_ = seed
        else:
            raise ValueError("Non compatible seed : {}".format(seed))

    def rand(self, *size, type_as=None):
        if type_as is not None:
            return torch.rand(size=size, generator=self.rng_, dtype=type_as.dtype, device=type_as.device)
        else:
            return torch.rand(size=size, generator=self.rng_)

    def randn(self, *size, type_as=None):
        if type_as is not None:
            return torch.randn(size=size, dtype=type_as.dtype, generator=self.rng_, device=type_as.device)
        else:
            return torch.randn(size=size, generator=self.rng_)

    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        if type_as is None:
            return torch.sparse_coo_tensor(torch.stack([rows, cols]), data, size=shape)
        else:
            return torch.sparse_coo_tensor(
                torch.stack([rows, cols]), data, size=shape,
                dtype=type_as.dtype, device=type_as.device
            )

    def issparse(self, a):
        return getattr(a, "is_sparse", False) or getattr(a, "is_sparse_csr", False)

    def tocsr(self, a):
        # Versions older than 1.9 do not support CSR tensors. PyTorch 1.9 and 1.10 offer a very limited support
        return self.todense(a)

    def eliminate_zeros(self, a, threshold=0.):
        if self.issparse(a):
            if threshold > 0:
                mask = self.abs(a) <= threshold
                mask = ~mask
                mask = mask.nonzero()
            else:
                mask = a._values().nonzero()
            nv = a._values().index_select(0, mask.view(-1))
            ni = a._indices().index_select(1, mask.view(-1))
            return self.coo_matrix(nv, ni[0], ni[1], shape=a.shape, type_as=a)
        else:
            if threshold > 0:
                a[self.abs(a) <= threshold] = 0
            return a

    def todense(self, a):
        if self.issparse(a):
            return a.to_dense()
        else:
            return a

    def where(self, condition, x, y):
        return torch.where(condition, x, y)

    def copy(self, a):
        return torch.clone(a)

    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def dtype_device(self, a):
        return a.dtype, a.device

    def assert_same_dtype_device(self, a, b):
        a_dtype, a_device = self.dtype_device(a)
        b_dtype, b_device = self.dtype_device(b)

        assert a_dtype == b_dtype, "Dtype discrepancy"
        assert a_device == b_device, f"Device discrepancy. First input is on {str(a_device)}, whereas second input is on {str(b_device)}"

    def squeeze(self, a, axis=None):
        if axis is None:
            return torch.squeeze(a)
        else:
            return torch.squeeze(a, dim=axis)

    def bitsize(self, type_as):
        return torch.finfo(type_as.dtype).bits

    def device_type(self, type_as):
        return type_as.device.type.replace("cuda", "gpu").upper()

    # def _bench(self, callable, *args, n_runs=1, warmup_runs=1):
    #     results = dict()
    #     for type_as in self.__type_list__:
    #         inputs = [self.from_numpy(arg, type_as=type_as) for arg in args]
    #         for _ in range(warmup_runs):
    #             callable(*inputs)
    #         if self.device_type(type_as) == "GPU":  # pragma: no cover
    #             torch.cuda.synchronize()
    #             start = torch.cuda.Event(enable_timing=True)
    #             end = torch.cuda.Event(enable_timing=True)
    #             start.record()
    #         else:
    #             start = time.perf_counter()
    #         for _ in range(n_runs):
    #             callable(*inputs)
    #         if self.device_type(type_as) == "GPU":  # pragma: no cover
    #             end.record()
    #             torch.cuda.synchronize()
    #             duration = start.elapsed_time(end) / 1000.
    #         else:
    #             end = time.perf_counter()
    #             duration = end - start
    #         key = ("Pytorch", self.device_type(type_as), self.bitsize(type_as))
    #         results[key] = duration / n_runs
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #     return results
