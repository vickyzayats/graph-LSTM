import numpy as np
import theano.tensor as T
from theano import gof
from theano.gof import Apply
from theano.gradient import grad_not_implemented

print 'Using a weighted masked loss'

class MaskedLossDx(gof.Op):

    def make_node(self, softmaxes, y_idxes, y_lengths, y_startidxes, g_costs, **kwargs):
        softmaxes = T.as_tensor_variable(softmaxes)
        y_idxes = T.as_tensor_variable(y_idxes)
        y_lengths = T.as_tensor_variable(y_lengths)
        y_startidxes = T.as_tensor_variable(y_startidxes)
        g_costs = T.as_tensor_variable(g_costs)

        if (softmaxes.type.ndim != 3 or
            softmaxes.type.dtype not in T.float_dtypes):
            raise ValueError('dy must be 3-d tensor of floats', softmaxes.type)

        if (y_idxes.type.ndim != 2 or
            y_idxes.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_idxes must be 2-d tensor of integers', y_idxes.type)

        if (y_lengths.type.ndim != 1 or
            y_lengths.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_lengths must be 1-d tensor of integers', y_lengths.type)

        if (y_startidxes.type.ndim != 1 or
            y_startidxes.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_startidxes must be 1-d tensor of integers', y_startidxes.type)

        if (g_costs.type.ndim != 1 or
            g_costs.type.dtype not in T.float_dtypes):
            raise ValueError('g_costs must be 1-d tensor of floats', g_costs.type)

        return Apply(self, [softmaxes, y_idxes, y_lengths, y_startidxes, g_costs],
                     [T.Tensor(dtype=softmaxes.dtype, broadcastable=softmaxes.type.broadcastable)()])

    def perform(self, node, input_storage, output_storage):
        softmaxes, y_idxes, y_lengths, y_startidxes, g_costs = input_storage

        dx = np.zeros_like(softmaxes)
        for i in range(y_lengths.shape[0]):
            # take the total cost to be the errors made
            #dx[i, y_startidxes[i]:y_startidxes[i]+y_lengths[i]] = softmaxes[i, y_startidxes[i]:y_startidxes[i]+y_lengths[i]] * g_costs[i]
            dx[i,
               np.arange(y_startidxes[i], y_startidxes[i] + y_lengths[i]),
               y_idxes[i, y_startidxes[i]:y_startidxes[i]+y_lengths[i]]
               ] -= 1./(softmaxes[i,
                              np.arange(y_startidxes[i], y_startidxes[i] + y_lengths[i]),
                              y_idxes[i, y_startidxes[i]:y_startidxes[i]+y_lengths[i]]] * g_costs[i])

        output_storage[0][0] = dx

#    def c_code_cache_version(self):
#        return (3,)

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return T.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

#    def c_code(self, node, name, inp, out, sub):
#        pass

    def grad(self, *args):
        raise NotImplementedError()

masked_loss_dx = MaskedLossDx()

class MaskedLoss(gof.Op):
    nin = 3
    nout = 1
    """Masked Loss for sequence"""

    def perform(self, node, input_storage, output_storage):
        softmaxes, y_idxes, y_lengths, y_startidxes = input_storage
        prediction_cost = np.zeros(y_lengths.shape[0], dtype=softmaxes.dtype)
        # for all lengths to be predicted
        for i in range(y_lengths.shape[0]):
            # take the total cost to be the errors made
            prediction_cost[i] -= np.log(softmaxes[i,
                                                   np.arange(y_startidxes[i], y_startidxes[i] + y_lengths[i]),
                                                   y_idxes[i, y_startidxes[i] :y_startidxes[i] + y_lengths[i]]
                                                   ]).sum()

        output_storage[0][0] = prediction_cost

#    def c_code(self, node, name, inp, out, sub):
#        pass

    def make_node(self, softmaxes, y_idxes, y_lengths, y_startidxes, **kwargs):
        softmaxes = T.as_tensor_variable(softmaxes)
        y_idxes = T.as_tensor_variable(y_idxes)
        y_lengths = T.as_tensor_variable(y_lengths)
        y_startidxes = T.as_tensor_variable(y_startidxes)
        if (softmaxes.type.ndim != 3 or
            softmaxes.type.dtype not in T.float_dtypes):
            raise ValueError('dy must be 3-d tensor of floats', softmaxes.type)

        if (y_idxes.type.ndim != 2 or
            y_idxes.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_idxes must be 2-d tensor of integers', y_idxes.type)

        if (y_lengths.type.ndim != 1 or
            y_lengths.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_lengths must be 1-d tensor of integers', y_lengths.type)

        if (y_startidxes.type.ndim != 1 or
            y_startidxes.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_startidxes must be 1-d tensor of integers', y_startidxes.type)

        return Apply(self, [softmaxes, y_idxes, y_lengths, y_startidxes], [
            T.Tensor(dtype=softmaxes.dtype, broadcastable=[False])()])

    def grad(self, inp, grads):
        softmaxes, y_idxes, y_lengths, y_startidxes = inp
        g_costs, = grads
        return [masked_loss_dx(softmaxes, y_idxes, y_lengths, y_startidxes, g_costs),
                grad_not_implemented(self, 1, y_idxes),
                grad_not_implemented(self, 1, y_lengths),
                grad_not_implemented(self, 1, y_startidxes)]

class MaskedSumDx(gof.Op):
    """
    Gradient of the sum of values along the third dimension
    for a 3d tensor for some subranges defined by a start dimension
    and a length along which the gradient is computed.
    """

    def make_node(self, y, y_starts, y_lengths, g_costs, **kwargs):
        y = T.as_tensor_variable(y)
        y_lengths = T.as_tensor_variable(y_lengths)
        y_starts = T.as_tensor_variable(y_starts)
        g_costs = T.as_tensor_variable(g_costs)

        if (y.type.ndim != 3 or
            y.type.dtype not in T.float_dtypes):
            raise ValueError('y must be 3-d tensor of floats', y.type)

        if (y_lengths.type.ndim != 1 or
            y_lengths.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_lengths must be 1-d tensor of integers', y_lengths.type)

        if (y_starts.type.ndim != 1 or
            y_starts.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_starts must be 1-d tensor of integers', y_starts.type)

        if (g_costs.type.ndim != 1 or
            g_costs.type.dtype not in T.float_dtypes):
            raise ValueError('g_costs must be 1-d tensor of floats', g_costs.type)

        return Apply(self, [y, y_starts, y_lengths, g_costs],
                     [T.Tensor(dtype=y.dtype, broadcastable=y.type.broadcastable)()])

    def perform(self, node, input_storage, output_storage):
        y, y_starts, y_lengths, g_costs = input_storage

        dx = np.zeros_like(y)
        for i in range(y_starts.shape[0]):
            # d/dx x = 1:
            dx[i, y_starts[i]:y_starts+y_lengths[i],:] = g_costs[i]

        output_storage[0][0] = dx

#    def c_code_cache_version(self):
#        return (3,)

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return T.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

#    def c_code(self, node, name, inp, out, sub):
#        pass

    def grad(self, *args):
        raise NotImplementedError()

masked_sum_dx = MaskedSumDx()

class MaskedSum(gof.Op):
    nin = 3
    nout = 1
    """Masked sum for sequence"""

    def make_node(self, y, y_starts, y_lengths, **kwargs):
        y = T.as_tensor_variable(y)
        y_lengths = T.as_tensor_variable(y_lengths)
        y_starts = T.as_tensor_variable(y_starts)

        if (y.type.ndim != 3 or
            y.type.dtype not in T.float_dtypes):
            raise ValueError('y must be 3-d tensor of floats', y.type)

        if (y_lengths.type.ndim != 1 or
            y_lengths.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_lengths must be 1-d tensor of integers', y_lengths.type)

        if (y_starts.type.ndim != 1 or
            y_starts.type.dtype not in T.discrete_dtypes):
            raise ValueError('y_starts must be 1-d tensor of integers', y_starts.type)

        return Apply(self, [y, y_starts, y_lengths],
                     [T.Tensor(dtype=y.dtype, broadcastable=y.type.broadcastable)()])

    def perform(self, node, input_storage, output_storage):
        y, y_starts, y_lengths = input_storage

        masked_acc = np.zeros([y.shape[0]], dtype=y.dtype)
        for i in range(y_starts.shape[0]):
            # sum along row / column i
            masked_acc[i] = y[i, y_starts[i]:y_starts+y_lengths[i],:].sum()

        output_storage[0][0] = masked_acc

#    def c_code(self, node, name, inp, out, sub):
#        pass

    def grad(self, inp, grads):
        y, y_starts, y_lengths, = inp
        g_costs, = grads
        return [masked_sum_dx(y, y_starts, y_lengths, g_costs),
                grad_not_implemented(self, 1, y_starts),
                grad_not_implemented(self, 1, y_lengths)]

masked_loss = MaskedLoss()
