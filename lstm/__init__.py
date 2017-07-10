"""
Small Theano LSTM recurrent network module.

@author: Jonathan Raiman
@date: December 10th 2014

Implements most of the great things that came out
in 2014 concerning recurrent neural networks, and
some good optimizers for these types of networks.

Note (from 5 January 2015): Dropout api is a bit sophisticated due to the way
random number generators are dealt with in Theano's scan.

"""

import theano, theano.tensor as T
from theano.tensor.signal import downsample
import numpy as np
from collections import OrderedDict
import code

srng = theano.tensor.shared_randomstreams.RandomStreams(1234)
np_rng = np.random.RandomState(1234)

from .weighted_masked_loss import masked_loss, masked_loss_dx
from .shared_memory import wrap_params, borrow_memory, borrow_all_memories

class GradClip(theano.compile.ViewOp):
    """
    Here we clip the gradients as Alex Graves does in his
    recurrent neural networks. In particular this prevents
    explosion of gradients during backpropagation.

    The original poster of this code was Alex Lamb,
    [here](https://groups.google.com/forum/#!topic/theano-dev/GaJwGw6emK0).

    """

    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        return [T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound) for g_out in g_outs]


def clip_gradient(x, bound):
    grad_clip = GradClip(-bound, bound)
    try:
        T.opt.register_canonicalize(theano.gof.OpRemove(grad_clip), name='grad_clip_%.1f' % (bound))
    except ValueError:
        pass
    return grad_clip(x)


def create_shared(out_size, in_size=None, name=None):
    """
    Creates a shared matrix or vector
    
    using the given in_size and out_size.

    Inputs
    ------

    out_size int            : outer dimension of the
                              vector or matrix
    in_size  int (optional) : for a matrix, the inner
                              dimension.

    Outputs
    -------

    theano shared : the shared matrix, with random numbers in it

    """

    if in_size is None:
        return theano.shared(random_initialization((out_size, )), name=name)
    else:
        return theano.shared(random_initialization((out_size, in_size)), name=name)


def random_initialization(size):
    return (np_rng.standard_normal(size) * 1. / size[0]).astype(theano.config.floatX)


def Dropout(shape, prob):
    """
    Return a dropout mask on x.

    The probability of a value in x going to zero is prob.

    Inputs
    ------

    x    theano variable : the variable to add noise to
    prob float, variable : probability of dropping an element.
    size tuple(int, int) : size of the dropout mask.


    Outputs
    -------

    y    theano variable : x with the noise multiplied.

    """

    mask = srng.binomial(n=1, p=1-prob, size=shape)
    return T.cast(mask, theano.config.floatX)


def MultiDropout(shapes, dropout = 0.):
    """
    Return all the masks needed for dropout outside of a scan loop.
    """
    return [Dropout(shape, dropout) for shape in shapes]


class Layer(object):
    """
    Base object for neural network layers.

    A layer has an input set of neurons, and
    a hidden activation. The activation, f, is a
    function applied to the affine transformation
    of x by the connection matrix W, and the bias
    vector b.

    > y = f ( W * x + b )

    """

    def __init__(self, input_size, hidden_size, activation, clip_gradients=False):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.activation  = activation
        self.clip_gradients = clip_gradients
        self.is_recursive = False
        self.create_variables()

    def create_variables(self):
        """
        Create the connection matrix and the bias vector
        """
        self.linear_matrix        = create_shared(self.hidden_size, self.input_size, name="Layer.linear_matrix")
        self.bias_matrix          = create_shared(self.hidden_size, name="Layer.bias_matrix")

    def activate(self, x):
        """
        The hidden activation of the network
        """
        if self.clip_gradients is not False:
            x = clip_gradient(x, self.clip_gradients)

        if x.ndim > 1:
            return self.activation(
                T.dot(self.linear_matrix, x.T) + self.bias_matrix[:,None] ).T
        else:
            return self.activation(
                T.dot(self.linear_matrix, x) + self.bias_matrix )

    @property
    def params(self):
        return [self.linear_matrix, self.bias_matrix]

    @params.setter
    def params(self, param_list):
        self.linear_matrix.set_value(param_list[0].get_value())
        self.bias_matrix.set_value(param_list[1].get_value())


class Embedding(Layer):
    """
    A Matrix useful for storing word vectors or other distributed
    representations.

    use #activate(T.iscalar()) or #activate(T.ivector()) to embed
    a symbol.
    """
    def __init__(self, vocabulary_size, hidden_size):
        """
        Vocabulary size is the number of different symbols to store,
        and hidden_size is the size of their embedding.
        """
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.create_variables()
        self.is_recursive = False

    def create_variables(self):
        self.embedding_matrix = create_shared(self.vocabulary_size, self.hidden_size, name='Embedding.embedding_matrix')

    def activate(self, x):
        """
        Inputs
        ------

        x T.ivector() or T.iscalar() : indices to embed

        Output
        ------

        embedding : self.embedding_matrix[x]

        """
        return self.embedding_matrix[x]

    @property
    def params(self):
        return [self.embedding_matrix]

    @params.setter
    def params(self, param_list):
        self.embedding_matrix.set_value(param_list[0].get_value())

class FeatureEmbedding(Layer):

    def __init__(self, feature_sizes, hidden_sizes):
        self.feature_sizes = feature_sizes
        self.hidden_sizes = hidden_sizes
        self.feat_dim = len(hidden_sizes)
        self.create_variables()
        self.is_recursive = False

    def create_variables(self):
        feature_size = sum(self.feature_sizes)
        hidden_size = max(self.hidden_sizes)
        self.embedding_matrix = create_shared(feature_size, hidden_size, name='FeatureEmbedding.embedding_matrix')
        
    def activate(self, feat_list):
        embeddings = []
        sdim = 0 # starting position in the embedding_matrix
        
        for i in range(self.feat_dim):
            moved_x = feat_list[:,i].astype('int32') + sdim
            hdim = self.hidden_sizes[i]
            embeddings.append(self.embedding_matrix[moved_x,:hdim])
            sdim += self.feature_sizes[i]
        embedding = T.concatenate(embeddings, axis=1)
        # add convolution filters outputs
        embedding = T.concatenate([embedding, feat_list[:,i+1:]], axis=1)

        return embedding

    @property
    def params(self):
        return [self.embedding_matrix]

    @params.setter
    def params(self, param_list):
        self.embedding_matrix.set_value(param_list[0].get_value())

class CommentEmbedding(Layer):

    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.create_variables()
        self.is_recursive = False

    def create_variables(self):
        self.embedding_matrix = create_shared(self.vocab_size, self.hidden_size, name='CommentWordEmbedding.embedding_matrix')
        self.embedding_bias = create_shared(self.hidden_size, name='CommentWordEmbedding.embedding_bias')
        
    def activate(self, words_list):
        # average of words for now
        words_len = words_list.shape[1] / 2
        # first half are the words indexes
        words = words_list[:,:words_len]
        # second half is a mask
        mask = words_list[:,words_len:]
        embeddings = self.embedding_matrix[words] * mask.dimshuffle(0,1,'x')
        # take into accout only non-zero elements
        avg = embeddings.sum(axis=1) / T.maximum(mask.sum(axis=1),
                                                 T.ones_like(mask.sum(axis=1))).dimshuffle(0,'x')
                                                 
        return avg

    @property
    def params(self):
        return [self.embedding_matrix]

    @params.setter
    def params(self, param_list):
        self.embedding_matrix.set_value(param_list[0].get_value())

class CommentAndFeatureEmbedding(Layer):

    def __init__(self, vocab_sizes, hidden_size):
        self.vocab_sizes = vocab_sizes
        self.hidden_size = hidden_size
        self.create_variables()
        self.is_recursive = False

    def create_variables(self):
        self.embedding_matrix = create_shared(sum(self.vocab_sizes), self.hidden_size, name='CommentWordEmbedding.embedding_matrix')
        self.embedding_bias = create_shared(self.hidden_size, name='CommentWordEmbedding.embedding_bias')
        
    def activate(self, words_list):
        # average of words for now
        words_len = (words_list.shape[1] - self.vocab_sizes[1])/ 2
        # first half are the words indexes
        words = words_list[:,:words_len]
        # second half is a mask
        mask = words_list[:,words_len:-self.vocab_sizes[1]]
        embeddings = self.embedding_matrix[words] * mask.dimshuffle(0,1,'x')
        # take into accout only non-zero elements
        avg = embeddings.sum(axis=1) / T.maximum(mask.sum(axis=1),
                                                 T.ones_like(mask.sum(axis=1))).dimshuffle(0,'x')
                                                 
        return T.concatenate([avg, words_list[:,-self.vocab_sizes[1]:]], axis=1)

    @property
    def params(self):
        return [self.embedding_matrix]

    @params.setter
    def params(self, param_list):
        self.embedding_matrix.set_value(param_list[0].get_value())


class RNN(Layer):
    """
    Special recurrent layer than takes as input
    a hidden activation, h, from the past and
    an observation x.

    > y = f ( W * [x, h] + b )

    Note: x and h are concatenated in the activation.

    """
    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__(*args, **kwargs)
        self.is_recursive = True

    def create_variables(self):
        """
        Create the connection matrix and the bias vector,
        and the base hidden activation.

        """
        self.linear_matrix        = create_shared(self.hidden_size, self.input_size+ self.hidden_size, name="RNN.linear_matrix")
        self.bias_matrix          = create_shared(self.hidden_size, name="RNN.bias_matrix")
        self.initial_hidden_state = create_shared(self.hidden_size, name="RNN.initial_hidden_state")

    def activate(self, x, h):
        """
        The hidden activation of the network
        """
        if self.clip_gradients is not False:
            x = clip_gradient(x, self.clip_gradients)
            h = clip_gradient(h, self.clip_gradients)
        if x.ndim > 1:
            return self.activation(
                T.dot(
                    self.linear_matrix,
                    T.concatenate([x, h], axis=1).T
                ) + self.bias_matrix[:,None] ).T
        else:
            return self.activation(
                T.dot(
                    self.linear_matrix,
                    T.concatenate([x, h])
                ) + self.bias_matrix )

    @property
    def params(self):
        return [self.linear_matrix, self.bias_matrix]

    @params.setter
    def params(self, param_list):
        self.linear_matrix.set_value(param_list[0].get_value())
        self.bias_matrix.set_value(param_list[1].get_value())

class GRU(RNN):
    def create_variables(self):
        self.reset_layer = theano_lstm.RNN(self.input_size, self.hidden_size, activation = T.nnet.sigmoid)
        self.memory_interpolation_layer = theano_lstm.RNN(self.input_size, self.hidden_size, activation = T.nnet.sigmoid)
        self.memory_to_memory_layer = theano_lstm.RNN(self.input_size, self.hidden_size, activation = T.tanh)
        self.internal_layers = [
            self.reset_layer,
            self.memory_interpolation_layer,
            self.memory_to_memory_layer
        ]

    @property
    def params(self):
        return [param for layer in self.internal_layers for param in layer.params]

    @params.setter
    def params(self, param_list):
        assert(len(param_list) == 6)
        self.reset_layer.params                = param_list[0:2]
        self.memory_interpolation_layer.params = param_list[2:4]
        self.memory_to_memory_layer.params     = param_list[4:6]
    
    def activate(self, x, h):
        reset_gate = self.reset_layer.activate(
            x,
            h
        )

        # the new state dampened by resetting
        reset_h = reset_gate * h;

        # the new hidden state:
        candidate_h = self.memory_to_memory_layer.activate(
            x,
            reset_h
        )

        # how much to update the new hidden state:
        update_gate = self.memory_interpolation_layer.activate(
            x,
            h
        )

        # the new state interploated between candidate and old:
        new_h = (
            h      * (1.0 - update_gate) +
            candidate_h * update_gate
        )
        return new_h


class LSTM(RNN):
    """
    The structure of the LSTM allows it to learn on problems with
    long term dependencies relatively easily. The "long term"
    memory is stored in a vector of memory cells c.
    Although many LSTM architectures differ in their connectivity
    structure and activation functions, all LSTM architectures have
    memory cells that are suitable for storing information for long
    periods of time. Here we implement the LSTM from Graves et al.
    (2013).
    """

    def create_variables(self):
        """
        Create the different LSTM gates and
        their variables, along with the initial
        hidden state for the memory cells and
        the initial hidden activation.

        """
        # input gate for cells
        self.in_gate     = Layer(self.input_size + 2 * self.hidden_size, self.hidden_size, T.nnet.sigmoid, self.clip_gradients)
        # forget gate number 1 for cells (hierarchical)
        self.forget_gate1 = Layer(self.input_size + 2 * self.hidden_size, self.hidden_size, T.nnet.sigmoid, self.clip_gradients)
        # forget gate numer 2 for cells (time)
        self.forget_gate2 = Layer(self.input_size + 2 * self.hidden_size, self.hidden_size, T.nnet.sigmoid, self.clip_gradients)
        # input modulation for cells
        self.in_gate2    = Layer(self.input_size + 2 * self.hidden_size, self.hidden_size, self.activation, self.clip_gradients)
        # output modulation
        self.out_gate    = Layer(self.input_size + 2 * self.hidden_size, self.hidden_size, T.nnet.sigmoid, self.clip_gradients)

        self.empty_h_bias  = create_shared(2 * self.hidden_size, name="Layer.Empty_h_bias")
        self.empty_t_bias  = create_shared(2 * self.hidden_size, name="Layer.Empty_t_bias")
        
        # keep these layers organized
        self.internal_layers = [self.in_gate, self.forget_gate1, self.forget_gate2, \
                                self.in_gate2, self.out_gate]

        # store the memory cells in first n spots, and store the current
        # output in the next n spots:
        self.initial_hidden_state = create_shared(self.hidden_size * 2, name="LSTM.initial_hidden_state")
    @property
    def params(self):
        """
        Parameters given by the 4 gates and the
        initial hidden activation of this LSTM cell
        layer.
        """
        return [param for layer in self.internal_layers for param in layer.params] + \
                                        [self.empty_h_bias, self.empty_t_bias]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end
        self.empty_h_bias = param_list[-2]
        self.empty_t_bias = param_list[-1]

    def postprocess_activation(self, x, *args):
        if x.ndim > 1:
            return x[:, self.hidden_size:]
        else:
            return x[self.hidden_size:]

    def activate(self, x, h):
        """
        The hidden activation, h, of the network, along
        with the new values for the memory cells, c,
        Both are concatenated as follows:

        >      y = f( x, past )

        Or more visibly, with past = [prev_c, prev_h]

        > [c, h] = f( x, [prev_c, prev_h] )

        """

        if h.ndim > 1:
            #previous memory cell values (hierarchical)
            prev_c1 = h[:, :self.hidden_size]

            #previous memory cell values (time)
            prev_c2 = h[:, 2 * self.hidden_size:3 * self.hidden_size]

            #previous activations of the hidden layer
            prev_h = T.concatenate([h[:, self.hidden_size: 2 * self.hidden_size], \
                                    h[:, 3 * self.hidden_size:]],
                                    axis=1)
        else:

            #previous memory cell values (hierarchical)
            prev_c1 = h[:self.hidden_size]

            #previous memory cell values (time)
            prev_c2 = h[2 * self.hidden_size:3 * self.hidden_size]

            #previous activations of the hidden layer
            prev_h = T.concatenate([h[self.hidden_size: 2 * self.hidden_size], \
                                    h[3 * self.hidden_size:]])
        # input and previous hidden constitute the actual
        # input to the LSTM:
        if h.ndim > 1:
            obs = T.concatenate([x, prev_h], axis=1)
        else:
            obs = T.concatenate([x, prev_h])
        # TODO could we combine these 4 linear transformations for efficiency? (e.g., http://arxiv.org/pdf/1410.4615.pdf, page 5)
        # how much to add to the memory cells
        in_gate = self.in_gate.activate(obs)

        # how much to forget the current contents of the memory (hierarchical)
        forget_gate1 = self.forget_gate1.activate(obs)

        # how much to forget the current contents of the memory (time)
        forget_gate2 = self.forget_gate2.activate(obs)

        # modulate the input for the memory cells
        in_gate2 = self.in_gate2.activate(obs)

        # new memory cells
        next_c = forget_gate1 * prev_c1 + forget_gate2 * prev_c2 + in_gate2 * in_gate

        # modulate the memory cells to create the new output
        out_gate = self.out_gate.activate(obs)

        # new hidden output
        next_h = out_gate * T.tanh(next_c)

        if h.ndim > 1:
            return T.concatenate([next_c, next_h], axis=1)
        else:
            return T.concatenate([next_c, next_h])


class ConvolutionFeatures(Layer):
    """
    Convolution Layer
    """

    def __init__(self, rng, filter_shape, image_shape):
        '''
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
        filter height, filter width)
        
        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
        image height, image width)
        '''
        
        assert image_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.create_variables(rng)
        self.is_recursive = False

    def create_variables(self, rng):
        fan_in = np.prod(self.filter_shape[1:])
        fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high = W_bound, size=self.filter_shape), dtype=theano.config.floatX), borrow=True, name='convolution.filters')
        b_values = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name='convolution.bias')


    def activate(self, input_list):
        filter_size = self.filter_shape[2:]
        conv_out = T.nnet.conv.conv2d(
            input=input_list,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape,
            border_mode='full',
            )

#        max_pool = T.diagonal(conv_out, axis1=2, axis2=3)
        pool = (1, self.image_shape[3] + self.filter_shape[3] - 1)
        diag = conv_out * T.eye(pool[1]).dimshuffle('x','x', 0, 1)
        max_pool = T.signal.downsample.max_pool_2d(input=diag, ds=pool)
        max_pool = max_pool[:,:,filter_size[0]/2:self.image_shape[2] + filter_size[0]/2,:]
#        max_pool = max_pool[:,:,filter_size[0]/2:self.image_shape[2] + filter_size[0]/2]

        output = T.tanh(max_pool + self.b.dimshuffle('x', 0, 'x', 'x'))
#        output = T.tanh(max_pool + self.b.dimshuffle('x', 0, 'x'))
        return output[:,:,:,0].transpose(0,2,1), conv_out, max_pool
    
    @property
    def params(self):
        return [self.W, self.b]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        self.b.set_value(param_list[1].get_value())


class GatedInput(RNN):
    def create_variables(self):
        # input gate for cells
        self.in_gate     = Layer(self.input_size + self.hidden_size, 1, T.nnet.sigmoid, self.clip_gradients)
        self.internal_layers = [self.in_gate]

    @property
    def params(self):
        """
        Parameters given by the 4 gates and the
        initial hidden activation of this LSTM cell
        layer.

        """
        return [param for layer in self.internal_layers
                for param in layer.params]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end

    def activate(self, x, h):
        # input and previous hidden constitute the actual
        # input to the LSTM:
        if h.ndim > 1:
            obs = T.concatenate([x, h], axis=1)
        else:
            obs = T.concatenate([x, h])

        gate = self.in_gate.activate(obs)
        if h.ndim > 1:
            gate = gate[:,0][:,None]
        else:
            gate = gate[0]

        return gate

    def postprocess_activation(self, gate, x, h):
        return gate * x


def apply_dropout(x, mask):
    if mask is not None:
        return mask * x
    else:
        return x


class StackedCells(object):
    """
    Sequentially connect several recurrent layers.

    celltypes can be RNN or LSTM.

    """
    def __init__(self, input_size, celltype=RNN, layers=None,
                 activation=lambda x:x, clip_gradients=False):
        if layers is None:
            layers = []
        self.input_size = input_size
        self.clip_gradients = clip_gradients
        self.create_layers(layers, activation, celltype)

    def create_layers(self, layer_sizes, activation_type, celltype):
        self.layers = []
        prev_size   = self.input_size
        for k, layer_size in enumerate(layer_sizes):
            layer = celltype(prev_size, layer_size, activation_type,
                             clip_gradients=self.clip_gradients)
            self.layers.append(layer)
            prev_size = layer_size

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end

    def forward(self, x, prev_hiddens=None, dropout=None):
        """
        Return new hidden activations for all stacked RNNs
        """
        if dropout is None:
            dropout = []
        if prev_hiddens is None:
            prev_hiddens = [(T.repeat(T.shape_padleft(layer.initial_hidden_state),
                                      x.shape[0], axis=0)
                             if x.ndim > 1 else layer.initial_hidden_state)
                            if hasattr(layer, 'initial_hidden_state') else None
                            for layer in self.layers]

        out = []
        layer_input = x
        for k, layer in enumerate(self.layers):
            level_out = layer_input
            if len(dropout) > 0:
                level_out = apply_dropout(layer_input, dropout[k])
            if layer.is_recursive:
                level_out = layer.activate(level_out, prev_hiddens[k])
            else:
                level_out = layer.activate(level_out)
            out.append(level_out)
            # deliberate choice to change the upward structure here
            # in an RNN, there is only one kind of hidden values
            if hasattr(layer, 'postprocess_activation'):
                # in this case the hidden activation has memory cells
                # that are not shared upwards
                # along with hidden activations that can be sent
                # updwards
                if layer.is_recursive:
                    level_out = layer.postprocess_activation(level_out, layer_input, prev_hiddens[k])
                else:
                    level_out = layer.postprocess_activation(level_out, layer_input)

            layer_input = level_out

        return out


def create_optimization_updates(cost, params, updates=None, max_norm=5.0,
                                lr=0.01, eps=1e-6, rho=0.95,
                                method = "adadelta", gradients = None):
    """
    Get the updates for a gradient descent optimizer using
    SGD, AdaDelta, or AdaGrad.

    Returns the shared variables for the gradient caches,
    and the updates dictionary for compilation by a
    theano function.

    Inputs
    ------

    cost     theano variable : what to minimize
    params   list            : list of theano variables
                               with respect to which
                               the gradient is taken.
    max_norm float           : cap on excess gradients
    lr       float           : base learning rate for
                               adagrad and SGD
    eps      float           : numerical stability value
                               to not divide by zero
                               sometimes
    rho      float           : adadelta hyperparameter.
    method   str             : 'adagrad', 'adadelta', or 'sgd'.


    Outputs:
    --------

    updates  OrderedDict   : the updates to pass to a
                             theano function
    gsums    list          : gradient caches for Adagrad
                             and Adadelta
    xsums    list          : gradient caches for AdaDelta only
    lr       theano shared : learning rate
    max_norm theano_shared : normalizing clipping value for
                             excessive gradients (exploding).

    """
    lr = theano.shared(np.float64(lr).astype(theano.config.floatX))
    eps = np.float64(eps).astype(theano.config.floatX)
    rho = theano.shared(np.float64(rho).astype(theano.config.floatX))
    if max_norm is not None and max_norm is not False:
        max_norm = theano.shared(np.float64(max_norm).astype(theano.config.floatX))

    gsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) if (method == 'adadelta' or method == 'adagrad') else None for param in params]
    xsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) if method == 'adadelta' else None for param in params]

    gparams = T.grad(cost, params) if gradients is None else gradients

    if updates is None:
        updates = OrderedDict()

    for gparam, param, gsum, xsum in zip(gparams, params, gsums, xsums):
        # clip gradients if they get too big
        if max_norm is not None and max_norm is not False:
            grad_norm = gparam.norm(L=2)
            gparam = (T.minimum(max_norm, grad_norm)/ (grad_norm + eps)) * gparam

        if method == 'adadelta':
            updates[gsum] = T.cast(rho * gsum + (1. - rho) * (gparam **2), theano.config.floatX)
            dparam = -T.sqrt((xsum + eps) / (updates[gsum] + eps)) * gparam
            updates[xsum] = T.cast(rho * xsum + (1. - rho) * (dparam **2), theano.config.floatX)
            updates[param] = T.cast(param + dparam, theano.config.floatX)
        elif method == 'adagrad':
            updates[gsum] =  T.cast(gsum + (gparam ** 2), theano.config.floatX)
            updates[param] =  T.cast(param - lr * (gparam / (T.sqrt(updates[gsum] + eps))), theano.config.floatX)
        else:
            updates[param] = param - gparam * lr

    if method == 'adadelta':
        lr = rho

    return updates, gsums, xsums, lr, max_norm


__all__ = [
    "create_optimization_updates",
    "masked_loss",
    "masked_loss_dx",
    "clip_gradient",
    "create_shared",
    "Dropout",
    "apply_dropout",
    "StackedCells",
    "Layer",
    "LSTM",
    "RNN",
    "GatedInput",
    "Embedding",
    "MultiDropout",
    "wrap_params",
    "borrow_memory",
    "borrow_all_memories"
    ]
