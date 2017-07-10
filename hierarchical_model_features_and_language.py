import theano, theano.tensor as T
import numpy as np
import random
import code
import lstm
from lstm import CommentAndFeatureEmbedding,LSTM, RNN, StackedCells, Layer, create_optimization_updates, masked_loss, MultiDropout

### Utilities:
class Vocab:
    __slots__ = ["word2index", "index2word", "unknown"]
    
    def __init__(self, index2word = None):
        self.word2index = {}
        self.index2word = []
        self.counts = {}
        
        # add unknown word:
        self.add_words(["**UNKNOWN**"])
        self.unknown = 0
        
        if index2word is not None:
            self.add_words(index2word)
            
    def add_words(self, words):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)
                self.counts[word] = 1
            else:
                self.counts[word] += 1


    def cut(self, min_num):
        deleted = []
        for word in self.counts:
            if self.counts[word] < min_num and word != "**UNKNOWN**":
                deleted.append((self.word2index[word], word))
                
        deleted = sorted(deleted, key=lambda x: x[0], reverse=True)
        self.word2index = {}
        for ind, word in deleted:
            self.index2word.pop(ind)
            del self.counts[word]
        for ind,word in enumerate(self.index2word):
            self.word2index[word] = ind
            
                
    def __call__(self, line):
        """
        Convert from numerical representation to words
        and vice-versa.
        """
        if type(line) is np.ndarray:
            return " ".join([self.index2word[word] for word in line])
        if type(line) is list:
            if len(line) > 0:
                if line[0] is int:
                    return " ".join([self.index2word[word] for word in line])
            indices = np.zeros(len(line), dtype=np.int32)
        else:
            line = line.split(" ")
            indices = np.zeros(len(line), dtype=np.int32)
            
        for i, word in enumerate(line):
            indices[i] = self.word2index.get(word, self.unknown)
            
        return indices

    @property
    def size(self):
        return len(self.index2word)
    
    def __len__(self):
        return len(self.index2word)

def pad_into_matrix(rows, padding = 0):
    if len(rows) == 0:
        return np.array([0, 0], dtype=np.int32)
    lengths = map(len, rows)
    width = max(lengths)
    height = len(rows)
    mat = np.empty([height, width], dtype=int)#, dtype=rows[0].dtype)
    mat.fill(padding)
    for i, row in enumerate(rows):
        mat[i, 0:len(row)] = row
    return mat, list(lengths)

def pad_into_variable_tensor(rows, padding = 0):
    if len(rows) == 0:
        return np.array([[0], [0]], dtype=np.int32)
    lengths = map(len, rows)
    width = max(lengths)
    height = len(rows)
    depth = max([max(x) for x in [[len(c) for c in row] for row in rows]])
    mat = np.empty([height, width, depth], dtype=int)#, dtype=rows[0].dtype)
    mat.fill(padding)
    for i, row in enumerate(rows):
        for j, com in enumerate(row):
            mat[i, j, 0:len(com)] = com[:]
    return mat, list(lengths)


def softmax(x):
    """
    Wrapper for softmax, helps with
    pickling, and removing one extra
    dimension that Theano adds during
    its exponential normalization.
    """
    return T.nnet.softmax(x.T)

def has_hidden(layer):
    """
    Whether a layer has a trainable
    initial hidden state.
    """
    return hasattr(layer, 'initial_hidden_state')

def matrixify(vector, n):
    return T.repeat(T.shape_padleft(vector), n, axis=0)

def initial_state(layer, dimensions = None):
    """
    Initalizes the recurrence relation with an initial hidden state
    if needed, else replaces with a "None" to tell Theano that
    the network **will** return something, but it does not need
    to send it to the next step of the recurrence
    """
    if dimensions is None:
        return layer.initial_hidden_state if has_hidden(layer) else None
    else:
        return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None
    
def initial_state_with_taps(layer, dimensions = None):
    """Optionally wrap tensor variable into a dict with taps=[-1]"""
    state = initial_state(layer, dimensions)
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None


class HierarchicalModel(object):

    def __init__(self, hidden_size, input_size, output_size, batch_size,
                                      vocab_sizes, celltype=LSTM):
        self.embedding = CommentAndFeatureEmbedding(vocab_sizes, input_size)
        self.forward_model = StackedCells(input_size + vocab_sizes[1], celltype=celltype, layers=[hidden_size])
        self.forward_model.layers.insert(0, self.embedding)
        self.backward_model = StackedCells(input_size + vocab_sizes[1], celltype=celltype, layers=[hidden_size])
        self.backward_model.layers.insert(0, self.embedding)
        self.output_layer = Layer(hidden_size * 2, output_size, activation=softmax)
        self.layers_shape = (input_size, hidden_size)
        self.output_shape = output_size
        self.for_how_long = T.imatrix()
        self.input_mat = T.itensor3()
        self.forward_taps = T.itensor3()
        self.backward_taps = T.itensor3()
        self.output_mat = T.imatrix()
        self.output_weighted_mat = T.itensor3()
        self.batch_size = batch_size
        self.srng = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))
        self.predictions = self.create_prediction()
        self.create_cost_fun()
        self.create_training_function()
        self.create_predict_function()

    @property
    def params(self):
        return self.forward_model.params + self.backward_model.params[1:] + self.output_layer.params
    
    def set_params(self, param_list):
        len_forward = len(self.forward_model.params)
        len_backward = len(self.backward_model.params) - 1
        self.forward_model.params = param_list[:len_forward]
        self.backward_model.params = [param_list[0]] + param_list[len_forward:len_forward + len_backward]
        self.output_layer.params = param_list[len_forward + len_backward:]
        
    def create_prediction(self):
        num_examples = self.input_mat.shape[0]
        input_len = self.input_mat.shape[1]
        
        # for now it is just one example per update
        inputs = self.input_mat.transpose((1,0,2))
        forward_taps = self.forward_taps.transpose((1,0,2))
        backward_taps = self.backward_taps.transpose((1,0,2))

        forward_h_bias = self.forward_model.layers[1].empty_h_bias
        forward_t_bias = self.forward_model.layers[1].empty_t_bias
        backward_h_bias = self.backward_model.layers[1].empty_h_bias
        backward_t_bias = self.backward_model.layers[1].empty_t_bias
                                
        # the problem solved here: https://groups.google.com/forum/#!topic/theano-users/tPLHgDp96O4

        def forward_step(idx, taps, l, h):
            prev_hiddens = [None, T.concatenate([h[taps[:,0],T.arange(num_examples),:] + \
                                taps[:,2].dimshuffle(0,'x') * forward_t_bias.dimshuffle('x', 0) + \
                                taps[:,3].dimshuffle(0,'x') * forward_h_bias.dimshuffle('x', 0), \
                                h[taps[:,1],T.arange(num_examples),:] + \
                                taps[:,4].dimshuffle(0,'x') * forward_t_bias.dimshuffle('x', 0) + \
                                taps[:,5].dimshuffle(0,'x') * forward_h_bias.dimshuffle('x', 0)],
                                axis=1), None]
            hiddens_t = self.forward_model.forward(idx, prev_hiddens=prev_hiddens)
            h = T.set_subtensor(h[l+1], hiddens_t[1])
            return h

        def backward_step(idx, taps, l, h):
            prev_hiddens = [None, T.concatenate([h[taps[:,0],T.arange(num_examples),:] + \
                                taps[:,2].dimshuffle(0,'x') * backward_t_bias.dimshuffle('x', 0) + \
                                taps[:,3].dimshuffle(0,'x') * backward_h_bias.dimshuffle('x', 0), \
                                h[taps[:,1],T.arange(num_examples),:] + \
                                taps[:,4].dimshuffle(0,'x') * backward_t_bias.dimshuffle('x', 0) + \
                                taps[:,5].dimshuffle(0,'x') * backward_h_bias.dimshuffle('x', 0)],
                                axis=1), None]
            hiddens_t = self.backward_model.forward(idx, prev_hiddens=prev_hiddens)
            h = T.set_subtensor(h[l+1], hiddens_t[1])
            return h

        def step_output(hiddens):
            new_states = self.output_layer.activate(hiddens)
            return new_states
        
        initial_hiddens = T.zeros((input_len + 1, num_examples, self.layers_shape[-1] * 2))
        initial_forward_hidden_state = self.forward_model.layers[1].initial_hidden_state
        initial_backward_hidden_state = self.backward_model.layers[1].initial_hidden_state
        forward_initial_hiddens = T.set_subtensor(initial_hiddens[0], initial_forward_hidden_state.dimshuffle('x',0))
        backward_initial_hiddens = T.set_subtensor(initial_hiddens[0], initial_backward_hidden_state.dimshuffle('x',0))

        result_forward, _ = theano.scan(fn=forward_step,
                                        sequences=[inputs, forward_taps, T.arange(input_len)],
                                        outputs_info=forward_initial_hiddens)

        result_backward, _ = theano.scan(fn=backward_step,
                                        sequences=[inputs, backward_taps, T.arange(input_len)],
                                        outputs_info=backward_initial_hiddens, go_backwards=True)

        output_hiddens = T.concatenate([result_forward[-1][1:,:,self.layers_shape[-1]:],
                                        result_backward[-1][1:,:,self.layers_shape[-1]:]], axis=2)

        result_output, _ = theano.scan(fn=step_output, sequences=[output_hiddens])

        return result_output.transpose((2,0,1))

    def create_cost_fun (self):
        what_to_predict = self.output_mat
        for_how_long = self.for_how_long
        #starting_when = T.zeros_like(self.for_how_long)
        preds_shape = self.predictions.shape[0] *  self.predictions.shape[1]
        pos = T.arange(preds_shape) * self.predictions.shape[2] + what_to_predict.flatten()
        self.cost = (-T.log(self.predictions.flatten()[pos]) * for_how_long.flatten()).sum()
        #self.cost = masked_loss(self.predictions,
        #                        what_to_predict,
        #                        for_how_long,
        #                        starting_when)
      
    def create_predict_function(self):
        self.pred_fun = theano.function(
            inputs=[self.input_mat, self.forward_taps, self.backward_taps],
            outputs =self.predictions,
            allow_input_downcast=True
            )

    def create_training_function(self):
        updates, _, _, _, _ = create_optimization_updates(self.cost.sum(), self.params, method="adadelta")
        self.update_fun = theano.function(
            inputs=[self.input_mat, self.forward_taps, self.backward_taps,
                    self.for_how_long, self.output_mat],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)
        
    def train(self, input_mat, forward_taps, backward_taps,
              for_how_long, output_mat):
        self.weighted_cost = False
        mask = np.zeros(np.shape(input_mat)[0:2])
        for i,l in enumerate(for_how_long):
            mask[i,1:l] = 1
        return self.update_fun(input_mat, forward_taps, backward_taps, mask, output_mat)

    def train_weighted(self, input_mat, forward_taps, backward_taps,
              for_how_long, output_mat):
        self.weighted_cost = True
        mask = np.zeros(np.shape(input_mat)[0:2])
        for i,l in enumerate(for_how_long):
            mask[i,1:l] = 1
        what_to_predict = np.zeros()
        return self.update_fun(input_mat, forward_taps, backward_taps, mask, output_mat)

    
    
    def __call__(self, x, y):
        return self.pred_fun(x, y)
