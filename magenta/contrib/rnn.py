# Copyright 2023 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forked classes and functions from `tf.contrib.rnn`."""
import tensorflow as tf
import collections
import warnings

LSTMStateTuple = collections.namedtuple('LSTMStateTuple', ['c', 'h'])

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

ASSERT_LIKE_RNNCELL_ERROR_REGEXP = "is not an RNNCell"

def _hasattr(obj, attr_name):
    try:
        getattr(obj, attr_name)
    except AttributeError:
        return False
    else:
        return True

def _is_sequence(maybe_seq):
    return isinstance(maybe_seq, (list, tuple, dict))

def assert_like_rnncell(cell_name, cell):
    conditions = [
        _hasattr(cell, "output_size"),
        _hasattr(cell, "state_size"),
        _hasattr(cell, "get_initial_state") or _hasattr(cell, "zero_state"),
        callable(cell),
    ]
    errors = [
        "'output_size' property is missing", "'state_size' property is missing",
        "either 'zero_state' or 'get_initial_state' method is required",
        "is not callable"
    ]
    if not all(conditions):
        errors = [error for error, cond in zip(errors, conditions) if not cond]
        raise TypeError("The argument {!r} ({}) is not an RNNCell: {}.".format(
            cell_name, cell, ", ".join(errors)))

class CompatRNNCell(tf.keras.layers.AbstractRNNCell):
    """Minimal wrapper over `tf.keras.layers.AbstractRNNCell`."""

    def zero_state(self, batch_size, dtype):
        return tf.nest.map_structure(
            lambda s: tf.zeros([batch_size, s], dtype=dtype), self.state_size)

class _Linear(tf.keras.layers.Layer):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable."""

    def __init__(self, output_size, build_bias, **kwargs):
        super().__init__(**kwargs)
        self._output_size = output_size
        self._build_bias = build_bias
        self._is_sequence = False

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            total_arg_size = sum(int(s[-1]) for s in input_shape)
            self._is_sequence = True
        else:
            total_arg_size = int(input_shape[-1])
            self._is_sequence = False
        self._kernel = self.add_weight(
            _WEIGHTS_VARIABLE_NAME,
            shape=[total_arg_size, self._output_size],
            initializer="glorot_uniform"
        )
        if self._build_bias:
            self._biases = self.add_weight(
                _BIAS_VARIABLE_NAME,
                shape=[self._output_size],
                initializer="zeros"
            )
        super().build(input_shape)

    def call(self, args):
        if not self._is_sequence:
            args = [args]
        inputs = args[0] if len(args) == 1 else tf.concat(args, axis=1)
        res = tf.matmul(inputs, self._kernel)
        if self._build_bias:
            res = tf.nn.bias_add(res, self._biases)
        return res

class InputProjectionWrapper(CompatRNNCell):
    """Operator adding an input projection to the given cell."""

    def __init__(self,
                 cell,
                 num_proj,
                 activation=None,
                 input_size=None,
                 reuse=None):
        super(InputProjectionWrapper, self).__init__(name="InputProjectionWrapper")
        if input_size is not None:
            warnings.warn(f"{self}: The input_size parameter is deprecated.")
        assert_like_rnncell("cell", cell)
        self._cell = cell
        self._num_proj = num_proj
        self._activation = activation
        self._linear = None

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        if self._linear is None:
            self._linear = _Linear(self._num_proj, True)
        projected = self._linear(inputs)
        if self._activation:
            projected = self._activation(projected)
        return self._cell(projected, state)

class AttentionCellWrapper(CompatRNNCell):
    """Basic attention cell wrapper."""

    def __init__(self,
                 cell,
                 attn_length,
                 attn_size=None,
                 attn_vec_size=None,
                 input_size=None,
                 state_is_tuple=True,
                 reuse=None):
        super(AttentionCellWrapper, self).__init__(name="AttentionCellWrapper")
        assert_like_rnncell("cell", cell)
        if _is_sequence(cell.state_size) and not state_is_tuple:
            raise ValueError(
                "Cell returns tuple of states, but the flag "
                "state_is_tuple is not set. State size is: %s" % str(cell.state_size))
        if attn_length <= 0:
            raise ValueError(
                "attn_length should be greater than zero, got %s" % str(attn_length))
        if not state_is_tuple:
            warnings.warn(
                f"{self}: Using a concatenated state is slower and will soon be "
                "deprecated.  Use state_is_tuple=True.")
        if attn_size is None:
            attn_size = cell.output_size
        if attn_vec_size is None:
            attn_vec_size = attn_size
        self._state_is_tuple = state_is_tuple
        self._cell = cell
        self._attn_vec_size = attn_vec_size
        self._input_size = input_size
        self._attn_size = attn_size
        self._attn_length = attn_length
        self._reuse = reuse
        self._linear1 = None
        self._linear2 = None
        self._linear3 = None
        self._k = None
        self._v = None

    @property
    def state_size(self):
        size = (self._cell.state_size, self._attn_size,
                self._attn_size * self._attn_length)
        if self._state_is_tuple:
            return size
        else:
            return sum(list(size))

    @property
    def output_size(self):
        return self._attn_size

    def call(self, inputs, state):
        if self._state_is_tuple:
            state, attns, attn_states = state
        else:
            states = state
            state = tf.slice(states, [0, 0], [-1, self._cell.state_size])
            attns = tf.slice(states, [0, self._cell.state_size],
                             [-1, self._attn_size])
            attn_states = tf.slice(
                states, [0, self._cell.state_size + self._attn_size],
                [-1, self._attn_size * self._attn_length])
        attn_states = tf.reshape(attn_states,
                                 [-1, self._attn_length, self._attn_size])
        input_size = self._input_size
        if input_size is None:
            input_size = inputs.shape[1]
        if self._linear1 is None:
            self._linear1 = _Linear(input_size, True)
        inputs = self._linear1([inputs, attns])
        cell_output, new_state = self._cell(inputs, state)
        if self._state_is_tuple:
            new_state_cat = tf.concat(tf.nest.flatten(new_state), 1)
        else:
            new_state_cat = new_state
        new_attns, new_attn_states = self._attention(new_state_cat, attn_states)
        if self._linear2 is None:
            self._linear2 = _Linear(self._attn_size, True)
        output = self._linear2([cell_output, new_attns])
        new_attn_states = tf.concat(
            [new_attn_states, tf.expand_dims(output, 1)], 1)
        new_attn_states = tf.reshape(
            new_attn_states, [-1, self._attn_length * self._attn_size])
        new_state = (new_state, new_attns, new_attn_states)
        if not self._state_is_tuple:
            new_state = tf.concat(list(new_state), 1)
        return output, new_state

    def _attention(self, query, attn_states):
        conv2d = tf.nn.conv2d
        reduce_sum = tf.reduce_sum
        softmax = tf.nn.softmax
        tanh = tf.tanh

        if self._k is None:
            self._k = self.add_weight(
                "attn_w",
                shape=[1, 1, self._attn_size, self._attn_vec_size],
                initializer="glorot_uniform"
            )
            self._v = self.add_weight(
                "attn_v",
                shape=[self._attn_vec_size],
                initializer="zeros"
            )
        k = self._k
        v = self._v
        hidden = tf.reshape(attn_states,
                            [-1, self._attn_length, 1, self._attn_size])
        hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
        if self._linear3 is None:
            self._linear3 = _Linear(self._attn_vec_size, True)
        y = self._linear3(query)
        y = tf.reshape(y, [-1, 1, 1, self._attn_vec_size])
        s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
        a = softmax(s)
        d = reduce_sum(
            tf.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
        new_attns = tf.reshape(d, [-1, self._attn_size])
        new_attn_states = tf.slice(attn_states, [0, 1, 0], [-1, -1, -1])
        return new_attns, new_attn_states

def stack_bidirectional_dynamic_rnn(cells_fw,
                                    cells_bw,
                                    inputs,
                                    initial_states_fw=None,
                                    initial_states_bw=None,
                                    dtype=None,
                                    sequence_length=None,
                                    parallel_iterations=None,
                                    time_major=False,
                                    scope=None,
                                    swap_memory=False):
    if not cells_fw:
        raise ValueError("Must specify at least one fw cell for BidirectionalRNN.")
    if not cells_bw:
        raise ValueError("Must specify at least one bw cell for BidirectionalRNN.")
    if not isinstance(cells_fw, list):
        raise ValueError("cells_fw must be a list of RNNCells (one per layer).")
    if not isinstance(cells_bw, list):
        raise ValueError("cells_bw must be a list of RNNCells (one per layer).")
    if len(cells_fw) != len(cells_bw):
        raise ValueError("Forward and Backward cells must have the same depth.")
    if (initial_states_fw is not None and
        (not isinstance(initial_states_fw, list) or
         len(initial_states_fw) != len(cells_fw))):
        raise ValueError(
            "initial_states_fw must be a list of state tensors (one per layer).")
    if (initial_states_bw is not None and
        (not isinstance(initial_states_bw, list) or
         len(initial_states_bw) != len(cells_bw))):
        raise ValueError(
            "initial_states_bw must be a list of state tensors (one per layer).")

    states_fw = []
    states_bw = []
    prev_layer = inputs

    for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
        initial_state_fw = None
        initial_state_bw = None
        if initial_states_fw:
            initial_state_fw = initial_states_fw[i]
        if initial_states_bw:
            initial_state_bw = initial_states_bw[i]

        outputs = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(cell_fw, return_sequences=True, return_state=True),
            backward_layer=tf.keras.layers.RNN(cell_bw, return_sequences=True, return_state=True),
            merge_mode='concat'
        )(prev_layer, initial_state=[initial_state_fw, initial_state_bw] if initial_state_fw and initial_state_bw else None)
        # outputs: [output, fw_state..., bw_state...]
        prev_layer = outputs[0]
        state_fw = outputs[1:1+len(cell_fw.state_size) if hasattr(cell_fw.state_size, '__len__') else 2]
        state_bw = outputs[1+len(cell_fw.state_size) if hasattr(cell_fw.state_size, '__len__') else 2:]
        states_fw.append(state_fw)
        states_bw.append(state_bw)

    return prev_layer, tuple(states_fw), tuple(states_bw)

def _lstm_block_cell(x,
                     cs_prev,
                     h_prev,
                     w,
                     b,
                     wci=None,
                     wcf=None,
                     wco=None,
                     forget_bias=None,
                     cell_clip=None,
                     use_peephole=None,
                     name=None):
    if wci is None:
        cell_size = int(cs_prev.shape[1])
        if cell_size is None:
            raise ValueError("cell_size from `cs_prev` should not be None.")
        wci = tf.zeros([cell_size], dtype=tf.float32)
        wcf = wci
        wco = wci

    xh = tf.concat([x, h_prev], axis=1)
    gates = tf.matmul(xh, w) + b
    i, ci, f, o = tf.split(gates, num_or_size_splits=4, axis=1)
    f = f + forget_bias

    if use_peephole:
        i = tf.sigmoid(cs_prev * wci + i)
        f = tf.sigmoid(cs_prev * wcf + f)
    else:
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)

    ci = tf.tanh(ci)
    cs = ci * i + cs_prev * f
    if cell_clip is not None and cell_clip > 0:
        cs = tf.clip_by_value(cs, -cell_clip, cell_clip)

    if use_peephole:
        o = tf.sigmoid(cs * wco + o)
    else:
        o = tf.sigmoid(o)
    co = tf.tanh(cs)
    h = co * o
    return i, cs, f, o, ci, co, h

class LayerRNNCell(CompatRNNCell):
    """Subclass of RNNCells that act like proper `tf.Layer` objects."""

    def __call__(self, inputs, state, *args, scope=None, **kwargs):
        return tf.keras.layers.Layer.__call__(
            self, inputs, state, *args, **kwargs)

class LSTMBlockCell(LayerRNNCell):
    """Basic LSTM recurrent network cell."""

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 cell_clip=None,
                 use_peephole=False,
                 dtype=None,
                 reuse=None,
                 name="lstm_cell"):
        super(LSTMBlockCell, self).__init__(dtype=dtype, name=name)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._use_peephole = use_peephole
        self._cell_clip = cell_clip if cell_clip is not None else -1
        self._names = {
            "W": "kernel",
            "b": "bias",
            "wci": "w_i_diag",
            "wcf": "w_f_diag",
            "wco": "w_o_diag",
            "scope": "lstm_cell"
        }
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if not inputs_shape[1]:
            raise ValueError(
                "Expecting inputs_shape[1] to be set: %s" % str(inputs_shape))
        input_size = int(inputs_shape[1])
        self._kernel = self.add_weight(
            self._names["W"],
            shape=[input_size + self._num_units, self._num_units * 4],
            initializer="glorot_uniform"
        )
        self._bias = self.add_weight(
            self._names["b"],
            shape=[self._num_units * 4],
            initializer=tf.constant_initializer(0.0)
        )
        if self._use_peephole:
            self._w_i_diag = self.add_weight(self._names["wci"], [self._num_units], initializer="zeros")
            self._w_f_diag = self.add_weight(self._names["wcf"], [self._num_units], initializer="zeros")
            self._w_o_diag = self.add_weight(self._names["wco"], [self._num_units], initializer="zeros")
        self.built = True

    def call(self, inputs, state):
        if len(state) != 2:
            raise ValueError("Expecting state to be a tuple with length 2.")

        if self._use_peephole:
            wci = self._w_i_diag
            wcf = self._w_f_diag
            wco = self._w_o_diag
        else:
            wci = wcf = wco = tf.zeros([self._num_units], dtype=self.dtype)

        (cs_prev, h_prev) = state
        (_, cs, _, _, _, _, h) = _lstm_block_cell(
            inputs,
            cs_prev,
            h_prev,
            self._kernel,
            self._bias,
            wci=wci,
            wcf=wcf,
            wco=wco,
            forget_bias=self._forget_bias,
            cell_clip=self._cell_clip,
            use_peephole=self._use_peephole)

        new_state = LSTMStateTuple(cs, h)
        return h, new_state

class LayerNormBasicLSTMCell(CompatRNNCell):
    """LSTM unit with layer normalization and recurrent dropout."""

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 input_size=None,
                 activation=tf.tanh,
                 layer_norm=True,
                 norm_gain=1.0,
                 norm_shift=0.0,
                 dropout_keep_prob=1.0,
                 dropout_prob_seed=None,
                 reuse=None):
        super(LayerNormBasicLSTMCell, self).__init__(name="LayerNormBasicLSTMCell")

        if input_size is not None:
            warnings.warn(f"{self}: The input_size parameter is deprecated.")

        self._num_units = num_units
        self._activation = activation
        self._forget_bias = forget_bias
        self._keep_prob = dropout_keep_prob
        self._seed = dropout_prob_seed
        self._layer_norm = layer_norm
        self._norm_gain = norm_gain
        self._norm_shift = norm_shift
        self._reuse = reuse
        self._norm_vars = {}
        self._kernel = None
        self._bias = None

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def _norm(self, inp, scope, dtype=tf.float32):
        shape = inp.shape[-1:]
        if scope not in self._norm_vars:
            gamma_init = tf.constant_initializer(self._norm_gain)
            beta_init = tf.constant_initializer(self._norm_shift)
            gamma = self.add_weight(
                scope + "_gamma", shape=shape, initializer=gamma_init, dtype=dtype)
            beta = self.add_weight(
                scope + "_beta", shape=shape, initializer=beta_init, dtype=dtype)
            self._norm_vars[scope] = (gamma, beta)
        gamma, beta = self._norm_vars[scope]
        mean, var = tf.nn.moments(inp, axes=[1], keepdims=True)
        normalized = (inp - mean) / tf.sqrt(var + 1e-12)
        return normalized * gamma + beta

    def _linear(self, args):
        out_size = 4 * self._num_units
        proj_size = args.shape[-1]
        dtype = args.dtype
        if self._kernel is None:
            self._kernel = self.add_weight(
                "kernel", shape=[proj_size, out_size], dtype=dtype, initializer="glorot_uniform")
        out = tf.matmul(args, self._kernel)
        if not self._layer_norm:
            if self._bias is None:
                self._bias = self.add_weight(
                    "bias", shape=[out_size], dtype=dtype, initializer=tf.zeros_initializer())
            out = tf.nn.bias_add(out, self._bias)
        return out

    def call(self, inputs, state):
        c, h = state
        args = tf.concat([inputs, h], 1)
        concat = self._linear(args)
        dtype = args.dtype

        i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)
        if self._layer_norm:
            i = self._norm(i, "input", dtype=dtype)
            j = self._norm(j, "transform", dtype=dtype)
            f = self._norm(f, "forget", dtype=dtype)
            o = self._norm(o, "output", dtype=dtype)

        g = self._activation(j)
        if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
            g = tf.nn.dropout(g, rate=1.0 - self._keep_prob, seed=self._seed)

        new_c = (
            c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * g)
        if self._layer_norm:
            new_c = self._norm(new_c, "state", dtype=dtype)
        new_h = self._activation(new_c) * tf.sigmoid(o)

        new_state = LSTMStateTuple(new_c, new_h)
        return new_h, new_state