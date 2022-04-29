import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from .kwargs import PCNET_INITIALIZATION_KWARGS


def error_gate_movingMV(step, m1, m2, val, gate, gate_below, stop_gradient=True, epsilon=1e-8):
    step_prev = step
    m1_prev = m1
    m2_prev = m2
    gate_prev = gate
    step += 1

    _m1 = m1
    m1 += (val - m1) / step
    m2 += (val - m1) * (val - _m1)

    sd = tf.maximum(tf.sqrt(m2 / step + epsilon), epsilon)
    gate = (val - m1) / sd  # streaming z-score
    gate = tf.clip_by_value(gate, 0., np.inf)
    gate = tf.tanh(gate)  # Valid because z-scores are clipped to be positive
    if stop_gradient:
        gate = tf.stop_gradient(gate)

    filter = 1 - gate
    step *= filter
    m2 *= filter

    gate_below_comp = 1 - gate_below
    step = step * gate_below + step_prev * gate_below_comp
    m1 = m1 * gate_below + m1_prev * gate_below_comp
    m2 = m2 * gate_below + m2_prev * gate_below_comp
    # gate = gate * gate_below + gate_prev * gate_below_comp
    gate *= gate_below

    return (step, m1, m2, gate)

def error_gate_ExpMV(step, m1, m2, val, stop_gradient=False, beta=0.9, epsilon=1e-8):
    step += 1

    _m1 = val
    _m2 = tf.square(val)
    m1 = m1 * beta + _m1 * (1. - beta)
    m2 = m2 * beta + _m2 * (1. - beta)
    sd = tf.maximum(tf.sqrt(m2), epsilon)
    gate = (val - m1) / sd  # streaming z-score
    gate = tf.clip_by_value(gate, 0., np.inf)
    gate = tf.tanh(gate)  # Valid because z-scores are clipped to be positive
    if stop_gradient:
        gate = tf.stop_gradient(gate)

    filter = 1 - gate
    step *= filter
    m2 *= filter

    return (step, m1, m2, gate)


def incremental_cosine_distance(x):
    """
    Cosine distance from one timestep to the next.
    Assumes the final two dimensions are <time, features>.

    :param x: A tensor of dimension D of vector representations over time. Dimensions <= D-2 are treated as batch dimensions.
    :return: A tensor of dimension D-1 (feature dimension reduced) containing cosine distances from one timestep to the next. Note that the time dimension will be reduced by 1 (i.e. if there are T timesteps, there will be T-1 similarities).
    """

    x_tm1 = tf.nn.l2_normalize(x[..., :-1, :], axis=-1)
    x_t = tf.nn.l2_normalize(x[..., 1:, :], axis=-1)
    sim = tf.reduce_sum(x_tm1 * x_t, axis=-1)
    dist = (1 - sim) / 2
    paddings = []
    for _ in x.shape[:-2]:
        paddings.append((0,0))
    paddings.append((0,1))
    dist = tf.pad(dist, paddings)

    return dist


def autocorrelation(x):
    """
    First order autocorrelation by feature dimension.
    Assumes the final two dimensions are <time, features>.

    :param x: A tensor of dimension D of vector representations over time. Dimensions <= D-2 are treated as batch dimensions.
    :return: A tensor of dimension D-1 (time dimension reduced) containing an autocorrelation value for each feature dimension.
    """

    x_tm1 = tf.nn.l2_normalize(x[..., :-1, :], axis=-1)
    x_t = tf.nn.l2_normalize(x[..., 1:, :], axis=-1)
    rho = tf.reduce_sum(x_tm1 * x_t, axis=-2) / (tf.cast(tf.shape(x_t)[-2] - 1, tf.float32))

    return rho


def swap_gradient(f, g=None):
    if g is None:
        g = tf.identity

    def out(x, f=f, g=g):
        a = f(x)
        b = g(x)

        return b + tf.stop_gradient(a - b)

    return out


def clip_by_value_preserve_gradient(x, clip_value_min, clip_value_max):
    def f(x, clip_value_min=clip_value_min, clip_value_max=clip_value_max):
        return tf.clip_by_value(x, clip_value_min=clip_value_min, clip_value_max=clip_value_max)
    return swap_gradient(f)(x)


class ProximityRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l1_min=0.01, l1_max=0.1):
        self.l1_min = tf.cast(l1_min, tf.float32)
        self.l1_max = tf.cast(l1_max, tf.float32)

    def __call__(self, x):
        a = tf.linspace(0., self.l1_max - self.l1_min, int(x.shape[-2]))
        b = tf.linspace(0., self.l1_max - self.l1_min, int(x.shape[-1]))
        W = tf.abs(a[..., None] - b[..., None]) + self.l1_min

        return tf.reduce_sum(tf.abs(x) * W)

class PCRNNCell(tf.keras.layers.AbstractRNNCell):
    def __init__(
            self,
            n_features,
            n_units,
            n_layers=2,
            hyperspheric=False,
            residual_update=False,
            bptt=True,
            backprop_into_targets=True,
            concentration=None,
            noise_sd=1e-1,
            kernel_regularizer=None,
            activation='tanh',
            dam=False,
            cutoff=3.,
            epsilon=1e-8,
            **kwargs
    ):
        self.n_features = n_features
        self.n_units = n_units
        self.n_layers = n_layers
        self.hyperspheric = hyperspheric
        self.residual_update = residual_update
        self.bptt = bptt
        self.backprop_into_targets = backprop_into_targets
        self.concentration = concentration
        self.noise_sd = noise_sd
        self.cutoff = cutoff
        self.epsilon = epsilon
        super(PCRNNCell, self).__init__(**kwargs)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        if activation:
            self.activation = getattr(tf, activation)
        else:
            self.activation = tf.identity
        self.dam = dam

    @property
    def state_size(self):
        states = []
        n_target_units = self.n_features
        for l in range(self.n_layers):
            #              state         preds           step err_m1  err_m2  gate
            states.append((self.n_units, n_target_units, 1,   1,      1,      1))
            n_target_units = self.n_units
        states = tuple(states)

        return states

    @property
    def output_size(self):
        out = []
        n_target_units = self.n_features
        for l in range(self.n_layers):
            #           state         err           preds           resid           loss   gate
            out.append((self.n_units, self.n_units, n_target_units, n_target_units, 1,     1))
            n_target_units = self.n_units
        out = tuple(out)

        return out

    def initialize_mlp(self, name, n_inputs, n_outputs, n_layers=1):
        layers = []
        for d in range(n_layers):
            kernel = self.add_weight(
                shape=(n_inputs, n_outputs),
                initializer='glorot_uniform',
                regularizer=self.kernel_regularizer,
                name='%s_d%d_kernel' % (name, d + 1)
            )
            bias = self.add_weight(
                shape=(n_outputs,),
                initializer='zeros',
                regularizer=None,
                name='%s_d%d_bias' % (name, d + 1)
            )
            if d < n_layers - 1:
                layers.append(lambda x, kernel=kernel, bias=bias: tf.tanh(tf.matmul(x, kernel) + bias[None, ...]))
            else:
                layers.append(lambda x, kernel=kernel, bias=bias: tf.matmul(x, kernel) + bias[None, ...])

            n_inputs = n_outputs

        def fn(x, layers=tuple(layers)):
            out = x
            for l, layer in enumerate(layers):
                out = layer(out)
            return out

        return fn

    def build(self, input_shape):

        self.transition = []
        self.topdown = []
        self.prediction = []

        self.noise_dist = tfd.VonMisesFisher

        L = self.n_layers

        n_features = self.n_features

        for l in range(L):
            # Kernels
            self.transition.append(
                self.initialize_mlp(
                    'transition_l%d' % (l + 1),
                    self.n_units + n_features,
                    self.n_units
                )
            )
            self.prediction.append(
                self.initialize_mlp(
                    'prediction_l%d' % (l + 1),
                    self.n_units,
                    n_features
                )
            )

            n_features = self.n_units

            if self.dam:
                self.memory_matrix.append(
                    self.add_weight(
                        shape=(self.n_units, self.n_units),
                        initializer='zeros',
                        name='memory_matrix_l%d' % (l + 1)
                    )
                )

        self.built = True

    @tf.function
    def call(self, inputs, states, training=None):
        gate_below = 1.
        outputs = []
        new_states = []

        for l in range(self.n_layers):
            _state, _preds, _step, _err_m1, _err_m2, _gate = states[l]

            if not self.backprop_into_targets:
                inputs = tf.stop_gradient(inputs)

            # Predictions
            preds = self.prediction[l](_state)
            resid = inputs - preds
            err = tf.square(resid)
            loss = tf.reduce_mean(err, axis=-1, keepdims=True)
            step, err_m1, err_m2, gate = error_gate_movingMV(
                _step, _err_m1, _err_m2, loss, _gate, gate_below
            )

            # New state (recurrent)
            state = tf.concat([_state, resid], axis=-1)
            state = self.transition[l](state)
            if self.residual_update:
                state += _state

            # New state (topdown)
            if l < self.n_layers - 1:
                topdown = states[l+1][1]
                state = gate * topdown + (1. - gate) * state
                # state = (state + topdown) / 2
            state = self.activation(state)
            state = gate_below * state + (1 - gate_below) * _state

            outputs.append((state, err, preds, resid, loss, gate))
            new_states.append((state, preds, step, err_m1, err_m2, gate))
            inputs = state
            gate_below = gate

        outputs = tuple(outputs)
        new_states = tuple(new_states)

        return outputs, new_states


class PCRNNLayer(tf.keras.layers.RNN):
    def __init__(
        self,
        n_features,
        n_units,
        n_layers=2,
        activation='tanh',
        kernel_regularizer=None,
        return_sequences=True,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        time_major=False,
        **kwargs
    ):
        self.n_features = n_features
        self.n_units = n_units
        self.n_layers = n_layers
        self.activation=activation
        self.kernel_regularizer = kernel_regularizer

        cell = PCRNNCell(
            self.n_features,
            self.n_units,
            n_layers=self.n_layers,
            activation=self.activation,
            kernel_regularizer=self.kernel_regularizer
        )
        super(PCRNNLayer, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            time_major=time_major,
            **kwargs
        )


class PCRNNModel(tf.keras.Model):
    _INITIALIZATION_KWARGS = PCNET_INITIALIZATION_KWARGS

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(PCRNNModel, self).__init__()

        ## Store initialization settings
        for kwarg in PCRNNModel._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        if self.sparsity_regularizer_scale:
            self.kernel_regularizer = tf.keras.regularizers.l1(l1=self.sparsity_regularizer_scale)
        elif self.kernel_regularizer_scale:
            self.kernel_regularizer = tf.keras.regularizers.l2(l2=self.sparsity_regularizer_scale)
        else:
            self.kernel_regularizer = None
        self.rnn = PCRNNLayer(self.n_features, self.n_units, n_layers=self.n_layers, kernel_regularizer=self.kernel_regularizer)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=True,
            name="Adam",
        )
        self.compile(optimizer=self.optimizer)

    def call(
            self,
            inputs,
            return_states=True,
            return_predictions=False,
            return_errors=True,
            return_gates=True,
            return_distances=False,
            return_autocorrelation=False,
            training=None
    ):
        outputs = self.rnn(inputs)

        # Group by variable type instead of layer
        if self.state_regularizer_scale or return_states or return_distances or return_autocorrelation:
            states = tuple([x[0] for x in outputs])
            if self.state_regularizer_scale:
                for x in states:
                    self.add_loss(self.state_regularizer_scale * tf.reduce_mean(tf.abs(x)))
            # for x in states:
            #     tf.print('state mean', tf.reduce_mean(x))
            #     tf.print('state max', tf.reduce_max(x))
            #     tf.print('state min', tf.reduce_min(x), '\n')
        else:
            states = None
        if return_predictions:
            predictions = tuple([x[2] for x in outputs])
        else:
            predictions = None
        residuals = tuple([x[3] for x in outputs])
        loss = tuple([x[4] for x in outputs])
        for l, x in enumerate(loss):
            _loss = tf.reduce_mean(x)
            tf.debugging.assert_all_finite(_loss, 'Non-finite loss.')
            self.add_loss(_loss)
            # tf.print('loss mean', tf.reduce_mean(x))
            # tf.print('loss max', tf.reduce_max(x))
            # tf.print('loss min', tf.reduce_min(x), '\n')
        if self.gate_regularizer_scale or return_gates:
            gates = tuple([x[5][..., 0] for x in outputs])
            if self.gate_regularizer_scale:
                for x in gates:
                    self.add_loss(self.gate_regularizer_scale * tf.reduce_mean(x))
        else:
            gates = None

        # Collect outputs
        out = []

        if return_states:
            out.append(states)
        if return_predictions:
            out.append(predictions)
        if return_errors:
            out.append(residuals)
        if return_gates:
            out.append(gates)
        if return_distances:
            out.append(tuple([incremental_cosine_distance(x) for x in residuals]))
        if return_autocorrelation:
            out.append(tuple([autocorrelation(x) for x in residuals]))

        if not out:
            out = None
        elif len(out) == 1:
            out = out[0]
        else:
            out = tuple(out)

        return out

    def save(self, path):
        if self.saver is None:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    self.saver = tf.compat.v1.train.Saver()
        self.saver.save(self.sess, path)

    def restore(self, path):
        self.saver.load(self.sess, path)
