import numpy as np
import tensorflow as tf
from baselines.a2c.utils import ortho_init
from baselines.common.models import register


@register("ppo_lstm")
def ppo_lstm(nlstm=128, layer_norm=False):
    """
    Builds LSTM (Long-Short Term Memory) network to be used in a policy.
    Note that the resulting function returns not only the output of the LSTM
    (i.e. hidden state of lstm for each step in the sequence), but also a dictionary
    with auxiliary tensors to be set as policy attributes.

    Specifically,
        S is a placeholder to feed current state (LSTM state has to be managed outside policy)
        M is a placeholder for the mask (used to mask out observations after the end of the episode, but can be used for other purposes too)
        initial_state is a numpy array containing initial lstm state (usually zeros)
        state is the output LSTM state (to be fed into S at the next call)


    An example of usage of lstm-based policy can be found here: common/tests/test_doc_examples.py/test_lstm_example

    Parameters:
    ----------

    nlstm: int          LSTM hidden state size

    layer_norm: bool    if True, layer-normalized version of LSTM is used

    Returns:
    -------

    function that builds LSTM with a given input tensor / placeholder
    """

    def network_fn(X, nenv=1):
        nbatch = X.shape[0]

        mask = tf.placeholder(tf.float32, [nbatch], name='mask')  # mask (done t-1)
        state = tf.placeholder(tf.float32, [nbatch, 2 * nlstm], name='state')  # states

        xs = tf.layers.flatten(X)
        ms = mask

        h, snew = lstm(xs, ms, state, scope='lstm', nh=nlstm)
        initial_state = np.zeros(state.shape.as_list(), dtype=float)

        return h, {'prev': {'state': state, 'mask': mask},
                   'post': {'state': snew},
                   'init': {'state': initial_state}}

    return network_fn


@register("ppo_cnn_lstm")
def ppo_cnn_lstm(nlstm=128, layer_norm=False, **conv_kwargs):
    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        mask = tf.placeholder(tf.float32, [nbatch], name='mask')
        state = tf.placeholder(tf.float32, [nbatch, 2 * nlstm], name='state')

        init = tf.constant_initializer(np.sqrt(2))

        h = tf.contrib.layers.conv2d(X,
                                     num_outputs=32,
                                     kernel_size=8,
                                     stride=4,
                                     padding="VALID",
                                     weights_initializer=init)
        h2 = tf.contrib.layers.conv2d(h,
                                      num_outputs=64,
                                      kernel_size=4,
                                      stride=2,
                                      padding="VALID",
                                      weights_initializer=init)
        h3 = tf.contrib.layers.conv2d(h2,
                                      num_outputs=64,
                                      kernel_size=3,
                                      stride=1,
                                      padding="VALID",
                                      weights_initializer=init)
        X = tf.layers.flatten(h3)
        X = tf.layers.dense(X, units=512, activation=tf.nn.relu, kernel_initializer=init)

        h, snew = lstm(X, mask, state, scope='lstm', nh=nlstm)
        initial_state = np.zeros(state.shape.as_list(), dtype=float)

        return h, {'prev': {'state': state, 'mask': mask},
                   'post': {'state': snew},
                   'init': {'state': initial_state}}

    return network_fn


@register("ppo_mlp")
def ppo_mlp(num_layers=2, num_hidden=128, activation=tf.nn.leaky_relu, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """

    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = tf.layers.dense(h,
                                units=num_hidden,
                                kernel_initializer=tf.constant_initializer(np.sqrt(2)),
                                name='mlp_fc{}'.format(i))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)
        return h

    return network_fn


def lstm(x, m, s, scope, nh, init_scale=1.0):
    x = tf.layers.flatten(x)
    nin = x.get_shape()[1]

    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh * 4], initializer=ortho_init(init_scale))
        wh = tf.get_variable("wh", [nh, nh * 4], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh * 4], initializer=tf.constant_initializer(0.0))

    m = tf.tile(tf.expand_dims(m, axis=-1), multiples=[1, nh])
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)

    c = c * (1 - m)
    h = h * (1 - m)
    z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
    i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
    i = tf.nn.sigmoid(i)
    f = tf.nn.sigmoid(f)
    o = tf.nn.sigmoid(o)
    u = tf.tanh(u)
    c = f * c + i * u
    h = o * tf.tanh(c)
    s = tf.concat(axis=1, values=[c, h])
    return h, s
