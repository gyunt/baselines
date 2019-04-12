import numpy as np
import tensorflow as tf
from baselines.a2c.utils import ortho_init

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def mlp(num_layers=2, num_hidden=64, activation=tf.keras.activations.tanh, layer_norm=False):
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

    @tf.function
    def network_fn(X):
        h = tf.keras.layers.Flatten()(X)

        for i in range(num_layers):
            h = tf.keras.layers.Dense(num_hidden,
                                      name='mlp_fc{}'.format(i),
                                      kernel_initializer=ortho_init(np.sqrt(2)))(h)
            if layer_norm:
                h = tf.layers.BatchNormalization(h)
            h = activation(h)
        return h

    return network_fn


# @tf.function
def func(X):
    h = tf.keras.layers.Flatten()(X)

    for i in range(2):
        h = tf.keras.layers.Dense(64,
                                  name='mlp_fc{}'.format(i),
                                  kernel_initializer=ortho_init(np.sqrt(2)))(h)
        h = tf.tanh(h)
    return h


class Net(tf.Module):

    def __init__(self):
        self.y = None

    @tf.function
    def add(self, x):
        if self.y is None:
            self.y = tf.Variable(2.)
        return x + self.y


class Adder(tf.train.Checkpoint):

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 50, 512), dtype=tf.float32)])
    def add(self, x):
        x = tf.keras.layers.Flatten()(x)
        return x + x + 1.


def p():
    from datetime import datetime
    # The function to be traced.
    @tf.function
    def my_func(x, y):
        # A simple hand-rolled layer.
        return tf.nn.relu(tf.matmul(x, y))

    # Set up logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'i:\\tmp\\eager\\%s' % stamp
    writer = tf.summary.create_file_writer(logdir)

    # Sample data for your function.
    x = tf.random.uniform((3, 3))
    y = tf.random.uniform((3, 3))

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().
    tf.summary.trace_on(graph=True, profiler=True)
    # Call only one tf.function when tracing.
    z = my_func(x, y)
    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir)


class UnitEmbed():
    def __init__(self):
        self.input = keras.Input(shape=(26,), name='unit_embed')
        x = tf.keras.layers.Dense(64, activation='relu')(self.input)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(26)(x)
        self.output = x
        self.model = keras.Model(self.input, self.output, name='encoder')
        self.model.summary()


def unit_embed():
    unit_embed_input = keras.Input(shape=(26,), name='unit_feature')
    x = tf.keras.layers.Dense(64, activation='relu')(unit_embed_input)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    unit_embed_output = x
    unit_embed = keras.Model(encoder_input, unit_embed_output, name='unit_embed')
    return unit_embed


def p2():
    keras = tf.keras
    layers = keras.layers

    unit_embed = UnitEmbed()
    unit_embed_2 = UnitEmbed()

    a = unit_embed.model(unit_embed_2.model(tf.random.uniform((2, 26))))


if __name__ == '__main__':
    keras = tf.keras
    layers = keras.layers

    encoder_input = keras.Input(shape=(26,), name='original_img')
    x = tf.keras.layers.Dense(64, activation='relu')(encoder_input)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    # x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
    # x = layers.Conv2D(32, 3, activation='relu')(x)
    # x = layers.MaxPooling2D(3)(x)
    # x = layers.Conv2D(32, 3, activation='relu')(x)
    # x = layers.Conv2D(16, 3, activation='relu')(x)
    # x = layers.GlobalMaxPooling2D()(x)
    encoder_output = x
    encoder = keras.Model(encoder_input, encoder_output, name='encoder')

    import inspect
    from pprint import pprint

    pprint(inspect.getmembers(encoder, predicate=inspect.ismethod))

    encoder(tf.random.uniform((1, 26)))

    p()
    #
    # encoder = keras.Model(encoder_input, encoder_output, name='encoder')
    # encoder.summary()
    #
    # decoder_input = keras.Input(shape=(16,), name='encoded_img')
    # x = layers.Reshape((4, 4, 1))(decoder_input)
    # x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    # x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
    # x = layers.UpSampling2D(3)(x)
    # x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    # decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)
    #
    # decoder = keras.Model(decoder_input, decoder_output, name='decoder')
    # decoder.summary()
    #
    # autoencoder_input = keras.Input(shape=(28, 28, 1), name='img')
    # encoded_img = encoder(autoencoder_input)
    # decoded_img = decoder(encoded_img)
    # autoencoder = keras.Model(autoencoder_input, decoded_img, name='autoencoder')
    # autoencoder.summary()
