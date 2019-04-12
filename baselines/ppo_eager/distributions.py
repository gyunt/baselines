import numpy as np
import tensorflow as tf


class Pd(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return - self.neglogp(x)

    def get_shape(self):
        return self.flatparam().shape

    @property
    def shape(self):
        return self.get_shape()

    def __getitem__(self, idx):
        return self.__class__(self.flatparam()[idx])


class PdType(object):
    """
    Parametrized family of probability distributions
    """

    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def pdfromlatent(self, latent_vector, init_scale, init_bias):
        raise NotImplementedError

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.__dict__ == other.__dict__)


class CategoricalPd(Pd):
    def __init__(self, logits, depth=None):
        self.logits = logits
        self.depth = depth or self.logits.get_shape().as_list()[-1]

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        return tf.nn.softmax(self.logits)

    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            # one-hot encoding
            x = tf.one_hot(x, self.depth, axis=1)
        # else:
        #     # already encoded
        #     assert x.shape.as_list() == self.logits.shape.as_list(), (x.shape.as_list(), self.logits.shape.as_list())

        return tf.keras.backend.categorical_crossentropy(target=x, output=self.logits)

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    def sample(self):
        u = tf.random.uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.math.log(-tf.math.log(u)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class MultiCategoricalPd(Pd):
    def __init__(self, nvec, flat):
        self.flat = flat
        self.categoricals = []

        for tensor, depth in zip(tf.split(flat, np.array(nvec, dtype=np.int32), axis=1), nvec):
            self.categoricals.append(CategoricalPd(tensor, depth))

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)

    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])

    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    def sample(self):
        return tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)

    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError


def make_pdtype(ac_space):
    from gym import spaces

    if isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPdType(ac_space.nvec)
    else:
        raise NotImplementedError


class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat

    # def pdfromflat(self, flat):
    #     return self.pdclass()(flat)

    def pdclass(self):
        return CategoricalPd

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = tf.keras.layers.Dense(self.ncat,
                                        activation=tf.nn.softmax,
                                        kernel_initializer=tf.keras.initializers.Constant(1.0),
                                        bias_initializer=tf.keras.initializers.zeros(), )(latent_vector)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int32


class MultiCategoricalPdType(PdType):
    def __init__(self, nvec):
        self.ncats = nvec.astype('int32')
        assert (self.ncats > 0).all()

    def pdclass(self):
        return MultiCategoricalPd

    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.ncats, flat)

    def pdfromlatent(self, latent, init_scale=1.0, init_bias=0.0):
        pdparam = tf.keras.layers.Dense(self.ncats.sum(),
                                        activation=tf.nn.softmax,
                                        kernel_initializer=tf.keras.initializers.Constant(1.0),
                                        bias_initializer=tf.keras.initializers.zeros(), )(latent)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [sum(self.ncats)]

    def sample_shape(self):
        return [len(self.ncats)]

    def sample_dtype(self):
        return tf.int32
