import gym
import numpy as np
import tensorflow as tf
from gym import spaces

from baselines.a2c.utils import fc
from baselines.common import tf_util
from baselines.common.distributions import PdType, Pd
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.models import get_network_builder
from baselines.common.tf_util import adjust_shape
from baselines.ppo2.layers import RNN, max_pool


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and two value function estimation with shared parameters
    """

    def __init__(self, observations, action_space, latent, dones, is_training, states=None, estimate_q=False,
                 vf_latent=None,
                 embed_dim=32,
                 sess=None):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """
        self.X = observations
        self.dones = dones
        self.is_training = is_training
        self.pdtype = SC2ActionsPdType(action_space.nvec)
        self.states = states
        self.sess = sess or tf.get_default_session()

        vf_latent = vf_latent if vf_latent is not None else latent

        with tf.variable_scope('policy'):
            latent = tf.layers.flatten(latent)
            # Based on the action space, will select what probability distribution type

            avail_actions = tf.layers.flatten(observations['avail_actions'])

            self.pd, self.pi = self.pdtype.pdfromlatent(latent, availables=avail_actions)

            with tf.variable_scope('sample_action'):
                self.action = self.pd.sample()

            with tf.variable_scope('negative_log_probability'):
                # Calculate the neg log of our probability
                self.neglogp = self.pd.neglogp(self.action)

        with tf.variable_scope('value'):
            obs_states = observations['obs_states']
            n_agents = 3
            embed_dim = 32

            vf_latent = tf.layers.flatten(vf_latent)

            self.value_net = tf.make_template('value_pdparam_net',
                                              _value_pdparam_net,
                                              num_hidden=1)
            values = []
            for i in range(0, 128 * 3, 128):
                values.append(self.value_net(vf_latent[:, i: i + 128]))
            values = tf.concat(values, axis=1)
            values = tf.reshape(values, shape=(-1, 1, n_agents))
            self.value = mixing_network(obs_states, values)

            # if estimate_q:
            #     assert isinstance(action_space, gym.spaces.Discrete)
            #     self.q = fc(vf_latent, 'q', action_space.n)
            #     self.value = self.q
            # else:
            #     vf_latent = tf.layers.flatten(vf_latent)
            #     self.value = fc(vf_latent, 'value', 1, init_scale=0.01)
            #     self.value = self.value[:, 0]

        if isinstance(self.X, dict):
            self.step_input = {
                'dones': self.dones,
                'is_training': self.is_training,
            }
            self.step_input.update(observations)
        else:
            self.step_input = {
                'observations': observations,
                'dones': self.dones,
            }

        self.step_output = {
            'actions': self.action,
            'values': self.value,
            'neglogpacs': self.neglogp,
        }
        if self.states:
            self.initial_state = np.zeros(self.states['current'].get_shape().as_list()[1:])
            self.step_input.update({'states': self.states['current']})
            self.step_output.update({'states': self.states['current'],
                                     'next_states': self.states['next']})
        else:
            self.initial_state = None

    def feed_dict(self, **kwargs):
        feed_dict = {}
        for key in kwargs:
            if key in self.step_input:
                feed_dict[self.step_input[key]] = adjust_shape(self.step_input[key], kwargs[key])
        return feed_dict

    def step(self, **kwargs):
        kwargs['is_training'] = False
        return self.sess.run(self.step_output,
                             feed_dict=self.feed_dict(**kwargs))

    def values(self, **kwargs):
        return self.sess.run({'values': self.value},
                             feed_dict=self.feed_dict(**kwargs))

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def mixing_network(obs_states, chosen_action_qvals,
                   embed_dim=32,
                   n_agents=3,
                   agent_dim=5,
                   n_enemy=4,
                   enemy_dim=4,
                   n_actions=10,
                   num_units=256,
                   num_output_dims=64):
    ally_states = obs_states[:, :n_agents * agent_dim]
    enemy_states = obs_states[:, n_agents * agent_dim: n_agents * agent_dim + n_enemy * enemy_dim]
    last_actions = obs_states[:, n_agents * agent_dim + n_enemy * enemy_dim:]

    ally_states = tf.reshape(ally_states, shape=(-1, n_agents, agent_dim))
    enemy_states =tf.reshape(enemy_states, shape=(-1, n_enemy, enemy_dim))
    last_actions = tf.reshape(last_actions, shape=(-1, n_agents, n_actions))
    ally_states = tf.concat([ally_states, last_actions], axis=2)

    ally_embed_net = tf.make_template('ally_embed_network',
                                      _unit_embed_net,
                                        num_hidden=num_units,
                                        layer_norm=True)

    enemy_embed_net = tf.make_template('enemy_embed_network',
                                       _unit_embed_net,
                                       num_hidden=num_units,
                                       layer_norm=True)

    ally_embeds = ally_embed_net(ally_states)
    enemy_embeds = enemy_embed_net(enemy_states)

    ally_embeds = max_pool(ally_embeds, ally_states)
    enemy_embeds = max_pool(enemy_embeds, enemy_states)

    states_embeds = tf.concat([ally_embeds, enemy_embeds], axis=1)

    # hyper_w_1 = tf.layers.dense(states_embeds, embed_dim * n_agents, None, True,
    #                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    # hyper_w_final = tf.layers.dense(states_embeds, embed_dim, None, True,
    #                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
    # hyper_b_1 = tf.layers.dense(states_embeds, embed_dim, None, True,
    #                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    # v = tf.layers.dense(states_embeds, embed_dim, tf.nn.relu, True,
    #                     kernel_initializer=tf.contrib.layers.xavier_initializer())
    # v = tf.layers.dense(v, 1, None, True,
    #                     kernel_initializer=tf.contrib.layers.xavier_initializer())
    # v = tf.reshape(v, shape=(-1, 1, 1))
    #
    # # w1 = tf.abs(hyper_w_1)
    # w1 = hyper_w_1
    # b1 = hyper_b_1
    # w1 = tf.reshape(w1, shape=(-1, n_agents, embed_dim))
    # b1 = tf.expand_dims(b1, axis=1)
    #
    # hidden = tf.nn.elu(tf.matmul(chosen_action_qvals, w1) + b1)
    #
    # # w_final = tf.abs(hyper_w_final)
    # w_final = hyper_w_final
    # w_final = tf.reshape(w_final, shape=(-1, embed_dim, 1))
    # return tf.squeeze(tf.reshape(tf.matmul(hidden, w_final) + v, shape=(-1, 1)), axis=1)
    embed = tf.layers.dense(states_embeds, num_units, tf.nn.relu, True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    embed = tf.layers.dense(embed, 1, None, True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    return tf.squeeze(embed, axis=1)




def _unit_embed_net(unit_embed,
                    activation_fn=tf.nn.relu,
                    num_hidden=64,
                    layer_norm=False,
                    ):
    embed = tf.layers.dense(unit_embed, num_hidden, activation_fn, True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    embed = tf.layers.dense(embed, num_hidden, activation_fn, True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    if layer_norm:
        embed = tf.contrib.layers.layer_norm(embed, center=True, scale=True)

    return embed


def get_elements(data, indices):
    shape = tf.shape(data)
    batch_size, num_agents, num_actions = shape[0], shape[1], shape[2]

    base = tf.reshape(tf.range(0, batch_size * num_agents * num_actions, num_actions), shape=(batch_size, num_agents))
    indices = base + indices

    data = tf.reshape(data, shape=(-1,))
    indices = tf.reshape(indices, shape=(-1, 1))
    return tf.reshape(tf.gather_nd(data, indices=indices), shape=(batch_size, 1, num_agents))


class SC2ActionsPdType(PdType):
    def __init__(self, nvec):
        self.ncats = nvec.astype('int32')
        self.pdparam_net = tf.make_template('test_pdparam_net',
                                            _value_pdparam_net,
                                            num_hidden=10)

        assert (self.ncats > 0).all()

    def pdclass(self):
        return SC2ActionsPd

    def pdfromflat(self, flat, availables=None):
        return SC2ActionsPd(self.ncats, flat, availables)

    def pdfromlatent(self, latent, init_scale=1.0, init_bias=0.0, availables=None):
        pdparam = tf.layers.dense(latent, self.ncats.sum(), None, True,
                                  name='pi',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        # pdparam = []
        # for i in range(0, 128 * 3, 128):
        #     pdparam.append(self.pdparam_net(latent[:, i: i+128]))
        # pdparam = tf.concat(pdparam, axis=1)
        return self.pdfromflat(pdparam, availables), pdparam

    def param_shape(self):
        return [sum(self.ncats)]

    def sample_shape(self):
        return [len(self.ncats)]

    def sample_dtype(self):
        return tf.int32


def _value_pdparam_net(x,
                       num_hidden=64, ):
    embed = tf.layers.dense(x, num_hidden, None, True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    return embed


class SC2ActionsPd(Pd):
    def __init__(self, nvec, flat, availables=None):
        self.flat = flat

        self.categoricals = []
        index = 0
        for size in nvec:
            self.categoricals.append(
                CategoricalPd(flat[:, index: index + size], availables[:, index: index + size]))
            index += size

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


class CategoricalPd(Pd):
    def __init__(self, logits, available=None):
        # self.logits = tf.nn.softmax(logits)
        self.logits = logits
        self.available = available

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
            x_shape_list = x.shape.as_list()
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

            x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        else:
            # already encoded
            assert x.shape.as_list() == self.logits.shape.as_list()

        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=x)

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        if self.available is None:
            return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
        return tf.argmax(tf.where(tf.cast(self.available, dtype=tf.bool),
                                  self.logits - tf.log(-tf.log(u)),
                                  -99999999 * tf.ones_like(self.logits)
                                  ), axis=-1)

        # tfp.distributions.OneHotCategorical(logit=self.logits)
        # aa = tfp.distributions.Categorical(logits=self.logits)
        # return aa.sample()#tf.argmax(aa.sample(), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


def build_ppo_policy(env, policy_network, value_network=None, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    if value_network is None:
        value_network = 'shared'

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None, ob_space=None, ac_space=None):
        if ob_space is None:
            ob_space = env.observation_space
        if ac_space is None:
            ac_space = env.action_space

        next_states_list = []
        state_map = {}
        state_placeholder = None
        interesting_field = ['obs', 'obs_states', 'avail_actions']

        if isinstance(ob_space, spaces.Dict):
            if observ_placeholder is not None:
                assert isinstance(observ_placeholder, dict)
                X = observ_placeholder
            else:
                X = dict()
                encoded_x = dict()
                for name in ob_space.spaces:
                    if name in interesting_field:
                        try:
                            X[name] = observation_placeholder(ob_space.spaces[name], name=name)
                            encoded_x[name] = encode_observation(ob_space.spaces[name], X[name])
                        except AssertionError:
                            pass
        else:
            X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space)
            encoded_x = encode_observation(ob_space, X)
        dones = tf.placeholder(tf.float32, shape=(None,), name='dones')
        is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

        with tf.variable_scope('current_rnn_memory'):
            if value_network == 'shared':
                value_network_ = value_network
            else:
                if value_network == 'copy':
                    value_network_ = policy_network
                elif isinstance(value_network, str):
                    network_type = value_network
                    value_network_ = get_network_builder(network_type)()
                else:
                    assert callable(value_network)
                    value_network_ = value_network

            policy_memory_size = policy_network.memory_size if isinstance(policy_network, RNN) else 0
            value_memory_size = value_network_.memory_size if isinstance(value_network_, RNN) else 0
            state_size = policy_memory_size + value_memory_size

            if state_size > 0:
                state_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, state_size),
                                                   name='states')

                state_map['policy'] = state_placeholder[:, 0:policy_memory_size]
                state_map['value'] = state_placeholder[:, policy_memory_size:]

        with tf.variable_scope('policy_latent', reuse=tf.AUTO_REUSE):
            if isinstance(policy_network, RNN):
                assert policy_memory_size > 0
                policy_latent, next_policy_state = \
                    policy_network(encoded_x, dones, state_map['policy'], is_training=is_training)
                next_states_list.append(next_policy_state)
            else:
                policy_latent = policy_network(encoded_x, is_training=is_training)

        with tf.variable_scope('value_latent', reuse=tf.AUTO_REUSE):
            if value_network_ == 'shared':
                value_latent = policy_latent
            elif isinstance(value_network_, RNN):
                assert value_memory_size > 0
                value_latent, next_value_state = \
                    value_network_(encoded_x, dones, state_map['value'], is_training=is_training)
                next_states_list.append(next_value_state)
            else:
                value_latent = value_network_(encoded_x, is_training=is_training)

        with tf.name_scope("next_rnn_memory"):
            if state_size > 0:
                next_states = tf.concat(next_states_list, axis=1)
                state_info = {'current': state_placeholder,
                              'next': next_states, }
            else:
                state_info = None

        policy = PolicyWithValue(
            observations=X,
            action_space=ac_space,
            dones=dones,
            is_training=is_training,
            latent=policy_latent,
            vf_latent=value_latent,
            states=state_info,
            sess=sess,
            estimate_q=estimate_q,
        )
        return policy

    return policy_fn
