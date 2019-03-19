import inspect

import gym
import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common import tf_util
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.models import get_network_builder


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and two value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None, sess=None, states=None, prev=None,
                 post=None,
                 init=None,
                 **tensors):
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
        self.initial_state = None
        self.__dict__.update(tensors)
        self.states = states
        self.prev = prev or {}
        self.post = post or {}
        self.init = init or {}
        self.pdtype = make_pdtype(env.action_space)

        vf_latent = vf_latent if vf_latent is not None else latent
        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.value = fc(vf_latent, 'value', 1, init_scale=0.01)
            self.value = self.value[:, 0]

        self.step_output = {
            'actions': self.action,
            'values': self.value,
            'neglogpacs': self.neglogp}
        self.step_output.update(self.post)

        self.step_input = {
            'observations': observations
        }
        self.step_input.update(self.prev)

        self.mapping = {
            'dones': ['policy_mask', 'value_mask'],
        }

    def get_initial_state(self):
        return self.init.copy()

    def feed_dict(self, **kwargs):
        feed_dict = {}
        for key in kwargs:
            if key in self.step_input:
                feed_dict[self.step_input[key]] = kwargs[key]
            elif key in self.mapping:
                for ph_name in self.mapping[key]:
                    if ph_name in self.step_input:
                        feed_dict[self.step_input[ph_name]] = kwargs[key]
        return feed_dict

    def step(self, **kwargs):
        return self.sess.run(self.step_output,
                             feed_dict=self.feed_dict(**kwargs))

    def values(self, **kwargs):
        return self.sess.run({'values': self.value},
                             feed_dict=self.feed_dict(**kwargs))

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def build_ppo_policy(env, policy_network, value_network=None, normalize_observations=False, estimate_q=False,
                     **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        prev = {}
        post = {}
        init = {}

        ob_space = env.observation_space
        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space,
                                                                                              batch_size=nbatch)

        # TODO @gyunt
        # if normalize_observations and X.dtype == tf.float32:
        #     encoded_x, rms = _normalize_clip_observation(X)
        #     extra_tensors['rms'] = rms
        # else:
        encoded_x = X
        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            if is_rnn_network(policy_network):
                nenv = nbatch // nsteps
                policy_latent, network_infos = policy_network(encoded_x, nenv)
                prev_ = network_infos['prev']
                post_ = network_infos['post']
                init_ = network_infos['init']

                _add_prefix((prev_, post_, init_), prefix='policy_')

                prev.update(prev_)
                post.update(post_)
                init.update(init_)
            else:
                policy_latent = policy_network(encoded_x)

        value_network_ = value_network

        if value_network_ is None or value_network_ == 'shared':
            value_latent = policy_latent
        else:
            if value_network_ == 'copy':
                value_network_ = policy_network
            else:
                assert callable(value_network_)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                if is_rnn_network(value_network_):
                    nenv = nbatch // nsteps
                    value_latent, network_infos = value_network_(encoded_x, nenv)

                    prev_ = network_infos['prev']
                    post_ = network_infos['post']
                    init_ = network_infos['init']

                    _add_prefix((prev_, post_, init_), prefix='value_')

                    prev.update(prev_)
                    post.update(post_)
                    init.update(init_)
                else:
                    value_latent = value_network_(encoded_x)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=value_latent,
            sess=sess,
            estimate_q=estimate_q,
            prev=prev,
            post=post,
            init=init,
        )
        return policy

    return policy_fn


def is_rnn_network(network):
    return 'nenv' in inspect.getfullargspec(network).args


def _add_prefix(dicts, prefix=''):
    if isinstance(dicts, dict):
        dicts = [dicts]

    for dict_ in dicts:
        prefixed = {}
        for key in dict_:
            prefixed[prefix + key] = dict_[key]
        dict_.clear()
        dict_.update(prefixed)
