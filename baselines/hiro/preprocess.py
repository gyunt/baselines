from copy import copy

import numpy as np
import tensorflow as tf
from baselines.common.input import observation_placeholder
from tensorflow.python.ops.template import Template

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None


class StatePreprocess(object):
    STATE_PREPROCESS_NET_SCOPE = 'state_process_net'
    ACTION_EMBED_NET_SCOPE = 'action_embed_net'
    META_ACTION_EMBED_NET_SCOPE = 'meta_action_embed_net'

    def __init__(self,
                 ob_space,
                 subgoal_space,
                 act_space,
                 meta_action_every_n,
                 state_preprocess_net=lambda states: states,
                 action_embed_net=lambda actions, *args, **kwargs: actions,
                 meta_action_embed_net=lambda meta_actions,: meta_actions,
                 name='state_preprocess',
                 sampling_size=1024,
                 sess=None,
                 max_grad_norm=None,
                 ndims=None):
        self.sess = sess or tf.get_default_session()
        with tf.variable_scope(name) as scope:
            self.scope = scope
            self._ndims = ndims
            self.sampling_size = sampling_size
            assert len(subgoal_space.shape) == 1
            self.goal_dims = subgoal_space.shape[0]

            with tf.variable_scope('model'):
                if isinstance(state_preprocess_net, Template):
                    def _fn(observation):
                        return tf.stop_gradient(state_preprocess_net(observation))

                    self._state_preprocess_net = _fn
                else:
                    self._state_preprocess_net = tf.make_template(
                        self.STATE_PREPROCESS_NET_SCOPE, state_preprocess_net,
                        num_output_dims=self.goal_dims,
                        create_scope_now_=True)
                self._action_embed_net = tf.make_template(
                    self.ACTION_EMBED_NET_SCOPE, action_embed_net,
                    num_output_dims=self.goal_dims,
                    create_scope_now_=True)
                self._meta_action_embed_net = tf.make_template(
                    self.META_ACTION_EMBED_NET_SCOPE, meta_action_embed_net,
                    num_output_dims=self.goal_dims,
                    create_scope_now_=True)

                ob_space = copy(ob_space)
                ob_space.dtype = np.float
                self.begin_high_observations = observation_placeholder(ob_space)
                self.end_high_observations = observation_placeholder(ob_space)
                self.high_observations = observation_placeholder(ob_space)
                self.next_high_observations = observation_placeholder(ob_space)
                self.meta_actions = observation_placeholder(subgoal_space)
                self.low_all_actions = tf.placeholder(tf.float32, [None, meta_action_every_n] + list(act_space.shape))
                self.low_all_high_observations = tf.placeholder(tf.float32,
                                                                [None, meta_action_every_n] + list(ob_space.shape))
                self.actions = tf.placeholder(tf.float32, [None] + list(act_space.shape))
                self.discounts = tf.placeholder(tf.float32, shape=(None,))

                with tf.variable_scope('embed_state'):
                    self.embed_begin_states = self._state_preprocess_net(
                        tf.layers.flatten(self.begin_high_observations))
                    self.embed_end_states = self._state_preprocess_net(tf.layers.flatten(self.end_high_observations))
                    self.embed_states = self._state_preprocess_net(tf.layers.flatten(self.high_observations))
                    self.embed_next_states = self._state_preprocess_net(tf.layers.flatten(self.next_high_observations))

                sampled_states = tf.get_variable('sampled_states',
                                                 [self.sampling_size, ob_space.shape[0]],
                                                 collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                                 initializer=tf.zeros_initializer())

                sampled_embed_displacement = tf.get_variable('sampled_embed_displacement',
                                                             [self.sampling_size,
                                                              self.goal_dims],
                                                             collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                                             initializer=tf.ones_initializer())

                with tf.variable_scope('translated_goal'):
                    self.goal_states = self._meta_action_embed_net(self.meta_actions, sampled_embed_displacement)

                with tf.variable_scope('embed_action'):
                    self.embed_all_action = tf.add_n(
                        [self._action_embed_net(tf.layers.flatten(self.low_all_actions[:, i, :]),
                                                states=tf.layers.flatten(self.embed_begin_states))
                         for i in range(meta_action_every_n)])

                with tf.variable_scope('inverse_goal'):
                    self.inverse_goal = inverse_goal = self.embed_begin_states + self.embed_all_action

                self.t = tf.Variable(
                    tf.zeros(shape=(), dtype=tf.int64), name='num_timer_steps')
                self.t2 = tf.Variable(
                    tf.zeros(shape=(), dtype=tf.int64), name='num_timer_steps')

                upd2 = sampled_embed_displacement.assign(
                    tf.concat([sampled_embed_displacement, self.embed_next_states - self.embed_begin_states], axis=0)[
                    -self.sampling_size:])
                upd3 = sampled_states.assign(
                    tf.concat([sampled_states, self.next_high_observations], axis=0)[-self.sampling_size:])
                self.upd = upd = tf.group([upd2, upd3])

                with tf.control_dependencies([
                    upd,
                    tf.assign_add(self.t2, 1),
                    tf.cond(tf.equal(tf.mod(self.t2, 128 * 4), 0),
                            lambda: tf.print(
                                '\nrewards:', -distance(self.embed_next_states, self.goal_states)[0],
                                '\ndiff_states:', self.embed_end_states[0] - self.embed_begin_states[0],
                                '\ngoal(diff)  :', self.goal_states[0],
                                tf.reduce_mean(
                                    tf.squared_difference(self.embed_end_states[0] - self.embed_begin_states[0],
                                                          self.goal_states)),
                                '\nbegin:', self.embed_begin_states[0],
                                '\nend:', self.embed_end_states[0],
                                '\ndisplacement:', tf.reduce_max(tf.abs(sampled_embed_displacement), axis=0),
                                # '\naa:', aa,
                            ),
                            lambda: tf.constant(False, dtype=tf.bool))
                ]):
                    with tf.variable_scope('low_rewards'):
                        # original rewards
                        self._low_rewards = -distance(self.embed_next_states, self.goal_states)

                with tf.variable_scope('loss'):
                    embed_sampled_states = self._state_preprocess_net(tf.layers.flatten(sampled_states))

                    with tf.variable_scope('estimated_log_partition'):
                        self.estimated_log_partition = \
                            tf.reduce_logsumexp(
                                - distance(embed_sampled_states[None, :, :],
                                           inverse_goal[:, None, :]),
                                axis=-1) \
                            - np.log(self.sampling_size)

                    # original implementation
                    # self.loss_attractive = distance(inverse_goal, self.embed_next_states)
                    # self.loss_repulsive = tf.exp(
                    #     # - distance(self.embed_next_states, inverse_goal) \
                    #     # - tf.stop_gradient(self.estimated_log_partition)
                    #     tf.clip_by_value(
                    #         - distance(self.embed_next_states, inverse_goal) \
                    #         - tf.stop_gradient(self.estimated_log_partition),
                    #         -10, 10
                    #     )
                    # )

                    repulsive = tf.equal(sampled_states[None, :, :], self.next_high_observations[:, None, :])
                    repulsive = tf.reduce_all(repulsive, axis=2)
                    repulsive = tf.to_float(repulsive)

                    # eq 13
                    self.loss_attractive = distance(inverse_goal, self.embed_next_states)
                    self.loss_repulsive = \
                        tf.reduce_mean(
                            1e-8 +
                            (1 - repulsive) * \
                            tf.exp(
                                - distance(embed_sampled_states[None, :, :], inverse_goal[:, None, :]) \
                                - tf.stop_gradient(self.estimated_log_partition)[:, None]),
                            axis=1)

                    diff = self.embed_end_states - self.embed_all_action
                    # self.meta_loss = \
                    #     tf.maximum(tf.maximum(diff, 0) - tf.maximum(self.goal_states, 0), 0) - \
                    #     tf.minimum(tf.minimum(diff, 0) - tf.minimum(self.goal_states, 0), 0)
                    #
                    # self.meta_loss = tf.reduce_mean(self.meta_loss)

                    self.meta_loss = tf.squared_difference(tf.reduce_mean(tf.norm(diff, axis=-1)), tf.reduce_mean(tf.norm(self.goal_states, axis=-1)))

                    self.representation_loss = tf.reduce_mean(
                        tf.squared_difference(inverse_goal, self.embed_next_states)) + self.meta_loss

            with tf.variable_scope('optimizer'):
                self.LR = LR = tf.placeholder(tf.float32, [], name='learning_rate')

                # UPDATE THE PARAMETERS USING LOSS
                # 1. Get the model parameters
                params = tf.trainable_variables(self.scope.name)

                # 2. Build our trainer
                if MPI is not None:
                    self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
                else:
                    self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

                # 3. Calculate the gradients
                grads_and_var = self.trainer.compute_gradients(self.representation_loss, params)
                grads, var = zip(*grads_and_var)

                if max_grad_norm is not None:
                    # Clip the gradients (normalize)
                    grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                grads_and_var = list(zip(grads, var))

                self.grads = grads
                self.var = var
                self._train_op = self.trainer.apply_gradients(grads_and_var)
                self.stats_names = ['loss_total',
                                    'loss_meta',
                                    'goal_size',
                                    'embed_size',
                                    ]
                self.stats_list = [self.representation_loss,
                                   self.meta_loss,
                                   tf.reduce_mean(tf.abs(self.goal_states)),
                                   tf.reduce_mean(tf.abs(self.embed_states)),
                                   ]

            with tf.variable_scope('initialization'):
                self.sess.run(tf.initializers.variables(tf.global_variables(self.scope.name)))
                self.sess.run(tf.initializers.variables(tf.local_variables(self.scope.name)))
                global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)
                if MPI is not None:
                    sync_from_root(sess, global_variables)  # pylint: disable=E1101

    def embedded_state(self, high_observations):
        return self.sess.run(self.embed_states, {self.high_observations: high_observations, })

    def low_rewards(self, begin_high_observations, high_observations, next_high_observations, end_high_observations,
                    meta_actions, discounts, low_all_actions):

        return self.sess.run(self._low_rewards, {
            self.begin_high_observations: begin_high_observations,
            self.high_observations: high_observations,
            self.next_high_observations: next_high_observations,
            self.meta_actions: meta_actions,
            self.discounts: discounts,
            self.end_high_observations: end_high_observations,
            self.low_all_actions: low_all_actions,
        })

    def update_states(self, next_high_observations):
        pass
        # return self.sess.run(self.upd, {
        #     self.next_high_observations: next_high_observations,
        # })

    def update_displacement(self, begin_high_observations, end_high_observations):
        pass
        # return self.sess.run(self.upd_displacement, {
        #     self.begin_high_observations: begin_high_observations,
        #     self.end_high_observations: end_high_observations,
        # })

    def get_goal_states(self, meta_actions):
        return self.sess.run(self.goal_states, {
            self.meta_actions: meta_actions,
        })

    def train(self,
              lr,
              begin_high_observations,
              end_high_observations,
              high_observations,
              next_high_observations,
              actions,
              low_all_actions,
              discounts,
              meta_actions,
              low_all_high_observations,
              **_kwargs):
        td_map = {
            self.LR: lr,
            self.begin_high_observations: begin_high_observations,
            self.end_high_observations: end_high_observations,
            self.high_observations: high_observations,
            self.next_high_observations: next_high_observations,
            self.actions: actions,
            self.low_all_actions: low_all_actions,
            self.discounts: discounts,
            self.meta_actions: meta_actions,
            self.low_all_high_observations: low_all_high_observations,
        }

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]


def state_preprocess_net(
    states,
    num_output_dims=2,
    states_hidden_layers=(64, 64),
    normalizer_fn=None,
    activation_fn=tf.nn.relu,
    zero_time=True,
    images=False):
    """Creates a simple feed forward net for embedding states.
    """
    states_shape = tf.shape(states)
    states_dtype = states.dtype
    states = tf.to_float(states)
    orig_states = states
    if images:  # Zero-out x-y
        states *= tf.constant([0.] * 2 + [1.] * (states.shape[-1] - 2), dtype=states.dtype)

    # if zero_time:
    #     states *= tf.constant([1.] * (states.shape[-1] - 1) + [0.], dtype=states.dtype)

    embed = tf.layers.dense(states, 64, activation_fn,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                factor=1.0 / 3.0, mode='FAN_IN', uniform=True))
    embed = tf.layers.dense(embed, 64, activation_fn, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
        factor=1.0 / 3.0, mode='FAN_IN', uniform=True))
    embed = tf.layers.dense(embed, num_output_dims, None, kernel_initializer=tf.random_uniform_initializer(
        minval=-0.003, maxval=0.003))
    output = embed
    output = tf.cast(output, states_dtype)
    return output


def action_embed_net(
    actions,
    states=None,
    num_output_dims=2,
    hidden_layers=(64, 64),
    normalizer_fn=None,
    activation_fn=tf.nn.relu,
    zero_time=True,
    images=False):
    """Creates a simple feed forward net for embedding actions.
    """

    actions = tf.to_float(actions)
    if states is not None:
        if images:  # Zero-out x-y
            states *= tf.constant([0.] * 2 + [1.] * (states.shape[-1] - 2), dtype=tf.float32)
        if zero_time:
            states *= tf.constant([1.] * (states.shape[-1] - 1) + [0.], dtype=tf.float32)

        if not states.dtype.is_floating:
            states = tf.to_float(states)
        if states.shape.ndims != 2:
            states = tf.layers.flatten(states)
        actions = tf.concat([actions, states], -1)

    embed = actions
    embed = tf.layers.dense(embed, 64, activation_fn, True,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                factor=1.0 / 3.0, mode='FAN_IN', uniform=True))
    embed = tf.layers.dense(embed, 64, activation_fn, True,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                factor=1.0 / 3.0, mode='FAN_IN', uniform=True))
    embed = tf.layers.dense(embed, num_output_dims, None, True, kernel_initializer=tf.random_uniform_initializer(
        minval=-0.003, maxval=0.003))
    return embed


def meta_action_embed_net(
    meta_actions,
    sampled_embed_displacement,
    num_output_dims=2,
    hidden_layers=None,
    normalizer_fn=None,
    activation_fn=tf.nn.leaky_relu):
    """Creates a simple feed forward net for embedding actions.
    """
    #
    # displacement = tf.math.reduce_max(tf.abs(sampled_embed_displacement), axis=0)
    # embed = meta_actions * displacement

    # embed = meta_actions
    # embed = tf.layers.dense(embed, 64, activation_fn, True,
    #                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
    #                             factor=1.0 / 3.0, mode='FAN_IN', uniform=True))
    # embed = tf.layers.dense(embed, 64, activation_fn, True,
    #                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
    #                             factor=1.0 / 3.0, mode='FAN_IN', uniform=True))
    # embed = tf.layers.dense(embed, num_output_dims, None, True, kernel_initializer=tf.random_uniform_initializer(
    #     minval=-0.003, maxval=0.003))

    aa = tf.get_variable('sampled_states',
                         (),
                         initializer=tf.ones_initializer())
    embed = meta_actions * tf.abs(aa)

    return meta_actions


def distance(a, b, tau=1, delta=.1, ):
    return tau * tf.reduce_sum(huber(a - b, delta=delta), -1)


def huber(x, delta=0.1):
    return ((0.5 * tf.square(x)) * tf.to_float(tf.abs(x) <= delta) + \
            (delta * (tf.abs(x) - 0.5 * delta)) * tf.to_float(tf.abs(x) > delta)) / delta
