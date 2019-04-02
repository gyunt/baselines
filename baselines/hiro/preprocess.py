from copy import copy

import numpy as np
import tensorflow as tf
from baselines.common.input import observation_placeholder

slim = tf.contrib.slim

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None


class StatePreprocess(object):
    STATE_PREPROCESS_NET_SCOPE = 'state_process_net'
    ACTION_EMBED_NET_SCOPE = 'action_embed_net'

    def __init__(self,
                 ob_space,
                 subgoal_space,
                 act_space,
                 meta_action_every_n,
                 state_preprocess_net=lambda states: states,
                 action_embed_net=lambda actions, *args, **kwargs: actions,
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
                self._state_preprocess_net = tf.make_template(
                    self.STATE_PREPROCESS_NET_SCOPE, state_preprocess_net,
                    num_output_dims=self.goal_dims,
                    create_scope_now_=True)
                self._action_embed_net = tf.make_template(
                    self.ACTION_EMBED_NET_SCOPE, action_embed_net,
                    num_output_dims=self.goal_dims,
                    create_scope_now_=True)

                ob_space = copy(ob_space)
                ob_space.dtype = np.float
                self.begin_high_observations = observation_placeholder(ob_space)
                self.end_high_observations = observation_placeholder(ob_space)
                self.high_observations = observation_placeholder(ob_space)
                self.next_high_observations = observation_placeholder(ob_space)
                self.goal_states = observation_placeholder(subgoal_space)
                self.low_all_actions = tf.placeholder(tf.float32, [None, meta_action_every_n] + list(act_space.shape))
                self.discounts = tf.placeholder(tf.float32, shape=(None,))

                with tf.variable_scope('embed_state'):
                    self.embed_begin_states = self._state_preprocess_net(
                        tf.layers.flatten(self.begin_high_observations))
                    self.embed_end_states = self._state_preprocess_net(tf.layers.flatten(self.end_high_observations))
                    self.embed_states = self._state_preprocess_net(tf.layers.flatten(self.high_observations))
                    self.embed_next_states = self._state_preprocess_net(tf.layers.flatten(self.next_high_observations))

                with tf.variable_scope('embed_action'):
                    flatten_begin_high_observations = tf.layers.flatten(self.begin_high_observations)
                    self.embed_all_action = tf.add_n(
                        [self._action_embed_net(tf.layers.flatten(self.low_all_actions[:, i, :]),
                                                states=flatten_begin_high_observations)
                         for i in range(meta_action_every_n)])

                with tf.variable_scope('inverse_goal'):
                    self.inverse_goal = inverse_goal = self.embed_begin_states + self.embed_all_action

                sampled_embedded_states = tf.get_variable('sampled_embedded_states',
                                                          [self.sampling_size, self.goal_dims],
                                                          collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                                          initializer=tf.zeros_initializer())

                self.t = tf.Variable(
                    tf.zeros(shape=(), dtype=tf.int64), name='num_timer_steps')
                self.t2 = tf.Variable(
                    tf.zeros(shape=(), dtype=tf.int64), name='num_timer_steps')

                upd = sampled_embedded_states.assign(
                    tf.concat([sampled_embedded_states, self.embed_next_states], axis=0)[-self.sampling_size:])

                with tf.variable_scope('estimated_log_partition'):
                    self.estimated_log_partition = \
                        tf.reduce_logsumexp(
                            - distance(tf.stop_gradient(sampled_embedded_states[None, :, :]),
                                       inverse_goal[:, None, :]),
                            axis=-1) \
                        - np.log(self.sampling_size)

                with tf.control_dependencies([
                    upd,
                    # tf.assign_add(self.t2, 1),
                    # tf.cond(tf.equal(tf.mod(self.t2, 128 * 4), 0),
                    #         lambda: tf.print('\ndifference:', (distance(self.embed_states, self.goal_states) \
                    #                                           - distance(self.embed_next_states, self.goal_states))[0],
                    #                          '\ncurrent_state:', distance(self.embed_states, self.goal_states)[0],
                    #                          '\nnext_state:', distance(self.embed_next_states, self.goal_states)[0],
                    #                          '\nembed_states:', self.embed_states[0],
                    #                          '\nembed_next_states:', self.embed_next_states[0],
                    #                          '\ngoal_sates:', self.goal_states[0],
                    #                          ),
                    #         lambda: tf.constant(False, dtype=tf.bool))
                ]):
                    with tf.variable_scope('low_rewards'):
                        # original rewards
                        # self._low_rewards = -distance(self.embed_next_states, self.goal_states)

                        # modified rewards
                        self._low_rewards = distance(self.embed_states, self.goal_states) \
                                            - distance(self.embed_next_states, self.goal_states)

                with tf.variable_scope('loss'):
                    self.prior_log_probs = tf.reduce_mean(
                        self.estimated_log_partition + distance(self.embed_next_states, inverse_goal))

                    # original implementation
                    self.loss_attractive = distance(inverse_goal, self.embed_next_states)
                    self.loss_repulsive = tf.exp(
                        tf.clip_by_value(
                            - distance(self.embed_next_states, inverse_goal) \
                            - tf.stop_gradient(self.estimated_log_partition),
                            -7, 0
                        )
                    )

                    loss = self.loss_attractive + self.loss_repulsive
                    discount_loss = tf.reduce_mean(
                        self.discounts * (self.loss_attractive + self.loss_repulsive))
                    self.representation_loss = tf.reduce_mean(loss)

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
                # with tf.control_dependencies([
                #     tf.assign_add(self.t, 1),
                #     tf.cond(tf.equal(tf.mod(self.t, 32), 0),
                #             lambda: tf.print('\ndiff   :',
                #                              tf.nn.moments(self.goal_states[0] - self.embed_states[0], axes=0),
                #                              self.goal_states[0] - self.embed_states[0],
                #                              '\nembed  :', tf.nn.moments(self.embed_states[0], axes=0),
                #                              self.embed_states[0],
                #                              '\ngoal   :', tf.nn.moments(self.goal_states[0], axes=0),
                #                              self.goal_states[0],
                #                              # sampled_embedded_states[-10:]
                #                              ),
                #             lambda: tf.constant(False, dtype=tf.bool))
                # ]):
                self._train_op = self.trainer.apply_gradients(grads_and_var)
                self.stats_names = ['loss_total',
                                    'loss_attractive',
                                    'loss_repulsive',
                                    'dst_goal',
                                    'goal_size',
                                    'embed_size',
                                    ]
                self.stats_list = [self.representation_loss,
                                   tf.reduce_mean(self.loss_attractive),
                                   tf.reduce_mean(self.loss_repulsive),
                                   tf.reduce_mean(distance(self.embed_next_states, self.goal_states)),
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

    def low_rewards(self, begin_high_observations, high_observations, next_high_observations,
                    goal_states,
                    discounts, **_kwargs):
        return self.sess.run(self._low_rewards, {
            self.begin_high_observations: begin_high_observations,
            self.high_observations: high_observations,
            self.next_high_observations: next_high_observations,
            self.goal_states: goal_states,
            self.discounts: discounts,
        })

    def train(self,
              lr,
              begin_high_observations,
              end_high_observations,
              high_observations,
              next_high_observations,
              low_all_actions,
              discounts,
              goal_states,
              **_kwargs):
        td_map = {
            self.LR: lr,
            self.begin_high_observations: begin_high_observations,
            self.end_high_observations: end_high_observations,
            self.high_observations: high_observations,
            self.next_high_observations: next_high_observations,
            self.low_all_actions: low_all_actions,
            self.discounts: discounts,
            self.goal_states: goal_states,
        }

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]


def join_scope(parent_scope, child_scope):
    """Joins a parent and child scope using `/`, checking for empty/none.

    Args:
      parent_scope: (string) parent/prefix scope.
      child_scope: (string) child/suffix scope.
    Returns:
      joined scope: (string) parent and child scopes joined by /.
    """
    if not parent_scope:
        return child_scope
    if not child_scope:
        return parent_scope
    return '/'.join([parent_scope, child_scope])


def state_preprocess_net(
    states,
    num_output_dims=2,
    states_hidden_layers=(128, 128),
    normalizer_fn=None,
    activation_fn=tf.nn.relu,
    zero_time=True,
    images=False):
    """Creates a simple feed forward net for embedding states.
    """
    with slim.arg_scope(
        [slim.fully_connected],
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        weights_initializer=slim.variance_scaling_initializer(
            factor=1.0 / 3.0, mode='FAN_IN', uniform=True),
    ):

        states_shape = tf.shape(states)
        states_dtype = states.dtype
        states = tf.to_float(states)
        orig_states = states
        if images:  # Zero-out x-y
            states *= tf.constant([0.] * 2 + [1.] * (states.shape[-1] - 2), dtype=states.dtype)
        # if zero_time:
        #     states *= tf.constant([1.] * (states.shape[-1] - 1) + [0.], dtype=states.dtype)
        embed = states
        if states_hidden_layers:
            embed = slim.stack(embed, slim.fully_connected, states_hidden_layers,
                               scope='states')

        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=None,
                            weights_initializer=tf.random_uniform_initializer(
                                minval=-0.003, maxval=0.003)):
            embed = slim.fully_connected(embed, num_output_dims,
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         scope='value')

        output = embed
        output = tf.cast(output, states_dtype)
        return output


def action_embed_net(
    actions,
    states=None,
    num_output_dims=2,
    hidden_layers=(128, 128),
    normalizer_fn=None,
    activation_fn=tf.nn.relu,
    zero_time=True,
    images=False):
    """Creates a simple feed forward net for embedding actions.
    """
    with slim.arg_scope(
        [slim.fully_connected],
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        weights_initializer=slim.variance_scaling_initializer(
            factor=1.0 / 3.0, mode='FAN_IN', uniform=True),
        # biases_initializer=tf.initializers.constant(1)
    ):

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
        if hidden_layers:
            embed = slim.stack(embed, slim.fully_connected, hidden_layers,
                               scope='hidden')

        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=None,
                            weights_initializer=tf.random_uniform_initializer(
                                minval=-0.003, maxval=0.003)):
            embed = slim.fully_connected(embed, num_output_dims,
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         scope='value')
            if num_output_dims == 1:
                return embed[:, 0, ...]
            else:
                return embed


def distance(a, b, tau=2, delta=0.1, ):
    return tau * tf.reduce_mean(huber(a - b, delta=delta), -1)


def huber(x, delta=0.1):
    return ((0.5 * tf.square(x)) * tf.to_float(tf.abs(x) <= delta) + \
            (delta * (tf.abs(x) - 0.5 * delta)) * tf.to_float(tf.abs(x) > delta)) / delta
