import tensorflow as tf
import numpy as np

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
                    num_output_dims=4,
                    create_scope_now_=True)
                self._action_embed_net = tf.make_template(
                    self.ACTION_EMBED_NET_SCOPE, action_embed_net,
                    num_output_dims=4,
                    create_scope_now_=True)

                self.states = observation_placeholder(ob_space)
                self.next_states = observation_placeholder(ob_space)
                self.low_states = observation_placeholder(subgoal_space)
                self.goal_states = observation_placeholder(subgoal_space)
                self.low_actions = observation_placeholder(act_space)
                flatten_state = tf.layers.flatten(self.states)

                with tf.variable_scope('embed_state'):
                    self.embed_state = embed_state = tf.to_float(self._state_preprocess_net(flatten_state))
                    self.embed_next_state = embed_next_state = tf.to_float(
                        self._state_preprocess_net(tf.layers.flatten(self.next_states)))
                with tf.variable_scope('embed_action'):
                    self.action_embed = action_embed = self._action_embed_net(tf.layers.flatten(self.low_actions),
                                                                              states=tf.to_float(flatten_state))
                with tf.variable_scope('inverse_goal'):
                    self.inverse_goal = inverse_goal = embed_state + action_embed

                tau = 2.0

                def distance(a, b):
                    return tau * tf.reduce_sum(huber(a - b), -1)

                sampled_embedded_states = tf.get_variable('sampled_embedded_states',
                                                          [self.sampling_size, self.goal_dims],
                                                          collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                                          initializer=tf.zeros_initializer())
                upd = sampled_embedded_states.assign(
                    tf.concat([sampled_embedded_states, embed_next_state], axis=0)[-self.sampling_size:])

                with tf.variable_scope('estimated_log_partition'):
                    self.estimated_log_partition = estimated_log_partition = \
                        tf.reduce_logsumexp(
                            -distance(inverse_goal[:, None, :], sampled_embedded_states[None, :, :]), axis=-1) \
                        - np.log(self.sampling_size)

                # with tf.variable_scope('representation_log_probability'):
                #     repr_log_probs = tf.stop_gradient(
                #         -distance(inverse_goal, embed_next_state) - estimated_log_partition) / tau
                #     self.repr_log_probs = repr_log_probs

                with tf.control_dependencies([upd]):
                    with tf.variable_scope('low_rewards'):
                        self._low_rewards = \
                            -distance(embed_next_state, self.goal_states) + \
                            distance(embed_next_state, inverse_goal) + estimated_log_partition

            with tf.variable_scope('loss'):
                with tf.variable_scope('representation_loss'):
                    attractive = tf.reduce_mean(distance(embed_next_state, inverse_goal))
                    repulsive = tf.reduce_mean(
                        tf.reduce_mean(tf.exp(distance(inverse_goal[:, None, :], sampled_embedded_states[None, :, :])),
                                       axis=1) - tf.stop_gradient(estimated_log_partition))
                    self.representation_loss = (attractive + repulsive + 1e-8) ** -1
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

            with tf.variable_scope('initialization'):
                self.sess.run(tf.initializers.variables(tf.global_variables(self.scope.name)))
                self.sess.run(tf.initializers.variables(tf.local_variables(self.scope.name)))
                global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)
                if MPI is not None:
                    sync_from_root(sess, global_variables)  # pylint: disable=E1101

    def embedded_state(self, states):
        return self.sess.run(self.embed_state, {self.states: states})

    def low_rewards(self, states, next_states, low_actions, goal_states):
        return self.sess.run(self._low_rewards, {self.states: states,
                                                 self.next_states: next_states,
                                                 self.low_actions: low_actions,
                                                 self.goal_states: goal_states})

    def train(self,
              lr,
              high_observations,
              next_high_observations,
              actions,
              **_kwargs):
        td_map = {
            self.LR: lr,
            self.states: high_observations,
            self.next_states: next_high_observations,
            self.low_actions: actions
        }

        return self.sess.run(
            [self._train_op],
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
    states_hidden_layers=(100,),
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
            factor=1.0 / 3.0, mode='FAN_IN', uniform=True)):

        states_shape = tf.shape(states)
        states_dtype = states.dtype
        states = tf.to_float(states)
        orig_states = states
        if images:  # Zero-out x-y
            states *= tf.constant([0.] * 2 + [1.] * (states.shape[-1] - 2), dtype=states.dtype)
        if zero_time:
            states *= tf.constant([1.] * (states.shape[-1] - 1) + [0.], dtype=states.dtype)
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
    hidden_layers=(400, 300),
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
            factor=1.0 / 3.0, mode='FAN_IN', uniform=True)):

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


def huber(x, kappa=0.1):
    return (0.5 * tf.square(x) * tf.to_float(tf.abs(x) <= kappa) +
            kappa * (tf.abs(x) - 0.5 * kappa) * tf.to_float(tf.abs(x) > kappa)
            ) / kappa
