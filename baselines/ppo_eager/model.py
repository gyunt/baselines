from collections import OrderedDict

import numpy as np
import tensorflow as tf
from baselines.ppo_eager.distributions import make_pdtype


# try:
#     from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
#     from mpi4py import MPI
#     from baselines.common.mpi_util import sync_from_root
# except ImportError:
#     MPI = None


class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    """
    Encapsulates fields and methods for RL policy and two value function estimation with shared parameters
    """

    def __init__(self, ob_space, ac_space, ent_coef, vf_coef, estimate_q=False, max_grad_norm=None):
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
        self.pdtype = make_pdtype(ac_space)
        self.estimate_q = estimate_q
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.initial_state = None

        observation = tf.keras.Input(shape=ob_space.shape, name='unit_feature', dtype=tf.float32)

        x = tf.keras.layers.Flatten()(observation)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        latent = tf.keras.layers.Dense(64, activation='relu', name='latent')(x)

        if estimate_q:
            q = tf.keras.layers.Dense(ac_space.n, activation='relu', name='value')(latent)
            value = q
        else:
            value = tf.keras.layers.Dense(1, activation='relu', name='value')(latent)
            value = value[:, 0]

        self.pd, self.pdparam = self.pdtype.pdfromlatent(latent, init_scale=0.01)
        action = self.pd.sample()
        neglogp = self.pd.neglogp(action)

        sampled_actions = tf.keras.Input(shape=action.shape, name='sampled_actions', dtype=tf.float32)

        self.model = tf.keras.Model(inputs=[observation],
                                    outputs={'actions': action,
                                     'values': value,
                                     'neglogpacs': neglogp,
                                     }, name='model')
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def compute_loss(self,
                     lr,
                     cliprange,
                     observations,
                     advs,
                     returns,
                     actions,
                     values,
                     neglogpacs,
                     **_kwargs):
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        outputs = self.model(observations.astype(np.float32))
        action_new = outputs['actions']
        value_new = outputs['values']
        neglopacs_new = outputs['neglogpacs']

        with tf.name_scope("entropy"):
            # Calculate the entropy
            # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
            entropy = tf.reduce_mean(self.pd.entropy())
            entropy_loss = (- self.ent_coef) * entropy

        with tf.name_scope("value_loss"):
            # CALCULATE THE LOSS
            value_clipped = values + tf.clip_by_value(value_new - values, -cliprange, cliprange)
            vf_losses1 = tf.keras.metrics.mean_squared_error(value_new, returns)
            vf_losses2 = tf.keras.metrics.mean_squared_error(value_clipped, returns)
            vf_loss = 0.5 * self.vf_coef * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        with tf.name_scope("policy_loss"):
            # Calculate ratio (pi current policy / pi old policy)
            ratio = tf.exp(neglogpacs - neglopacs_new)
            pg_losses = -advs * ratio
            pg_losses2 = -advs * tf.clip_by_value(ratio, 1.0 - cliprange, 1.0 + cliprange)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        with tf.name_scope("approxkl"):
            approxkl = .5 * tf.reduce_mean(tf.keras.metrics.mean_squared_error(neglopacs_new, neglogpacs))

        with tf.name_scope("clip_fraction"):
            clipfrac = tf.reduce_mean(tf.keras.backend.cast(tf.greater(tf.abs(ratio - 1.0), cliprange), tf.float32))

        with tf.name_scope("total_loss"):
            loss = pg_loss + entropy_loss + vf_loss

        return OrderedDict({'total_loss': loss,
                            'entropy_loss': entropy_loss,
                            'value_loss': vf_loss,
                            'policy_loss': pg_loss,
                            })

    def compute_gradients(self, **kwargs):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(**kwargs)
            return tape.gradient(loss['total_loss'], self.model.trainable_variables), loss

    def apply_gradients(self, gradients, variables):
        self.optimizer.apply_gradients(zip(gradients, variables))

    def step_with_dict(self, observations, dones=None, **_kwargs):
        outputs = self.model(observations)
        outputs = {key: outputs[key].numpy() for key in outputs}
        return outputs

    def step(self, obs, M=None, S=None, **kwargs):
        kwargs.update({'observations': obs})
        if M is not None and S is not None:
            kwargs.update({'dones': M})
            kwargs.update({'states': S})
        transition = self.step_with_dict(**kwargs)
        states = transition['next_states'] if 'next_states' in transition else None
        return transition['actions'], transition['values'], states, transition['neglogpacs']

    def train(self,
              lr,
              cliprange,
              observations,
              advs,
              returns,
              actions,
              values,
              neglogpacs,
              **_kwargs):
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        gradients, loss = self.compute_gradients(
            lr=lr,
            cliprange=cliprange,
            observations=observations,
            advs=advs,
            returns=returns,
            actions=actions,
            values=values,
            neglogpacs=neglogpacs, )

        self.apply_gradients(gradients, self.model.trainable_variables)
        return {key: np.array(loss[key]) for key in loss}


def print_graph():
    tf.summary.trace_on(graph=True, profiler=False)
    # Call only one tf.function when tracing.
    logdir = 'i:/tmp/eager/'
    writer = tf.summary.create_file_writer(logdir)
    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir)
