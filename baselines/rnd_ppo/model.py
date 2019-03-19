import functools

import tensorflow as tf

from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize
from baselines.rnd_ppo.RND import RND

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None


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

    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm, rnd_update_rate,
                 int_coef, ext_coef,
                 rnd_network=None,
                 microbatch_size=None):
        self.sess = sess = get_session()

        with tf.variable_scope('rnd_ppo_model', reuse=tf.AUTO_REUSE):
            with tf.name_scope('act_model'):
                # CREATE OUR TWO MODELS
                # act_model that is used for sampling
                act_model = policy(nbatch_act, 1, sess)

            with tf.name_scope('train_model'):
                # Train model for training
                if microbatch_size is None:
                    train_model = policy(nbatch_train, nsteps, sess)
                else:
                    train_model = policy(microbatch_size, nsteps, sess)

                rnd_model = RND(list(ob_space.shape),
                                rnd_update_rate,
                                rnd_network=rnd_network)

            # CREATE THE PLACEHOLDERS
            self.A = A = train_model.pdtype.sample_placeholder([None], name='action')
            self.ADV = ADV = tf.placeholder(tf.float32, [None], name='advantage')
            self.INT_R = INT_R = tf.placeholder(tf.float32, [None], name='intrinsic_reward')
            self.EXT_R = EXT_R = tf.placeholder(tf.float32, [None], name='extrinsic_reward')
            # self.R = R = tf.placeholder(tf.float32, [None])
            # Keep track of old actor
            self.OLDV_EXT = OLDV_EXT = tf.placeholder(tf.float32, [None], name='extrinsic_value_old')
            self.OLDV_INT = OLDV_INT = tf.placeholder(tf.float32, [None], name='intrinsic_value_old')

            self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name='negative_log_p_action_old')

            self.LR = LR = tf.placeholder(tf.float32, [], name='learning_rate')
            # Cliprange
            self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [], name='clip_range')

            with tf.name_scope("neglogpac"):
                neglogpac = train_model.pd.neglogp(A)

            with tf.name_scope("entropy"):
                # Calculate the entropy
                # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
                entropy = tf.reduce_mean(train_model.pd.entropy())
                entropy_loss = (- ent_coef) * entropy

            with tf.name_scope("value_loss"):
                # CALCULATE THE LOSS
                value = train_model.vf_int + train_model.vf_ext
                value_prev = (OLDV_INT + OLDV_EXT)
                value_clipped = value_prev + tf.clip_by_value(value - value_prev, -CLIPRANGE, CLIPRANGE)
                vf_losses1 = tf.square(value - (INT_R + EXT_R))
                vf_losses2 = tf.square(value_clipped - (INT_R + EXT_R))
                vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            with tf.name_scope("policy_loss"):
                # Calculate ratio (pi current policy / pi old policy)
                ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
                pg_losses = -ADV * ratio
                pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

            with tf.name_scope("approxkl"):
                approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))

            with tf.name_scope("clip_fraction"):
                clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

            with tf.name_scope("total_loss"):
                # Total loss
                loss = pg_loss + entropy_loss + vf_loss + rnd_model.rnd_loss

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('rnd_ppo_model')

        # 2. Build our trainer
        if MPI is not None:
            self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)

        self.loss_names = ['policy_loss', 'value_loss', 'rnd_loss', 'entropy_loss', 'approxkl', 'clipfrac',
                           'int_return',
                           'ext_return',
                           'total_loss']
        self.stats_list = [pg_loss, vf_loss, rnd_model.rnd_loss, entropy_loss, approxkl, clipfrac,
                           tf.reduce_mean(INT_R),
                           tf.reduce_mean(EXT_R),
                           loss]

        self.train_model = train_model
        self.act_model = act_model
        self.rnd_model = rnd_model
        self.reward_rms = RunningMeanStd()

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables)  # pylint: disable=E1101

    def get_initial_state(self):
        return self.act_model.get_initial_state()

    def step(self, *args, **kwargs):
        return self.act_model.step(*args, **kwargs)

    def values(self, *args, **kwargs):
        return self.act_model.step(*args, **kwargs)

    def train(self,
              lr,
              cliprange,
              obs,
              advs,
              int_returns,
              ext_returns,
              actions,
              int_values,
              ext_values,
              returns,
              neglogpacs,
              ext_advs,
              int_advs,
              states=None, **_kwargs):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')

        # Normalize the advantages
        # advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X: obs,
            self.A: actions,
            self.ADV: advs,
            # self.R: returns,
            self.INT_R: int_returns,
            self.EXT_R: ext_returns,
            self.LR: lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC: neglogpacs,
            self.OLDV_EXT: ext_values,
            self.OLDV_INT: int_values,
            self.rnd_model.OBS: obs,
            self.rnd_model.RND_OBS_MEAN: self.rnd_model.rnd_ob_rms.mean,
            self.rnd_model.RND_OBS_STD: self.rnd_model.rnd_ob_rms.var ** 0.5,
            # The standard deviation is the square root of the variance.
        }

        td_map.update(self.train_model.feed_dict(**_kwargs))

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]
