import tensorflow as tf
from baselines.common.models import get_network_builder
from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.tf_util import get_session


# Helpers functions
def conv2d(inputs, filters, kernel_size, strides, activation):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            activation=activation)


def fc(inputs, units, activation):
    return tf.layers.dense(inputs=inputs,
                           units=units,
                           activation=activation)


class RND():
    def __init__(self, obs_shape, rnd_update_rate, rnd_network='mlp', normalize_obs=True,
                 scope='RND',
                 layer_size=128):
        self.sess = sess = get_session()
        self.rnd_update_rate = rnd_update_rate
        obs_batch_shape = [None] + obs_shape

        with tf.variable_scope(scope):
            # CREATE THE PLACEHOLDERS
            # Remember that we pass only one frame, not a stack of frame hence ([None, 84, 84, 1])
            self.OBS = tf.placeholder(tf.float32, shape=obs_batch_shape, name="OBS")

            # These two are for the observation normalization (mean and std)
            self.RND_OBS_MEAN = tf.placeholder(tf.float32, obs_shape)
            self.RND_OBS_STD = tf.placeholder(tf.float32, obs_shape)

            # Build our RunningMeanStd object for observation normalization
            self.rnd_ob_rms = RunningMeanStd(shape=obs_shape)

            # Build our RunningMeanStd for intrinsic reward normalization
            # (mandatory since IR are non-stationary rewards)
            self.rnd_ir_rms = RunningMeanStd()

            # These two are for the observation normalization (mean and std)
            self.pred_next_feature_ = tf.placeholder(tf.float32, [None, 512])
            self.target_next_feature_ = tf.placeholder(tf.float32, [None, 512])

            obs = self.OBS
            self.layer_size = layer_size

            if normalize_obs:
                with tf.name_scope('normalized_obs'):
                    obs = tf.clip_by_value((obs - self.RND_OBS_MEAN) / self.RND_OBS_STD, -5.0, 5.0)

            if isinstance(rnd_network, str):
                network_type = rnd_network
                rnd_network = get_network_builder(network_type)()

            with tf.variable_scope('target'):
                target_feature = rnd_network(obs)
                target_feature = tf.layers.flatten(target_feature)
                target_feature = fc(target_feature, layer_size, None)
                self.target_feature = tf.stop_gradient(target_feature)

            with tf.variable_scope('predictor'):
                predictor_feature = rnd_network(obs)
                predictor_feature = tf.layers.flatten(predictor_feature)
                predictor_feature = fc(predictor_feature, layer_size, tf.nn.leaky_relu)
                predictor_feature = fc(predictor_feature, layer_size, tf.nn.leaky_relu)
                self.predictor_feature = fc(predictor_feature, layer_size, None)

            with tf.name_scope('intrinsic_rewards'):
                self.int_reward = tf.reduce_mean(tf.square(self.target_feature - self.predictor_feature), axis=-1)

            with tf.name_scope('rand_loss'):
                mask = tf.random_uniform(shape=tf.shape(self.int_reward), minval=0., maxval=1., dtype=tf.float32)
                mask = tf.cast(mask < self.rnd_update_rate, tf.float32)
                self.rnd_loss = tf.reduce_sum(mask * self.int_reward) / tf.maximum(tf.reduce_sum(mask), 1.)

    def int_rewards(self, obs):
        intrinsic_rewards = self.sess.run(self.int_reward, {
            self.OBS: obs,
            self.RND_OBS_MEAN: self.rnd_ob_rms.mean,
            self.RND_OBS_STD: self.rnd_ob_rms.var ** 0.5
        })

        return intrinsic_rewards
