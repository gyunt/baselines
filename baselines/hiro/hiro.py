import os
import time
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from baselines import logger
from baselines.common import explained_variance
from baselines.common import set_global_seeds
from baselines.common.tf_util import display_var_info, save_variables, load_variables
from baselines.hiro.preprocess import StatePreprocess, state_preprocess_net, action_embed_net
from baselines.hiro.runner import Runner
from baselines.ppo2.policies import build_ppo_policy

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def constfn(val):
    def f(_):
        return val

    return f


def learn(*, network, env, total_timesteps, eval_env=None, seed=None, nsteps=128, ent_coef=0.0, lr=3e-4,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95, sub_goal_dim=8,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2, meta_action_every_n=4,
          save_interval=10, load_path=None, model_fn=None, **network_kwargs):
    """
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor for rewards

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.
    """

    set_global_seeds(seed)

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_ppo_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    subgoal_space = gym.spaces.Box(low=-.1, high=.1, shape=(sub_goal_dim,), dtype=np.float32)

    with tf.Session() as sess:
        high_model = model_fn(name='high_model', policy=policy, ob_space=ob_space, ac_space=subgoal_space,
                              nbatch_act=nenvs,
                              nbatch_train=nbatch_train // meta_action_every_n,
                              sess=sess,
                              nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)
        low_model = model_fn(name='low_model', policy=policy, ob_space=subgoal_space, ac_space=ac_space,
                             nbatch_act=nenvs,
                             nbatch_train=nbatch_train,
                             sess=sess,
                             nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)

        state_preprocess = StatePreprocess(
            ob_space=ob_space,
            subgoal_space=subgoal_space,
            act_space=ac_space,
            meta_action_every_n=meta_action_every_n,
            state_preprocess_net=state_preprocess_net,
            action_embed_net=action_embed_net,
            max_grad_norm=max_grad_norm)

        def save_graph(path, sess=None, graph=None):
            if graph is None:
                if sess is None:
                    sess = tf.get_default_session()
                graph = sess.graph
            writer = tf.summary.FileWriter(path, graph)
            writer.flush()

        save_graph("i:/tmp/")

        if load_path is not None:
            load_variables(load_path, sess=sess )

        high_allvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, high_model.scope.name)
        display_var_info(high_allvars)

        low_allvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, low_model.scope.name)
        display_var_info(low_allvars)

        state_preprocess_all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, state_preprocess.scope.name)
        display_var_info(state_preprocess_all_vars)

        # Instantiate the runner object
        runner = Runner(env=env, high_model=high_model, low_model=low_model, state_preprocess=state_preprocess,
                        nsteps=nsteps, meta_action_every_n=meta_action_every_n, gamma=gamma, ob_space=ob_space,
                        sess=sess,
                        lam=lam)

        epinfobuf = deque(maxlen=100)

        # Start total timer
        tfirststart = time.time()
        nupdates = total_timesteps // nbatch

        for update in range(1, nupdates + 1):
            assert nbatch % nminibatches == 0
            # Start timer
            tstart = time.time()
            frac = 1.0 - (update - 1.0) / nupdates
            # Calculate the learning rate
            lrnow = lr(frac)
            # Calculate the cliprange
            cliprangenow = cliprange(frac)

            # Get minibatch
            high_minibatch, low_minibatch = runner.run()

            epinfobuf.extend(high_minibatch.pop('epinfos'))

            # Here what we're going to do is for each minibatch calculate the loss and append it.
            high_loss_vals = []
            low_loss_vals = []
            repr_loss_vals = []

            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch // meta_action_every_n)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch // meta_action_every_n, nbatch_train):
                    end = start + nbatch_train // meta_action_every_n
                    mbinds = inds[start:end]
                    slices = {key: high_minibatch[key][mbinds] for key in high_minibatch}
                    high_loss_vals.append(high_model.train(lrnow, cliprangenow, **slices))

            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = {key: low_minibatch[key][mbinds] for key in low_minibatch}
                    low_loss_vals.append(low_model.train(lrnow, cliprangenow, **slices))
                    repr_loss_vals.append(state_preprocess.train(lrnow, **slices))

            # Feedforward --> get losses --> update
            high_loss_vals = np.mean(high_loss_vals, axis=0)
            low_loss_vals = np.mean(low_loss_vals, axis=0)
            repr_loss_vals = np.mean(repr_loss_vals, axis=0)

            # End timer
            tnow = time.time()
            # Calculate the fps (frame per second)
            fps = int(nbatch / (tnow - tstart))
            if update % log_interval == 0 or update == 1:
                # Calculates if value function is a good predicator of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
                ev = explained_variance(high_minibatch['values'], high_minibatch['returns'])
                logger.logkv("serial_timesteps", update * nsteps)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps", update * nbatch)
                logger.logkv("fps", fps)
                logger.logkv("explained_variance", float(ev))
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger.logkv('high_rewards_per_step ', safemean(high_minibatch['rewards']))
                logger.logkv('low_rewards_per_step', safemean(low_minibatch['rewards']))
                logger.logkv('advantages_per_step', safemean(high_minibatch['advs']))
                logger.logkv('sampled_estimated_log_partition',
                             safemean(low_minibatch['sampled_estimated_log_partition']))
                logger.logkv('time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(high_loss_vals, high_model.loss_names):
                    logger.logkv('high_' + lossname, lossval)
                for (lossval, lossname) in zip(low_loss_vals, low_model.loss_names):
                    logger.logkv('low_' + lossname, lossval)
                for (lossval, lossname) in zip(repr_loss_vals, state_preprocess.stats_names):
                    logger.logkv('repr_' + lossname, lossval)
                if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                    logger.dumpkvs()

            if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and (
                MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
                checkdir = os.path.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = os.path.join(checkdir, '%.5i' % update)
                print('Saving to', savepath)
                save_variables(savepath, sess=sess)
            del high_minibatch
            del low_minibatch
    return tuple([high_model, low_model])


# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
