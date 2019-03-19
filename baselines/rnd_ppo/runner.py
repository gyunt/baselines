import numpy as np

from baselines.common.runners import AbstractEnvRunner


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """

    def __init__(self, *, env, model, nsteps, gamma, gamma_int, int_coef, ext_coef, ob_space, lam):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv * nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.get_initial_state()
        self.states.update({'dones': np.array([0 for _ in range(nenv)], dtype=np.float)})

        self.lam = lam  # Lambda used in GAE (General Advantage Estimation)
        self.gamma_ext = gamma  # Discount rate for extrinsic rewards
        self.gamma_int = gamma_int
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.ob_space = ob_space

    def run(self):
        minibatch = {
            "obs": [],
            "actions": [],
            "ext_rewards": [],
            "int_rewards": [],
            "ext_values": [],
            "int_values": [],
            "values": [],
            "dones": [],
            "neglogpacs": [],
        }

        data_type = {
            "obs": self.obs.dtype,
            "actions": np.float32,
            "ext_rewards": np.float32,
            "int_rewards": np.float32,
            "ext_values": np.float32,
            "int_values": np.float32,
            "values": np.float32,
            "dones": np.float32,
            "neglogpacs": np.float32,
        }

        prev_state = self.states
        dones = self.states['dones']
        epinfos = []
        # For n in range number of steps

        for _ in range(self.nsteps):
            step = {}
            step['obs'] = self.obs.copy()
            step['dones'] = dones
            step.update(self.model.step(observations=self.obs, **prev_state))
            step['values'] = step['int_values'] + step['ext_values']

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], ext_rewards, dones, infos = self.env.step(step['actions'])
            dones = np.array(dones, dtype=np.float)
            int_rewards = self.model.rnd_model.int_rewards(self.obs) - self.model.rnd_model.int_rewards(step['obs'])

            step.update({"ext_rewards": ext_rewards,
                         "int_rewards": int_rewards, })

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            for key in step:
                dtype = data_type[key] if key in data_type else np.float
                step[key] = np.array(step[key], dtype=dtype)

            for key in step:
                if key not in minibatch:
                    minibatch[key] = []
                minibatch[key].append(step[key])
            prev_state = step

        self.states = step
        step['obs'] = self.obs.copy()
        step['dones'] = dones

        # batch of steps to batch of rollouts
        for key in minibatch:
            dtype = data_type[key] if key in data_type else np.float
            minibatch[key] = np.asarray(minibatch[key], dtype=dtype)

        last_values = self.model.values(observations=self.obs, **self.states)
        last_int_values, last_ext_values = last_values['int_values'], last_values['ext_values']

        # Update reward normalization parameters
        self.model.rnd_model.rnd_ir_rms.update(minibatch['int_rewards'].ravel())
        self.model.reward_rms.update(minibatch['ext_rewards'].ravel())

        # Normalize the rewards
        minibatch['int_rewards'] = self.int_coef * minibatch['int_rewards'] / np.sqrt(
            self.model.rnd_model.rnd_ir_rms.var)
        minibatch['ext_rewards'] = self.ext_coef * minibatch['ext_rewards']  # / np.sqrt(self.model.reward_rms.var)

        # Calculate intrinsic returns and advantages.
        minibatch['int_advs'], minibatch['int_returns'] = \
            self.temporal_difference(values=minibatch['int_values'],
                                     rewards=minibatch['int_rewards'],
                                     dones=minibatch['dones'],
                                     last_value=last_int_values,
                                     use_non_episodic_rewards=True)
        minibatch['ext_advs'], minibatch['ext_returns'] = \
            self.temporal_difference(values=minibatch['ext_values'],
                                     rewards=minibatch['ext_rewards'],
                                     dones=minibatch['dones'],
                                     last_value=last_int_values)

        # Combine the extrinsic and intrinsic advantages.
        minibatch['advs'] = minibatch['int_advs'] + minibatch['ext_advs']
        minibatch['returns'] = minibatch['int_returns'] + minibatch['ext_returns']

        # Update norm parameters after the rollout is completed
        obs_ = minibatch['obs'].reshape((-1, *self.ob_space.shape))
        self.model.rnd_model.rnd_ob_rms.update(obs_)

        for key in minibatch:
            minibatch[key] = sf01(minibatch[key])

        minibatch['epinfos'] = epinfos
        return minibatch

    def init_obs_rnd_norm(self, obs_rnd_norm_nsteps, nenvs, ob_space):
        # This function is used to have initial normalization parameters by stepping
        # a random agent in the environment for a small nb of steps.
        print("Start to initialize the normalization parameters by stepping a random agent in the environment")

        all_obs = []
        _step = {}
        for _ in range(obs_rnd_norm_nsteps * nenvs):
            self.states = self.model.step(observations=self.obs, **self.states)
            all_obs.append(self.obs)
        ob_ = np.asarray(all_obs).astype(np.float32).reshape((-1, *ob_space.shape))
        self.model.rnd_model.rnd_ob_rms.update(ob_)
        print("Initialization finished")

        all_obs.clear()

    def temporal_difference(self, values, rewards, dones, last_value, use_non_episodic_rewards=True):
        """
        from: https://openai-public.s3-us-west-2.amazonaws.com/blog/2017-07/ppo/ppo-arxiv.pdf

        Minibatch style requires an advantage estimator that does not look beyond timestep T. The estimator used by [Mni+16] is
            A_hat_t = -V(s_t) + r_t + [gamma * r_(t+1)] + ... + [gamma^(T-t+1) * r_(t-1)] + [gamma^(T-t) * V(s_T)]  (10)
        where t specifies the time index in [0, T], within a given length-T trajectory segment.
        Generalizing this choice, we can use a truncated version of generalized advantage estimation, which reduces to
        Equation (10) when lambda = 1:
            A_hat_t = delta_t + [(gamma * lambda) * delta_(t+1)] + ... + [(gamma * lambda)^(T-t+1) * delta_(T-1)]   (11)
            where delta_t = r_t + gamma * V(s_(t+1)) - V(s_t)
        """
        advantages = np.zeros_like(rewards)
        lastgaelam = 0  # Lambda used in General Advantage Estimation
        for t in reversed(range(self.nsteps)):
            if not use_non_episodic_rewards:
                if t == self.nsteps - 1:
                    next_non_terminal = 1.0 - dones
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
            else:
                next_non_terminal = 1.0  # No dones for intrinsic reward.
            next_value = values[t + 1] if t < self.nsteps - 1 else last_value
            delta = rewards[t] + self.gamma_int * next_value * next_non_terminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma_int * self.lam * next_non_terminal * lastgaelam
        returns = advantages + values
        return advantages, returns


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
