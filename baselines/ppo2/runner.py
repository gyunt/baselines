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

    def __init__(self, *, env, model, nsteps, gamma, ob_space, lam):
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
        self.gamma = gamma  # Discount rate for rewards
        self.ob_space = ob_space

    def run(self):
        minibatch = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "dones": [],
            "neglogpacs": [],
        }

        data_type = {
            "obs": self.obs.dtype,
            "actions": np.float32,
            "rewards": np.float32,
            "values": np.float32,
            "dones": np.float32,
            "neglogpacs": np.float32,
        }

        prev_state = self.states
        dones = self.states['dones']
        epinfos = []

        # For n in range number of steps
        for _ in range(self.nsteps):
            transition = {}
            transition['obs'] = self.obs.copy()
            transition['dones'] = dones
            transition.update(self.model.step(observations=self.obs, **prev_state))
            transition['values'] = transition['values']

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], transition['rewards'], dones, infos = self.env.step(transition['actions'])
            dones = np.array(dones, dtype=np.float)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            for key in transition:
                dtype = data_type[key] if key in data_type else np.float
                transition[key] = np.array(transition[key], dtype=dtype)

            for key in transition:
                if key not in minibatch:
                    minibatch[key] = []
                minibatch[key].append(transition[key])
            prev_state = transition
        transition['obs'] = self.obs.copy()
        transition['dones'] = dones
        self.states = transition

        for key in minibatch:
            dtype = data_type[key] if key in data_type else np.float
            minibatch[key] = np.asarray(minibatch[key], dtype=dtype)

        last_values = self.model.step(observations=self.obs, **self.states)['values']

        # Calculate returns and advantages.
        minibatch['advs'], minibatch['returns'] = \
            self.temporal_difference(values=minibatch['values'],
                                     rewards=minibatch['rewards'],
                                     dones=minibatch['dones'],
                                     last_values=last_values,
                                     last_dones=dones,
                                     gamma=self.gamma)

        for key in minibatch:
            minibatch[key] = sf01(minibatch[key])

        minibatch['epinfos'] = epinfos
        return minibatch

    def temporal_difference(self, values, rewards, dones, last_values, last_dones, gamma,
                            use_non_episodic_rewards=False):
        """
        from: PPO Paper, https://openai-public.s3-us-west-2.amazonaws.com/blog/2017-07/ppo/ppo-arxiv.pdf

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
                    next_non_terminal = 1.0 - last_dones
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
            else:
                next_non_terminal = 1.0
            next_value = values[t + 1] if t < self.nsteps - 1 else last_values
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * self.lam * next_non_terminal * lastgaelam
        returns = advantages + values
        return advantages, returns


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
