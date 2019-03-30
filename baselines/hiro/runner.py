import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.tf_util import get_session


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """

    def __init__(self, *, env, high_model, low_model, state_preprocess, meta_action_every_n, nsteps, gamma, ob_space,
                 lam, sess):
        self.env = env
        self.high_model = high_model
        self.low_model = low_model
        self.running_mean = RunningMeanStd(shape=ob_space.shape)
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv * nsteps,) + env.observation_space.shape
        self.observations = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.observations = env.reset()
        self.running_mean.update(self.observations)
        self.observations = (self.observations - self.running_mean.mean) / np.sqrt(self.running_mean.var + 1e-8)

        self.nsteps = nsteps
        self.dones = np.array([False for _ in range(self.nenv)])
        self.lam = lam  # Lambda used in GAE (General Advantage Estimation)
        self.gamma = gamma  # Discount rate for rewards
        self.ob_space = ob_space
        self.state_preprocess = state_preprocess
        self.meta_action_every_n = meta_action_every_n

        discount = np.zeros((self.meta_action_every_n, self.nenv))
        discount[0] = 1

        for i in range(self.meta_action_every_n - 1):
            discount[i + 1] = discount[i] * self.gamma
        discount[-1] /= (1 - self.gamma)
        self.discount = discount

        if sess is None:
            sess = get_session()
        self.sess = sess

    def run(self):
        high_minibatch = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "dones": [],
            "neglogpacs": [],
        }

        low_minibatch = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "dones": [],
            "neglogpacs": [],
        }

        data_type = {
            "observations": self.observations.dtype,
            "actions": np.float32,
            "rewards": np.float32,
            "values": np.float32,
            "dones": np.float32,
            "neglogpacs": np.float32,
        }

        epinfos = []
        prev_high_transition = dict()

        # For n in range number of steps
        for i in range(self.nsteps // self.meta_action_every_n):
            high_transitions = dict()
            high_transitions['observations'] = self.observations.copy()
            high_transitions['dones'] = self.dones
            if 'next_states' in prev_high_transition:
                high_transitions['states'] = prev_high_transition['next_states']
            high_transitions.update(self.high_model.step_as_dict(**high_transitions))
            high_transitions['rewards'] = [0] * self.nenv
            sub_goals = high_transitions['actions']

            low_actions = []
            prev_low_transition = dict()

            for j in range(self.meta_action_every_n):
                low_transitions = dict()
                low_transitions['begin_high_observations'] = high_transitions['observations']
                low_transitions['observations'] = self.state_preprocess.embedded_state(self.observations)
                low_transitions['dones'] = self.dones
                low_transitions['high_observations'] = self.observations.copy()
                low_transitions['discounts'] = self.discount[j]
                low_transitions['goal_states'] = sub_goals

                if 'next_states' in prev_low_transition:
                    low_transitions['states'] = prev_low_transition['next_states']
                low_transitions.update(self.low_model.step_as_dict(**low_transitions))
                low_actions.append(low_transitions['actions'])

                # Take actions in env and look the results
                # Infos contains a ton of useful informations
                self.observations, reward, self.dones, infos = self.env.step(low_transitions['actions'])
                self.running_mean.update(self.observations)
                self.observations = (self.observations - self.running_mean.mean) / np.sqrt(self.running_mean.var + 1e-8)

                low_transitions['next_high_observations'] = self.observations.copy()
                high_transitions['rewards'] += reward

                self.dones = np.array(self.dones, dtype=np.float)

                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo:
                        epinfos.append(maybeepinfo)

                for key in low_transitions:
                    if key not in low_minibatch:
                        low_minibatch[key] = []
                    low_minibatch[key].append(low_transitions[key])
                prev_low_transition = low_transitions

            if 'low_actions' not in low_minibatch:
                low_minibatch['low_actions'] = []
            if 'rewards' not in low_minibatch:
                low_minibatch['rewards'] = []
            low_actions = np.array(low_actions).swapaxes(0, 1)

            for j in range(self.meta_action_every_n):
                low_minibatch['low_actions'].append(low_actions)

            for j in reversed(range(self.meta_action_every_n)):
                rewards, sampled_estimated_log_partition = self.state_preprocess.low_rewards(
                    begin_high_observations=low_minibatch['begin_high_observations'][-j],
                    high_observations=low_minibatch['high_observations'][-j],
                    next_high_observations=low_minibatch['next_high_observations'][-j],
                    low_actions=low_minibatch['low_actions'][-j],
                    goal_states=low_minibatch['goal_states'][-j],
                    discounts=low_minibatch['discounts'][-j])
                low_minibatch['rewards'].append(rewards)

            for key in high_transitions:
                if key not in high_minibatch:
                    high_minibatch[key] = []
                high_minibatch[key].append(high_transitions[key])
            prev_high_transition = high_transitions

        high_transitions['observations'] = self.observations.copy()
        high_transitions['dones'] = self.dones
        if 'states' in high_transitions:
            high_transitions['states'] = high_transitions.pop('next_states')

        # Calculate returns and advantages.
        high_minibatch['advs'], high_minibatch['returns'] = \
            self.advantage_and_returns(nsteps=self.nsteps // self.meta_action_every_n,
                                       values=high_minibatch['values'],
                                       rewards=high_minibatch['rewards'],
                                       dones=high_minibatch['dones'],
                                       last_values=self.high_model.step_as_dict(**high_transitions)['values'],
                                       last_dones=self.dones,
                                       gamma=self.gamma)

        low_minibatch['advs'], low_minibatch['returns'] = \
            self.advantage_and_returns(nsteps=self.nsteps,
                                       values=low_minibatch['values'],
                                       rewards=low_minibatch['rewards'],
                                       dones=low_minibatch['dones'],
                                       last_values=self.low_model.step_as_dict(**low_transitions)['values'],
                                       last_dones=self.dones,
                                       gamma=self.gamma)

        for key in high_minibatch:
            dtype = data_type[key] if key in data_type else np.float
            high_minibatch[key] = sf01(np.array(high_minibatch[key], dtype=dtype))

        for key in low_minibatch:
            dtype = data_type[key] if key in data_type else np.float
            low_minibatch[key] = sf01(np.array(low_minibatch[key], dtype=dtype))

        high_minibatch['epinfos'] = epinfos
        return high_minibatch, low_minibatch

    def advantage_and_returns(self, nsteps, values, rewards, dones, last_values, last_dones, gamma,
                              use_non_episodic_rewards=False):
        """
        calculate Generalized Advantage Estimation (GAE), https://arxiv.org/abs/1506.02438
        see also Proximal Policy Optimization Algorithms, https://arxiv.org/abs/1707.06347
        """

        advantages = np.zeros_like(rewards)
        lastgaelam = 0  # Lambda used in General Advantage Estimation
        for t in reversed(range(nsteps)):
            if not use_non_episodic_rewards:
                if t == nsteps - 1:
                    next_non_terminal = 1.0 - last_dones
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
            else:
                next_non_terminal = 1.0
            next_value = values[t + 1] if t < nsteps - 1 else last_values
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
