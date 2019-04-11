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
                 ac_space,
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
        self.ac_space = ac_space
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

        observation_shape = self.observations.shape
        multi = 1
        for i in range(1, len(observation_shape)):
            multi *= observation_shape[i]
        env_observation_space = [observation_shape[0], multi]
        env_observations = self.observations

        low_all_actions = np.zeros(shape=[self.meta_action_every_n] + [self.nenv, int(np.prod(self.ac_space.shape))],
                                   dtype=np.float)
        low_all_states = np.zeros(shape=[self.meta_action_every_n] + [self.nenv, int(np.prod(self.ob_space.shape))],
                                  dtype=np.float)

        # For n in range number of steps
        for i in range(self.nsteps // self.meta_action_every_n):
            high_transitions = dict()
            high_transitions['observations'] = np.concatenate(
                [
                    env_observations.reshape(env_observation_space),
                    # self.state_preprocess.embedded_state(env_observations),
                    # low_all_states.swapaxes(0, 1).reshape(low_all_states.shape[1], -1),
                    # low_all_actions.swapaxes(0, 1).reshape(low_all_actions.shape[1], -1),
                ], axis=1)
            high_transitions['dones'] = self.dones
            if 'next_states' in prev_high_transition:
                high_transitions['states'] = prev_high_transition['next_states']
            high_transitions.update(self.high_model.step_with_dict(**high_transitions))
            high_transitions['rewards'] = [0] * self.nenv

            meta_actions = high_transitions['actions']

            prev_low_transition = dict()
            context = self.state_preprocess.get_goal_states(meta_actions=meta_actions)
            begin_env_observations = env_observations

            for j in range(self.meta_action_every_n):
                low_transitions = dict()
                low_transitions['begin_high_observations'] = begin_env_observations
                low_transitions['dones'] = self.dones
                low_transitions['high_observations'] = env_observations
                low_transitions['discounts'] = self.discount[j]
                low_transitions['meta_actions'] = meta_actions
                low_transitions['observations'] = np.concatenate(
                    [
                        env_observations.reshape(env_observation_space),
                        self.state_preprocess.embedded_state(low_transitions['high_observations']),
                        context,
                    ], axis=1)

                if 'next_states' in prev_low_transition:
                    low_transitions['states'] = prev_low_transition['next_states']
                low_transitions.update(self.low_model.step_with_dict(**low_transitions))
                low_all_actions[j, :] = low_transitions['actions']
                low_all_states[j, :] = env_observations.reshape(env_observation_space)

                # Take actions in env and look the results
                # Infos contains a ton of useful informations
                self.observations, high_rewards, self.dones, infos = self.env.step(low_transitions['actions'])
                self.observations = self.observations.copy()
                self.running_mean.update(self.observations)
                self.observations = (self.observations - self.running_mean.mean) / np.sqrt(self.running_mean.var + 1e-8)
                env_observations = self.observations

                low_transitions['next_high_observations'] = env_observations
                high_transitions['rewards'] += high_rewards
                context = context - (self.state_preprocess.embedded_state(low_transitions['next_high_observations']) \
                                     - self.state_preprocess.embedded_state(low_transitions['high_observations']))

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

            if 'low_all_actions' not in low_minibatch:
                low_minibatch['low_all_actions'] = []
            if 'rewards' not in low_minibatch:
                low_minibatch['rewards'] = []
            if 'end_high_observations' not in low_minibatch:
                low_minibatch['end_high_observations'] = []
            if 'low_all_high_observations' not in low_minibatch:
                low_minibatch['low_all_high_observations'] = []
            #
            # for key in ['low_all_actions', 'rewards', 'end_high_observations', 'low_all_high_observations']:
            #     low_minibatch[key] = []

            high_transitions['rewards'] += (-np.linalg.norm(
                high_transitions['actions'] - self.state_preprocess.embedded_state(
                    low_transitions['next_high_observations']), axis=-1))

            swapped_low_all_actions = np.array(low_all_actions).swapaxes(0, 1)
            swapped_low_high_observations = np.array(low_all_states).swapaxes(0, 1)
            for j in range(self.meta_action_every_n):
                low_minibatch['end_high_observations'].append(self.observations)
                low_minibatch['low_all_actions'].append(swapped_low_all_actions)
                low_minibatch['low_all_high_observations'].append(swapped_low_high_observations)

            for j in range(1, self.meta_action_every_n + 1):
                low_rewards = self.state_preprocess.low_rewards(
                    begin_high_observations=low_minibatch['begin_high_observations'][-j],
                    high_observations=low_minibatch['high_observations'][-j],
                    next_high_observations=low_minibatch['next_high_observations'][-j],
                    end_high_observations=low_minibatch['end_high_observations'][-j],
                    meta_actions=low_minibatch['meta_actions'][-j],
                    low_all_actions=swapped_low_all_actions,
                    discounts=low_minibatch['discounts'][-j])
                low_minibatch['rewards'].append(low_rewards)

            for key in high_transitions:
                if key not in high_minibatch:
                    high_minibatch[key] = []
                high_minibatch[key].append(high_transitions[key])
            prev_high_transition = high_transitions

        self.state_preprocess.update_displacement(begin_high_observations=begin_env_observations,
                                                  end_high_observations=self.observations)

        high_transitions['observations'] = np.concatenate(
            [
                env_observations.reshape(env_observation_space),
                # self.state_preprocess.embedded_state(env_observations),
                # low_all_states.swapaxes(0, 1).reshape(low_all_states.shape[1], -1),
                # low_all_actions.swapaxes(0, 1).reshape(low_all_actions.shape[1], -1),
            ], axis=1)
        high_transitions['dones'] = self.dones
        if 'states' in high_transitions:
            high_transitions['states'] = high_transitions.pop('next_states')

        # Calculate returns and advantages.
        high_minibatch['advs'], high_minibatch['returns'] = \
            self.advantage_and_returns(nsteps=self.nsteps // self.meta_action_every_n,
                                       values=high_minibatch['values'],
                                       rewards=high_minibatch['rewards'],
                                       dones=high_minibatch['dones'],
                                       last_values=self.high_model.step_with_dict(**high_transitions)['values'],
                                       last_dones=self.dones,
                                       gamma=self.gamma)

        low_minibatch['advs'], low_minibatch['returns'] = \
            self.advantage_and_returns(nsteps=self.nsteps,
                                       values=low_minibatch['values'],
                                       rewards=low_minibatch['rewards'],
                                       dones=low_minibatch['dones'],
                                       last_values=self.low_model.step_with_dict(**low_transitions)['values'],
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
