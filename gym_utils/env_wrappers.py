"""OpenAI gym environment wrappers."""
import collections

import gym
import numpy as np
from gym import spaces

__all__ = [
    'FlatBoxView',
    'FlattenObservations',
    'BufferObservations',
]


def _flatten_space(space):
    """Flaten a space to a 1D box.

    Args:
        space: A `gym.Space` instance.

    Returns:
        low: Lower bounds on the box. A 1D `numpy.ndarray`.
        high: Upper bounds on the box. A 1D `numpy.ndarray`.
        transformer: A function mapping samples from `space` to samples from
            `flat_space`.
    """
    if isinstance(space, spaces.Box):
        # Flatten to a 1D box
        low = space.low.flatten()
        high = space.high.flatten()

        def transformer(x):
            return np.asarray(x).flatten()

    elif isinstance(space, spaces.Discrete):
        # Convert to 1-hot vectors
        n = space.n
        low = np.zeros((n, ))
        high = np.ones((n, ))
        ident = np.eye(n)

        def transformer(x):
            return ident[x]

    elif isinstance(space, spaces.Tuple):
        # Append dimensions
        lows, highs, transformers = zip(*(_flatten_space(subspace)
                                          for subspace in space.spaces))
        low = np.concatenate(lows)
        high = np.concatenate(highs)

        def transformer(x):
            return np.concatenate(
                [sub_t(sub_x) for sub_t, sub_x in zip(transformers, x)])

    else:
        raise ValueError('Flattening not supported for space {}'.format(space))

    return low, high, transformer


class FlatBoxView(spaces.Box):
    """"Wraps a space to appear as a 1D box space."""

    def __init__(self, space):
        self.original_space = space
        low, high, self._transformer = _flatten_space(space)
        super().__init__(low=low, high=high)

    def sample(self):
        return self.convert(self.original_space.sample())

    def convert(self, x):
        """Convert a sample from original space to a sample from this space.

        Args:
            x: A sample from `self.original_space`.

        Returns:
            The associated sample from this space.
        """
        return self._transformer(x)


class FlattenObservations(gym.ObservationWrapper):
    """Wrapper that flattens the observation space into a 1D Box space."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = FlatBoxView(self.env.observation_space)

    def _observation(self, observation):
        return self.observation_space.convert(observation)

    def _ensure_no_double_wrap(self):
        # I'll wrap as many times as I please, you're not the boss of me.
        pass


class BufferObservations(gym.ObservationWrapper):
    """Environment wrapper, observations are a buffer of recent observations.
    """

    def __init__(self, env, buffer_size):
        """Initialize BufferObservations.

        Args:
            env: Environment to wrap.
            buffer_size: Number of `env` observations to include in the buffer.
        """
        super().__init__(env)
        original_observation_space = env.observation_space
        try:
            low = original_observation_space.low
            high = original_observation_space.high
        except AttributeError as e:
            raise ValueError('Environment observation space must be '
                             'an instance of gym.spaces.Box') from e
        # Prepend a new 'history' dimension
        reps = (buffer_size, ) + (1, ) * low.ndim
        low = np.tile(low, reps)
        high = np.tile(high, reps)
        self.observation_space = spaces.Box(low=low, high=high)
        self.buffer_size = buffer_size
        self._observation_buffer = collections.deque([], maxlen=buffer_size)

    def _reset(self):
        observation = self.env.reset()
        self._observation_buffer.clear()
        for i in range(self.buffer_size):
            self._observation_buffer.append(observation)
        return self._get_observation()

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._observation_buffer.append(observation)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return np.concatenate(self._observation_buffer)


class SemiSupervisedFiniteReward(gym.Wrapper):
    """Convert an environment into a finite-reward semi-supervised RL env.

    At most `max_rewards` nonzero reward observations are given. This may
    either be the all reward observations until the limit, or reward may be
    specifically requested. All other reward observations are zero.

    The reward limit persists through resets.
    """

    def __init__(self,
                 env,
                 max_rewards,
                 reward_indicator_observation=False,
                 reward_on_request=False):
        """Initialize SemiSupervisedFixedFiniteReward environment.

        Args:
            env: Environment to wrap.
            max_rewards: Maximum allowed reward observations.
            reward_indicator_observation: Augment observation space with a bit
                that indicates whether the reward is the true reward (1) or
                constant zero (0).
            reward_on_request: If `True`, augment the action space with a bit
                that requests the true reward for the current step.
        """
        super().__init__(env)
        self.rewards_given = 0
        self.max_rewards = max_rewards
        self.reward_indicator_observation = reward_indicator_observation
        self.reward_on_request = reward_on_request

        if reward_indicator_observation:
            self.observation_space = spaces.tuple(self.observation_space,
                                                  spaces.Discrete(2))

        if reward_on_request:
            self.action_space = spaces.tuple(self.action_space,
                                             spaces.Discrete(2))

        # Ensure 0 is contained in the reward range.
        self.reward_range = (min(self.reward_range[0], 0), max(
            self.reward_range[1], 0))

    def _step(self, action):
        can_give_reward = self.rewards_given < self.max_rewards
        if self.reward_on_request:
            action, requesting_reward = action
            give_reward = can_give_reward and requesting_reward
        else:
            give_reward = can_give_reward

        observation, reward, done, info = self.env.step(action)
        if give_reward:
            self.rewards_given += 1
        else:
            reward = 0

        if info is None:
            info = {}
        info['true_reward'] = give_reward
        if self.reward_indicator_observation:
            observation = (observation, give_reward)

        return observation, reward, done, info
