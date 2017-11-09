import pytest

import numpy as np
import gym
from gym import spaces

from gym_utils import env_wrappers

flat_box_test_spaces = [
    (spaces.Box(0, 1, ()), 1),
    (spaces.Box(0, 1, (0, )), 0),
    (spaces.Box(0, 1, (4, )), 4),
    (spaces.Box(0, 1, (2, 3)), 6),
    (spaces.Discrete(5), 5),
    (spaces.Tuple((spaces.Box(0, 1, (2)), spaces.Discrete(3))), 5),
    (spaces.Tuple(()), 0),
]


class TestFlatBoxView():
    @pytest.mark.parametrize('space,n', flat_box_test_spaces)
    def test_shape(self, space, n):
        low = np.zeros(n)
        high = np.ones(n)
        flat_space = env_wrappers.FlatBoxView(space)
        assert flat_space.shape == (n, )
        assert np.array_equal(flat_space.low, low)
        assert np.array_equal(flat_space.high, high)

    @pytest.mark.parametrize('space,n', flat_box_test_spaces)
    def test_sample(self, space, n):
        flat_space = env_wrappers.FlatBoxView(space)
        x = flat_space.sample()
        assert x.shape == (n, )
        assert all(x >= np.zeros(n))
        assert all(x <= np.ones(n))
        flat_space.contains(x)

    def test_convert(self):
        flat_space = env_wrappers.FlatBoxView(spaces.Box(0, 1, ()))
        assert flat_space.convert(0.3) == 0.3

        flat_space = env_wrappers.FlatBoxView(spaces.Box(0, 1, (2, 3)))
        assert np.array_equal(
            flat_space.convert(np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])),
            np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))

        flat_space = env_wrappers.FlatBoxView(spaces.Discrete(4))
        assert np.array_equal(
            flat_space.convert(2), np.array([0, 0, 1, 0], dtype=float))

        flat_space = env_wrappers.FlatBoxView(
            spaces.Tuple((spaces.Box(0, 1, (2, )), spaces.Discrete(2))))
        assert np.array_equal(
            flat_space.convert((np.array([0.3, 0.7]), 1)),
            np.array([0.3, 0.7, 0.0, 1.0]))


@pytest.mark.parametrize('env_name,obs_size', [
    ('CartPole-v0', 4),
    ('FrozenLake8x8-v0', 8 * 8),
    ('Hex9x9-v0', 3 * 9 * 9),
])
def test_flatten_observations(env_name, obs_size):
    env = gym.make(env_name)
    flat_env = env_wrappers.FlattenObservations(env)

    assert isinstance(flat_env.observation_space, spaces.Box)
    assert flat_env.observation_space.shape == (obs_size, )
    assert flat_env.observation_space.sample().shape == (obs_size, )

    obs = flat_env.reset()
    assert obs.shape == (obs_size, )


@pytest.mark.parametrize('env_name,obs_shape,buffer_size', [
    ('CartPole-v0', (4, ), 2),
    ('CartPole-v0', (4, ), 0),
    ('Hex9x9-v0', (3, 9, 9), 2),
])
def test_buffer_observations(env_name, obs_shape, buffer_size):
    env = gym.make(env_name)
    buffered_obs_env = env_wrappers.BufferObservations(
        env, buffer_size=buffer_size)

    assert isinstance(buffered_obs_env.observation_space, spaces.Box)
    assert buffered_obs_env.observation_space.shape == (
        (buffer_size, ) + obs_shape)

    obs = buffered_obs_env.reset()
    assert obs.shape == (buffer_size, ) + obs_shape


def _assert_cartpole_with_reward(step_returns, reward_indicator_observation):
    obs, reward, done, info = step_returns
    if done:
        raise RuntimeError('Test case does not yet deal with resetting '
                           'a completed environment.')
    # CartPole-v0 always has reward 1
    assert reward == 1
    assert info['is_true_reward']
    assert info['true_reward'] == 1
    if reward_indicator_observation:
        _, true_reward = obs
        assert true_reward == 1


def _assert_cartpole_without_reward(step_returns,
                                    reward_indicator_observation):
    obs, reward, done, info = step_returns
    if done:
        raise RuntimeError('Test case does not yet deal with resetting '
                           'a completed environment.')
    assert reward == 0
    assert not info['is_true_reward']
    assert info['true_reward'] == 1
    if reward_indicator_observation:
        _, true_reward = obs
        assert true_reward == 0


@pytest.mark.parametrize('reward_indicator_observation', [(False, True)])
def test_semi_supervised_finite_reward(reward_indicator_observation):
    env = gym.make('CartPole-v0')
    ss_env = env_wrappers.SemiSupervisedFiniteReward(
        env,
        max_rewards=3,
        reward_indicator_observation=reward_indicator_observation,
        reward_on_request=False)

    action_space = ss_env.action_space
    ss_env.reset()
    for _ in range(3):
        _assert_cartpole_with_reward(
            ss_env.step(action_space.sample()), reward_indicator_observation)

    # Have used up the 3 rewards, now ensure no more
    for _ in range(2):
        _assert_cartpole_without_reward(
            ss_env.step(action_space.sample()), reward_indicator_observation)

    # Ensure still no more reward after reset
    ss_env.reset()
    for _ in range(2):
        _assert_cartpole_without_reward(
            ss_env.step(action_space.sample()), reward_indicator_observation)


@pytest.mark.parametrize('reward_indicator_observation', [(False, True)])
def test_semi_supervised_finite_reward_request(reward_indicator_observation):
    env = gym.make('CartPole-v0')
    ss_env = env_wrappers.SemiSupervisedFiniteReward(
        env,
        max_rewards=2,
        reward_indicator_observation=reward_indicator_observation,
        reward_on_request=True)

    action_space = ss_env.action_space
    assert isinstance(action_space, spaces.Tuple)
    sub_action_space, request_action_space = action_space.spaces
    assert request_action_space.contains(0)
    assert request_action_space.contains(1)

    ss_env.reset()

    # Not requesting reward
    _assert_cartpole_without_reward(
        ss_env.step((sub_action_space.sample(), 0)),
        reward_indicator_observation)
    # Request reward x2
    _assert_cartpole_with_reward(
        ss_env.step((sub_action_space.sample(), 1)),
        reward_indicator_observation)
    _assert_cartpole_with_reward(
        ss_env.step((sub_action_space.sample(), 1)),
        reward_indicator_observation)
    # Not requesting reward
    _assert_cartpole_without_reward(
        ss_env.step((sub_action_space.sample(), 0)),
        reward_indicator_observation)
    # Request reward - past limit, no reward given
    _assert_cartpole_without_reward(
        ss_env.step((sub_action_space.sample(), 1)),
        reward_indicator_observation)
    # Still no reward after reset
    ss_env.reset()
    _assert_cartpole_without_reward(
        ss_env.step((sub_action_space.sample(), 0)),
        reward_indicator_observation)
