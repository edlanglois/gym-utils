"""RL Agents."""
import os
import os.path

from baselines import deepq

from gym_utils import env_wrappers


class Agent():
    """RL Agent base class."""

    def __init__(self, env):
        """Initialize agent.

        Args:
            env: The gym environment. An instance of `gym.Env`.
        """
        self.env = env

    @staticmethod
    def wrap_env(env):
        return env

    def train(self, max_timesteps, env=None):
        """Self-directed training on the environment.
        Args:
            max_timesteps: Maximum number of training timesteps.
            env: Env to use for training. Uses `self.env` by default.

        Returns:
            True if training occurred.
        """
        return False

    def save(self, directory):
        """Save the agent state to disk."""
        raise NotImplementedError

    def load(self, directory):
        """Load agent state from disk."""
        raise NotImplementedError

    def act(self, observation):
        """Select an action for `observation`."""
        raise NotImplementedError

    def update(self, observation, reward, done):
        """Update the agent."""
        pass


class RandomAgent(Agent):
    """Takes a random action."""

    def act(self, observation):
        del observation
        return self.env.action_space.sample()

    def save(self, directory):
        pass

    def load(self, directory):
        pass


class DeepQAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.model = deepq.models.mlp([64], layer_norm=True)
        self.actor = None
        self._model_filename = 'dqn_model.pkl'

    @staticmethod
    def wrap_env(env):
        return env_wrappers.FlattenObservations(env)

    def train(self, max_timesteps, env=None):
        self.actor = deepq.learn(
            env or self.env,
            q_func=self.model,
            max_timesteps=max_timesteps,
            param_noise=True)
        return True

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_file = os.path.join(directory, self._model_filename)
        print('Saving model to:', model_file)
        self.actor.save(model_file)

    def load(self, directory):
        model_file = os.path.join(directory, self._model_filename)
        print('Loading model from:', model_file)
        self.actor = deepq.load(model_file)

    def act(self, observation):
        if self.actor is None:
            raise RuntimeError(
                'DeepQAgent must be trained with train() before it can be '
                'used.')
        try:
            obs_vector = observation[None, :]
        except TypeError as e:
            raise TypeError('Environment not supported. '
                            'Wrap it with `wrap_env` first.') from e
        return self.actor(obs_vector)[0]
