"""RL Agents."""


class Agent():
    """RL Agent base class."""

    def __init__(self, env):
        """Initialize agent.

        Args:
            env: The gym environment. An instance of `gym.Env`.
        """

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
