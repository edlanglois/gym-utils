"""RL Agents."""


class Agent():
    """RL Agent base class."""

    def __init__(self, env):
        """Initialize agent.

        Args:
            env: The gym environment. An instance of `gym.Env`.
        """
        self.env = env

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
