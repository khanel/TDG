from typing import Any, Dict, Optional
from stable_baselines3.common.base_class import BaseAlgorithm
import gymnasium as gym

class SB3AgentWrapper:
    """
    Thin wrapper for Stable Baselines 3 (SB3) agents.
    Provides a unified interface for model creation, training, prediction, saving, loading,
    and switching environments. This class does not implement RL logic itself; it delegates
    to the underlying SB3 model.
    """
    def __init__(self, model: BaseAlgorithm):
        """
        Initialize the agent wrapper with an SB3 model instance.
        Args:
            model: An instance of a Stable Baselines 3 RL algorithm (e.g., PPO, DQN).
        """
        self.model = model

    def learn(self, total_timesteps: int, **kwargs) -> None:
        """
        Train the model for a given number of timesteps.
        Args:
            total_timesteps: Number of timesteps to train for.
            **kwargs: Additional arguments for SB3's learn().
        """
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def predict(self, observation, deterministic: bool = False) -> Any:
        """
        Predict the action for a given observation.
        Args:
            observation: The observation from the environment.
            deterministic: Whether to use deterministic actions.
        Returns:
            The predicted action (and optionally, state info).
        """
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str) -> None:
        """
        Save the model to disk.
        Args:
            path: Path to save the model.
        """
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env: Optional[gym.Env] = None, **kwargs) -> 'SB3AgentWrapper':
        """
        Load a model from disk and return a new wrapper instance.
        Args:
            path: Path to load the model from.
            env: (Optional) Environment to bind to the loaded model.
            **kwargs: Additional arguments for SB3's load().
        Returns:
            SB3AgentWrapper instance with loaded model.
        """
        from stable_baselines3.common.base_class import BaseAlgorithm
        model = BaseAlgorithm.load(path, env=env, **kwargs)
        return cls(model)

    def set_env(self, env: gym.Env) -> None:
        """
        Set the environment for the model.
        Args:
            env: The Gymnasium environment.
        """
        self.model.set_env(env)

    def get_env(self) -> Optional[gym.Env]:
        """
        Get the environment currently bound to the model.
        Returns:
            The Gymnasium environment.
        """
        return self.model.get_env()

    def get_policy(self):
        """
        Get the policy object from the model.
        Returns:
            The SB3 policy object.
        """
        return self.model.policy

    @property
    def sb3_model(self) -> BaseAlgorithm:
        """
        Access the underlying SB3 model directly.
        Returns:
            The SB3 model instance.
        """
        return self.model

