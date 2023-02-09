import abc
from typing import Any, Dict, Tuple, TypeVar

import torch


class DynamicsModel(abc.ABC, torch.nn.Module):
    """Abstract base class for a dynamics model."""

    @abc.abstractmethod
    def step(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Steps the dynamics forward one timestep.

        Args:
            obs: The observation at this timestep.
                This is assumed be to batched, so shape (B, ...).
            act: The action at this timestep.
                Also assumed to be batched, shape (B, ...).

        Returns:
            A tuple containing the next observation and a
            dictionary of information about the step.
        """

    def reset(self) -> None:
        """Resets any internal state; called at the beginning of an episode."""
        pass
