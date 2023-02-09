import abc
from typing import Any, Dict, Generic, Tuple, TypeVar

import torch.nn as nn

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class DynamicsModel(abc.ABC, Generic[ObsType, ActType], nn.Module):
    """Abstract base class for a dynamics model."""

    @abc.abstractmethod
    def step(self, obs: ObsType, act: ActType) -> Tuple[ObsType, Dict]:
        """Steps the dynamics forward one timestep.

        Args:
            obs: The observation at this timestep.
            act: The action at this timestep.

        Returns:
            A tuple containing the next observation and a
            dictionary of information about the step.
        """

    def reset(self) -> None:
        """Resets any internal state; called at the beginning of an episode."""
        pass
