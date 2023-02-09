from typing import Dict, Tuple

import torch
import torch.nn as nn

from mani_skill2.dynamics.base import DynamicsModel


class NetworkDynamicsModel(DynamicsModel):
    """A dynamics model that uses an underlying network."""

    def __init__(self, network: nn.Module, is_residual: bool):
        """
        Args:
            network: The network used to generate the next state or a residual of it.
            is_residual: If True, the network output is interpreted as a residual.
        """
        super().__init__()
        self.network = network
        self.is_residual = is_residual

    def step(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if self.is_residual:
            pred_residual = self.network(obs, act)
            pred_next_obs = obs + pred_residual
        else:
            pred_next_obs = self.network(obs, act)
            pred_residual = pred_next_obs - obs

        info = dict(pred_residual=pred_residual)
        return pred_next_obs, info
