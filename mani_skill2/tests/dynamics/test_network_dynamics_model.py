import pytest
import torch
import torch.nn as nn

from mani_skill2.dynamics.network_dynamics_model import NetworkDynamicsModel


class MockFCNetwork(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.layer = nn.Linear(obs_size + act_size, obs_size)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat((obs, act), dim=-1)
        x = self.layer(x)
        return x


@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("obs_size", (1, 3))
@pytest.mark.parametrize("act_size", (1, 2))
@pytest.mark.parametrize("is_residual", (True, False))
def test_network_dynamics_model_step(batch_size, obs_size, act_size, is_residual):
    net = MockFCNetwork(obs_size, act_size)
    dyn = NetworkDynamicsModel(
        net,
        is_residual=is_residual,
    )

    obs = torch.zeros((batch_size, obs_size))
    act = torch.zeros((batch_size, act_size))

    next_obs, info = dyn.step(obs, act)
    assert next_obs.shape == (batch_size, obs_size)
