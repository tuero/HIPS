"""
Implements the different neural networks for Custom envs
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from .blocks import FilmResBlock, GroupOfBlocks, GroupOfResidualBlocks, ResNetBlock
from .layers import MLP, ResidualBlock, ResidualHead, conv1x1
from .networks import FilmDecoder, VQEmbedding, weights_init

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ObservationShape:
    c: int
    h: int
    w: int

    def flat_size(self) -> int:
        return self.c * self.h * self.w

    def to_list(self) -> list[int]:
        return [self.c, self.h, self.w]

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.c, self.h, self.w)


# Typedef for observations
Observation = NDArray[np.float32]


class PolicyConvNet(nn.Module):
    def __init__(
        self,
        obs_shape: ObservationShape,
        num_actions: int,
        resnet_channels: int,
        resnet_blocks: int,
        policy_channels: int,
        policy_mlp_layers: list[int],
        use_batchnorm: bool,
    ):
        """A Residual conv network with policy."""
        super(PolicyConvNet, self).__init__()
        self.resnet_head = ResidualHead(obs_shape.c, resnet_channels, use_batchnorm)
        self.resnet_body = nn.Sequential(
            *[
                ResidualBlock(resnet_channels, use_batchnorm)
                for _ in range(resnet_blocks)
            ]
        )
        self.policy1x1 = conv1x1(resnet_channels, policy_channels)
        self.policy_mlp_input_size = policy_channels * obs_shape.h * obs_shape.w
        self.policy_mlp = MLP(
            self.policy_mlp_input_size, policy_mlp_layers, num_actions
        )

    def forward(self, x: torch.Tensor):
        output = self.resnet_head(x)
        output = self.resnet_body(output)
        pol = self.policy1x1(output)
        pol = pol.reshape(-1, self.policy_mlp_input_size)
        pol = self.policy_mlp(pol).squeeze(dim=1)
        return pol


class CustomDetectorNetwork(nn.Module):
    """
    Detector for Custom Environments
    """

    def __init__(self, obs_shape: ObservationShape, norm=nn.BatchNorm2d):
        super().__init__()
        self.in_block = nn.Sequential(
            ResNetBlock(
                in_channels=obs_shape.c * 2,
                out_channels=32,
                kernel_size=5,
                padding=2,
                norm=norm,
            ),
            ResNetBlock(
                in_channels=32, out_channels=32, kernel_size=3, padding=1, norm=norm
            ),
            ResNetBlock(
                in_channels=32, out_channels=32, kernel_size=3, padding=1, norm=norm
            ),
            ResNetBlock(
                in_channels=32, out_channels=32, kernel_size=3, padding=1, norm=norm
            ),
            ResNetBlock(
                in_channels=32, out_channels=8, kernel_size=3, padding=1, norm=norm
            ),
        )

        self._obs_shape = obs_shape
        self.converter = nn.Linear(self._obs_shape.h * self._obs_shape.w * 8, 128)
        self.policy_head = nn.Linear(128, 1)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, compute_val=True):
        x = self.in_block(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.converter(x))

        pol = self.policy_head(x).squeeze(dim=1)
        if compute_val:
            val = self.value_head(x)
            return pol, val
        else:
            return pol


class CustomPolicy(nn.Module):
    def __init__(
        self,
        obs_shape: ObservationShape,
        num_actions: int,
        resnet_channels: int,
        resnet_blocks: int,
        policy_channels: int,
        policy_mlp_layers: list[int],
        use_batchnorm: bool,
    ):
        super(CustomPolicy, self).__init__()
        # Policy channels get doubled
        obs_shape = ObservationShape(obs_shape.c * 2, obs_shape.h, obs_shape.w)
        self._net = PolicyConvNet(
            obs_shape,
            num_actions,
            resnet_channels,
            resnet_blocks,
            policy_channels,
            policy_mlp_layers,
            use_batchnorm,
        )

    def forward(self, x):
        return self._net.forward(x)


class CustomModel(nn.Module):
    """
    Dynamics model for Custom environment
    """

    def __init__(self, dim, channels, out_dim, expand=False):
        super().__init__()
        self.act_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
        )

        self.obs_block = nn.Sequential(
            GroupOfBlocks(channels, dim, 2, norm=nn.InstanceNorm2d),
            GroupOfBlocks(dim, dim, 2, norm=nn.InstanceNorm2d),
        )

        self.trunk = nn.ModuleList(
            [
                FilmResBlock(dim, dim, 8, norm=nn.InstanceNorm2d),
            ]
        )

        self.out_block = nn.Sequential(
            GroupOfBlocks(dim, 3 * dim if expand else dim, 2, norm=nn.InstanceNorm2d),
            nn.Conv2d(
                3 * dim if expand else dim, channels * out_dim, kernel_size=1, padding=0
            ),
        )

        self.out_dim = out_dim
        self.channels = channels

    def forward(self, obs, acts):
        acts = self.act_mlp(acts[:, None])
        obs = self.obs_block(obs)
        for b in self.trunk:
            obs = b(obs, acts)
        out = self.out_block(obs)
        out = out.view(
            out.shape[0], self.channels, self.out_dim, out.shape[2], out.shape[3]
        ).permute(0, 1, 3, 4, 2)
        return out


class CustomPrior(nn.Module):
    """
    Prior for custom environments, both high-level and low-level (unconditional BC-policy)
    """

    def __init__(self, dim, K, n_acts, obs_shape: ObservationShape):
        super().__init__()

        input_nc = obs_shape.c
        if obs_shape.h == 14:
            flat_size = 2048
        elif obs_shape.h == 20:
            flat_size = 3200

        self.in_block = nn.Sequential(
            nn.Conv2d(input_nc, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            GroupOfBlocks(dim, dim, 4, stride=2, norm=nn.InstanceNorm2d),
            GroupOfBlocks(dim, dim, 4, stride=2, norm=nn.InstanceNorm2d),
            nn.Flatten(1, 3),
        )

        self.code_block = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, K),
        )

        self.act_block = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_acts),
        )

    def forward(self, obs):
        obs = self.in_block(obs)
        out_c = self.code_block(obs)
        out_a = self.act_block(obs)
        return out_c, out_a


class VQVAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        resnet_channels: int,
        blocks_per_group: int,
        square_shape: int,
    ):
        super(VQVAEEncoder, self).__init__()
        self.head = ResidualHead(in_channels, resnet_channels, False)
        body: list[nn.Module] = [
            GroupOfResidualBlocks(resnet_channels, blocks_per_group, False)
        ]
        while square_shape > 1:
            square_shape = int(square_shape / 2)
            body.append(
                nn.Conv2d(resnet_channels, resnet_channels, 4, stride=2, padding=1)
            )
            body.append(nn.ReLU())
            body.append(GroupOfResidualBlocks(resnet_channels, blocks_per_group, False))
        self.body = nn.Sequential(*body)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.head(obs)
        x = self.body(x)
        return x


class CustomVQVAE(nn.Module):
    """
    VQVAE for custom environments
    """

    # Complete VQVAE model
    def __init__(self, dim, K, obs_shape: ObservationShape, output_dim=256):
        super().__init__()
        input_nc = obs_shape.c
        self.encoder = VQVAEEncoder(obs_shape.c * 2, dim, 2, obs_shape.w)
        self.codebook = VQEmbedding(K, dim)
        self.decoder = FilmDecoder(dim, input_nc, output_dim)
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, decoder_input, latent_idxs):
        # Generate subgoals given previous state and discrete codes
        z_q_x = self.codebook.embedding(latent_idxs)[:, :, None, None]
        x_tilde = self.decoder(decoder_input, z_q_x)
        return x_tilde

    def reinit_codes(self, D=None, K=None):
        self.codebook.reinit_codes(D, K)

    def forward(self, encoder_input, decoder_input, continuous=False):
        z_e_x = self.encoder(encoder_input)
        if continuous:
            z_q_x_st, z_q_x, n_codes, codes = z_e_x, z_e_x, 1, []
        else:
            z_q_x_st, z_q_x, n_codes, codes = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(decoder_input, z_q_x_st)
        return x_tilde, z_e_x, z_q_x, n_codes, codes
