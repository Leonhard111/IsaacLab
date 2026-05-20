# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import NetworkBuilder


class ForgeCustomActorNet(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)

        self.actions_num = kwargs.pop("actions_num")
        self.input_shape = kwargs.pop("input_shape")
        self.value_size = kwargs.pop("value_size", 1)
        self.num_seqs = kwargs.pop("num_seqs", 1)

        self.central_value = params.get("central_value", False)

        self.frame_stack_size = 4
        self.frame_channels = self.input_shape[-1] // self.frame_stack_size
        if self.input_shape[-1] % self.frame_stack_size != 0:
            raise ValueError(f"Expected the channel dimension to be divisible by {self.frame_stack_size}, got {self.input_shape[-1]}")

        self.cnn_channels = 256
        self.rnn_units = 256
        self.rnn_layers = 2

        self.conv1 = nn.Conv2d(in_channels=self.frame_channels, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=self.cnn_channels, kernel_size=3, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_projection = nn.Linear(self.cnn_channels, self.cnn_channels)

        self.actor_rnn = nn.LSTM(input_size=self.cnn_channels, hidden_size=self.rnn_units, num_layers=self.rnn_layers, batch_first=True)
        self.actor_mlp = nn.Sequential(
            nn.Linear(self.rnn_units, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
        )

        self.mu = nn.Linear(256, self.actions_num)
        self.sigma = nn.Parameter(torch.zeros(self.actions_num, dtype=torch.float32), requires_grad=True)
        self.value = nn.Linear(256, self.value_size)

        self._conv_activation = nn.ReLU()

    def _split_tactile_frames(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 4:
            raise ValueError(f"Expected a 4D tactile observation, got shape {tuple(obs.shape)}")

        # rl_games passes tactile images as NHWC; split the channel axis into 4 temporal frames.
        obs = obs.permute(0, 3, 1, 2).contiguous()
        batch_size, channels, height, width = obs.shape

        left_right_channels = self.frame_stack_size * self.frame_channels
        if channels != left_right_channels:
            raise ValueError(f"Expected {left_right_channels} tactile channels, got {channels}")

        left_frames = obs[:, : left_right_channels // 2].reshape(batch_size, self.frame_stack_size, self.frame_channels // 2, height, width)
        right_frames = obs[:, left_right_channels // 2 :].reshape(batch_size, self.frame_stack_size, self.frame_channels // 2, height, width)
        return torch.cat((left_frames, right_frames), dim=2)

    def _encode_frame(self, frame_obs: torch.Tensor) -> torch.Tensor:
        x = self._conv_activation(self.conv1(frame_obs))
        x = self._conv_activation(self.conv2(x))
        x = self._conv_activation(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self._conv_activation(self.cnn_projection(x))
        return x

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        frame_obs = self._split_tactile_frames(obs)
        batch_size, frame_count, frame_channels, height, width = frame_obs.shape
        frame_features = self._encode_frame(frame_obs.reshape(batch_size * frame_count, frame_channels, height, width))
        return frame_features.reshape(batch_size, frame_count, -1)

    def forward(self, obs_dict):
        obs = obs_dict["obs"]

        if not isinstance(obs, torch.Tensor):
            raise TypeError(f"Expected obs to be a tensor, got {type(obs)}")

        frame_features = self._encode_obs(obs)
        rnn_out, _ = self.actor_rnn(frame_features)
        policy_features = self.actor_mlp(rnn_out[:, -1])

        mu = self.mu(policy_features)
        logstd = self.sigma.unsqueeze(0).expand_as(mu)
        value = self.value(policy_features)

        if self.central_value:
            return mu, logstd, value, None

        return mu, logstd, value, None


class ForgeCustomActorBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return ForgeCustomActorNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)


model_builder.register_network("forge_custom_actor", ForgeCustomActorBuilder)