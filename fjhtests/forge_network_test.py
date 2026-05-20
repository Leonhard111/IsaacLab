# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


parser = argparse.ArgumentParser(description="Smoke test for the Forge custom rl_games network.")
parser.add_argument("--batch_size", type=int, default=2, help="Dummy batch size used for the forward pass.")
parser.add_argument("--actions_num", type=int, default=7, help="Expected action dimension for Forge.")
parser.add_argument("--height", type=int, default=20, help="Tactile image height.")
parser.add_argument("--width", type=int, default=20, help="Tactile image width.")
parser.add_argument("--channels", type=int, default=24, help="Tactile channel count.")
args = parser.parse_args()


def _load_custom_network_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "direct" / "forge" / "agents" / "forge_custom_network.py"
    spec = importlib.util.spec_from_file_location("forge_custom_network", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_custom_network_module()

    from rl_games.algos_torch import model_builder

    assert "forge_custom_actor" in model_builder.NETWORK_REGISTRY, "forge_custom_actor was not registered"

    net = module.ForgeCustomActorNet(
        {},
        actions_num=args.actions_num,
        input_shape=(args.height, args.width, args.channels),
        value_size=1,
        num_seqs=args.batch_size,
    )

    obs = torch.randn(args.batch_size, args.height, args.width, args.channels)
    frame_obs = net._split_tactile_frames(obs)
    assert frame_obs.shape == (args.batch_size, 4, args.channels // 4, args.height, args.width), f"Unexpected frame shape: {frame_obs.shape}"

    mu, logstd, value, states = net({"obs": obs})
    assert mu.shape == (args.batch_size, args.actions_num), f"Unexpected mu shape: {mu.shape}"
    assert logstd.shape == (args.batch_size, args.actions_num), f"Unexpected logstd shape: {logstd.shape}"
    assert value.shape == (args.batch_size, 1), f"Unexpected value shape: {value.shape}"
    assert states is None, f"Expected states to be None, got {type(states)}"

    print("[OK] forge_custom_actor is registered")
    print(f"[OK] frame shape: {tuple(frame_obs.shape)}")
    print(f"[OK] mu shape: {tuple(mu.shape)}")
    print(f"[OK] logstd shape: {tuple(logstd.shape)}")
    print(f"[OK] value shape: {tuple(value.shape)}")


if __name__ == "__main__":
    main()