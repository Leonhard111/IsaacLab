# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#python tests/envtest.py --task=Isaac-Forge-PegInsert-Direct-v0 --num_envs=1 --steps=200 --action_mode=random

# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for loading and stepping a Forge direct RL environment."""

"""Launch Isaac Sim Simulator first."""
import argparse
import os
import random
import sys
from typing import Any, cast

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Smoke test a Forge environment with a simple control loop.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during the run.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps to run after reset.")
parser.add_argument(
    "--action_mode",
    type=str,
    default="zero",
    choices=("zero", "nudge", "random"),
    help="Action pattern to send to the environment.",
)
parser.add_argument("--nudge_scale", type=float, default=0.05, help="Scale for the nudge action pattern.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401


def _make_action(env, step_index: int):
    if args_cli.action_mode == "zero":
        return torch.zeros(env.action_space.shape, dtype=torch.float32)
    if args_cli.action_mode == "random":
        return torch.as_tensor(env.action_space.sample(), dtype=torch.float32)

    action = torch.zeros(env.action_space.shape, dtype=torch.float32)
    action[0] = args_cli.nudge_scale * np.sin(step_index * 0.1)
    return action


def _print_stage_prims(stage: Any, keywords: tuple[str, ...] = ("Robot", "panda", "centered")):
    """Print all prim paths and a filtered subset for quick inspection."""
    prim_paths = []
    for prim in stage.Traverse():
        prim_paths.append(str(prim.GetPath()))

    print("[INFO] Stage prim paths:")
    for prim_path in prim_paths:
        print(f"[PRIM] {prim_path}")

    filtered_paths = [prim_path for prim_path in prim_paths if any(keyword in prim_path for keyword in keywords)]
    print(f"[INFO] Filtered prim paths ({', '.join(keywords)}):")
    for prim_path in filtered_paths:
        print(f"[PRIM-FILTERED] {prim_path}")


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg: dict):
    """Load the environment and execute a short control loop."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    env_cfg.seed = args_cli.seed

    env_cfg.log_dir = os.path.abspath(os.path.join("logs", "envtest", args_cli.task))

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(cast(DirectMARLEnv, env.unwrapped))

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(env_cfg.log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env_unwrapped = cast(Any, env.unwrapped)
    # _print_stage_prims(env_unwrapped.scene.stage)

    obs, _ = env.reset(seed=args_cli.seed)
    print("env.reset test pass !")
    
    if isinstance(obs, dict):
        print(f"[INFO] Observation keys: {list(obs.keys())}")
    else:
        print(f"[INFO] Observation shape: {getattr(obs, 'shape', None)}")

    # while(1):
    #     pass
    for step_index in range(args_cli.steps):
        action = _make_action(env, step_index)
        obs, reward, terminated, truncated, _ = env.step(action)

        if step_index % 10 == 0:
            print(f"[INFO] step={step_index:04d}, reward={reward}, terminated={terminated}, truncated={truncated}")

        if torch.any(terminated).item() or torch.any(truncated).item():
            print(f"[INFO] Episode ended at step={step_index}, resetting environment.")
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()
