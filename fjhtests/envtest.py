# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#python tests/envtest.py --task=Isaac-Forge-PegInsert-Direct-v0 --num_envs=1 --steps=200 --action_mode=random
#python fjhtests/envtest.py --task=Isaac-Forge-PegInsert-Direct-v0 --num_envs=1 --steps=400 --action_mode=zero --save_viz
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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

import cv2


# add argparse arguments
parser = argparse.ArgumentParser(description="Smoke test a Forge environment with a simple control loop.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during the run.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps to run after reset.")
parser.add_argument("--save_viz", action="store_true", help="Save tactile visualization images during the run.")
parser.add_argument("--save_viz_dir", type=str, default="tactile_record", help="Directory to save tactile images.")
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
# TacSL uses a camera-based tactile sensor, so rendering must be enabled even when not recording video.
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import compute_tactile_shear_image
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401




def save_viz_helper(
    dir_path_list: tuple[str, str],
    count: int,
    tactile_data: VisuoTactileSensorData,
    num_envs: int,
    nrows: int,
    ncols: int,
):
    """Save visualization of tactile sensor data.

    Args:
        dir_path_list: A tuple containing paths to the force field directory and RGB image directory.
        count: The current simulation step count, used for naming saved files.
        tactile_data: The data object containing tactile sensor readings (forces, images).
        num_envs: Number of environments in the simulation.
        nrows: Number of rows in the tactile array.
        ncols: Number of columns in the tactile array.
    """
    # Only save the first 2 environments

    tactile_force_field_dir, tactile_rgb_image_dir = dir_path_list

    if tactile_data.tactile_shear_force is not None and tactile_data.tactile_normal_force is not None:
        # visualize tactile forces
        tactile_normal_force = tactile_data.tactile_normal_force.view((num_envs, nrows, ncols))
        tactile_shear_force = tactile_data.tactile_shear_force.view((num_envs, nrows, ncols, 2))

        tactile_image = compute_tactile_shear_image(
            tactile_normal_force[0, :, :].detach().cpu().numpy(), tactile_shear_force[0, :, :].detach().cpu().numpy()
        )

        if tactile_normal_force.shape[0] > 1:
            tactile_image_1 = compute_tactile_shear_image(
                tactile_normal_force[1, :, :].detach().cpu().numpy(),
                tactile_shear_force[1, :, :].detach().cpu().numpy(),
            )
            combined_image = np.vstack([tactile_image, tactile_image_1])
            cv2.imwrite(
                os.path.join(tactile_force_field_dir, f"{count:04d}.png"), (combined_image * 255).astype(np.uint8)
            )
        else:
            cv2.imwrite(
                os.path.join(tactile_force_field_dir, f"{count:04d}.png"), (tactile_image * 255).astype(np.uint8)
            )

    if tactile_data.tactile_rgb_image is not None:
        tactile_rgb_data = tactile_data.tactile_rgb_image.cpu().numpy()
        tactile_rgb_data = np.transpose(tactile_rgb_data, axes=(0, 2, 1, 3))
        # print(f"tactile_rgb_data shape:{tactile_rgb_data.shape}")
        # print(f"tactile_rgb_data type:{tactile_rgb_data.dtype}")
        tactile_rgb_data_first_2 = tactile_rgb_data[:2] if len(tactile_rgb_data) >= 2 else tactile_rgb_data
        tactile_rgb_tiled = np.concatenate(tactile_rgb_data_first_2, axis=0)
        # print(tactile_rgb_tiled.max() <= 1.0)
        # Convert to uint8 if not already
        if tactile_rgb_tiled.dtype != np.uint8:
            tactile_rgb_tiled = (
                (tactile_rgb_tiled * 255).astype(np.uint8)
                if tactile_rgb_tiled.max() <= 1.0
                else tactile_rgb_tiled.astype(np.uint8)
            )
        cv2.imwrite(os.path.join(tactile_rgb_image_dir, f"{count:04d}.png"), tactile_rgb_tiled)




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


def mkdir_helper(dir_path: str) -> tuple[str, str]:
    """Create output folders for tactile visualizations."""
    tactile_img_folder = dir_path
    os.makedirs(tactile_img_folder, exist_ok=True)

    tactile_force_field_dir = os.path.join(tactile_img_folder, "tactile_force_field")
    os.makedirs(tactile_force_field_dir, exist_ok=True)

    tactile_rgb_image_dir = os.path.join(tactile_img_folder, "tactile_rgb_image")
    os.makedirs(tactile_rgb_image_dir, exist_ok=True)

    return tactile_force_field_dir, tactile_rgb_image_dir


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
        print(f"[INFO] Observation keys shape: {getattr(obs['policy'], 'shape', None)}")
    else:
        print(f"[INFO] Observation shape: {getattr(obs, 'shape', None)}")
    env_unwrapped.scene["tactile_sensor_left"].get_initial_render()
    tactile_data = env_unwrapped.scene["tactile_sensor_left"].data
    print(type(tactile_data).__name__)

    if args_cli.save_viz:
        print(f"[INFO] Saving tactile visualizations to: {args_cli.save_viz_dir}")
        dir_path_list = mkdir_helper(args_cli.save_viz_dir)

    # while(1):
    #     pass
    for step_index in range(args_cli.steps):
        action = _make_action(env, step_index)
        obs, reward, terminated, truncated, _ = env.step(action)

        if step_index % 10 == 0:
            print(f"[INFO] step={step_index:04d}, reward={reward}, terminated={terminated}, truncated={truncated}")

        if torch.as_tensor(terminated).any().item() or torch.as_tensor(truncated).any().item():
            print(f"[INFO] Episode ended at step={step_index}, resetting environment.")
            obs, _ = env.reset()

        if args_cli.save_viz:
            tactile_data = env_unwrapped.scene["tactile_sensor_left"].data
            nrows = env_unwrapped.scene["tactile_sensor_left"].cfg.tactile_array_size[0]
            ncols = env_unwrapped.scene["tactile_sensor_left"].cfg.tactile_array_size[1]
            save_viz_helper(dir_path_list, step_index, tactile_data, args_cli.num_envs, nrows, ncols)

    env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()


