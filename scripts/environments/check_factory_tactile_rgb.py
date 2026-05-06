# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test the PegInsert tactile RGB pipeline and save tactile images to disk."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from typing import cast
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check tactile RGB observations in the Factory PegInsert environment.")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Direct-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=8, help="Number of environment steps before saving tactile RGB.")
parser.add_argument("--save_dir", type=str, default="outputs/factory_tactile_rgb", help="Directory to save images.")
parser.add_argument(
    "--robot_usd",
    type=str,
    default=None,
    help="Path to the tactile-enabled robot USD. It must contain elastomer and camera prims.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.enable_cameras:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import save_images_to_file
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.timer import Timer
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryEnvCfg
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = cast(
        FactoryEnvCfg,
        parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        ),
    )

    if args_cli.robot_usd is None:
        args_cli.robot_usd = f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_r15_finger/gelsight_r15_finger.usd"

    env_cfg.tactile.enable_sensor = True
    env_cfg.tactile.enable_camera_tactile = True
    env_cfg.tactile.enable_force_field = False
    env_cfg.tactile.debug_vis = False

    if args_cli.robot_usd is not None:
        env_cfg.task.robot_cfg.robot_usd = args_cli.robot_usd

    if not env_cfg.task.robot_cfg.robot_usd:
        raise ValueError(
            "A tactile-enabled robot USD is required. Pass it with --robot_usd so the environment can find "
            "the elastomer and camera prims."
        )

    os.makedirs(args_cli.save_dir, exist_ok=True)

    env = cast(FactoryEnv, gym.make(args_cli.task, cfg=env_cfg).unwrapped)

    print(f"[INFO]: Observation space: {env.observation_space}")
    print(f"[INFO]: Action space: {env.action_space}")
    print(f"[INFO]: Using robot USD: {env.cfg.task.robot_cfg.robot_usd}")
    print(f"[INFO]: Tactile sensor prim: {env.cfg.tactile.sensor_prim_path}")
    print(f"[INFO]: Tactile camera prim: {env.cfg.tactile.camera_prim_path}")

    env.reset()

    with torch.inference_mode():
        for _ in range(args_cli.num_steps):
            actions = torch.zeros((env.num_envs, cast(int, env.cfg.action_space)), device=env.device)
            env.step(actions)

        tactile_sensor = env.scene.sensors[env.cfg.tactile.sensor_name]
        tactile_rgb = tactile_sensor.data.tactile_rgb_image

        if tactile_rgb is None:
            raise RuntimeError("Tactile RGB image is None. Check the tactile camera prim and robot USD asset.")

        print(f"[INFO]: Tactile RGB tensor shape: {tuple(tactile_rgb.shape)}")

        raw_path = os.path.join(args_cli.save_dir, "tactile_rgb_raw.png")
        save_images_to_file(tactile_rgb, raw_path)
        print(f"[INFO]: Saved tactile RGB image grid to: {raw_path}")

        if env.num_envs > 0:
            first_env_path = os.path.join(args_cli.save_dir, "tactile_rgb_env0.png")
            save_images_to_file(tactile_rgb[:1], first_env_path)
            print(f"[INFO]: Saved first-environment tactile RGB image to: {first_env_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
