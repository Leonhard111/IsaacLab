# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Diagnostic test for TacSL tactile sensor configuration and environment matching."""

import argparse
import os
import random
import sys
import traceback
from typing import Any, cast

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Diagnose TacSL tactile sensor cfg matching in a Forge environment.")
parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-Direct-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--replicate_physics",
    type=str,
    default=None,
    choices=("true", "false"),
    help="Override scene replicate_physics. Leave unset to use the task default.",
)
parser.add_argument(
    "--clone_in_fabric",
    type=str,
    default=None,
    choices=("true", "false"),
    help="Override scene clone_in_fabric. Leave unset to use the task default.",
)
parser.add_argument(
    "--skip_reset",
    action="store_true",
    help="Create the environment and inspect sensors without calling env.reset().",
)
parser.add_argument(
    "--strict",
    action="store_true",
    help="Fail the test if a tactile sensor does not report the same number of instances as the scene.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# TacSL requires rendering even if we only inspect sensor metadata.
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401


def _describe_sensor(sensor: Any, scene_num_envs: int, sensor_name: str) -> None:
    print(f"[INFO] {sensor_name}.cfg.prim_path = {sensor.cfg.prim_path}")
    print(f"[INFO] {sensor_name}.cfg.contact_object_prim_path_expr = {sensor.cfg.contact_object_prim_path_expr}")
    print(f"[INFO] {sensor_name}.is_initialized = {sensor.is_initialized}")
    print(f"[INFO] {sensor_name}.num_instances = {sensor.num_instances}")
    print(f"[INFO] scene.num_envs = {scene_num_envs}")

    parent_prims = getattr(sensor, "_parent_prims", None)
    if parent_prims is not None:
        parent_paths = [str(prim.GetPath()) for prim in parent_prims]
        print(f"[INFO] {sensor_name} parent prims ({len(parent_paths)}):")
        for prim_path in parent_paths:
            print(f"[PRIM] {prim_path}")

    parent_pattern = sensor.cfg.prim_path.rsplit("/", 1)[0]
    matched_parent_paths = sim_utils.find_matching_prim_paths(parent_pattern)
    print(f"[INFO] matched parent prims for pattern {parent_pattern}: {len(matched_parent_paths)}")
    for prim_path in matched_parent_paths:
        print(f"[MATCHED-PARENT] {prim_path}")

    if sensor.cfg.contact_object_prim_path_expr is not None:
        matched_contact_paths = sim_utils.find_matching_prim_paths(sensor.cfg.contact_object_prim_path_expr)
        print(
            "[INFO] matched contact prims for pattern "
            f"{sensor.cfg.contact_object_prim_path_expr}: {len(matched_contact_paths)}"
        )
        for prim_path in matched_contact_paths:
            print(f"[MATCHED-CONTACT] {prim_path}")

    if sensor.num_instances != scene_num_envs:
        message = (
            f"{sensor_name} num_instances mismatch: sensor={sensor.num_instances}, scene={scene_num_envs}. "
            f"This usually means the sensor prim path matched only a subset of cloned envs, or the sensor "
            f"was initialized before all envs were visible in the stage."
        )
        if args_cli.strict:
            raise RuntimeError(message)
        print(f"[WARN] {message}")


def _describe_scene(env_unwrapped: Any) -> None:
    print(f"[INFO] requested num_envs = {args_cli.num_envs}")
    print(f"[INFO] configured scene.num_envs = {env_unwrapped.scene.num_envs}")
    print(f"[INFO] scene replicate_physics = {env_unwrapped.scene.cfg.replicate_physics}")
    print(f"[INFO] scene clone_in_fabric = {env_unwrapped.scene.cfg.clone_in_fabric}")


def _apply_scene_overrides(env_cfg: Any) -> None:
    if args_cli.replicate_physics is not None:
        env_cfg.scene.replicate_physics = args_cli.replicate_physics == "true"
    if args_cli.clone_in_fabric is not None:
        env_cfg.scene.clone_in_fabric = args_cli.clone_in_fabric == "true"


def _inspect_sensors(env_unwrapped: Any) -> None:
    if not hasattr(env_unwrapped.scene, "sensors"):
        print("[WARN] scene has no sensors attribute")
        return

    for sensor_name in ("tactile_sensor_left", "tactile_sensor_right"):
        if sensor_name in env_unwrapped.scene.sensors:
            _describe_sensor(env_unwrapped.scene[sensor_name], env_unwrapped.scene.num_envs, sensor_name)
        else:
            print(f"[WARN] {sensor_name} not found in scene.sensors")


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _apply_scene_overrides(env_cfg)

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    env_cfg.seed = args_cli.seed

    env_cfg.log_dir = os.path.abspath(os.path.join("logs", "sensor_cfg_test", args_cli.task))

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(cast(DirectMARLEnv, env.unwrapped))

    env_unwrapped = cast(Any, env.unwrapped)

    _describe_scene(env_unwrapped)

    if args_cli.skip_reset:
        print("[INFO] skipping env.reset() by request")
        _inspect_sensors(env_unwrapped)
        env.close()
        return

    try:
        obs, _ = env.reset(seed=args_cli.seed)
    except Exception:
        print("[ERROR] env.reset failed; dumping sensor state before re-raising")
        _inspect_sensors(env_unwrapped)
        traceback.print_exc()
        raise

    print("[INFO] env.reset completed")
    if isinstance(obs, dict):
        print(f"[INFO] observation keys = {list(obs.keys())}")
        print(f"[INFO] policy observation shape = {getattr(obs['policy'], 'shape', None)}")
    else:
        print(f"[INFO] observation shape = {getattr(obs, 'shape', None)}")

    _inspect_sensors(env_unwrapped)

    env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()