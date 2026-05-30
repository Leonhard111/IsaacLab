#!/usr/bin/env bash
set -e

# Visualize a trained RL-Games policy for Forge.
#
# Usage:
#   bash fjhtests/play.sh
#   bash fjhtests/play.sh /abs/path/to/checkpoint.pth
#   NUM_ENVS=4 DEVICE=cuda:1 bash fjhtests/play.sh

TASK="${TASK:-Isaac-Forge-PegInsert-Direct-v0}"
NUM_ENVS="${NUM_ENVS:-1}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT="${1:-logs/rl_games/Forge/test/nn/last_Forge_ep_200_rew_23.87163.pth}"

python scripts/reinforcement_learning/rl_games/play.py \
  --task="${TASK}" \
  --num_envs="${NUM_ENVS}" \
  --device="${DEVICE}" \
  --checkpoint="${CHECKPOINT}" \
  --real-time
