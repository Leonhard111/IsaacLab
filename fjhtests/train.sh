#!/usr/bin/env bash
set -e

# export CUDA_VISIBLE_DEVICES=3

python scripts/reinforcement_learning/rl_games/train.py \
  --task=Isaac-Forge-PegInsert-Direct-v0 \
  --num_envs=128 \
  --device=cuda:0 \
  --headless