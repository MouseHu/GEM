#!/usr/bin/env bash

declare algo_alias="gem+tbp"
declare algo="GEM"

declare root="$HOME/gem_icml2021/gem_mujoco/"
declare max_steps=1000


#declare -a gpus=(0 1 2)
#declare -a envs=("Hopper-v2" "Walker2d-v2" "Humanoid-v2")
#declare -a envs_alias=("hopper" "walker" "humanoid")

declare -a gpus=(0)
declare -a envs=("Ant-v2")
declare -a envs_alias=("ant")

export PYTHONPATH=$root

for ((i = 0; i < ${#gpus[@]}; i++)); do
  for ((seed = 0; seed < 5; seed++)); do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${gpus[$i]} OPENAI_LOGDIR=$HOME/log_gem/mujoco/$algo_alias nohup python $root/run/train.py --max_steps=$max_steps --agent=$algo --comment="${envs_alias[$i]}"_${algo_alias}_$1_"$seed" --env-id="${envs[$i]}" >./logs/"${envs_alias[$i]}"_${algo_alias}_$1_"$seed".out &
  done
  sleep 1
done
