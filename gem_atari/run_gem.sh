#!/usr/bin/env bash

declare alg="ddq6"
declare -a gpus=(6 7)
declare -a envs=("Zaxxon" "Frostbite")

#declare -a gpus=(0 1 2 3 4 5)
#declare -a envs=("Alien" "Frostbite" "Zaxxon" "BattleZone" "WizardOfWor" "MsPacman")
echo $alg

for ((i = 0; i < ${#gpus[@]}; i++)); do
  for ((seed = 0; seed < 2; seed++)); do
    # shellcheck disable=SC2086
    echo "Env:${envs[$i]},Seed:$seed,GPU:${gpus[$i]}"
    tag="${alg}_${envs[$i]}_exploration_10_meta_q_loss_lr=2e-4_$seed"
    echo "$tag"
    DDQ_LOGDIR=$HOME/log_gem/atari/ CUDA_VISIBLE_DEVICES=${gpus[$i]} nohup python train.py --tag=$tag --env="${envs[$i]}" --alg=$alg --learn=atari_ddq --save_rews=True --buffer=episodic --save_Q=True --beta=-1 --test_eps=0.05 --eps_r=0.1 --eps_decay=1000000 --q_lr=2e-4 > ./logs/$tag.out &
  done
#  sleep 3
done