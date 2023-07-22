#!/usr/bin/env bash

source pre_run.sh

omp_thread=16
actors=128
num_cpus=1

export CUDA_VISIBLE_DEVICES=0,1

./build/deep_cfr/run_sbescfr --use_regret_net=true --use_policy_net=true --use_tabular=false --num_gpus=0 \
--num_cpus=$num_cpus --actors=$actors --memory_size=1000000 --policy_memory_size=10000000 --cfr_batch_size=1000 \
--train_batch_size=128 --global_value_train_steps=2 --train_steps=2 --policy_train_steps=16 \
--inference_batch_size=$actors --inference_threads=$num_cpus --inference_cache=1000000 \
--omp_threads=$omp_thread --exp_evaluation_window=true --game=leduc_poker --evaluation_window=10 \
--average_type=Opponent --weight_type=Linear \
--checkpoint_freq=1000000 --max_steps=10000000 --graph_def= --verbose=false --cfr_rm_scale=0.001 --suffix=leduc_poker_osomdb_0.001_2216uawl --nfsp_eta=0.4 
