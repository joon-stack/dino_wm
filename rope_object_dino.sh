#!/bin/bash

#SBATCH --job-name=rope_6_1                 # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:2                         # Using 1 gpu
#SBATCH --time=0-48:00:00                     # 1 hour timelimit
#SBATCH --mem=80GB                         # Using 10GB CPU Memory
#SBATCH --partition=laal_a6000                        # Using "b" partition 
#SBATCH --cpus-per-task=8                     # Using 4 maximum processor

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate dino_wm
export DATASET_DIR=/home/shared/robotics
export HYDRA_FULL_ERROR=1
accelerate launch --main_process_port 29532  train.py  --config-name train_object.yaml env=deformable_env_dino_small frameskip=1 num_hist=1 encoder=dinov1 training.save_every_x_epoch=10 env.dataset.object_name=rope num_clusters=6 num_features=1
