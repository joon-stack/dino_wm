#!/bin/bash

#SBATCH --job-name=maze_full                 # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:2                         # Using 1 gpu
#SBATCH --time=0-72:00:00                     # 1 hour timelimit
#SBATCH --mem=80GB                        # Using 10GB CPU Memory
#SBATCH --partition=laal_3090                        # Using "b" partition 
#SBATCH --cpus-per-task=8                     # Using 4 maximum processor

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate dino_wm
export DATASET_DIR=/home/shared/robotics
export HYDRA_FULL_ERROR=1
accelerate launch --main_process_port 29523 train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3 encoder=dinov1 training.save_every_x_epoch=10 has_decoder=True model.train_decoder=True training.epochs=20 
