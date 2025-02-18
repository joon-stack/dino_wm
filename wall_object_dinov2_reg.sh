#!/bin/bash

#SBATCH --job-name=wall_obj_reg                 # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:2                         # Using 1 gpu
#SBATCH --time=0-48:00:00                     # 1 hour timelimit
#SBATCH --mem=80G                         # Using 10GB CPU Memory
#SBATCH --partition=laal_a6000                        # Using "b" partition 
#SBATCH --cpus-per-task=8                     # Using 4 maximum processor

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate dino_wm
export DATASET_DIR=/home/shared/robotics
export HYDRA_FULL_ERROR=1
accelerate launch --main_process_port 29504 train.py --config-name train_object.yaml env=wall_object_dinov2_reg_small frameskip=5 num_hist=1 encoder=dino_reg training.save_every_x_epoch=10
