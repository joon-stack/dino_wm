#!/bin/bash

#SBATCH --job-name=pusht_cls                # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:4                         # Using 1 gpu
#SBATCH --time=5-00:00:00                     # 1 hour timelimit
#SBATCH --mem=80GB                        # Using 10GB CPU Memory
#SBATCH --partition=laal_3090                        # Using "b" partition 
#SBATCH --cpus-per-task=16                     # Using 4 maximum processor

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate dino_wm
export DATASET_DIR=/home/shared/robotics
export HYDRA_FULL_ERROR=1
accelerate launch --main_process_port 29502 train.py --config-name train.yaml env=pusht frameskip=5 num_hist=3 encoder=dino_cls training.save_every_x_epoch=null training.save_every_x_steps=1000
