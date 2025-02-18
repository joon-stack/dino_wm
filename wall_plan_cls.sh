#!/bin/bash

#SBATCH --job-name=wall_plan_cls                 # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                         # Using 1 gpu
#SBATCH --time=0-48:00:00                     # 1 hour timelimit
#SBATCH --mem=40G                         # Using 10GB CPU Memory
#SBATCH --partition=laal_a6000                        # Using "b" partition 
#SBATCH --cpus-per-task=4                     # Using 4 maximum processor

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate dino_wm
export DATASET_DIR=/home/shared/robotics
export HYDRA_FULL_ERROR=1
source ~/.bashrc
# python plan.py model_name=2025-01-25/02-51-33 n_evals=25 planner=cem goal_H=5 goal_source='random_state' seed=1004
# python plan.py model_name=2025-01-25/02-51-33 n_evals=25 planner=cem goal_H=5 goal_source='random_state'  seed=99
python plan.py model_name=2025-01-22/15-29-53 n_evals=10 planner=mpc_cem goal_H=5 goal_source='random_state' seed=1 
python plan.py model_name=2025-01-22/15-29-53 n_evals=10 planner=mpc_cem goal_H=5 goal_source='random_state' seed=2
python plan.py model_name=2025-01-22/15-29-53 n_evals=10 planner=mpc_cem goal_H=5 goal_source='random_state' seed=3 
python plan.py model_name=2025-01-22/15-29-53 n_evals=10 planner=mpc_cem goal_H=5 goal_source='random_state' seed=4
python plan.py model_name=2025-01-22/15-29-53 n_evals=10 planner=mpc_cem goal_H=5 goal_source='random_state' seed=5

# python plan.py model_name=2025-01-22/15-29-53 n_evals=25 planner=cem goal_H=5 goal_source='random_state'  seed=1004
# python plan.py model_name=2025-01-22/15-29-53 n_evals=25 planner=cem goal_H=5 goal_source='random_state'  seed=99
# python plan.py model_name=2025-01-22/15-29-53 n_evals=25 planner=mpc_cem goal_H=5 goal_source='random_state'  seed=1004
# python plan.py model_name=2025-01-22/15-29-53 n_evals=25 planner=mpc_cem goal_H=5 goal_source='random_state'  seed=99



