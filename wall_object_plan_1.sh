#!/bin/bash

#SBATCH --job-name=wall_plan                 # Submit a job named "example"
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
# python plan.py model_name=2025-02-16/10-47-35 n_evals=10 planner=mpc_cem goal_H=5 goal_source='random_state' seed=1 planner.max_iter=20
# python plan.py model_name=2025-02-16/10-47-35 n_evals=10 planner=mpc_cem goal_H=5 goal_source='random_state' seed=2 planner.max_iter=20
python plan.py model_name=2025-02-16/10-47-35 n_evals=10 planner=mpc_cem goal_H=5 goal_source='random_state' seed=3 planner.max_iter=20
python plan.py model_name=2025-02-16/10-47-35 n_evals=10 planner=mpc_cem goal_H=5 goal_source='random_state' seed=4 planner.max_iter=20
python plan.py model_name=2025-02-16/10-47-35 n_evals=10 planner=mpc_cem goal_H=5 goal_source='random_state' seed=5 planner.max_iter=20

python plan.py model_name=2025-02-16/10-47-35 n_evals=10 planner=cem goal_H=5 goal_source='random_state' seed=1 planner.max_iter=20
python plan.py model_name=2025-02-16/10-47-35 n_evals=10 planner=cem goal_H=5 goal_source='random_state' seed=1 planner.max_iter=20
python plan.py model_name=2025-02-16/10-47-35 n_evals=10 planner=cem goal_H=5 goal_source='random_state' seed=1 planner.max_iter=20
python plan.py model_name=2025-02-16/10-47-35 n_evals=10 planner=cem goal_H=5 goal_source='random_state' seed=1 planner.max_iter=20
python plan.py model_name=2025-02-16/10-47-35 n_evals=10 planner=cem goal_H=5 goal_source='random_state' seed=1 planner.max_iter=20


