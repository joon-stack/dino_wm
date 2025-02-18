#!/bin/bash

#SBATCH --job-name=data                 # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                         # Using 1 gpu
#SBATCH --time=0-12:00:00                     # 1 hour timelimit
#SBATCH --mem=10G                         # Using 10GB CPU Memory
#SBATCH --partition=laal_a6000                      # Using "b" partition 
#SBATCH --cpus-per-task=4                    # Using 4 maximum processor

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate dino_wm

# python cluster_patch.py --encoder dino --dataset rope --end_idx 2000 --num_clusters 6 --stride 5 --num_features 1
# python cluster_patch.py --encoder dino --dataset rope --end_idx 2000 --num_clusters 6 --stride 5 --num_features 3
# python cluster_patch.py --encoder dino --dataset rope --end_idx 2000 --num_clusters 6 --stride 5 --num_features 5
python cluster_patch.py --encoder dino --dataset granular --end_idx 2000 --num_clusters 5 --stride 5 --num_features 1
python cluster_patch.py --encoder dino --dataset granular --end_idx 2000 --num_clusters 5 --stride 5 --num_features 3
# python cluster_patch.py --encoder dino --dataset point_maze --end_idx 2000 --num_clusters 3 --stride 5 --num_features 1
# python cluster_patch.py --encoder dino --dataset point_maze --end_idx 2000 --num_clusters 3 --stride 5 --num_features 3
# python cluster_patch.py --encoder dino --dataset point_maze --end_idx 2000 --num_clusters 3 --stride 5 --num_features 5
# python cluster_patch.py --encoder dino --dataset wall_single --end_idx 2000 --num_clusters 4 --stride 10 --num_features 3
# python cluster_patch.py --encoder dino --dataset wall_single --end_idx 2000 --num_clusters 4 --stride 10 --num_features 5
# python cluster_patch.py --encoder dino --dataset granular --end_idx 2000 --num_clusters 6 --stride 5 --num_features 3
# python cluster_patch.py --encoder dino --dataset wall_single --end_idx 2000 --num_clusters 4 --stride 20
# python cluster_patch.py --encoder dino --dataset granular --end_idx 2000 --num_clusters 6 --stride 5 --num_features 1
# python cluster_patch.py --encoder dino --dataset granular --end_idx 2000 --num_clusters 6 --stride 5 --num_features 5
# python cluster_patch.py --encoder dino --dataset granular --end_idx 2000 --num_clusters 6 --stride 5 --num_features 2
# python cluster_patch.py --encoder dino --dataset granular --end_idx 2000 --num_clusters 8 --stride 10
# python cluster_patch.py --encoder dino --dataset wall_single --end_idx 2000 --num_clusters 7
# python datasets/cluster_patch.py --encoder dinov2_reg --dataset wall_single --end_idx 2000 --num_clusters 4