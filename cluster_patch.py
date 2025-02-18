import numpy as np
import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
from PIL import Image
import pickle   
import argparse

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import pdist, squareform
from einops import rearrange
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import matplotlib.pyplot as plt
import cv2

from models.dino import DinoV2Encoder, DinoEncoder, DinoV2EncoderReg
from datasets.img_transforms import imagenet_transform



def load_pth(filename):
    return torch.load(filename)

def tensor_to_image(x: torch.Tensor) -> Image:
    x = x.detach().cpu().numpy()
    return x

def create_adjacency_matrix(grid_size=16):
    """
    Create adjacency matrix for DINO patches in a grid.
    Args:
        grid_size (int): Size of the grid (e.g., 16 for 16x16=256 patches)
    Returns:
        adjacency_matrix (np.ndarray): 256x256 connectivity matrix
    """
    num_patches = grid_size * grid_size
    adjacency_matrix = np.zeros((num_patches, num_patches), dtype=int)

    for idx in range(num_patches):
        # Convert index to (row, col) coordinates
        row = idx // grid_size
        col = idx % grid_size

        # Check all 4 possible neighbors
        neighbors = []
        if row > 0:
            neighbors.append((row-1, col))  # Top
        if row < grid_size-1:
            neighbors.append((row+1, col))  # Bottom
        if col > 0:
            neighbors.append((row, col-1))  # Left
        if col < grid_size-1:
            neighbors.append((row, col+1))  # Right

        # Convert neighbor coordinates back to indices
        for nr, nc in neighbors:
            neighbor_idx = nr * grid_size + nc
            adjacency_matrix[idx, neighbor_idx] = 1
            adjacency_matrix[neighbor_idx, idx] = 1  # Ensure symmetry

    return adjacency_matrix

def get_vit_patch_coordinates(num_patches=196, coord_min=-1, coord_max=1):
    grid_size = int(np.sqrt(num_patches))

    patch_size = (coord_max - coord_min) / grid_size  # 패치 하나의 크기

    # 14개의 x, y 좌표 만들기 (중심점 기준)
    x_coords = torch.linspace(coord_min + patch_size / 2, coord_max - patch_size / 2, grid_size)
    y_coords = torch.linspace(coord_min + patch_size / 2, coord_max - patch_size / 2, grid_size)

    # 좌표 meshgrid 생성 (14x14)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')

    # 좌표를 (196, 2) 형태로 변환
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    coords = coords.numpy()

    return coords

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1)
    parser.add_argument("--num_clusters", type=int, default=4)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--dataset", type=str, default="point_maze")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--encoder", type=str, default="dinov2")
    parser.add_argument("--size", type=str, default="small")
    parser.add_argument("--num_features", type=int, default=1) # Number of features per clusters

    args = parser.parse_args()

    demo_path = f'/home/shared/robotics/{args.dataset}/test_images_{args.encoder}_{args.size}_{args.num_clusters}_feat{args.num_features}/'
    output_path = f'/home/shared/robotics/{args.dataset}/objects_{args.encoder}_{args.size}_{args.num_clusters}_feat{args.num_features}/'
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("INFO: Using device: ", device)
    if args.dataset == "point_maze":
        data_dir = "/home/shared/robotics/point_maze/obses"
    elif "wall" in args.dataset:
        data_dir = f"/home/shared/robotics/{args.dataset}/obses"
    elif args.dataset == "pusht":
        data_dir = f"/home/s2/youngjoonjeong/github/dino-wm/dataset/pointmaze/pusht_3000_{args.split}.pkl"
    elif args.dataset in ["rope", "granular"]:
        data_dir = f"/home/shared/robotics/deformable/{args.dataset}"
        demo_path = f"/home/shared/robotics/deformable/{args.dataset}_test_images_{args.encoder}_{args.size}_{args.num_clusters}_feat{args.num_features}"
        output_path = f"/home/shared/robotics/deformable/{args.dataset}_objects_{args.encoder}_{args.size}_{args.num_clusters}_feat{args.num_features}"
    else:
        data_dir = '/home/youngjoon/github/dino_wm/misc/test_images'
        
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(demo_path, exist_ok=True)

    

    if args.encoder == "dinov2":
        name = f"dinov2_vit{args.size[0]}14"
        encoder = DinoV2Encoder(name=name, feature_key='x_norm_patchtokens').to(device)
    elif args.encoder == "dino":
        name = f"dino_vit{args.size[0]}16"
        encoder = DinoEncoder(name=name, feature_key='x_norm_patchtokens').to(device)
    elif args.encoder == "dinov2_reg":
        name = f"dinov2_vit{args.size[0]}14_reg"
        encoder = Dinov2EncoderReg(sname=name, feature_key='x_norm_patchtokens').to(device)

    normalize = imagenet_transform()


    np.random.seed(72)

    color_map = [np.concatenate([np.random.random(3), [1.0]]) for _ in range(100)]
    # color_map = [np.array([1, 0, 0, 1.0]), np.array([0, 1, 0, 1.0]), np.array([0, 0, 1, 1.0]), np.array([1, 1, 0, 1.0]), np.array([1, 0, 1, 1.0]), np.array([0, 1, 1, 1.0]), np.array([1, 1, 0.5, 1.0]), np.array([1, 0.5, 1, 1.0]), np.array([0.5, 1, 1, 1.0])]
    for fidx in range(args.start_idx, args.end_idx):
        if args.dataset in ["rope", "granular"]:
            demo_i_path = os.path.join(demo_path, f"{str(fidx).zfill(6)}")
            fpath = os.path.join(data_dir, f"{str(fidx).zfill(6)}", "obses.pth")
        else:
            demo_i_path = os.path.join(demo_path, f"episode_{str(fidx).zfill(5)}")
            fpath = os.path.join(data_dir, f"episode_{str(fidx).zfill(3)}.pth")

        os.makedirs(demo_i_path, exist_ok=True)

        video = load_pth(fpath) # (T, H, W, 3)
        if video.shape[1] == 3:
            video = rearrange(video, 't c h w -> t h w c')
        print("INFO: video.shape: ", video.shape, video.min(), video.max())
        if video.max() > 1.0:
            video = video.float() / 255.0
        print("INFO: video.shape: ", video.shape, video.min(), video.max())
        if video.shape[-1] == 3:
            video_copy = rearrange(video, 't h w c -> t c h w')
        else:
            video_copy = video
        video_norm = torch.stack([normalize(frame) for frame in video_copy], dim=0).to(device) # (T, 3, H, W)
        print("INFO: video_norm.shape: ", video_norm.shape, video_norm.min(), video_norm.max(), video_norm.mean())
        frames = [tensor_to_image(frame) for frame in video]

        video_norm = video_norm.to(device) # (T, H, W, 3)
        
        video_features = encoder(video_norm)
        patch_num = video_features.shape[1]
        print("INFO: Video features shape: ", video_features.shape)


        T, H, W, _ = video.shape
        print("INFO: Video shape: ", video.shape)

        
        
        episode_features = []  # 에피소드 전체 특징을 저장할 리스트
        episode_coords = []  # 에피소드 전체 좌표를 저장할 리스트
        episode_patch_info = []  # 에피소드 전체 패치 정보를 저장할 리스트

        # 개별 프레임 저장할 리스트 추가
        all_original_frames = []  # 원본 이미지
        all_overlay_frames = []   # Overlay 이미지

        coords = get_vit_patch_coordinates(num_patches=video_features.shape[1])

        for t in range(video_features.shape[0]):
            frame = frames[t]
            features = video_features[t]
            features = features.detach().cpu().numpy()  # (num_patches, feature_dim)
            grid_size = H // encoder.patch_size
            step_y = H // grid_size
            step_x = W // grid_size
            token_positions = np.array([
                (int(y * step_y + step_y//2), int(x * step_x + step_x//2))
                for y in range(grid_size) for x in range(grid_size)
            ])
            # distance_matrix = squareform(pdist(features))
            
            # distance_matrix = cosine_distances(np.concatenate([features, coords], axis=1))
            distance_matrix = cosine_distances(features)
            connectivity_matrix = create_adjacency_matrix(grid_size) if args.connect else None
            
            if args.num_clusters == -1:
                 clustering = AgglomerativeClustering(
                    # n_clusters=args.num_clusters, 
                    n_clusters = None,
                    metric='precomputed', 
                    linkage='average',
                    distance_threshold=0.5,
                    connectivity=connectivity_matrix
                                        )
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=args.num_clusters, 
                    # n_clusters = None,
                    metric='precomputed', 
                    linkage='average',
                    # distance_threshold=
                    connectivity=connectivity_matrix)
            
            cluster_labels = clustering.fit_predict(distance_matrix)

            if t > 0:
                prev_object_groups = object_groups

            # Group mask indices by their cluster labels
            object_groups = []
            for cluster_id in np.unique(cluster_labels):
                group = np.where(cluster_labels == cluster_id)[0].tolist()
                object_groups.append(group)
            
            # 프레임 간 클러스터 순서 정렬
            if t > 0:
                reordered_groups = []
                used_indices = set()

                for prev_group in prev_object_groups:
                    # 현재 프레임의 그룹과 이전 프레임의 그룹 간의 유사도를 비교
                    best_match_idx = -1
                    best_overlap = 0

                    for i, curr_group in enumerate(object_groups):
                        if i in used_indices:
                            continue  # 이미 매칭된 그룹은 건너뜀
                        
                        overlap = len(set(prev_group) & set(curr_group))  # 교집합 크기 계산
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match_idx = i

                    if best_match_idx != -1:
                        reordered_groups.append(object_groups[best_match_idx])
                        used_indices.add(best_match_idx)

                # 아직 매칭되지 않은 그룹들은 그대로 추가
                for i, group in enumerate(object_groups):
                    if i not in used_indices:
                        reordered_groups.append(group)

                object_groups = reordered_groups  # 업데이트
                    
            if t == 0:
                object_groups.sort(key=len, reverse=True)
            # print(f"INFO: Frame {t} has {len(object_groups)} clusters")
            # print(f"INFO: Frame {t} groups: ", object_groups)

            overlay = np.ones((frame.shape[0], frame.shape[1], 4))
            overlay[:, :, 3] = 0
            patch_h = int(step_y)
            patch_w = int(step_x)
            legend_items = []
            for group_idx, group in enumerate(object_groups):
            

                for token_idx in group:
                    y, x = token_positions[token_idx]
                    color_bgr = color_map[group_idx] 
                    legend_items.append((group_idx, color_bgr))
                    
                    overlay[
                        max(0, y-patch_h//2):min(H, y+patch_h//2),
                        max(0, x-patch_w//2):min(W, x+patch_w//2)
                    ] = color_map[group_idx]

            if t % args.stride == 0:
            # 결과 시각화
                plt.figure(figsize=(8, 8))
                # frame = frame.astype(np.uint8)
                plt.imshow(frame)
                plt.axis('off')
                plt.savefig(os.path.join(demo_i_path ,f"{str(t).zfill(5)}_ori.png"))
                
                print(f"INFO: Saved {os.path.join(demo_i_path ,f'{str(t).zfill(5)}_ori.png')}")
                plt.imshow(overlay, alpha=0.8)
                plt.title(f"Frame {t} with Clusters")
                

                plt.show()

                plt.savefig(os.path.join(demo_i_path ,f"{str(t).zfill(5)}.png"))
                plt.close()
                print(f"INFO: Saved {os.path.join(demo_i_path ,f'{str(t).zfill(5)}.png')}")

                 # 전체 이미지 만들기 위해 리스트에 추가
                all_original_frames.append(frame)
                all_overlay_frames.append(overlay)

            # [추가 부분 2] 클러스터별 특징 평균 계산 및 저장
            frame_features = []  # 프레임별 클러스터 특징 저장
            frame_coords = []  # 프레임별 클러스터 좌표 저장
            frame_patch_info = []  # 프레임별 클러스터 패치 정보 저장
            for group_idx, group in enumerate(object_groups):
                # 해당 클러스터에 속한 패치들의 특징 추출
                group = sorted(group)                
                cluster_features = features[group]  # (num_patches_in_cluster, feature_dim)
                cluster_coords = coords[group]

                cluster_feature_chunks = np.array_split(cluster_features, args.num_features)
                cluster_coord_chunks = np.array_split(cluster_coords, args.num_features)
                patch_chunks = np.array_split(np.array(group), args.num_features)

                for j, feature_chunk in enumerate(cluster_feature_chunks):
                    k = j
                    feature_chunk = cluster_feature_chunks[k]
                    coord_chunk = cluster_coord_chunks[k]
                    patch_chunk = patch_chunks[k]
                    
                    while feature_chunk.shape[0] == 0:
                        print("group : ", group)
                        k -= 1
                        feature_chunk = cluster_feature_chunks[k]
                        coord_chunk = cluster_coord_chunks[k]
                        patch_chunk = patch_chunks[k]
                        
                    
                    avg_feature = np.mean(feature_chunk, axis=0)
                    frame_features.append(avg_feature)

                    avg_coord = np.mean(coord_chunk, axis=0)
                    std_coord = np.std(coord_chunk, axis=0)
                    avg_coord = np.concatenate([avg_coord, std_coord])
                    frame_coords.append(avg_coord)

                    patch_info_chunk = np.zeros(patch_num).astype(float)
                    patch_info_chunk[patch_chunk] = 1.0
                    frame_patch_info.append(patch_info_chunk)
                    
            episode_features.append(np.array(frame_features)) 
            episode_coords.append(np.array(frame_coords))
            episode_patch_info.append(np.array(frame_patch_info))

        # 모든 프레임을 가로로 붙이기
        full_original = np.hstack(all_original_frames)
        full_overlay = np.hstack(all_overlay_frames)

        # 전체 프레임 한 장으로 저장
        plt.imsave(os.path.join(demo_i_path, "full_original.png"), full_original)
        plt.imsave(os.path.join(demo_i_path, "full_overlay.png"), full_overlay)

        np.save(os.path.join(output_path, f"episode_{fidx:05d}.npy"), np.array(episode_features))
        np.save(os.path.join(output_path, f"coord_{fidx:05d}.npy"), np.array(episode_coords))
        np.save(os.path.join(output_path, f"patch_{fidx:05d}.npy"), np.array(episode_patch_info))
        # np.save(os.path.join(output_path, f"full_{i:05d}.npy"), video_features.detach().cpu().numpy())
        print(f"INFO: Saved {os.path.join(output_path, f'episode_{fidx:05d}.npy')}")
        print(f"INFO: Saved {os.path.join(output_path, f'coord_{fidx:05d}.npy')}")
        print(f"INFO: Saved {os.path.join(output_path, f'patch_{fidx:05d}.npy')}")
        # print("INFO: Episode features shape: ", np.array(episode_features).shape)
        # print("INFO: Episode features: ", np.array(episode_features).min(), np.array(episode_features).max(), np.array(episode_features).mean())
        # print("INFO: Episode coords shape: ", np.array(episode_coords).shape)
        # print("INFO: Episode coords: ", np.array(episode_coords).min(), np.array(episode_coords).max(), np.array(episode_coords).mean())
        print("INFO: Episode patch info shape: ", np.array(episode_patch_info).shape)

