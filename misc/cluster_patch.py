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

class DINOv2ImageEncoder(nn.Module):
    def __init__(self, size="small", patch_size=14):
        super().__init__()
        """Initialize the DINOv2 encoder."""
        self.size = size
        self.patch_size = patch_size
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vit" + self.size[0] + str(self.patch_size))
        # self.model = torch.hub.load('facebookresearch/dino:main', f'dino_vit{self.size[0]}{self.patch_size}')  # 16은 patch_size
        self.embed_dim = self.model.embed_dim
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, images, cls_token=False):
        """
            Encode a batch of images.
            images: (B, 3, H, W)
            return: (B, token, dim)
        """
        assert images.shape[2] % self.patch_size == 0
        assert images.shape[3] % self.patch_size == 0

        self.model.eval()
        try:
            x = self.model.prepare_tokens_with_masks(images)
        except:
            x = self.model.get_intermediate_layers(images, n=1)[0]
        for block in self.model.blocks:
            x = block(x)
        if cls_token:
            return x[:, 0].unsqueeze(1) # (B, 1, dim)
        else:
            return x[:, 1:] # (B, token, dim)

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

def load_pth(filename):
    return torch.load(filename)

def normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float() / 255.0
    # print("INFO: x.min, x.max: ", x.min(), x.max())
    x = x.permute(2, 0, 1)
    # print("INFO: x.shape: ", x.shape)
    normalize = transforms.Compose([
        # transforms.Resize(224),
        # transforms.CenterCrop(224),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return normalize(x)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1)
    parser.add_argument("--num_clusters", type=int, default=4)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--dataset", type=str, default="point_maze")

    args = parser.parse_args()

    demo_path = '/shared/s2/lab01/dataset/robotics/temp'
    output_path = f'/shared/s2/lab01/dataset/robotics/objects/{args.dataset}'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(demo_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("INFO: Using device: ", device)
    if args.dataset == "point_maze":
        data_dir = "/shared/s2/lab01/dataset/robotics/point_maze/obses"
    elif args.dataset == "pusht":
        data_dir = f"/home/s2/youngjoonjeong/github/dino-wm/dataset/pointmaze/pusht_3000_{args.split}.pkl"

    encoder = DINOv2ImageEncoder(size="small", patch_size=14).to(device)

    np.random.seed(72)

    color_map = [np.concatenate([np.random.random(3), [1.0]]) for _ in range(100)]
    # color_map = [np.array([1, 0, 0, 1.0]), np.array([0, 1, 0, 1.0]), np.array([0, 0, 1, 1.0]), np.array([1, 1, 0, 1.0]), np.array([1, 0, 1, 1.0]), np.array([0, 1, 1, 1.0]), np.array([1, 1, 0.5, 1.0]), np.array([1, 0.5, 1, 1.0]), np.array([0.5, 1, 1, 1.0])]
    for i in range(args.start_idx, args.end_idx):
        demo_i_path = os.path.join(demo_path, f"episode_{str(i).zfill(5)}")
        os.makedirs(demo_i_path, exist_ok=True)

        fpath = os.path.join(data_dir, f"episode_{str(i).zfill(3)}.pth")
        video = load_pth(fpath) # (T, H, W, 3)
        video_norm = torch.stack([normalize(frame) for frame in video], dim=0).to(device) # (T, 3, H, W)
        frames = [tensor_to_image(frame) for frame in video]

        video_norm = video_norm.to(device) # (T, H, W, 3)
        video_features = encoder(video_norm) # (T, token, dim)

        T, H, W, _ = video.shape

        for t in range(video_features.shape[0]):
            frame = frames[t]
            features = video_features[t]
            grid_size = H // encoder.patch_size
            step_y = H // grid_size
            step_x = W // grid_size
            token_positions = np.array([
                (int(y * step_y + step_y//2), int(x * step_x + step_x//2))
                for y in range(grid_size) for x in range(grid_size)
            ])
            features = features.detach().cpu().numpy()
            # distance_matrix = squareform(pdist(features))
            distance_matrix = cosine_distances(features)
            connectivity_matrix = create_adjacency_matrix(grid_size) if args.connect else None
            clustering = AgglomerativeClustering(
                n_clusters=args.num_clusters, 
                # n_clusters = None,
                metric='precomputed', 
                linkage='average',
                # distance_threshold=0.5,
                connectivity=connectivity_matrix)
            
            cluster_labels = clustering.fit_predict(distance_matrix)

            # Group mask indices by their cluster labels
            object_groups = []
            for cluster_id in np.unique(cluster_labels):
                group = np.where(cluster_labels == cluster_id)[0].tolist()
                object_groups.append(group)
            object_groups.sort(key=len, reverse=True)
            print(f"INFO: Frame {t} has {len(object_groups)} clusters")

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
            # 결과 시각화
            plt.figure(figsize=(8, 8))
            plt.imshow(frame)
            plt.imshow(overlay, alpha=0.8)
            plt.title(f"Frame {t} with Clusters")
            plt.axis('off')

            plt.show()

            plt.savefig(os.path.join(demo_i_path ,f"{str(t).zfill(5)}.png"))
            plt.close()
            print(f"INFO: Saved {os.path.join(demo_i_path ,f'{str(t).zfill(5)}.png')}")
    