import numpy as np
import torch

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment # for Hungarian algorithm
import os
from scipy.stats import spearmanr
import argparse

def match_objects(tgt, pred):
    """
    input : tgt, pred : (b, num_frames, num_objects, emb_dim)
    output: pred      : (b, num_frames, num_objects, emb_dim)
    """
    B, T, N, D = pred.shape

    # 배치와 시간 차원 병합
    pred_flat = pred.reshape(B*T, N, D)
    tgt_flat = tgt.reshape(B*T, N, D)

    cost_matrices = torch.cdist(tgt_flat, pred_flat)  # (B*T, N, N)
    matched_pred_flat = torch.zeros_like(pred_flat)
    

    # 디버그용 샘플 선택 (첫 2개 배치 & 첫 시간대만 출력)
    debug_samples = min(1, B*T)  # 최대 2개 샘플만 출력

    res = []
    for i in range(B*T):
        cost_matrix = cost_matrices[i].detach().cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        matched_pred_flat[i] = pred_flat[i, col_idx]
        matched_cost = cost_matrix[row_idx, col_idx].mean()
        res.append(matched_cost)

        # 디버그 정보 출력 -------------------------------------------------
        if i < debug_samples:
            batch_idx = i // T
            time_idx = i % T
            original_order = torch.arange(N).numpy()
            
            # 매칭 전후 Cost 계산
            original_cost = cost_matrix[original_order, original_order].mean()
            

            if original_cost > matched_cost:
                print(f"\n=== Sample [Batch {batch_idx}, Time {time_idx}] ===")
                print(f"Object Count: {N}")
                print(f"Original Object Order: {original_order, original_order}")
                print(f"Matched Object Order: {row_idx, col_idx}")
                print(f"Cost Matrix:\n{np.round(cost_matrix, 4)}")
                print(f"Original Cost (Diagonal): {original_cost:.4f}")
                print(f"Matched Cost (Hungarian): {matched_cost:.4f}")
                print(f"Cost Improvement: {original_cost - matched_cost:.4f}\n")
    res = np.array(res)
    return res

def compute_spearman_by_diagonal(arr: np.ndarray) -> dict:
    """
    각 컬럼에 대해, 대각 원소를 기준으로 아래쪽(대각원소 포함, 기대: 오름차순)과 
    위쪽(대각원소 포함, 기대: 내림차순) 부분의 데이터 순서와 순위 간의 스피어만 상관계수를 계산합니다.
    
    Parameters:
    -----------
    arr : np.ndarray
        입력 배열. (예: 20×20 배열)
        
    Returns:
    --------
    results : dict
        각 컬럼별로 계산된 결과를 담은 딕셔너리입니다.
        결과 형식은 다음과 같습니다.
            {
              column_index: {
                  'upper_rho': 스피어만 상관계수 (내림차순 기대),
                  'upper_p': 해당 p-value,
                  'lower_rho': 스피어만 상관계수 (오름차순 기대),
                  'lower_p': 해당 p-value
              },
              ...
            }
    """
    results = {}
    
    # 행/열 수 (정사각형 배열임을 가정)
    n_rows, n_cols = arr.shape

    for j in range(n_cols):
        # --- 아래쪽 부분 (대각 원소부터 마지막 행까지) ---
        # 기대: 오름차순 (행 인덱스가 증가하면 값도 증가)
        lower_segment = arr[j:, j]
        x_lower = np.arange(len(lower_segment))
        if len(lower_segment) > 1:
            rho_lower, p_lower = spearmanr(x_lower, lower_segment)
        else:
            rho_lower, p_lower = np.nan, np.nan

        # --- 위쪽 부분 (맨 위부터 대각 원소까지) ---
        # 기대: 내림차순 (대각 원소에서 위로 갈수록 값이 커져야 함)
        # spearmanr은 오름차순 비교이므로, 순서를 뒤집어 비교합니다.
        upper_segment = arr[:j+1, j]
        x_upper = np.arange(len(upper_segment))
        upper_segment_reversed = upper_segment[::-1]
        if len(upper_segment) > 1:
            rho_upper, p_upper = spearmanr(x_upper, upper_segment_reversed)
        else:
            rho_upper, p_upper = np.nan, np.nan

        results[j] = {
            'upper_rho': rho_upper,   # 위쪽 부분 (내림차순 기대)
            'upper_p': p_upper,
            'lower_rho': rho_lower,   # 아래쪽 부분 (오름차순 기대)
            'lower_p': p_lower
        }
    upper_rho = [results[res]['upper_rho'] for res in results]
    lower_rho = [results[res]['lower_rho'] for res in results]
    upper_rho = np.array(upper_rho).reshape(1, -1)
    lower_rho = np.array(lower_rho).reshape(1, -1)
    rho = np.concatenate([upper_rho, lower_rho], axis=0)
    return results, rho

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()
    idx = args.idx
    path = "/home/shared/robotics/wall_single/objects_dino_small_4"
    feat_path = os.path.join(path, f"episode_{str(idx).zfill(5)}.npy")
    coord_path = os.path.join(path, f"coord_{str(idx).zfill(5)}.npy")
    full_path = os.path.join(path, f"full_{str(idx).zfill(5)}.npy")

    feat = np.load(feat_path)  # (20, 6, 384)
    coord = np.load(coord_path)  # (20, 6, 4)
    full = np.load(full_path)  # (20, 196, 384)


    matched_dist = np.zeros((feat.shape[0], feat.shape[0]))
    matched_coord = np.zeros((feat.shape[0], feat.shape[0]))
    matched_coord_coord = np.zeros((feat.shape[0], feat.shape[0]))
    matched_coord_var = np.zeros((feat.shape[0], feat.shape[0]))

    for i in range(feat.shape[0]):
        for j in range(i+1, feat.shape[0]):
            matched_cost = match_objects(torch.tensor(np.expand_dims(feat[i:i+1], 0)), torch.tensor(np.expand_dims(feat[j:j+1], 0)))
            matched_c = match_objects(torch.tensor(np.expand_dims(coord[i:i+1], 0)), torch.tensor(np.expand_dims(coord[j:j+1], 0)))
            matched_c_co = match_objects(torch.tensor(np.expand_dims(coord[i:i+1, :, :2], 0)), torch.tensor(np.expand_dims(coord[j:j+1, :, :2], 0)))
            matched_c_var = match_objects(torch.tensor(np.expand_dims(coord[i:i+1, :, 2:], 0)), torch.tensor(np.expand_dims(coord[j:j+1, :, 2:], 0)))
            
            matched_dist[i, j] = matched_cost
            matched_dist[j, i] = matched_cost
            matched_coord[i, j] = matched_c
            matched_coord[j, i] = matched_c
            matched_coord_coord[i, j] = matched_c_co
            matched_coord_coord[j, i] = matched_c_co
            matched_coord_var[i, j] = matched_c_var
            matched_coord_var[j, i] = matched_c_var
            

    feat_matrices = []
    coord_matrices = []
    coord_var_matrices = []
    coord_coord_matrices = []
    full_matrices = []

    for i in range(feat.shape[1]):
        # feat_matrices.append(cdist(feat[:, i, :], feat[:, i, :], metric='euclidean'))
        # coord_matrices.append(cdist(coord[:, i, :], coord[:, i, :], metric='euclidean'))
        full_matrices.append(cdist(full[:, i, :], full[:, i, :], metric='euclidean'))
        # coord_coord_matrices.append(cdist(coord[:, i, :2], coord[:, i, :2], metric='euclidean'))
        # coord_var_matrices.append(cdist(coord[:, i, 2:], coord[:, i, 2:], metric='euclidean'))

    # feat_distance_matrix = np.mean(feat_matrices, axis=0)
    # coord_distance_matrix = np.mean(coord_matrices, axis=0)
    feat_distance_matrix = matched_dist
    coord_distance_matrix = matched_coord
    full_distance_matrix = np.mean(full_matrices, axis=0)
    feat_coord_distance_matrix = feat_distance_matrix + coord_distance_matrix * 500
    coord_coord_distance_matrix = matched_coord_coord
    coord_var_distance_matrix = matched_coord_var


    # CSV 파일 저장 (Pandas 없이 NumPy로 저장)
    feat_csv_filename = "feat_distance_matrix.csv"
    coord_csv_filename = "coord_distance_matrix.csv"
    full_csv_filename = "full_distance_matrix.csv"
    # matched_csv_filename = "matched_distance_matrix.csv"
    # matched_coord_filename = "matched_coord_distance_matrix.csv"
    feat_coord_csv_filename = "feat_coord_distance_matrix.csv"
    coord_coord_csv_filename = "coord_coord_distance_matrix.csv"
    coord_var_csv_filename = "coord_var_distance_matrix.csv"

    np.savetxt(feat_csv_filename, feat_distance_matrix, delimiter=",", fmt="%.6f")
    np.savetxt(coord_csv_filename, coord_distance_matrix, delimiter=",", fmt="%.6f")
    np.savetxt(full_csv_filename, full_distance_matrix, delimiter=",", fmt="%.6f")
    # np.savetxt(matched_csv_filename, matched_dist, delimiter=",", fmt="%.6f")
    # np.savetxt(matched_coord_filename, matches_coord, delimiter=",", fmt="%.6f")
    np.savetxt(feat_coord_csv_filename, feat_coord_distance_matrix, delimiter=",", fmt="%.6f")
    np.savetxt(coord_coord_csv_filename, coord_coord_distance_matrix, delimiter=",", fmt="%.6f")
    np.savetxt(coord_var_csv_filename, coord_var_distance_matrix, delimiter=",", fmt="%.6f")


    for i, f in enumerate(feat_matrices):
        np.savetxt(f"feat_{i}_distance_matrix.csv", f, fmt="%.6f")
        np.savetxt(f"coord_{i}_distance_matrix.csv", coord_matrices[i], fmt="%.6f")


    # 단조 증가/감소 여부 확인
    _, feat_rho = compute_spearman_by_diagonal(feat_distance_matrix)
    _, coord_rho = compute_spearman_by_diagonal(coord_distance_matrix)
    _, full_rho = compute_spearman_by_diagonal(full_distance_matrix)
    # _, matched_rho = compute_spearman_by_diagonal(matched_dist)
    # _, matched_coord_rho = compute_spearman_by_diagonal(matches_coord)
    _, feat_coord_rho = compute_spearman_by_diagonal(feat_coord_distance_matrix)
    _, coord_coord_rho = compute_spearman_by_diagonal(coord_coord_distance_matrix)
    _, coord_var_rho = compute_spearman_by_diagonal(coord_var_distance_matrix)

    np.savetxt("feat_rho.csv", feat_rho, delimiter=",", fmt="%.6f")
    np.savetxt("coord_rho.csv", coord_rho, delimiter=",", fmt="%.6f")
    np.savetxt("full_rho.csv", full_rho, delimiter=",", fmt="%.6f")
    # np.savetxt("matched_rho.csv", matched_rho, delimiter=",", fmt="%.6f")
    # np.savetxt("matched_coord_rho.csv", matched_coord_rho, delimiter=",", fmt="%.6f")
    np.savetxt("feat_coord_rho.csv", feat_coord_rho, delimiter=",", fmt="%.6f")
    np.savetxt("coord_coord_rho.csv", coord_coord_rho, delimiter=",", fmt="%.6f")
    np.savetxt("coord_var_rho.csv", coord_var_rho, delimiter=",", fmt="%.6f")


