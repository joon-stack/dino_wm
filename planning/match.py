import torch

def match_objects(tgt, pred):
    import torch
    from sklearn.metrics.pairwise import cosine_distances
    from scipy.optimize import linear_sum_assignment # for Hungarian algorithm
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
    debug_samples = min(0, B*T)  # 최대 2개 샘플만 출력
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

    return torch.tensor(res) 