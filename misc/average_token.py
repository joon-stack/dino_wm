import numpy as np
import torch
# x는 shape (18, 4, 384)를 가진 numpy 배열로 가정
# x = np.load('/home/youngjoon/github/dino_wm/misc/test_emb/temp/episode_00000.npy')
x = torch.load('/home/youngjoon/github/dino_wm/misc/test_images/episode_000.pth')
x = x.detach().cpu().numpy()
# 1. axis=1에 대해 평균 계산 후 L2 거리 행렬 저장
print(x.shape)  # (18, 4, 384)
avg_x = x.mean(axis=1)  # shape (18, 384)

# L2 거리 행렬 계산 함수
def compute_l2_dist(matrix):
    # matrix: (18, 384)
    diff = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]
    squared_diff = diff ** 2
    sum_squared = squared_diff.sum(axis=-1)
    distance = np.sqrt(sum_squared)
    return distance

# 평균에 대한 거리 계산 및 저장
avg_distance = compute_l2_dist(avg_x)
np.savetxt('average_distance_full.csv', avg_distance, delimiter=',')

# # 2. 각 인덱스(0, 1, 2, 3) 선택 후 L2 거리 행렬 저장
# for idx in range(4):
#     index_x = x[:, idx, :]  # shape (18, 384)
#     index_distance = compute_l2_dist(index_x)
#     filename = f'index_{idx}_distance.csv'
#     np.savetxt(filename, index_distance, delimiter=',')