from itertools import zip_longest

import numpy as np

import torch

def solve_b(G,C):

    G_pinv = torch.linalg.pinv(G)

    return G_pinv @ C @ G_pinv.T


def get_index_dict(C):
    indices = (C == 1).nonzero(as_tuple=False)
    return {(int(i), int(j)): True for i, j in indices}



def drop_ones_proportionally(connection_matrix: torch.Tensor, p: float) -> torch.Tensor:
    """
    按比例 p 把 connection_matrix 中的 1 随机置为 0。

    参数：
        connection_matrix: 0-1 矩阵 (torch.Tensor)
        p: 需要置0的比例 (0<=p<=1)

    返回：
        新矩阵，置0后的矩阵（不改变原矩阵）
    """
    assert 0 <= p <= 1, "p 必须在0和1之间"
    assert connection_matrix.dtype == torch.float or connection_matrix.dtype == torch.int, "矩阵必须为浮点或整型"

    conn = connection_matrix.clone()
    ones_indices = (conn == 1).nonzero(as_tuple=False)  # 所有1的位置索引

    total_ones = ones_indices.shape[0]
    num_to_zero = int(total_ones * p)

    if num_to_zero == 0:
        return conn

    # 随机选取需要置0的索引
    perm = torch.randperm(total_ones)
    zero_indices = ones_indices[perm[:num_to_zero]]

    # 将选中位置置为0
    for idx in zero_indices:
        conn[idx[0], idx[1]] = 0

    return conn

def top_n_diff_positions(mat1: torch.Tensor, mat2: torch.Tensor, n: int):
    """
    输入两个形状相同的矩阵，返回差异最大前 n 个元素的位置及差值。

    参数：
        mat1, mat2: 形状相同的 torch.Tensor
        n: 返回的最大差异数量

    返回：
        List[(i, j, diff_value)]，长度最多为 n，按差值从大到小排序
    """
    assert mat1.shape == mat2.shape, "两个矩阵形状必须相同"

    diff = torch.abs(mat1 - mat2)
    flat_diff = diff.flatten()

    # 如果 n 大于所有元素数目，返回所有元素
    n = min(n, flat_diff.numel())

    # 找出前 n 个最大差值及索引
    top_vals, top_idxs = torch.topk(flat_diff, n)

    results = []
    rows, cols = mat1.shape

    for val, idx in zip(top_vals, top_idxs):
        i = idx // cols
        j = idx % cols
        results.append((i.item(), j.item()))

    return results


def show_gene(b_matrix_list):
    header_titles = ["敲除100%", "敲除90%", "敲除80%", "敲除70%"]
    width = 16

    print("".join(f"{title:^{width}}" for title in header_titles))

    for row in zip_longest(*b_matrix_list, fillvalue=''):
        print("".join(f"{str(x):^{width}}" for x in row))


def generate_thresholds(threshold, step=0.1):
    return [round(x, 2) for x in np.arange(1.0, threshold - 0.001, -step)]


def common_gene(threshold,step, gene_matrix, connection_matrix, top_n=100):
    b0 = solve_b(gene_matrix, connection_matrix)
    thresholds = generate_thresholds(threshold,step)

    n_b = [
        set(top_n_diff_positions(solve_b(gene_matrix, drop_ones_proportionally(connection_matrix, t)), b0, top_n))
        for t in thresholds
    ]

    return set.intersection(*n_b)


if __name__ == "__main__":

    loaded_data = torch.load("data_save.pt")
    gene_matrix = loaded_data["gene_matrix"].cuda().float()
    connection_matrix = loaded_data["connection_matrix"].cuda().float()

    # b0 = solve_b(gene_matrix,connection_matrix)
    #
    # n_b = [top_n_diff_positions(solve_b(gene_matrix,drop_ones_proportionally(connection_matrix,threshold)),b0,100)
    #        for threshold in [1,0.9,0.8,0.7]]
    common = common_gene(threshold=0.7, step=0.1, gene_matrix=gene_matrix, connection_matrix=connection_matrix)

