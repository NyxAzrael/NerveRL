import torch
import random


import torch

def extract_high_discriminative_genes(connection_matrix, gene_matrix, top_n=20):
    num_neurons = connection_matrix.shape[0]
    num_genes = gene_matrix.shape[1]

    # 构造正负样本掩码，排除 i==j
    pos_mask = (connection_matrix == 1)
    neg_mask = (connection_matrix == 0)
    pre_idx, post_idx = torch.meshgrid(
        torch.arange(num_neurons), torch.arange(num_neurons), indexing='ij'
    )
    valid = pre_idx != post_idx
    pos_mask &= valid
    neg_mask &= valid

    # 提取索引对
    pos_pairs = torch.nonzero(pos_mask, as_tuple=False)
    neg_pairs = torch.nonzero(neg_mask, as_tuple=False)

    # 计算每个基因在正负样本上的出现频率
    pre_pos_freq = gene_matrix[pos_pairs[:, 0]].sum(dim=0) / len(pos_pairs)
    post_pos_freq = gene_matrix[pos_pairs[:, 1]].sum(dim=0) / len(pos_pairs)
    pre_neg_freq = gene_matrix[neg_pairs[:, 0]].sum(dim=0) / len(neg_pairs)
    post_neg_freq = gene_matrix[neg_pairs[:, 1]].sum(dim=0) / len(neg_pairs)

    # 计算正样本平均频率，用于过滤低频基因
    avg_pre_pos = pre_pos_freq.mean()
    avg_post_pos = post_pos_freq.mean()

    def get_top_indices(pos_freq, neg_freq, avg_threshold):
        # 只保留在正样本中频率不低于平均值的基因
        mask = pos_freq >= avg_threshold
        # 差异性指标：normalized difference
        diff = torch.zeros_like(pos_freq)
        diff[mask] = (pos_freq[mask] - neg_freq[mask]) / (pos_freq[mask] + neg_freq[mask] + 1e-6)
        # 取 top_n 最大值的索引
        topk = torch.topk(diff, k=min(top_n, mask.sum().item()))
        return topk.indices.tolist()

    pre_key_genes = get_top_indices(pre_pos_freq, pre_neg_freq, avg_pre_pos)
    post_key_genes = get_top_indices(post_pos_freq, post_neg_freq, avg_post_pos)

    return pre_key_genes, post_key_genes

# 示例调用


def get_random_common_gene(n=10):
    loaded_data = torch.load("data_save.pt")
    gene_matrix = loaded_data["gene_matrix"]
    connection_matrix = loaded_data["connection_matrix"]

    pre_key_genes, post_key_genes = extract_high_discriminative_genes(connection_matrix, gene_matrix, top_n=n)


    return pre_key_genes, post_key_genes

loaded_data = torch.load("data_save.pt")
gene_matrix = loaded_data["gene_matrix"]
connection_matrix = loaded_data["connection_matrix"]
pre_genes, post_genes = extract_high_discriminative_genes(connection_matrix, gene_matrix, top_n=200)







def extract_high_freq_key_genes(connection_matrix, gene_matrix, top_n=3, freq_threshold=0.05):
    num_neurons = connection_matrix.shape[0]
    num_genes = gene_matrix.shape[1]

    # 初始化频率字典
    pre_pos_freq = torch.zeros(num_genes)
    post_pos_freq = torch.zeros(num_genes)
    pre_neg_freq = torch.zeros(num_genes)
    post_neg_freq = torch.zeros(num_genes)

    pos_count = 0
    neg_count = 0

    for pre in range(num_neurons):
        for post in range(num_neurons):
            if pre == post:
                continue

            pre_gene = gene_matrix[pre]
            post_gene = gene_matrix[post]

            if connection_matrix[pre, post] == 1:
                pre_pos_freq += pre_gene
                post_pos_freq += post_gene
                pos_count += 1
            else:
                pre_neg_freq += pre_gene
                post_neg_freq += post_gene
                neg_count += 1

    # 归一化频率
    if pos_count > 0:
        pre_pos_freq /= pos_count
        post_pos_freq /= pos_count
    if neg_count > 0:
        pre_neg_freq /= neg_count
        post_neg_freq /= neg_count

    results = []

    for role, pos_freq, neg_freq in [
        ("pre", pre_pos_freq, pre_neg_freq),
        ("post", post_pos_freq, post_neg_freq)
    ]:
        # 找出差值最大的 top-n 基因
        freq_diff = pos_freq - neg_freq
        high_pos_mask = pos_freq >= freq_threshold
        low_neg_mask = neg_freq <= freq_threshold
        keep_mask = high_pos_mask & low_neg_mask

        if keep_mask.sum() == 0:
            print(f"{role}: 无满足条件的key_gene")
            continue

        candidate_indices = torch.nonzero(keep_mask).squeeze()
        top_indices = freq_diff[candidate_indices].argsort(descending=True)[:top_n]
        key_genes = candidate_indices[top_indices]

        print(f"{role} 高频 key_genes 索引: {key_genes.tolist()}")
        print(f"{role} 对应阳性频率: {[round(pos_freq[i].item(), 3) for i in key_genes]}")
        print(f"{role} 对应阴性频率: {[round(neg_freq[i].item(), 3) for i in key_genes]}")

        results.append({
            "role": role,
            "key_genes": key_genes.tolist()
        })

    return results


extract_high_freq_key_genes(connection_matrix,gene_matrix,freq_threshold=0.5)






