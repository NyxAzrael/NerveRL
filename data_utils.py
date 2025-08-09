
import os

import pandas

from collections import Counter
import numpy as np
from sympy.printing.pytorch import torch

connection_path_from = r'D:\code\neuralRL\GRN-main\data\flywire\neuron_type'
rna_mean_df = r'D:\code\neuralRL\GRN-main\data\gene_data\overlap_part\RNA_mean_df (expression_level_3).csv'



def load_gene_data(path_from='', type='IgSF'):
    """Load gene expression data for overlapping neuron types"""
    # Load gene expression data using existing function
    # IgSF=True

    RNA_mean_df = pandas.read_csv(rna_mean_df, index_col=0)
    gene_matrix = RNA_mean_df.values
    gene_names = list(RNA_mean_df.columns)
    return gene_matrix, gene_names


def load_connection_matrix():
    name = os.path.join(connection_path_from, 'cn_type_by_percent.npy')
    cn_type_percent_new = np.load(name)

    name = os.path.join(connection_path_from, 'ct_type_by_percent.npy')
    ct_type_percent_new = np.load(name)

    C = cn_type_percent_new
    A = ct_type_percent_new

    print('Matrix C: ', C.shape)  # (39, 39)
    print('Matrix A: ', A.shape)  # (39, 39)
    # adjacency_matrix, connection_matrix = A, C
    adjacency_matrix = (A > 0.5).astype(int)
    connection_matrix = (C > 0.5).astype(int)
    return adjacency_matrix, connection_matrix

def load_gene_data_for_RL4Con(type='IgSF'):
    # Load gene data
    gene_matrix, gene_names = load_gene_data('', type)
    # print(f"\nTotal neurons: {len(gene_matrix)}")
    # print(f"\nTotal unique genes: {len(gene_names)}")
    # print("Genes:", gene_names)
    return gene_matrix, gene_names


def load_data(threshold=0.5):
    gene_matrix, gene_names = load_gene_data_for_RL4Con()
    adjacency_matrix, connection_matrix = load_connection_matrix()

    binary_gene_matrix = (gene_matrix > threshold).astype(int)
    gene_matrix = torch.from_numpy(binary_gene_matrix)
    # 找出非全0列（即：列求和 > 0）
    col_sum = gene_matrix.sum(dim=0)
    non_zero_col_mask = col_sum > 0  # 布尔 mask，长度 17561

    # 保留对应的列
    filtered_gene_matrix = gene_matrix[:, non_zero_col_mask]

    # 保留对应的 gene_names
    filtered_gene_names = [name for name, keep in zip(gene_names, non_zero_col_mask.tolist()) if keep]

    adjacency_matrix = torch.from_numpy(adjacency_matrix)

    connection_matrix = torch.from_numpy(connection_matrix)


    return filtered_gene_matrix.float(),filtered_gene_names,adjacency_matrix.float(),connection_matrix.float()


import torch


def extract_connected_gene_pairs(connection_matrix: torch.Tensor, gene_matrix: torch.Tensor):
    """
    从连接矩阵和基因表达矩阵中提取连接的神经元对及其表达的基因。

    参数:
    - connection_matrix: (N, N) 的 Tensor，表示神经元之间的连接（1表示连接）
    - gene_matrix: (N, G) 的布尔型 Tensor，表示每个神经元是否表达对应基因

    返回:
    - 一个包含若干字典的列表，每个字典表示一个连接神经元对及其表达基因
    """

    connected_gene_pairs = []
    N = connection_matrix.shape[0]

    for i in range(N):
        for j in range(N):
            if connection_matrix[i, j] == 1:
                pre_genes = torch.nonzero(gene_matrix[i], as_tuple=True)[0].tolist()
                post_genes = torch.nonzero(gene_matrix[j], as_tuple=True)[0].tolist()

                entry = {
                    "pos": (i, j),
                    "pre_gene": pre_genes,
                    "post_gene": post_genes,
                }
                connected_gene_pairs.append(entry)
    return connected_gene_pairs

def get_global_gene_intersections(entries):
    if not entries:
        return [], []

    # 初始化为第一个连接的pre_gene和post_gene集合
    pre_shared = set(entries[0]["pre_gene"])
    post_shared = set(entries[0]["post_gene"])

    # 逐个求交集
    for entry in entries[1:]:
        pre_shared &= set(entry["pre_gene"])
        post_shared &= set(entry["post_gene"])

    return list(pre_shared), list(post_shared)



def extract_key_genes_with_freq(connection_matrix: torch.Tensor, gene_matrix: torch.Tensor):
    """
    connection_matrix: (N, N) 二值矩阵
    gene_matrix: (N, G) 二值矩阵
    返回：一个列表（每个连接的 key_gene 字典）和一个基因频率统计 Counter
    """
    result = []
    all_key_genes = []
    N, G = gene_matrix.shape

    for i in range(N):
        for j in range(N):
            if connection_matrix[i, j] == 1:
                pos_neurons = torch.logical_or(connection_matrix[i] == 1, connection_matrix[:, j] == 1)
                pos_gene_mask = gene_matrix[pos_neurons]
                pos_gene_set = (pos_gene_mask.sum(dim=0) > 0).nonzero(as_tuple=True)[0].tolist()

                neg_neurons = torch.logical_or(connection_matrix[i] == 0, connection_matrix[:, j] == 0)
                neg_gene_mask = gene_matrix[neg_neurons]
                neg_gene_set = (neg_gene_mask.sum(dim=0) > 0).nonzero(as_tuple=True)[0].tolist()

                key_gene = set(pos_gene_set) - set(neg_gene_set)

                result.append({
                    "pos": (i, j),
                    "key_gene": key_gene
                })

                all_key_genes.extend(key_gene)  # 收集全部 key_gene

    # 基因频率统计
    freq_counter = Counter(all_key_genes)

    return result, freq_counter



if __name__ == "__main__":

    gene_matrix, gene_names, adjacency_matrix, connection_matrix =load_data()

    # extract_connected_gene_pairs(connection_matrix,gene_matrix)
    #
    # pre_shared_genes, post_shared_genes = get_global_gene_intersections(extract_connected_gene_pairs(connection_matrix,gene_matrix))
    #
    # results, freq_counter = extract_key_genes_with_freq(connection_matrix, gene_matrix)
    #
    #

