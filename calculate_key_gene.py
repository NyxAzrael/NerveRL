import torch
from itertools import product
from collections import defaultdict
from data_utils import load_data


def filtered_gene():
    gene_matrix, gene_names, adjacency_matrix, connection_matrix = load_data(threshold=0.8)


    mask = ~(gene_matrix == 1).all(dim=0)  # True = 保留, False = 全1列


    filtered_gene_matrix = gene_matrix[:, mask]


    filtered_gene_names = [name for i, name in enumerate(gene_names) if mask[i]]

    return filtered_gene_matrix,filtered_gene_names,adjacency_matrix,connection_matrix


def connect_gene(gene_matrix,connection_matrix):

# 假设 gene_matrix: (39, num_genes)
# connection_matrix: (39, 39)
    num_neurons, num_genes = gene_matrix.shape

    threshold = 0.5
    gene_mask = gene_matrix > threshold

# 分割连接对和不连接对
    conn_pairs = [(i, j) for i in range(num_neurons) for j in range(num_neurons) if i != j and connection_matrix[i, j] == 1]
    unconn_pairs = [(i, j) for i in range(num_neurons) for j in range(num_neurons) if i != j and connection_matrix[i, j] == 0]

# 用字典记录每个基因组合对应的 (i,j) 列表
    conn_counts = defaultdict(list)

# 统计连接对里每个组合对应的神经元对
    for i, j in conn_pairs:
        pre_genes = torch.nonzero(gene_mask[i]).flatten().tolist()
        post_genes = torch.nonzero(gene_mask[j]).flatten().tolist()
        for g_pre, g_post in product(pre_genes, post_genes):
            conn_counts[(g_pre, g_post)].append((i, j))

# 筛掉在不连接对里出现的组合
    for i, j in unconn_pairs:
        pre_genes = torch.nonzero(gene_mask[i]).flatten().tolist()
        post_genes = torch.nonzero(gene_mask[j]).flatten().tolist()
        for g_pre, g_post in product(pre_genes, post_genes):
            if (g_pre, g_post) in conn_counts:
                del conn_counts[(g_pre, g_post)]

    remaining_pairs = set()  # 存所有需要覆盖的神经元对
    for pairs in conn_counts.values():
        remaining_pairs.update(pairs)


    sorted_combos = sorted(conn_counts.items(), key=lambda x: len(x[1]), reverse=True)




    selected_combos = []  # 最终筛选出的基因组合

    covered_pairs = set()

    for (g_pre, g_post), neuron_pairs in sorted_combos:
    # 找出当前组合可以覆盖的“未覆盖”神经元对
        new_pairs = [p for p in neuron_pairs if p not in covered_pairs]
        if len(new_pairs) > 0:
            selected_combos.append(((g_pre, g_post), new_pairs))
        # 更新已覆盖的神经元对
            covered_pairs.update(new_pairs)

    return selected_combos
    # for (g_pre, g_post), neuron_pairs in selected_combos:
    #     print(f"Selected gene pair ({g_pre}, {g_post}) covers neuron pairs: {neuron_pairs}")



def generate_prediction(gene_matrix, gene_rule, adjacency_matrix):

    num_neurons = gene_matrix.shape[0]

    # 初始化预测矩阵为 0/1 int
    pred_matrix = torch.zeros((num_neurons, num_neurons), dtype=torch.int)

    # 遍历基因组合
    for (g_pre, g_post), neuron_pairs in gene_rule:
        for i, j in neuron_pairs:
            # 如果基因表达值大于0，就认为表达
            if gene_matrix[i, g_pre] > 0 and gene_matrix[j, g_post] > 0:
                pred_matrix[i, j] = 1

    # 限制在可能连接范围内
    pred_matrix &= adjacency_matrix.to(torch.int)  # 将 float 或 bool 转成 int 后再按位与

    return pred_matrix.float()


def rule_to_text(gene_names, gene_rule):

    rule_texts = []

    for (g_pre, g_post), neuron_pairs in gene_rule:
        # 每条规则内部是 AND
        text = f"({gene_names[g_pre]} ^ {gene_names[g_post]})"
        rule_texts.append(text)

    # 规则列表内部是 OR
    return " V ".join(rule_texts)



if __name__ == "__main__":
    from data_utils import visualize_rule_performance
    gene_matrix, gene_names, adjacency_matrix, connection_matrix = filtered_gene()

    gene_rule = connect_gene(gene_matrix,connection_matrix)

    prediction = generate_prediction(gene_matrix,gene_rule,adjacency_matrix)

    print(rule_to_text(gene_names,gene_rule))


    visualize_rule_performance(prediction,connection_matrix)