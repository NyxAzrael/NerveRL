import torch
import numpy as np
from typing import List, Dict,Tuple, Optional
from torchrl.envs import EnvBase
from torchrl.data import  Composite, Categorical, Bounded,Unbounded
from tensordict import TensorDict
import copy
import time

CANDIDATE_NUM = 20
MAX_GENE_PER_CLAUSE = 6
MAX_CLAUSE_NUM = 5

ANIMAL = 'drosophila'


class GeneticRule:
    """Represents a logical rule for genetic connections with multiple pre/post genes per clause"""

    def __init__(self, clauses):
        self.clauses = clauses
        self.normalize()

    def __str__(self):
        self.normalize()
        return str(self.clauses)

    def normalize_clause(self, clause):
        """Normalize a clause by sorting the pre and post genes"""
        pre_genes, post_genes = clause
        pre_genes.sort()
        post_genes.sort()
        return (pre_genes, post_genes)

    def normalize(self):
        self.clauses = [self.normalize_clause(clause) for clause in self.clauses]
        self.remove_repeated_clauses()

    def remove_repeated_clauses(self):
        """Remove repeated clauses"""
        new_clauses = []
        for i, (pre1, post1) in enumerate(self.clauses):
            is_duplicate = False
            for j in range(i):
                pre2, post2 = self.clauses[j]
                if pre1 == pre2 and post1 == post2:
                    is_duplicate = True
                    break
            if not is_duplicate:
                new_clauses.append((pre1, post1))
        self.clauses = new_clauses

    def add_clause(self, pre_genes: List[int], post_genes: List[int]) -> None:
        """Add a new clause to the rule

        Args:
            pre_genes: List of pre-synaptic gene indices
            post_genes: List of post-synaptic gene indices
        """
        # Ensure inputs are lists
        pre_genes = [pre_genes] if isinstance(pre_genes, (int, np.integer)) else list(pre_genes)
        post_genes = [post_genes] if isinstance(post_genes, (int, np.integer)) else list(post_genes)
        self.clauses.append((pre_genes, post_genes))
        self.normalize()

    def remove_clause(self, clause_index: int) -> None:
        """Remove a clause from the rule"""
        if 0 <= clause_index < len(self.clauses):
            self.clauses.pop(clause_index)

    def add_gene_for_clause(self, clause_index: int, gene_index: int, pre_or_post: str) -> None:
        """Add a gene to a clause"""
        if 0 <= clause_index < len(self.clauses) or clause_index == -1:
            if pre_or_post == 'pre':
                if gene_index not in self.clauses[clause_index][0]:
                    self.clauses[clause_index][0].append(gene_index)
            elif pre_or_post == 'post':
              if gene_index not in self.clauses[clause_index][1]:
                    self.clauses[clause_index][1].append(gene_index)
        self.normalize()

    def evaluate(self, pre_genes: torch.Tensor, post_genes: torch.Tensor) -> bool:
        """Evaluate the rule for given pre and post synaptic genes using PyTorch.

        Each clause is evaluated as:
            (pre_gene1 AND pre_gene2 AND ...) AND (post_gene1 AND post_gene2 AND ...)
        The final result is the OR across all clauses.

        Args:
            pre_genes: 1D tensor of shape [num_genes] for presynaptic neuron
            post_genes: 1D tensor of shape [num_genes] for postsynaptic neuron

        Returns:
            Boolean indicating whether the rule is satisfied
        """
        if not self.clauses:
            return False

        result = False
        for pre_indices, post_indices in self.clauses:
            pre_indices = [pre_indices] if isinstance(pre_indices, int) else pre_indices
            post_indices = [post_indices] if isinstance(post_indices, int) else post_indices

            pre_expr = pre_genes[pre_indices] > 0.5
            post_expr = post_genes[post_indices] > 0.5

            pre_condition = pre_expr.all().item()
            post_condition = post_expr.all().item()

            clause_result = pre_condition and post_condition
            result = result or clause_result

        return result


class ConnectionEnv(EnvBase):
    def __init__(self, gene_matrix, adjacency_matrix, connection_matrix, gene_names, reward_lambda=0.01):
        super().__init__()

        # 强制转为 torch.Tensor
        self.gene_matrix = torch.tensor(gene_matrix, dtype=torch.float32)
        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.bool)
        self.connection_matrix = torch.tensor(connection_matrix, dtype=torch.bool)

        self.device = self.gene_matrix.device
        self.gene_names = gene_names
        self.reward_lambda = reward_lambda

        # 规则相关
        self.current_rule = GeneticRule([])
        self.gene_interactome = self._initialize_gene_interactome()
        self.candidate_pairs = self.gene_interactome
        self.candidate_pairs_codebook = self._initialize_candidate_pairs_from_codebook()

        self.max_rule_length = MAX_CLAUSE_NUM

        # 初始化当前状态
        self._current_predictions = None
        self._done = False
        self.metrics = {}

        # 初始化环境规范（observation_spec, action_spec, reward_spec）
        self._make_specs()

    def _initialize_gene_interactome(self) -> List[Tuple[int, int]]:
        """Initialize known gene interactions from Ig family"""

        # Get gene interactions
        gene_interactome = create_Ig_family_interaction()
        indexed_interactions = []

        # Convert gene names to indices
        for pre_gene, post_gene in gene_interactome:
            try:
                pre_idx = self.gene_names.index(pre_gene)
                post_idx = self.gene_names.index(post_gene)
                indexed_interactions.append((pre_idx, post_idx))
                indexed_interactions.append((post_idx, pre_idx))
            except ValueError:
                continue  # Skip if gene not in our dataset

        return indexed_interactions

    def _initialize_candidate_pairs_from_codebook(self, top_n: int = 100):
        """Initialize candidate gene pairs using pseudo-inverse method with torch.

        Returns:
            candidate_pairs: List of (i, j) index tuples
            weights: Corresponding weight values from interaction matrix
        """
        # 使用 torch.linalg.pinv 计算伪逆
        gene_matrix_inv = torch.linalg.pinv(self.gene_matrix)
        gene_matrix_t_inv = torch.linalg.pinv(self.gene_matrix.T)

        interaction_matrix = gene_matrix_inv @ self.connection_matrix.float() @ gene_matrix_t_inv

        # 选出前 top_n 大值的 index（flatten 后排序）
        flat_scores = interaction_matrix.flatten()
        topk = torch.topk(flat_scores, k=top_n)

        flat_indices = topk.indices
        weights = topk.values

        # 还原为二维索引
        num_cols = interaction_matrix.shape[1]
        candidate_pairs = [(idx.item() // num_cols, idx.item() % num_cols) for idx in flat_indices]

        return candidate_pairs, weights.tolist()


    def _make_specs(self):
        self.observation_spec = Composite({
            # 规则由 max_rule_length 条子句组成，每条子句是 (pre_gene, post_gene) 两个整数索引
            "rule": Categorical(
                minimum=0,
                maximum =self.gene_matrix.shape[1] - 1,
                shape=(self.max_rule_length, self.max_rule_length),
                dtype=torch.long,
                device=self.device
            ),

        }, device=self.device)

        # 离散动作空间，有3个动作：add_clause, enhance_clause, optimize_clauses
        self.action_spec = Categorical(n=3, device=self.device)

        # 奖励是标量float32
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32, device=self.device)

        # done是标量bool
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool, device=self.device)

    def calculate_metrics(self, predictions: torch.Tensor) -> Dict[str, float]:
        """Calculate precision, recall, F1 score, and reward using torch."""

        # 确保类型一致
        conn = self.connection_matrix.to(dtype=predictions.dtype)

        tp = torch.sum((predictions == 1) & (conn == 1)).item()
        fp = torch.sum((predictions == 1) & (conn == 0)).item()
        fn = torch.sum((predictions == 0) & (conn == 1)).item()
        tn = torch.sum((predictions == 0) & (conn == 0)).item()

        epsilon = 1e-8

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)

        # 规则复杂度惩罚项
        rule_complexity = len(self.current_rule.clauses)

        complexity_penalty = self.reward_lambda * rule_complexity / MAX_CLAUSE_NUM

        reward = f1 - complexity_penalty

        return {
            'reward': reward,
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'rule_complexity': rule_complexity,
        }

    def _reset(self, tensordict=None, **kwargs):
        self.current_rule = GeneticRule([])
        if self.candidate_pairs:
            pre_idx, post_idx = self.candidate_pairs[0]
            self.current_rule.add_clause(pre_idx, post_idx)
        self._current_predictions = self._evaluate_current_rule(self.current_rule)


        obs = TensorDict({
            "rule": self._encode_rule(self.current_rule)
        }, batch_size=[])

        return obs

    def _encode_rule(self, rule: GeneticRule) -> torch.Tensor:
        # 如果当前子句少于 max_rule_length，补 -1
        clauses = rule.clauses
        padded = torch.full(
            (self.max_rule_length, self.max_rule_length),  # 固定 shape
            fill_value=0,
            dtype=torch.long,
            device=self.device
        )
        for i, (pre, post) in enumerate(clauses):
            if i >= self.max_rule_length:
                break
            if len(pre) > 0 and len(post) > 0:
                padded[i, 0] = pre[-1]  # 这里只能选一个 gene，可调整策略
                padded[i, 1] = post[-1]
        return padded

    def _evaluate_current_rule(self, rule: GeneticRule) -> torch.Tensor:
        """Evaluate current rule on all neuron pairs using PyTorch"""
        num_neurons = self.gene_matrix.shape[0]
        device = self.gene_matrix.device
        predictions = torch.zeros_like(self.connection_matrix, dtype=torch.bool, device=device)

        for i in range(num_neurons):
            pre_gene = self.gene_matrix[i]
            for j in range(num_neurons):
                if i == j:
                    continue
                post_gene = self.gene_matrix[j]
                if rule.evaluate(pre_gene, post_gene):  # rule.evaluate needs to support torch tensors
                    predictions[i, j] = True

        # Only keep predictions where adjacency_matrix is 1, set others to 0
        predictions = predictions & (self.adjacency_matrix == 1)
        return predictions.to(dtype=torch.float32)

    def _make_specs(self):

        self.observation_spec = Composite(
            rule=Categorical(
                n=self.gene_matrix.shape[1],
                shape=torch.Size([5, 5]),
                dtype=torch.int64,
            )
        )

        # 离散动作空间，有3个动作：add_clause, enhance_clause, optimize_clauses
        self.action_spec = Categorical(n=3, device=self.device)

        # 奖励是标量float32
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32, device=self.device)

        # done是标量bool
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool, device=self.device)

    def _step(self, tensordict):
        t0 = time.time()
        action = int(tensordict["action"].argmax().item())
        # 修改规则

        self.current_predictions = self._evaluate_current_rule(self.current_rule)
        self.metrics = self.calculate_metrics(self.current_predictions)


        if action == 0:
            new_pre, new_post = self._select_genes_for_new_clause(self.metrics)
            self.current_rule .add_clause(new_pre, new_post)
        if action == 1:
            self.current_rule  = self._enhance_existing_clauses(self.current_rule, self.metrics)
        if action == 2:
            self.current_rule  = self._optimize_current_clauses(self.current_rule, self.metrics)


        # 重新计算预测和指标
        self.current_predictions = self._evaluate_current_rule(self.current_rule)
        metrics = self.calculate_metrics(self.current_predictions)

        reward = metrics['reward']

        # 返回符合 TorchRL 规范的 tensordict
        return TensorDict({
            "rule": self._encode_rule(self.current_rule),
            "reward": torch.tensor(reward, dtype=torch.float32, device=self.device),
            "done": torch.tensor(False, dtype=torch.bool, device=self.device),
        })
    def _get_random_gene_pair(self) -> Optional[Tuple[List[int], List[int]]]:
        """Get a random gene pair from known interactions using torch operations

        Returns:
            Tuple of ([pre_gene], [post_gene]) or None if no interactions exist
        """
        if self.gene_interactome is None or not self.gene_interactome:
            return None

        idx = torch.randint(
            low=0,
            high = len(self.gene_interactome),
            size=(1,),
            device=self.device
        ).item()

        pair = self.gene_interactome[idx]
        return [pair[0]], [pair[1]]

    def _select_genes_for_new_clause(self, metrics: Dict[str, float]) -> Tuple[List[int], List[int]]:
        """使用错误分析选择用于新子句的基因（PyTorch版本）"""

        # 取出必要矩阵
        conn = self.connection_matrix.bool()  # [N, N]
        pred = self.current_predictions.bool()  # [N, N]
        gene_matrix = self.gene_matrix.float()  # [N, G]

        # False Negatives: 连接为1但预测为0
        fn_mask = conn & (~pred)

        # 有 False Negative 的 presynaptic neuron 索引
        fn_pre_mask = fn_mask.any(dim=1)  # shape: [N]
        fn_post_mask = fn_mask.any(dim=0)  # shape: [N]

        if fn_pre_mask.any() and fn_post_mask.any():
            # FN pre neurons 表达的基因平均（每列代表一个基因）
            fn_pre_genes = gene_matrix[fn_pre_mask]  # [num_fn_pre, G]
            pre_gene_scores = fn_pre_genes.mean(dim=0)  # [G]
            pre_gene = int(torch.argmax(pre_gene_scores).item())

            # FN post neurons 表达的基因平均
            fn_post_genes = gene_matrix[fn_post_mask]  # [num_fn_post, G]
            post_gene_scores = fn_post_genes.mean(dim=0)  # [G]
            post_gene = int(torch.argmax(post_gene_scores).item())

            return [pre_gene], [post_gene]

        # 如果没有 false negative，返回随机基因对
        return self._get_random_gene_pair()

    def _enhance_existing_clauses(self, rule: GeneticRule, metrics: Dict[str, float]) -> GeneticRule:
        """基于False Positive或候选对权重，为已有子句添加基因（PyTorch版）"""

        random_num = torch.rand(1).item()
        CANDIDATE_NUM = 1000  # 你代码中的常量，确保定义

        if random_num < 0.5:
            # 1. 基于False Positive添加基因
            for clause_idx, (pre_genes, post_genes) in enumerate(rule.clauses):
                # FP mask: 连接为0但预测为1
                fp_mask = (~self.connection_matrix.bool()) & (self.current_predictions.bool())

                # FP pre neurons 索引
                fp_pre_mask = fp_mask.any(dim=1)  # [N]
                if fp_pre_mask.any():
                    fp_pre_genes = self.gene_matrix[fp_pre_mask]  # [num_fp_pre, G]
                    gene_means = fp_pre_genes.float().mean(dim=0)  # [G]

                    # 选出top CANDIDATE_NUM基因索引
                    top_pre_gene_vals, top_pre_gene_indices = torch.topk(gene_means,
                                                                         min(CANDIDATE_NUM, gene_means.size(0)))
                    weights = top_pre_gene_vals / top_pre_gene_vals.sum()

                    new_pre = top_pre_gene_indices[torch.multinomial(weights, 1)].item()
                    rule.add_gene_for_clause(clause_idx, new_pre, 'pre')

                # FP post neurons 索引
                fp_post_mask = fp_mask.any(dim=0)  # [N]
                if fp_post_mask.any():
                    fp_post_genes = self.gene_matrix[fp_post_mask]  # [num_fp_post, G]
                    gene_means = fp_post_genes.float().mean(dim=0)  # [G]

                    top_post_gene_vals, top_post_gene_indices = torch.topk(gene_means,
                                                                           min(CANDIDATE_NUM, gene_means.size(0)))
                    weights = top_post_gene_vals / top_post_gene_vals.sum()

                    new_post = top_post_gene_indices[torch.multinomial(weights, 1)].item()
                    rule.add_gene_for_clause(clause_idx, new_post, 'post')

        else:
            # 2. 基于伪逆候选对权重添加基因
            candidate_pairs_with_weight = self.candidate_pairs_codebook
            candidate_pairs, weights = candidate_pairs_with_weight  # candidate_pairs: List[Tuple[int,int]]，weights: List[float]

            # 转成torch tensor
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
            weights_tensor = weights_tensor / weights_tensor.sum()

            clause_idx = torch.randint(len(rule.clauses), (1,)).item()
            pair_idx = torch.multinomial(weights_tensor, 1).item()
            new_pre, new_post = candidate_pairs[pair_idx]

            rule.add_gene_for_clause(clause_idx, new_pre, 'pre')
            rule.add_gene_for_clause(clause_idx, new_post, 'post')

        return rule

    def _optimize_current_clauses(self, rule: GeneticRule, metrics: Dict[str, float]) -> GeneticRule:
        """用 PyTorch 优化已有子句，移除冗余基因"""

        for clause_idx, (pre_genes, post_genes) in enumerate(rule.clauses):
            # 超过最大基因数限制时尝试移除无贡献基因
            if (len(pre_genes) + len(post_genes)) > (MAX_GENE_PER_CLAUSE - 2):
                if len(pre_genes) > 1:
                    # 注意迭代时不能直接在列表中删除，先做拷贝
                    for gene in pre_genes[:]:
                        if not self._gene_contribution(gene, 'pre', metrics):
                            rule.clauses[clause_idx][0].remove(gene)
                            # print(f"Removed pre gene {gene} from clause {clause_idx}")

                if len(post_genes) > 1:
                    for gene in post_genes[:]:
                        if not self._gene_contribution(gene, 'post', metrics):
                            rule.clauses[clause_idx][1].remove(gene)
                            # print(f"Removed post gene {gene} from clause {clause_idx}")

        # 移除表现最差的子句
        if len(rule.clauses) > (MAX_CLAUSE_NUM - 2):
            clause_performance = self._evaluate_clause_performance(rule)  # 假设返回 torch.Tensor 或可转为tensor
            if isinstance(clause_performance, np.ndarray):
                clause_performance = torch.tensor(clause_performance, device=self.device)

            worst_clause = torch.argmin(torch.tensor(clause_performance)).item()
            if clause_performance[worst_clause] < 0.1:
                rule.remove_clause(worst_clause)

        return rule

    def _gene_contribution(self, gene_idx: int, role: str, metrics: Dict[str, float]) -> bool:
        """Check if a gene contributes to true positives, using torch tensors"""

        # connection_matrix和current_predictions假设是torch.Tensor，gene_matrix也是torch.Tensor
        if role == 'pre':
            # axis=1表示按行，any后得到的布尔向量
            tp_mask = (self.connection_matrix == 1).any(dim=1)  # [num_neurons_pre]
            tp_genes = self.gene_matrix[tp_mask, gene_idx]
            tp_contrib = tp_genes.float().mean().item()

            fp_mask = (self.current_predictions == 1).any(dim=1)
            fp_genes = self.gene_matrix[fp_mask, gene_idx]
            fp_contrib = fp_genes.float().mean().item()
        else:
            tp_mask = (self.connection_matrix == 1).any(dim=0)  # [num_neurons_post]
            tp_genes = self.gene_matrix[tp_mask, gene_idx]
            tp_contrib = tp_genes.float().mean().item()

            fp_mask = (self.current_predictions == 1).any(dim=0)
            fp_genes = self.gene_matrix[fp_mask, gene_idx]
            fp_contrib = fp_genes.float().mean().item()

        return tp_contrib >= fp_contrib

    def _evaluate_clause_performance(self, rule: GeneticRule) -> List[float]:
        """Evaluate individual clause performance, use torch for internal data"""

        clause_scores = []
        original_clauses = copy.deepcopy(rule.clauses)

        for i in range(len(original_clauses)):
            # 移除第i个子句
            rule.clauses = [c for j, c in enumerate(original_clauses) if j != i]
            rule = GeneticRule(clauses=rule.clauses)

            # 调用环境评估当前规则，返回tensor或numpy都可以
            predictions = self._evaluate_current_rule(rule)  # 预测tensor
            metrics = self.calculate_metrics(predictions)  # dict

            clause_scores.append(metrics['f1_score'])

        rule.clauses = original_clauses
        return clause_scores

    def _set_seed(self, seed: int) -> None:
        """Implements required abstract method to set seed."""
        self._seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(self, "rng"):
            self.rng.manual_seed(seed)

    def get_metrics(self):
        return self.metrics



def create_Ig_family_interaction():
    Dpr_DIP_gene_interactome = [
        # ['dpr12','DIP-delta'],
        ['dpr10', 'DIP-alpha'],
        ['dpr6', 'DIP-alpha'],
        ['dpr6', 'DIP-beta'],
        ['dpr6', 'DIP-zeta'],
        ['dpr6', 'DIP-epsilon'],

        ['dpr8', 'DIP-beta'],
        ['dpr9', 'DIP-beta'],
        ['dpr21', 'DIP-beta'],
        ['dpr11', 'DIP-beta'],
        ['dpr11', 'DIP-gamma'],

        ['dpr13', 'DIP-zeta'],
        ['dpr13', 'DIP-epsilon'],

        ['dpr16', 'DIP-zeta'],
        ['dpr16', 'DIP-epsilon'],
        ['dpr16', 'DIP-gamma'],

        ['dpr19', 'DIP-zeta'],
        ['dpr19', 'DIP-epsilon'],
        ['dpr20', 'DIP-zeta'],
        ['dpr20', 'DIP-epsilon'],
        ['dpr14', 'DIP-epsilon'],

        ['dpr17', 'DIP-epsilon'],
        ['dpr17', 'DIP-gamma'],
        ['dpr15', 'DIP-gamma'],

        ['dpr1', 'DIP-iota'],
        ['dpr1', 'DIP-theta'],
        ['dpr1', 'DIP-eta'],
        ['dpr2', 'DIP-theta'],
        ['dpr2', 'DIP-eta'],
        ['dpr3', 'DIP-theta'],
        ['dpr3', 'DIP-eta'],
        ['dpr4', 'DIP-theta'],
        ['dpr4', 'DIP-eta'],
        ['dpr5', 'DIP-theta'],
        ['dpr7', 'DIP-theta'],
        ['dpr7', 'DIP-eta'],
        # 943 dpr16
        # 1580 dpr19
        # 2556 dpr11
        # 2967 dpr5
        # 3035 dpr17
        # 3039 dpr15
        # 5614 dpr14
        # 7165 dpr7
        # 7223 dpr9
        # 7648 dpr18
        # 7760 dpr4
        # 7763 dpr3
        # 7796 dpr1
        # 8670 dpr13
        # 9222 dpr12
        # 9980 dpr21
        # 10253 dpr2
        # 11294 dpr20
        # 15248 dpr8
        # 16613 dpr10
        # 16626 dpr6
        #
        # 308 DIP-alpha
        # 1093 DIP-theta
        # 1094 DIP-eta
        # 1224 DIP-iota
        # 1466 DIP-zeta
        # 1936 DIP-kappa
        # 4672 DIP-gamma
        # 7468 DIP1
        # 9226 DIP-delta
        # 9667 DIP-beta
        # 9693 DIP-epsilon
        # 11248 DIP2
        # 13747 DIP-lambda
    ]

    Side_Beat_gene_interactome = [
        ['side', 'beat-Ia'],
        ['side', 'beat-Ib'],
        ['side', 'beat-Ic'],

        ['side-III', 'beat-Ic'],
        ['side-IV', 'beat-IIa'],
        ['side-IV', 'beat-IIb'],

        ['side-VII', 'beat-IV'],

        ['side-VI', 'beat-Va'],
        ['side-VI', 'beat-Vb'],
        ['side-VI', 'beat-Vc'],

        ['side-II', 'beat-VI'],
        ['side-II', 'side-III'],
        # 2841 side-VII
        # 4581 side
        # 8713 side-IV
        # 8816 side-III
        # 8817 side-VI
        # 9213 side-V
        # 9436 side-VIII
        # 9642 side-II

        # 2048 beat-Ic
        # 2051 beat-Ia
        # 2053 beat-Ib
        # 2129 beat-IIIb
        # 2145 beat-IIIc
        # 3119 beat-Vc
        # 3137 beat-Vb
        # 3546 beat-IIb
        # 3574 beat-IIa
        # 4135 beat-IV
        # 4638 beat-VI
        # 7803 beat-Va
        # 9583 beat-VII
        # 12317 beat-IIIa
    ]
    if ANIMAL == 'drosophila':
        gene_interactome = Side_Beat_gene_interactome + Dpr_DIP_gene_interactome
    elif ANIMAL == 'celegans':
        gene_interactome = [
            ['inx-3','inx-3'],
            ['inx-6','inx-6'],
            ['inx-10','inx-11'],
            ['inx-19','inx-19'],
            ['unc-9','unc-9']]
    return gene_interactome


if __name__ == "__main__":
    from torchrl.envs.utils import check_env_specs
    from data_utils import load_gene_data,load_connection_matrix

    gene_matrix, gene_names = load_gene_data()
    adjacency_matrix, connection_matrix = load_connection_matrix()
    env =ConnectionEnv(gene_matrix=gene_matrix,
                     gene_names=gene_names,
                     adjacency_matrix=adjacency_matrix,
                     connection_matrix=connection_matrix)
    check_env_specs(env)  # 确保实现符合 TorchRL 要求
