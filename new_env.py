import torch
from torchrl.envs.common import EnvBase
from tensordict import TensorDict
from torchrl.data import  Composite, Categorical,Unbounded
import matplotlib.pyplot as plt
import random
from torchrl.envs.utils import check_env_specs
from data_utils import load_data
from torchrl.envs import StepCounter,TransformedEnv
from test import common_gene

class ConnectionEnv(EnvBase):
    def __init__(self, gene_matrix, gene_names,
                 adjacency_matrix, connection_matrix,common_genes,
                 max_clauses: int = 5, max_pre_genes: int = 5 , max_post_genes: int =5,
                 device='cpu'):

        super().__init__(device=device)

        self.gene_matrix = gene_matrix.to(device)
        self.gene_names = gene_names
        self.common_gene = common_genes
        self.common_gene_idx = len(common_genes)
        self.adjacency_matrix = adjacency_matrix.to(device)
        self.connection_matrix = connection_matrix.to(device)

        self.max_clauses = max_clauses
        self.max_pre_genes = max_pre_genes
        self.max_post_genes = max_post_genes
        self.genes_per_clause = max_pre_genes + max_post_genes

        self.total_genes = gene_matrix.shape[1]
        self.rule_size = self.max_clauses * self.genes_per_clause

        # 初始化空规则矩阵（动作作用的地方）
        self.current_rule = torch.zeros(self.rule_size, dtype=torch.int64, device=self.device)
        self.best_rule = self.current_rule
        self.best_f1 = 0
        self.best_precision = 0
        self.best_metrics = {}

        self.metrics = {}

        # 初始化环境规范（observation_spec, action_spec, reward_spec）
        self._make_specs()

    def _make_specs(self):
        # Observation spec: 当前规则矩阵，类型为长整型向量（0 或 1）
        self.observation_spec = Composite({
            "rule": Categorical(
                n=self.total_genes+1,  # 0 or 1
                shape=torch.Size([self.rule_size]),
                dtype=torch.int64,
                device=self.device
            )
        })

        self.action_spec = Composite({
            "pair_idx": Categorical(
                # 0 is 删除或者什么也不干，1-self.total_genes + 1代表基因
                n=self.common_gene_idx + 1,
                shape=torch.Size([]),
                dtype=torch.int64,
                device=self.device
            ),
            "clause_idx": Categorical(
                n=self.max_clauses,
                shape=torch.Size([]),
                dtype=torch.int64,
                device=self.device
            )
        })

        self.reward_spec = Unbounded(shape=(torch.Size([1])), dtype=torch.float32, device=self.device)

        # done是标量bool
        self.done_spec = Unbounded(shape=(torch.Size([1])), dtype=torch.bool, device=self.device)


    def _set_seed(self, seed: int) -> int:
        # 设置当前环境的种子
        self._seed = seed
        torch.manual_seed(seed)
        return seed

    def _reset(self, tensordict=None, **kwargs):
        # 每次 reset 初始化规则矩阵
        self.current_rule = torch.zeros(self.rule_size, dtype=torch.int64, device=self.device)

        return TensorDict({
            "rule": self.current_rule.clone()
        }, batch_size=[])

    def _step(self, tensordict: TensorDict) -> TensorDict:
        pair_idx = tensordict["pair_idx"].item()
        clause_idx = tensordict["clause_idx"].item()

        new_rule = self.current_rule.clone().reshape(self.max_clauses, self.genes_per_clause)
        new_clause = new_rule[clause_idx]
        pre_gene = new_clause[:self.max_pre_genes]
        post_gene = new_clause[self.max_pre_genes:]

        if pair_idx > 0:
            pre_gen_idx, post_gen_idx = self.common_gene[pair_idx - 1]
        else:
            pre_gen_idx, post_gen_idx = 0, 0

        pre_pos = torch.nonzero(pre_gene == 0).view(-1)
        post_pos = torch.nonzero(post_gene == 0).view(-1)

        if pre_pos.numel() == 0:
            pre_pos = random.randint(0, len(pre_gene) - 1)
        else:
            pre_pos = pre_pos[0].item()

        if post_pos.numel() == 0:
            post_pos = random.randint(0, len(post_gene) - 1)
        else:
            post_pos = post_pos[0].item()

        pre_gene[pre_pos] = pre_gen_idx
        post_gene[post_pos] = post_gen_idx

        new_clause[:self.max_pre_genes] = pre_gene
        new_clause[self.max_pre_genes:] = post_gene
        new_rule[clause_idx] = new_clause
        new_rule = new_rule.view(-1)

        if self.has_duplicate_clause(new_rule) or not self.is_valid_action(new_rule):

            reward = -1.0
            return self._build_step_result(reward=reward)

        self.current_rule = new_rule
        self.metrics = self.evaluate(self.generate_prediction_matrix())

        # 可选：计算预测与 reward
        reward = (
                1.0 * self.metrics["TP"] # 鼓励找出真正例
                - 0.5 * self.metrics["FP"] # 惩罚错误地预测正例
                - 0.1 * self.metrics["FN"]  # 轻微惩罚漏掉正例（防止压制 recall）
                + 0.01 * self.metrics["TN"]  # 鼓励维持负例的正确性
        )

        # 最终奖励组合：引导奖励 + F1 奖励（混合）
        final_reward = reward + 5.0 * self.metrics["F1"]


        return self._build_step_result(final_reward)

    def is_valid_action(self, rule) -> bool:
        rule = rule.reshape(self.max_clauses, self.genes_per_clause)
        for clause in rule:
            pre_gene = clause[:self.max_pre_genes]
            post_gene = clause[self.max_pre_genes:]

            # 只取非零基因进行重复性检查
            pre_nonzero = pre_gene[pre_gene != 0]
            post_nonzero = post_gene[post_gene != 0]

            if len(set(pre_nonzero.tolist())) != len(pre_nonzero):
                return False

            if len(set(post_nonzero.tolist())) != len(post_nonzero):
                return False

        return True

    def has_duplicate_clause(self, rule) -> bool:
        """
        删除全为 0 的子句后，检查是否存在重复（pre/post 基因组合一样）
        返回 True 表示存在重复，False 表示无重复
        """
        rule = rule.reshape(self.max_clauses, self.genes_per_clause)

        # 删除所有全为 0 的子句
        non_zero_clauses = rule[torch.any(rule != 0, dim=1)]

        seen = set()
        for clause in non_zero_clauses:
            pre = clause[:self.max_pre_genes]
            post = clause[self.max_pre_genes:]

            pre_set = frozenset(pre[pre != 0].tolist())
            post_set = frozenset(post[post != 0].tolist())
            key = (pre_set, post_set)

            if key in seen:
                return True
            seen.add(key)

        return False

    def _build_step_result(self,reward = 0.0, done = False) -> TensorDict:
        return TensorDict({
            "rule": self.current_rule.clone(),
            "reward": torch.tensor(reward, dtype=torch.float32, device=self.device),
            "done": torch.tensor(done, dtype=torch.bool, device=self.device)
        }, batch_size=[])

    def generate_prediction_matrix(self,rule = None):
        """
        基于当前 self.current_rule 和 gene_matrix 生成预测连接矩阵
        Returns: prediction_matrix: Tensor[neuron_num, neuron_num]
        """
        if rule is None:
            rule = self.current_rule

        prediction_matrix = torch.zeros_like(self.connection_matrix, dtype=torch.int32)

        rule_matrix = rule.view(self.max_clauses, self.max_pre_genes + self.max_post_genes)

        for clause in rule_matrix:
            pre_gene_ids = clause[:self.max_pre_genes]
            post_gene_ids = clause[self.max_pre_genes:]

            # 找出非0基因索引
            # 找出非0基因索引，并转换为实际索引（0-based）
            pre_genes = (pre_gene_ids[pre_gene_ids != 0] - 1).unique()
            post_genes = (post_gene_ids[post_gene_ids != 0] - 1).unique()

            # 如果没有有效基因就跳过
            if len(pre_genes) == 0 or len(post_genes) == 0:
                continue

            # 逻辑与匹配：选出所有表达了这些基因的神经元
            # gene_matrix: [neuron_num, gene_num]
            pre_mask = self.gene_matrix[:, pre_genes].all(dim=1)  # 所有 pre_genes 都是1
            post_mask = self.gene_matrix[:, post_genes].all(dim=1)  # 所有 post_genes 都是1

            pre_indices = torch.where(pre_mask)[0]
            post_indices = torch.where(post_mask)[0]

            # 在 prediction_matrix[i, j] = 1
            for i in pre_indices:
                for j in post_indices:
                    prediction_matrix[i, j] = 1

        prediction_matrix = prediction_matrix.bool() & self.adjacency_matrix.bool()
        prediction_matrix = prediction_matrix.to(dtype=torch.int32)

        return prediction_matrix

    def evaluate(self, prediction: torch.Tensor) -> dict:
        """
        与 self.connection_matrix 进行比较，计算评价指标
        :param prediction: [num_neurons, num_neurons]，预测矩阵（0/1）
        :return: dict 包含 TP、FP、FN、TN、Precision、Recall、F1、Accuracy 等
        """
        conn = self.connection_matrix.to(dtype=prediction.dtype)

        TP = torch.sum((prediction == 1) & (conn == 1)).item()
        FP = torch.sum((prediction == 1) & (conn == 0)).item()
        FN = torch.sum((prediction == 0) & (conn == 1)).item()
        TN = torch.sum((prediction == 0) & (conn == 0)).item()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_rule = self.current_rule
            torch.save(self.current_rule, "best_rule.pt")

        if precision > self.best_precision:
            self.best_precision = precision


        return {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy
        }

    def visualize_rule_performance(self, rule) -> None:
        """
        Visualize performance of current rule
        """
        predictions = self.generate_prediction_matrix(rule)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # True connections
        im1 = ax1.imshow(self.connection_matrix, cmap='Blues')
        ax1.set_title('True Connections')
        ax1.set_xlabel('Post-synaptic')
        ax1.set_ylabel('Pre-synaptic')
        plt.colorbar(im1, ax=ax1)

        # Predicted connections
        im2 = ax2.imshow(predictions, cmap='Blues')
        ax2.set_title('Predicted Connections')
        ax2.set_xlabel('Post-synaptic')
        ax2.set_ylabel('Pre-synaptic')
        plt.colorbar(im2, ax=ax2)

        # Difference (errors)
        differences = predictions - self.connection_matrix
        im3 = ax3.imshow(differences, cmap='RdBu')
        ax3.set_title('Prediction Errors\n(Red: False Pos, Blue: False Neg)')
        ax3.set_xlabel('Post-synaptic')
        ax3.set_ylabel('Pre-synaptic')
        plt.colorbar(im3, ax=ax3)

        # Add rule text
        rule_str = self.print_the_rule(rule)
        rule_text = "Current Rule: " + rule_str
        plt.suptitle(rule_text)
        # if self.current_rule.clauses:
        #     rule_text = "Current Rule: "
        #     for i, (pre_genes, post_genes) in enumerate(self.current_rule.clauses):
        #         if i == 0:
        #             rule_text += f"({' ∧ '.join(map(str, pre_genes))}) AND ({' ∧ '.join(map(str, post_genes))})"
        #         else:
        #             rule_text += f" OR ({' ∧ '.join(map(str, pre_genes))}) AND ({' ∧ '.join(map(str, post_genes))})"
        #     plt.suptitle(rule_text)

        plt.tight_layout()
        plt.show()

    def print_the_rule(self,rule):
        """
        Print the final genetic rule and its performance metrics

        Args:
            rule_clauses: List of rule clauses
            performance_metrics: Dictionary containing performance metrics
            gene_names: Optional list of gene names for readable output
        """
        rule_str = ''
        rule_clauses = rule.reshape(self.max_clauses,self.genes_per_clause)
        for i,clause in enumerate(rule_clauses):
            pre_genes = clause[:self.max_pre_genes]
            post_genes = clause[self.max_pre_genes:]


                # Format pre-synaptic genes
            pre_terms = [f"g{idx}:{self.gene_names[idx]}" for idx in pre_genes]
            pre_expr = " ∧ ".join(pre_terms)

                # Format post-synaptic genes
            post_terms = [f"h{idx}:{self.gene_names[idx]}" for idx in post_genes]
            post_expr = " ∧ ".join(post_terms)


            if i == 0:
                rule_str += f"({pre_expr}) ∧ ({post_expr})"
            else:
                rule_str += f" ∨ ({pre_expr}) ∧ ({post_expr})"


        return rule_str


def make_env(seed=None,max_clauses = 5, max_pre_genes = 5 , max_post_genes =5, device=torch.device('cpu'),max_steps = 50):
    gene_matrix, gene_names, adjacency_matrix, connection_matrix = load_data()
    common = list(common_gene(threshold=0.7, step=0.1, gene_matrix=gene_matrix, connection_matrix=connection_matrix))
    print(common)

    env = ConnectionEnv(gene_matrix=gene_matrix,
                        gene_names=gene_names,
                        adjacency_matrix=adjacency_matrix,
                        connection_matrix=connection_matrix,
                        common_genes=common,
                        max_clauses=max_clauses,
                        max_pre_genes=max_pre_genes,
                        max_post_genes=max_post_genes,
                        device=device)

    if seed is not None:
        env.set_seed(seed)
    check_env_specs(env)

    env = TransformedEnv(env,StepCounter(max_steps=max_steps))

    return env


if __name__ == "__main__":


    env = make_env()  # 确保实现符合 TorchRL 要求




