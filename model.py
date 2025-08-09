import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    对输入张量 x 按最后一个维度做标准化：
    z = (x - mean) / std

    Args:
        x: 输入张量，形状为 (batch_size, feature_dim)
        eps: 防止除以0的小值

    Returns:
        标准化后的张量
    """
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=eps)
    return (x - mean) / std



class MLPResBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(x + residual)

class CompositePolicyNet(nn.Module):
    def __init__(self, obs_dim, total_genes, clauses, base_dim=128, num_blocks=4, dropout=0.1):
        super().__init__()
        # 初始映射到 base_dim
        self.initial = nn.Sequential(
            nn.Linear(obs_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # ResNet 风格残差块序列
        self.res_layers = nn.Sequential(*[
            MLPResBlock(base_dim, base_dim*2, dropout)
            if i == 0 else MLPResBlock(base_dim, dropout=dropout)
            for i in range(num_blocks)
        ])
        # 最终特征层
        self.feature = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 输出头
        self.pre_head = nn.Linear(base_dim, total_genes + 1)  # +1 删除
        self.clause_head = nn.Linear(base_dim, clauses)

    def forward(self, obs):
        # obs: [batch, obs_dim]
        obs = normalize_input(obs)
        x = self.initial(obs)
        x = self.res_layers(x)
        features = self.feature(x)

        pre_logits = self.pre_head(features)
        clause_logits = self.clause_head(features)
        return pre_logits, clause_logits

# Example:
# model = CompositePolicyNet(obs_dim=env.observation_spec.shape[0], total_genes=100, rule_size=env.max_clauses)



# Example initialization:
# model = CompositePolicyNet(obs_dim=env.observation_spec.shape[0], total_genes=100, rule_size=env.max_clauses)
class ValueNet(nn.Module):
    def __init__(self, output_dim, hidden_dim=128, device="cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim, device=device),
            nn.Tanh(),
            nn.LazyLinear(hidden_dim, device=device),
            nn.Tanh(),
            nn.LazyLinear(hidden_dim, device=device),
            nn.Tanh(),
            nn.LazyLinear(1, device=device),
        )

    def forward(self, x):
        return self.net(normalize_input(x))