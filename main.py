from envs import ConnectionEnv
from data_utils import load_gene_data, load_connection_matrix
from tensordict.nn import TensorDictModule
from torchrl.modules import Actor
from tqdm import tqdm
from torchrl.envs import StepCounter, TransformedEnv
import time
from torchrl._utils import logger as torchrl_logger
from torchrl.envs import  check_env_specs
from tensordict.nn import TensorDictSequential
from torchrl.modules import EGreedyModule,QValueModule
import torch

import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

def make_env(seed=None):
    gene_matrix, gene_names = load_gene_data()
    adj_matrix, conn_matrix = load_connection_matrix()
    env = ConnectionEnv(
        gene_matrix=gene_matrix,
        gene_names=gene_names,
        adjacency_matrix=adj_matrix,
        connection_matrix=conn_matrix,
    )
    # env = TransformedEnv(env, StepCounter())
    if seed is not None:
        env.set_seed(seed)
    check_env_specs(env)
    return env


env = make_env()
env = TransformedEnv(env, StepCounter(max_steps=50))

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 64),         # 全连接层，25->64
            nn.Tanh(),
            nn.Linear(64, 64),         # 第二层，64->64
            nn.Tanh(),
            nn.Linear(64, 3)           # 输出层，64->3
        )

    def norm_tensor(self,x):
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8
        x = (x - mean) / std  # 仅在 forward 中标准化


        return x

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1) # -> [1, 5, 5]
        else:
            x = x.view(x.shape[0], -1)

        return self.net(self.norm_tensor(x))

module= MLP()
value_net = TensorDictModule(module, in_keys=["rule"], out_keys=["action_value"])

policy = TensorDictSequential(value_net, QValueModule(spec=env.action_spec))

exploration_module = EGreedyModule(
    spec=env.action_spec, annealing_num_steps=50, eps_init=0.5
)
exploration_policy = TensorDictSequential(policy, exploration_module)



from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer

init_rand_steps = 500
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    exploration_policy,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,
)
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))

from torch.optim import Adam

from torchrl.objectives import DQNLoss, SoftUpdate

loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
optim = Adam(loss.parameters(), lr=0.02)
updater = SoftUpdate(loss, eps=0.99)

total_count = 0
total_episodes = 0
t0 = time.time()




# 假设 collector 是一个 iterable，包上 tqdm 进度条
for i, data in enumerate(tqdm(collector, desc="Collecting")):
    # 写入 replay buffer
    rb.extend(data)
    max_length = rb[:]["next", "step_count"].max()
    print("\n正在第{}轮数据收集,数据量{}".format(i+1,len(rb)))


    if len(rb) > init_rand_steps:
        print("start training.......")
        for _ in range(optim_steps):
            sample = rb.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            exploration_module.step(data.numel())
            updater.step()

            # 打印日志信息
            if (i + 1) % 1 == 0:
                metrics = env.get_metrics()
                torchrl_logger.info(f"Max num steps: {max_length}, rb length {len(rb)}")
                torchrl_logger.info("F1 Score now is {},Reward now is {}".
                                    format(metrics["f1_score"],metrics["reward"]))

            total_count += data.numel()
            total_episodes += data["next", "done"].sum()

    if max_length > 200:
        break

t1 = time.time()
torchrl_logger.info(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
)






