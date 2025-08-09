from new_env import make_env
import logging
from tensordict.nn import CompositeDistribution
from torch import distributions as d
from collections import defaultdict
from tensordict.nn import TensorDictModule
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
import torch
from torchrl.envs.utils import  ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")



env = make_env(device=device,max_steps=500)


from model import  CompositePolicyNet,ValueNet

value_mlp = CompositePolicyNet(
    obs_dim=env.rule_size,total_genes=env.common_gene_idx,clauses=env.max_clauses)

module = TensorDictModule(
    value_mlp,
    in_keys=["rule"],
    out_keys=[
        ("params", "pair_idx", "logits"),
        ("params", "clause_idx", "logits"),
    ]
)

policy_module = ProbabilisticActor(
    module=module,
    in_keys=["params"],
    distribution_class=CompositeDistribution,
    distribution_kwargs={
        "distribution_map": {
            "pair_idx": d.Categorical,
            "clause_idx": d.Categorical,
        },
    },
    return_log_prob=True,
)



value_net = ValueNet(env.rule_size)

value_module = ValueOperator(
    module=value_net,
    in_keys=["rule"],
)
print("Initializing policy and value network ")
policy_module(env.reset())
value_module(env.reset())

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer

init_rand_steps = 500
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,

)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)


sub_batch_size = 64
lr = 3e-4
max_grad_norm = 1.0
frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 10_000
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coeff=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coeff=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    logs["lr"].append(optim.param_groups[0]["lr"])

    if i % 100 == 0:
        # Evaluate policy every 10 batches
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"[Batch: {i}] "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f" F1-Score: {env.metrics['F1']} "
                f"Best F1 Score: {env.best_f1} "
                f"Best Precision: {env.best_precision} "
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str]))
    # print(", ".join([eval_str, cum_reward_str]))

    scheduler.step()

