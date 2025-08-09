# Genetic Rule Learning with PPO and TorchRL

## Features

* Custom **ConnectionEnv** environment (`EnvBase` subclass)
* Supports **PPO** (ClipPPO Loss)
* Multi-distribution action output: `CompositeDistribution` predicts both `pair_idx` and `clause_idx`
* Reward integrates:

  * True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN) from predicted matrices
  * Additional **F1-score bonus** for balanced optimization
* Saves best rule (`best_rule.pt`)
* Visualization of prediction results


## Project Structure

```
.
├── new_env.py         # Environment definition
├── model.py           # Policy & value networks
├── train.py           # Main PPO training script
├── data_utils.py      # Data loading utilities
├── test.py            # Common functions
├── best_rule.pt       # Saved best rule
└── README.md          # This document
```


## Installation

```bash
pip install torch torchvision torchaudio
pip install torchrl tensordict tqdm matplotlib
```

Make sure your PyTorch version matches your CUDA version if using GPU.


## Running Training

```bash
python train.py
```

Optional parameters (adjust in `make_env`):

* `max_clauses` (default 5)
* `max_pre_genes` (default 5)
* `max_post_genes` (default 5)
* `max_steps` (default 50)

## Environment Overview: Custom `ConnectionEnv` (EnvBase subclass)

The `ConnectionEnv` is a custom reinforcement learning environment designed to model genetic logic rules that predict neuronal connectivity. It extends `EnvBase` from TorchRL and encapsulates the following key features:

* **State (Observation):**
  Represents the current genetic rule as a fixed-size integer vector (`rule`), encoding multiple clauses. Each clause contains sets of “pre” and “post” gene indices.

* **Action Space:**
  Actions modify the current rule by selecting a gene pair (`pair_idx`) and the clause index (`clause_idx`) to update. The gene pair corresponds to a tuple of pre- and post-synaptic gene indices or deletion (represented by zero).

* **Reward Function:**
  Combines classical classification metrics computed by comparing the predicted connectivity matrix (generated from the current rule and gene expression data) with the ground-truth connectivity matrix:

  * True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
  * Includes an additional weighted F1-score bonus to balance precision and recall effectively

* **Algorithmic Insight:**
  The environment encodes the combinatorial search over genetic logic rules as a sequential decision process. The agent incrementally refines the rules by selecting which gene pairs to add or remove in each clause. Validity checks ensure no duplicates or contradictory gene combinations occur. This setup allows PPO to efficiently explore and optimize over a large discrete rule space, leveraging structured multi-distribution actions for flexible, interpretable rule learning.

* **Termination:**
  Episodes are controlled externally via a step counter (`StepCounter` transform), with no inherent done condition inside the environment.



## Training Flow

1. **Initialize**: load data, create environment, initialize policy and value networks
2. **Collect data**: interact with environment via `SyncDataCollector`
3. **Compute advantage**: use Generalized Advantage Estimation (`GAE`)
4. **Update networks**: calculate PPO loss (`ClipPPOLoss`), apply gradient clipping, optimize with Adam
5. **Evaluate & save**: periodically evaluate and save the best rule (`best_rule.pt`)

---

## Output & Visualization

Example training log:

```
[Batch: 0] (init: 0.0000),  F1-Score: 0.750  Best F1 Score: 0.780  Best Precision: 0.800 , average reward= 0.5123 (init= 0.3120)
```

Visualize rule performance:

```python
from new_env import make_env
import torch

env = make_env()
rule = torch.load("best_rule.pt")
env.visualize_rule_performance(rule)
```

This generates three plots:

* True connectivity matrix
* Predicted connectivity matrix
* Prediction errors (False Positives in red, False Negatives in blue)


## Notes

* The reward function heavily relies on F1-score; adjust reward weights if focusing on precision or recall
* For stable convergence, set `total_frames` ≥ 1,000,000
* Ensure data matrices (`gene_matrix`, `adjacency_matrix`, `connection_matrix`) are consistent and valid


