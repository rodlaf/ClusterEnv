# ClusterEnv

ClusterEnv is a lightweight interface for distributed reinforcement learning (RL) environment execution on Slurm-managed clusters. It decouples environment simulation from training logic, enabling scalable rollout collection without adopting a monolithic framework. ClusterEnv mirrors the Gymnasium API and introduces two core components: the **DETACH** architecture and **Adaptive Actor Policy Synchronization (AAPS)**.

## Core Concepts

**DETACH (Distributed Environment execution with Training Abstraction and Centralized Head)** separates rollout collection from the training loop. Remote workers run only `reset()` and `step()` methods, while the learner remains centralized.

**AAPS (Adaptive Actor Policy Synchronization)** addresses policy staleness by triggering weight updates only when divergence exceeds a user-defined KL threshold. This minimizes unnecessary communication while keeping behavior on-policy enough for efficient training.

## Installation

```bash
git clone https://github.com/rodlaf/ClusterEnv.git
cd ClusterEnv
pip install -r requirements.txt
```

## Requirements

* Slurm with `sbatch` submission access
* SSH access to allocated cluster nodes
* Python 3.8+
* RL agent implementing `.act(obs)` and `.get_parameters()`

## Basic Usage

```python
from clusterenv import ClusterEnv, SlurmConfig

env_config = {
    "type": "gymnasium",           # or "MujocoAnt", etc.
    "env_name": "LunarLander-v2",
    "kl_threshold": 0.05,
    "envs_per_node": 64
}

slurm_cfg = SlurmConfig(
    job_name="ppo_lander",
    nodes=4,
    gpus_per_node=2,
    partition="gpu",
    time_limit="02:00:00"
)

env = ClusterEnv(env_config, slurm_cfg)
env.launch()

obs = env.reset()
for _ in range(1000):
    obs, reward, done, info = env.step(agent)
```

Policy synchronization via AAPS is handled internally. Agents pull updated weights only when their local policy drifts too far from the central learner, as measured by KL divergence.

## Custom Environments

To add a custom environment:

1. Place your environment code in `clusterenv/envs/`
2. Register it:

```bash
python clusterenv/register_env.py --name MyEnv --path /absolute/path/to/my_env.py
```

