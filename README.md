# ClusterEnv

## Project Overview

ClusterEnv is an open-source framework designed to distribute reinforcement learning (RL) environment execution across Slurm-managed high-performance computing clusters. It enables seamless parallelization of both lightweight environments (e.g., CartPole) and computationally intensive physics-based simulations (e.g., Mujoco), optimizing training efficiency on HPC infrastructure.


## Getting Started

Prerequisites: 
Slurm workload manager installed on your cluster.
Python 3.8+, PyTorch (for RL integration), and OpenAI Gym/Mujoco libraries.
SSH access to submit jobs via sbatch.
Installation: 
```bash
git clone https://github.com/rodlaf/ClusterEnv.git
cd clusterenv
pip install -r requirements.txt
```

Basic Usage:
```python
from clusterenv import ClusterEnv, SlurmConfig

# Define environment configuration
env_config = {
    "type": "MujocoAnt",
    "n_parallel": 50,
    "slurm_partition": "gpu"
}

# Define SLURM job configuration
config = SlurmConfig(
    job_name="mujoco_training",
    time_limit="02:00:00",
    nodes=4,
    gpus_per_node=2,
    cpus_per_task=4,   # Optional
    mem_per_node="32G" # Optional
)

# Initialize and launch distributed environment
env = ClusterEnv(env_config, config)
env.launch()  # Submits the SLURM job internally
```

You can then interact with the environment from your training loop as usual:

```python
obs = env.reset()
for _ in range(1000):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
```

## Custom Environments

Add your environment to clusterenv/envs/ with a .yaml config file.
Use the CLI to register it:

```python
python clusterenv/register_env.py --name MyEnv --path /path/to/my_env.py
```
