# ClusterEnv

## Project Overview

ClusterEnv is an open-source framework designed to distribute reinforcement learning (RL) environment execution across Slurm-managed high-performance computing clusters. It enables seamless parallelization of both lightweight environments (e.g., CartPole) and computationally intensive physics-based simulations (e.g., Mujoco), optimizing training efficiency on HPC infrastructure.

## Key Features

* Slurm Integration: Leverages Slurm’s job scheduling, resource management, and fault tolerance capabilities to orchestrate environment execution.
* Environment Abstraction Layer: Supports diverse simulation libraries (Mujoco, Bullet, custom physics engines) through a unified API.
* Scalable Execution: Automatically distributes environments across nodes based on CPU/GPU availability, achieving near-linear scaling up to 32+ nodes.
* State Synchronization: Minimizes communication overhead via async/await patterns, ensuring efficient state updates between learners and workers.
* Open Source & Modular: Extensible architecture with pre-built templates for Mujoco, Bullet, and custom environments.

## Getting Started
Prerequisites: 
Slurm workload manager installed on your cluster.
Python 3.8+, PyTorch (for RL integration), and OpenAI Gym/Mujoco libraries.
SSH access to submit jobs via sbatch.
Installation: 
```bash
git clone https://github.com/cluster-gym.git
cd cluster-gym
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

config = SlurmConfig(
    job_name="mujoco_training",
    time_limit="02:00:00",
    nodes=4,
    gpus_per_node=2
)

# Initialize distributed environment
env = ClusterEnv(env_config, config)
```

Submitting Jobs to Slurm
Create a batch script (submit.sh):

```bash
#!/bin/bash
#SBATCH --job-name=mujoco_train
#SBATCH --nodes=4
#SBATCH --gpus=8
#SBATCH --time=02:00:00

python train.py --config env/mujoco_ant.yaml
```

Run:

```bash
sbatch submit.sh
```

## Custom Environments

Add your environment to clusterenv/envs/ with a .yaml config file.
Use the CLI to register it:

```python
python clusterenv/register_env.py --name MyEnv --path /path/to/my_env.py
```

## Performance Tuning Tips

Use Slurm’s partition-aware scheduling to prioritize GPU nodes for physics-based environments.
Adjust n_parallel based on node CPU/GPU availability to maximize utilization.
Enable fault tolerance by setting slurm_requeueable=True in your job scripts.
