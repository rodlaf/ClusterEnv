import subprocess
import os
import tempfile
import socket
from dataclasses import dataclass


@dataclass
class SlurmConfig:
    job_name: str = "clusterenv_job"
    nodes: int = 1
    gpus_per_node: int = 1
    time_limit: str = "01:00:00"
    partition: str = "gpu"
    gpu_type: str = "gpu"
    debug: bool = False


def launch_slurm_job(slurm_config: SlurmConfig, env_config_path: str):
    """
    Launches a SLURM job with the given config and environment config file.
    Assumes worker.py will be invoked on each node.
    """

    controller_ip = socket.gethostbyname(socket.gethostname())
    n_nodes = slurm_config.nodes
    job_name = slurm_config.job_name
    time_limit = slurm_config.time_limit
    partition = slurm_config.partition
    gpus_per_node = slurm_config.gpus_per_node
    gpu_type = slurm_config.gpu_type
    debug = slurm_config.debug

    debug_flag = "--debug" if debug else ""

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
##SBATCH --gres={gpu_type}:{gpus_per_node} # TODO
#SBATCH --ntasks={n_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --nodes={n_nodes}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --export=ALL

module load cuda

srun python -m clusterenv.worker \
    --config_path {env_config_path} \
    --controller_ip {controller_ip} \
    --controller_port 5555 {debug_flag}
"""

    os.makedirs("logs", exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".slurm", delete=False) as f:
        f.write(slurm_script)
        slurm_path = f.name

    print(f"[ClusterEnv] Submitting SLURM job: {slurm_path}")
    subprocess.run(["sbatch", slurm_path])

