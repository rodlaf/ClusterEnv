import gymnasium as gym
import os

def make_env(env_config):
    env_name = env_config["env_name"]
    num_envs = env_config.get("envs_per_node", 1)

    base_seed = env_config.get("seed")
    if base_seed is None:
        base_seed = 0

    # Ensure unique seed per node by using SLURM_PROCID or fallback to pid
    node_offset = int(os.environ.get("SLURM_PROCID"))
    base_seed += node_offset * num_envs

    def make_single_env(i):
        def _init():
            env = gym.make(env_name)
            env.reset(seed=base_seed + i)
            return env
        return _init

    return gym.vector.SyncVectorEnv([make_single_env(i) for i in range(num_envs)])

