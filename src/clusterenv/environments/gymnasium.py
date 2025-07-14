import gymnasium as gym

def make_env(env_config):
    env_name = env_config.get("env_name")
    num_envs = env_config.get("envs_per_node")

    def thunk():
        return gym.make(env_name)

    return gym.vector.SyncVectorEnv([thunk for _ in range(num_envs)])
