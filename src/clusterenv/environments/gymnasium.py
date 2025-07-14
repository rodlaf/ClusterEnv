import gymnasium as gym

def make_env(env_config):
    env_name = env_config["env_name"]
    num_envs = env_config.get("envs_per_node")

    def make_single_env():
        env = gym.make(env_name)
        return env

    return gym.vector.SyncVectorEnv([make_single_env for _ in range(num_envs)])
