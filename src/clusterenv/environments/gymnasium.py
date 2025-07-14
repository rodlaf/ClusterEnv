import gymnasium as gym

def make_env(env_config):
    env_name = env_config.get("env_name", "CartPole-v1")
    env = gym.make(env_name)

    class GymnasiumEnvWrapper:
        def __init__(self, env):
            self.env = env

        def reset(self):
            obs, _ = self.env.reset()
            return obs

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            return obs, reward, done, info

        def __getattr__(self, attr):
            return getattr(self.env, attr)

    return GymnasiumEnvWrapper(env)
