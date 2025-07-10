from clusterenv.launchers import SlurmConfig

class ClusterEnv():
    def __init__(self, env_config, config: SlurmConfig):
        self.env_config = env_config
        self.config = config

    def launch(self):
        pass