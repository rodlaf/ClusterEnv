from clusterenv.launchers import SlurmConfig
from clusterenv.worker import Worker
import zmq
import json
import socket
import time

class ClusterEnv:
    def __init__(self, env_config, config: SlurmConfig):
        self.env_config = env_config
        self.config = config
        self.n_parallel = env_config.get("n_parallel", 1)

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.port = self.socket.bind_to_random_port("tcp://*")
        self.workers = {}

    def launch(self):
        self.config.launch_workers(
            n_workers=self.n_parallel,
            controller_ip=self._get_ip(),
            controller_port=self.port,
            env_type=self.env_config.get("type", "DummyEnv")
        )
        self._wait_for_workers()

    def _get_ip(self):
        return socket.gethostbyname(socket.gethostname())

    def _wait_for_workers(self):
        print("Waiting for worker nodes to connect...")
        while len(self.workers) < self.n_parallel:
            ident, _, msg = self.socket.recv_multipart()
            data = json.loads(msg)
            if data.get("type") == "register":
                self.workers[ident] = {}
                print(f"Registered worker {ident}")

    def reset(self):
        for ident in self.workers:
            self.socket.send_multipart([ident, b"", json.dumps({"type": "reset"}).encode()])
        obs = {}
        for _ in self.workers:
            ident, _, msg = self.socket.recv_multipart()
            obs[ident] = json.loads(msg)["obs"]
        return obs

    def step(self, actions):
        for ident, action in actions.items():
            self.socket.send_multipart([ident, b"", json.dumps({"type": "step", "action": action}).encode()])

        results = {}
        for _ in self.workers:
            ident, _, msg = self.socket.recv_multipart()
            results[ident] = json.loads(msg)

        return results
