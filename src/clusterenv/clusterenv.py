from clusterenv.launchers import SlurmConfig
import zmq
import json
import uuid
import atexit
import os
import cloudpickle
import torch
import base64

from clusterenv.launchers.slurm import launch_slurm_job


class ClusterEnv:
    def __init__(self, env_config: dict, config: SlurmConfig):
        self.env_config = env_config
        self.slurm_config = config
        self.uuid = str(uuid.uuid4())[:8]

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:5555")  # Replace with dynamic port later if needed

        self.workers = []
        self.worker_ready = set()
        self.observations = dict()
        self.rewards = dict()
        self.dones = dict()
    
        atexit.register(self._cleanup)

    def set_agent(self, agent: torch.nn.Module):
        self.serialized_agent = cloudpickle.dumps(agent)

    def _cleanup(self):
        self.socket.close()

    def launch(self):
        """
        Launch SLURM worker jobs.
        """
        print("[ClusterEnv] Launching workers via SLURM...")

        shared_dir = os.path.expanduser("~/clusterenv_shared")
        os.makedirs(shared_dir, exist_ok=True)

        # Serialize env config + agent
        config = {
            "env_config": self.env_config,
        }

        config_path = os.path.join(shared_dir, f"config_{uuid.uuid4().hex}.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        launch_slurm_job(self.slurm_config, config_path)


    def _wait_for_worker_connections(self, expected):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while len(self.worker_ready) < expected:
            socks = dict(poller.poll(1000))
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                identity, _, message = self.socket.recv_multipart()
                msg = json.loads(message.decode())
                if msg.get("type") == "register":
                    self.worker_ready.add(identity)
                    self.workers.append(identity)
                    print(f"[ClusterEnv] Worker registered: {identity.decode()}")

    def reset(self):
        """
        Broadcast a reset command to all workers and collect observations.
        """
        for identity in self.workers:
            self.socket.send_multipart([identity, b"", json.dumps({"type": "reset"}).encode()])

        return self._gather("observation")

    def step(self, agent_input):
        if not hasattr(self, "agent_sent"):
            self.agent_sent = True
            agent_payload = self.serialized_agent.hex()
        else:
            agent_payload = None

        agent_state = agent_input.state_dict()
        agent_weights = {k: v.cpu().numpy().tolist() for k, v in agent_state.items()}

        for identity in self.workers:
            payload = {
                "type": "step",
                "agent_weights": agent_weights,
            }
            if agent_payload:
                payload["agent_serialized"] = agent_payload
            self.socket.send_multipart([identity, b"", json.dumps(payload).encode()])

    def _gather(self, mode):
        """
        Collect responses from workers for either reset or step.
        """
        obs, rews, dones, infos = [], [], [], []
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        remaining = set(self.workers)
        while remaining:
            socks = dict(poller.poll(1000))
            if self.socket in socks:
                identity, _, message = self.socket.recv_multipart()
                msg = json.loads(message.decode())
                if msg["type"] == "response":
                    remaining.remove(identity)
                    obs.append(msg["obs"])
                    if mode == "step":
                        rews.append(msg["reward"])
                        dones.append(msg["done"])
                        infos.append(msg.get("info", {}))

        if mode == "reset":
            return obs
        else:
            return obs, rews, dones, infos

