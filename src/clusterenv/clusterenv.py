import zmq
import json
import uuid
import atexit
import os
import cloudpickle
import torch
import importlib
from clusterenv.launchers import SlurmConfig
from clusterenv.launchers.slurm import launch_slurm_job


class ClusterEnv:
    def __init__(self, env_config: dict, config: SlurmConfig):
        self.env_config = env_config
        self.slurm_config = config
        self.uuid = str(uuid.uuid4())[:8]

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:5555")

        self.workers = []
        self.worker_ready = set()
        self.agent_sent = False

        self.serialized_agent = None

        self.debug = config.debug

        atexit.register(self._cleanup)

    def _cleanup(self):
        self.socket.close()

    def load_env(self, env_config):
        env_type = env_config["type"]
        mod = importlib.import_module(f"clusterenv.environments.{env_type.lower()}")
        return mod.make_env(env_config)

    def launch(self):
        if self.debug:
            print("[ClusterEnv] Launching workers via SLURM...")

        shared_dir = os.path.expanduser("~/clusterenv_shared")
        os.makedirs(shared_dir, exist_ok=True)

        config_path = os.path.join(shared_dir, f"config_{uuid.uuid4().hex}.json")
        with open(config_path, "w") as f:
            json.dump(self.env_config, f)

        launch_slurm_job(self.slurm_config, config_path)

        self._wait_for_worker_connections(expected=self.slurm_config.nodes)

        env = self.load_env(self.env_config)

        if hasattr(env, "single_observation_space"):
            self.observation_space = env.single_observation_space
            self.action_space = env.single_action_space
        else:
            self.observation_space = env.observation_space
            self.action_space = env.action_space

        return self.observation_space, self.action_space

    def _wait_for_worker_connections(self, expected):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        if self.debug:
            print(f"[ClusterEnv] Waiting for {expected} worker(s) to register...")

        import time
        start_time = time.time()
        timeout = 30  # seconds

        while len(self.worker_ready) < expected:
            socks = dict(poller.poll(1000))
            if self.socket in socks:
                parts = self.socket.recv_multipart()
                if len(parts) == 3:
                    identity, _, message = parts
                    msg = json.loads(message.decode())
                    if msg.get("type") == "register":
                        if identity not in self.worker_ready:
                            self.worker_ready.add(identity)
                            self.workers.append(identity)
                            if self.debug:
                                print(f"[ClusterEnv] Worker registered: {identity.decode()}")

            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"[ClusterEnv] Timed out waiting for {expected} workers. Only got {len(self.worker_ready)}."
                )

    def reset(self):
        for identity in self.workers:
            self.socket.send_multipart([
                identity,
                b"",
                json.dumps({"type": "reset"}).encode()
            ])
        return self._gather("reset")

    def step(self, agent_input: torch.nn.Module):
        if not self.serialized_agent:
            self.serialized_agent = cloudpickle.dumps(agent_input)

        if not self.agent_sent:
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
            self.socket.send_multipart([
                identity,
                b"",
                json.dumps(payload).encode()
            ])

        return self._gather("step")

    def _gather(self, mode):
        if self.debug:
            print(f"[ClusterEnv] Gathering responses for {mode}...", flush=True)

        obs, rews, dones, infos, logprobs, values, actions = [], [], [], [], [], [], []
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        remaining = set(self.workers)
        while remaining:
            socks = dict(poller.poll(1000))
            if self.socket in socks:
                parts = self.socket.recv_multipart()
                if len(parts) != 3:
                    if self.debug:
                        print(f"[ClusterEnv] Unexpected multipart length: {len(parts)} — skipping", flush=True)
                    continue

                identity, _, message = parts
                msg = json.loads(message.decode())

                if msg["type"] == "response":
                    if identity in remaining:
                        remaining.remove(identity)
                    else:
                        if self.debug:
                            print(f"[ClusterEnv] Warning: duplicate response from {identity.decode()}", flush=True)
                        continue

                    if self.debug:
                        print(f"[ClusterEnv] Got response from {identity.decode()} with {len(msg['obs'])} obs", flush=True)

                    obs.extend(msg["obs"])
                    if mode == "step":
                        rews.extend(msg["reward"])
                        dones.extend(msg["done"])
                        infos.append(msg.get("info", {}))
                        logprobs.extend(msg["logprob"])
                        values.extend(msg["value"])
                        actions.extend(msg["action"])

        if mode == "reset":
            return obs
        else:
            return obs, rews, dones, logprobs, values, actions, infos
