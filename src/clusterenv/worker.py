import argparse
import json
import zmq
import torch
import torch.nn.functional as F
import numpy as np
import importlib
import cloudpickle
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--controller_ip", type=str, required=True)
    parser.add_argument("--controller_port", type=int, default=5555)
    return parser.parse_args()

def load_env(env_config):
    env_type = env_config["type"]
    mod = importlib.import_module(f"clusterenv.environments.{env_type.lower()}")
    return mod.make_env(env_config)

def deserialize_weights(agent, weights_dict):
    state_dict = {k: torch.tensor(v) for k, v in weights_dict.items()}
    agent.load_state_dict(state_dict)

def compute_kl(old_logits, new_logits):
    old_probs = F.softmax(old_logits, dim=-1)
    new_log_probs = F.log_softmax(new_logits, dim=-1)
    return F.kl_div(new_log_probs, old_probs, reduction="batchmean")

def main():
    args = parse_args()

    with open(args.config_path, "r") as f:
        env_config = json.load(f)
        print(f"[Worker] Loaded env_config: {env_config}", file=sys.stderr)
        sys.stderr.flush()

    env = load_env(env_config)
    obs = env.reset()[0]  # assuming SyncVectorEnv
    agent = None

    ctx = zmq.Context()
    socket = ctx.socket(zmq.DEALER)
    identity = f"worker-{np.random.randint(10000)}".encode()
    socket.setsockopt(zmq.IDENTITY, identity)
    connect_addr = f"tcp://{args.controller_ip}:{args.controller_port}"
    socket.connect(connect_addr)

    print(f"[Worker] Socket identity: {identity}", file=sys.stderr)
    print(f"[Worker] Connecting to: {connect_addr}", file=sys.stderr)
    sys.stderr.flush()

    socket.send_multipart([
        b"",  # empty delimiter frame required for ROUTER socket
        json.dumps({"type": "register"}).encode()
    ])
    print("[Worker] Sent registration", file=sys.stderr)
    sys.stderr.flush()

    kl_threshold = env_config.get("kl_threshold", 0.1)
    old_logits = None

    while True:
        parts = socket.recv_multipart()
        print(f"[Worker] Received parts: {len(parts)}", file=sys.stderr)
        sys.stderr.flush()

        if len(parts) < 2:
            print("[Worker] Skipping invalid multipart message", file=sys.stderr)
            continue

        msg = parts[-1]
        payload = json.loads(msg.decode())

        if payload["type"] == "reset":
            obs = env.reset()[0]
            socket.send_json({
                "type": "response",
                "obs": obs.tolist()  # make JSON serializable
            })
            print("[Worker] Sent reset response", file=sys.stderr)
            sys.stderr.flush()

        elif payload["type"] == "step":
            if agent is None and "agent_serialized" in payload:
                agent = cloudpickle.loads(bytes.fromhex(payload["agent_serialized"]))
                print("[Worker] Agent deserialized.", file=sys.stderr)
                sys.stderr.flush()

            if agent is None:
                raise ValueError("Agent is not initialized and was not provided.")

            weights = payload["agent_weights"]
            deserialize_weights(agent, weights)

            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            logits = agent(obs_tensor)

            if old_logits is not None:
                kl = compute_kl(old_logits.detach(), logits.detach()).item()
                if kl > kl_threshold:
                    deserialize_weights(agent, weights)
            old_logits = logits.detach()

            actions = logits.argmax(dim=-1).cpu().numpy()
            obs_next, reward, terminated, truncated, infos = env.step(actions)

            response = {
                "type": "response",
                "obs": obs_next.tolist(),
                "reward": reward.tolist(),
                "done": (terminated | truncated).tolist(),
                "info": infos,
            }
            socket.send_json(response)
            obs = obs_next
            print("[Worker] Sent step response", file=sys.stderr)
            sys.stderr.flush()

if __name__ == "__main__":
    main()
