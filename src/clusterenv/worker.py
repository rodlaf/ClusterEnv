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

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

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
            socket.send_multipart([
                b"",
                json.dumps({
                    "type": "response",
                    "obs": obs.tolist()
                }).encode()
            ])
            print("[Worker] Sent reset response", file=sys.stderr)
            sys.stderr.flush()

        elif payload["type"] == "step":
            print("[Worker] Received step payload", file=sys.stderr)
            if "agent_serialized" in payload:
                print("[Worker] Deserializing agent...", file=sys.stderr)
                agent = cloudpickle.loads(bytes.fromhex(payload["agent_serialized"]))
                print("[Worker] Agent deserialized", file=sys.stderr)

            if agent is None:
                raise ValueError("Agent is not initialized and was not provided.")

            weights = payload["agent_weights"]
            print("[Worker] Loading weights...", file=sys.stderr)
            deserialize_weights(agent, weights)
            print("[Worker] Weights loaded", file=sys.stderr)

            print("[Worker] Running inference...", file=sys.stderr)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, logprob, _, value = agent(obs_tensor)
            print("[Worker] Inference done", file=sys.stderr)

            if old_logits is not None:
                with torch.no_grad():
                    logits = agent.actor(obs_tensor)
                    kl = compute_kl(old_logits.detach(), logits.detach()).item()
                    if kl > kl_threshold:
                        deserialize_weights(agent, weights)
                old_logits = logits.detach()
            else:
                old_logits = agent.actor(obs_tensor).detach()

            action_np = action.cpu().numpy().squeeze(0)
            print(f"[Worker] action_np shape: {action_np.shape}", file=sys.stderr)
            obs_next, reward, terminated, truncated, infos = env.step(action_np)

            response = {
                "type": "response",
                "obs": obs_next.tolist(),
                "reward": reward.tolist(),
                "done": (terminated | truncated).tolist(),
                "info": infos,
                "logprob": logprob.cpu().tolist(),
                "value": value.cpu().squeeze().tolist(),
                "action": action_np.tolist(),
            }

            socket.send_multipart([
                b"",
                json.dumps(response, default=to_serializable).encode()
            ])
            obs = obs_next
            print("[Worker] Sent step response", file=sys.stderr)
            sys.stderr.flush()

if __name__ == "__main__":
    main()
