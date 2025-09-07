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
    parser.add_argument("--debug", action="store_true")
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
    debug = args.debug

    sync_count = 0
    current_version = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config_path, "r") as f:
        env_config = json.load(f)
        if debug:
            print(f"[Worker] Loaded env_config: {env_config}", file=sys.stderr)
            sys.stderr.flush()

    env = load_env(env_config)
    obs = env.reset()[0]
    agent = None

    ctx = zmq.Context()
    socket = ctx.socket(zmq.DEALER)

    identity = f"worker-{np.random.randint(10000)}".encode()
    socket.setsockopt(zmq.IDENTITY, identity)

    connect_addr = f"tcp://{args.controller_ip}:{args.controller_port}"
    socket.connect(connect_addr)

    if debug:
        print(f"[Worker] Socket identity: {identity}", file=sys.stderr)
        print(f"[Worker] Connecting to: {connect_addr}", file=sys.stderr)
        sys.stderr.flush()

    socket.send_multipart([
        b"",
        json.dumps({"type": "register"}).encode()
    ])
    if debug:
        print("[Worker] Sent registration", file=sys.stderr)
        sys.stderr.flush()

    # Default to always syncing if no threshold was provided
    kl_threshold = env_config.get("kl_threshold", 0.0)

    while True:
        parts = socket.recv_multipart()
        if debug:
            print(f"[Worker] Received parts: {len(parts)}", file=sys.stderr, flush=True)

        if len(parts) < 2:
            if debug:
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
            if debug:
                print("[Worker] Sent reset response", file=sys.stderr)
                sys.stderr.flush()

        elif payload["type"] == "step":
            if debug:
                print(
                    f"[Worker] Received step payload (version={payload.get('weights_version')}, current={current_version})",
                    file=sys.stderr,
                    flush=True,
                )

            step_version = payload.get("weights_version", 0)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # === Handle weight updates ===
            if agent is None:
                if debug:
                    print("[Worker] No agent, requesting initial weights", file=sys.stderr, flush=True)
                socket.send_multipart([
                    b"",
                    json.dumps({"type": "weight_request"}).encode(),
                ])
                update_parts = socket.recv_multipart()
                update_msg = json.loads(update_parts[-1].decode())

                if "agent_serialized" in update_msg:
                    if debug:
                        print("[Worker] Deserializing agent...", file=sys.stderr, flush=True)
                    agent = cloudpickle.loads(bytes.fromhex(update_msg["agent_serialized"]))
                    agent.to(device)
                    agent.eval()

                deserialize_weights(agent, update_msg["agent_weights"])
                current_version = update_msg.get("weights_version", step_version)
                sync_count += 1
                if debug:
                    print(f"[Worker] Initial sync to version {current_version}", file=sys.stderr, flush=True)

            elif step_version != current_version:
                if debug:
                    print(
                        f"[Worker] Version mismatch (local={current_version}, head={step_version}). Requesting weights...",
                        file=sys.stderr,
                        flush=True,
                    )

                with torch.no_grad():
                    old_logits = agent.actor(obs_tensor)

                socket.send_multipart([
                    b"",
                    json.dumps({"type": "weight_request"}).encode(),
                ])
                update_parts = socket.recv_multipart()
                update_msg = json.loads(update_parts[-1].decode())
                weights = update_msg["agent_weights"]

                with torch.no_grad():
                    tmp_agent = cloudpickle.loads(cloudpickle.dumps(agent))
                    tmp_agent.to(device)
                    tmp_agent.eval()
                    deserialize_weights(tmp_agent, weights)
                    new_logits = tmp_agent.actor(obs_tensor)
                    kl = compute_kl(old_logits.detach(), new_logits.detach()).item()

                if debug:
                    print(f"[Worker] KL divergence={kl:.6f}", file=sys.stderr, flush=True)

                if kl > kl_threshold:
                    deserialize_weights(agent, weights)
                    current_version = update_msg.get("weights_version", step_version)
                    sync_count += 1
                    if debug:
                        print(
                            f"[Worker] Applied weight update to version {current_version} (KL={kl:.4f})",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    current_version = step_version
                    if debug:
                        print(
                            f"[Worker] KL below threshold (KL={kl:.4f} <= {kl_threshold}). Skipping update.",
                            file=sys.stderr,
                            flush=True,
                        )

            if agent is None:
                raise ValueError("Agent is not initialized after weight update.")

            # === Inference
            with torch.no_grad():
                action, logprob, _, value = agent(obs_tensor)

            # === Step environment
            action_np = action.cpu().numpy().squeeze(0)
            obs_next, reward, terminated, truncated, infos = env.step(action_np)

            if 'episode_rewards' not in locals():
                episode_rewards = [0.0] * env.num_envs
                episode_lengths = [0] * env.num_envs

            final_info = []
            for i in range(env.num_envs):
                episode_rewards[i] += reward[i]
                episode_lengths[i] += 1

                if terminated[i] or truncated[i]:
                    ep_info = {
                        "episode": {
                            "r": episode_rewards[i],
                            "l": episode_lengths[i]
                        }
                    }
                    if debug:
                        print(f"[Worker] Env {i} episode return: {episode_rewards[i]}", file=sys.stderr)
                    final_info.append(ep_info)
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
                else:
                    final_info.append(None)

            response = {
                "type": "response",
                "obs": obs_next.tolist(),
                "reward": reward.tolist(),
                "done": (terminated | truncated).tolist(),
                "info": {
                    "final_info": final_info,
                    "sync_count": sync_count,
                },
                "logprob": logprob.cpu().tolist(),
                "value": value.cpu().squeeze().tolist(),
                "action": action_np.tolist(),
            }

            socket.send_multipart([
                b"",
                json.dumps(response, default=to_serializable).encode()
            ])
            obs = obs_next
            if debug:
                print("[Worker] Sent step response", file=sys.stderr, flush=True)




if __name__ == "__main__":
    main()
