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

    kl_threshold = env_config.get("kl_threshold")
    old_logits = None

    while True:
        parts = socket.recv_multipart()
        if debug:
            print(f"[Worker] Received parts: {len(parts)}", file=sys.stderr)
            sys.stderr.flush()

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
                print("[Worker] Received step payload", file=sys.stderr)

            # Deserialize agent ONCE if needed
            if "agent_serialized" in payload and agent is None:
                if debug:
                    print("[Worker] Deserializing agent...", file=sys.stderr)
                agent = cloudpickle.loads(bytes.fromhex(payload["agent_serialized"]))
                agent.to(device)
                agent.eval()
                if debug:
                    print("[Worker] Agent deserialized", file=sys.stderr)

            if agent is None:
                raise ValueError("Agent is not initialized and was not provided.")

            # === Get candidate weights from payload ===
            weights = payload["agent_weights"]
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # === Handle KL-based sync ===
            if old_logits is not None:
                with torch.no_grad():
                    current_logits = agent.actor(obs_tensor)

                    # Save a copy of the current agent weights
                    current_params = {k: v.clone() for k, v in agent.state_dict().items()}
                    deserialize_weights(agent, weights)
                    new_logits = agent.actor(obs_tensor)
                    kl = compute_kl(current_logits.detach(), new_logits.detach()).item()

                    if kl > kl_threshold:
                        sync_count += 1
                        if debug:
                            print(f"[Worker] KL sync triggered. sync_count={sync_count}", file=sys.stderr)
                        # keep weights
                    else:
                        # restore old weights
                        agent.load_state_dict(current_params) # TODO: performance improvement still on the table
                        if debug:
                            print(f"[Worker] KL={kl:.4f} < threshold. Skipping sync.", file=sys.stderr)
            else:
                # First-time sync
                deserialize_weights(agent, weights)
                sync_count += 1
                if debug:
                    print("[Worker] First-time sync", file=sys.stderr)

            # === Inference
            with torch.no_grad():
                action, logprob, _, value = agent(obs_tensor)
                old_logits = agent.actor(obs_tensor).detach()

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
                print("[Worker] Sent step response", file=sys.stderr)
            sys.stderr.flush()



if __name__ == "__main__":
    main()
