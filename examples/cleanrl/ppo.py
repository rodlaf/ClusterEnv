import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from clusterenv import ClusterEnv
from clusterenv.launchers import SlurmConfig

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_steps: int = 128
    envs_per_node: int = 4
    num_nodes: int = 2
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

class Agent(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_shape, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(obs_shape, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

if __name__ == "__main__":
    args = tyro.cli(Args)

    num_envs = args.num_nodes * args.envs_per_node
    args.batch_size = int(num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Initialize ClusterEnv
    envs = ClusterEnv(
        env_config={
            "type": "gymnasium", 
            "env_name": args.env_id, 
            "envs_per_node": args.envs_per_node
        },
        config=SlurmConfig(
            job_name=run_name,
            time_limit="01:00:00",
            nodes=args.num_nodes,
            gpus_per_node=0,
            partition="normal", # replace with partition information
            # gpu_type="gpu:volta"
        )
    )
    obs_space, action_space = envs.launch()

    obs_shape = obs_space.shape
    obs_dim = int(np.prod(obs_shape))
    if isinstance(action_space, gym.spaces.Discrete):
        act_dim = action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        act_dim = action_space.shape[0]
    else:
        raise NotImplementedError("Unsupported action space type")

    agent: nn.Module = Agent(obs_dim, act_dim)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, num_envs, *obs_shape)).to(device)
    actions = torch.zeros((args.num_steps, num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, num_envs)).to(device)
    dones = torch.zeros((args.num_steps, num_envs)).to(device)
    values = torch.zeros((args.num_steps, num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.tensor(envs.reset()).float().to(device) # ClusterEnv
    print("next_obs.shape:", next_obs.shape)
    print(f"next_obs: {next_obs}")
    next_done = torch.zeros(num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Inference + stepping happens on the worker nodes
            next_obs_arr, reward_arr, done_arr, logprob_arr, value_arr, action_arr = envs.step(agent) # ClusterEnv

            next_done = torch.tensor(done_arr).to(device)
            rewards[step] = torch.tensor(reward_arr).to(device).view(-1)
            next_obs = torch.tensor(next_obs_arr).float().to(device)
             
            # must flatten given (num_nodes, envs_per_node) shape
            logprobs[step] = torch.tensor(logprob_arr).to(device).view(-1)
            values[step] = torch.tensor(value_arr).to(device).view(-1)
            actions[step] = torch.tensor(action_arr).to(device).view(-1)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss = torch.max(-mb_advantages * ratio, -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                if args.target_kl and approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

    envs.close()
    writer.close()
