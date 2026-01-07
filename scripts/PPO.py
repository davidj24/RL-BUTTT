import os
import random
import time
import sys
import wandb
import collections
import trueskill
import json
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from collections import deque
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.Agent import Agent
from src.EnvWrapper import SingleAgentTrainingWrapper
from src.Opponent import FrozenAgentOpponent
from dataclasses import dataclass
from typing import Optional
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

gym.register(
    id="UTTT-v0",
    entry_point="src.Env:UTTTEnv",
)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RL_BUTTT"
    """the wandb's project name"""
    wandb_entity: str = "davidj24-uc-berkeley"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model_every: int = 100000
    """How many steps are taken before model is saved"""
    model_save_folder: str = "models"
    """The folder to save the models to"""
    self_play: bool = True
    """Whether or not an opponent pool should be created and used to swap opponent"""
    swap_to_newest_model_prob: float = 0.8
    """The probability that the opponent is swapped to the newest model"""

    # Algorithm specific arguments
    env_id: str = "UTTT-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    min_lr: float = 5e-5
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def make_env(env_id, idx, capture_video, run_name, opponent_pool_paths):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video and idx == 0 else None)

        env = SingleAgentTrainingWrapper(env, opponent_pool_paths, args.swap_to_newest_model_prob)

        # Apply Gym Wrappers
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def save_ratings(agent_ratings, ratings_folder, exp_name):
    # Convert Rating objects to serializable dictionaries
    serializable_ratings = {}
    for key, rating in agent_ratings.items():
        serializable_ratings[str(key)] = {
            "mu": rating.mu,
            "sigma": rating.sigma
        }
    
    save_path = Path(ratings_folder) / f"{exp_name}_ratings.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(serializable_ratings, f, indent=4)

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}"

    if args.track:
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
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    agent = Agent().to(device)

    # Save initial agent
    folder = Path(args.model_save_folder) / args.exp_name
    folder.mkdir(parents=True, exist_ok=True)
    initial_agent_path = folder / f"{args.exp_name}_initial.pth"
    torch.save(
        agent.state_dict(),
        initial_agent_path
    )

    # Set up data structures for custom wandb charts
    recent_returns = deque(maxlen=100)
    recent_wins = deque(maxlen=100)
    if args.self_play:
        agent_ratings = {initial_agent_path: trueskill.Rating()}

    # env setup
    shared_opponent_pool_paths = deque([initial_agent_path] if args.self_play else [], maxlen=10)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, shared_opponent_pool_paths) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)


    # ==================== BEGIN TRAINING ====================
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr and optimizer.param_groups[0]["lr"] > args.min_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_training_info(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            
            # ==================== CUSTOM WANDB CHART LOGGING ====================
            # Check if episode ended and write to wandb charts if so
            if args.track and "episode" in infos:
                for env_idx in range(args.num_envs):
                    if infos["_episode"][env_idx]: # Only get results from envs where episode ended
                        ret = infos["episode"]["r"][env_idx]

                        recent_returns.append(ret)
                        recent_wins.append(1 if ret >= 0.7 else 0)
                        ep_rew_mean = 0
                        win_rate = 0

                        if len(recent_returns) > 0 and len(recent_wins) > 0:
                            ep_rew_mean = sum(recent_returns) / len(recent_returns)
                            win_rate = (sum(recent_wins) / len(recent_wins)) * 100

                            wandb.log({
                                "rollout/ep_rew_mean": ep_rew_mean, # This is now smoothed!
                                "rollout/win_rate": win_rate,
                                "global_step": global_step,
                            })

                        if args.self_play: # Track the frontier model's trueskill rating/elo
                            frontier_model_id = "training_frontier"
                            opponent_path = infos["opponent_path"][env_idx]

                            if frontier_model_id not in agent_ratings:
                                agent_ratings[frontier_model_id] = trueskill.Rating()
                            if opponent_path not in agent_ratings:
                                agent_ratings[opponent_path] = trueskill.Rating()

                            is_draw = -0.7 < ret < 0.7
                            winner, loser = (frontier_model_id, opponent_path) if ret >= 0.7 else (opponent_path, frontier_model_id)
                            agent_ratings[winner], agent_ratings[loser] = trueskill.rate_1vs1(
                                agent_ratings[winner], agent_ratings[loser], drawn=is_draw
                            )

                            mu = agent_ratings[frontier_model_id].mu
                            sigma = agent_ratings[frontier_model_id].sigma
                            conservative_trueskill = mu - (3*sigma)
                            wandb.log({
                                "eval/conservative_trueskill": conservative_trueskill,
                                "global_step": global_step,
                            })


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            next_value = next_value.reshape(1, -1)
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_training_info(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # Save and update the model every n global steps
        if global_step % args.save_model_every < args.batch_size:
            new_model_path = folder / f"{args.exp_name}_{global_step}.pth"
            torch.save(
                agent.state_dict(),
                new_model_path
            )

            if args.self_play:
                # Add model to opponent pool
                shared_opponent_pool_paths.append(new_model_path)
                frontier_rating = agent_ratings["training_frontier"]
                agent_ratings[new_model_path] = trueskill.Rating(
                    mu=frontier_rating.mu,
                    sigma=frontier_rating.sigma
                )
                save_ratings(agent_ratings, "ratings", args.exp_name)
            



        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()