import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
from agents.dqn_agent import DQNAgent
from agents.critic import CentralizedCritic
from agents.shared_replay_buffer import SharedReplayBuffer
import cv2

# --- Setup ---
os.makedirs("plots", exist_ok=True)

env = simple_spread_v3.env(N=3, max_cycles=100, render_mode='human')
env.reset(seed=42)
obs_size = env.observation_space(env.agents[0]).shape[0]
n_actions = env.action_space(env.agents[0]).n
n_agents = len(env.possible_agents)
agent_list = list(env.possible_agents)  # Store agent list for consistent access

# Update total_obs to account for message passing (+1 per agent)
total_obs = (obs_size * n_agents) + n_agents
total_actions = n_actions * n_agents

# === Centralized Critic ===
critic = CentralizedCritic(total_obs, total_actions)
critic_target = CentralizedCritic(total_obs, total_actions)
critic_target.load_state_dict(critic.state_dict())
critic_opt = optim.Adam(critic.parameters(), lr=1e-4)
crit_criterion = nn.MSELoss()

# === Shared Replay Buffer ===
shared_buffer = SharedReplayBuffer(capacity=10000)

# === Agents ===
# All agents use the full obs_size for DQNAgent, since all states are padded to this size
agents = {a: DQNAgent(obs_size, n_actions, shared_buffer) for a in env.possible_agents}

def modify_obs(agent, obs):
    if agent == "agent_0":
        # Use only the first half, pad to full obs_size
        masked = obs[:obs_size // 2]
        padded = np.zeros(obs_size)
        padded[:len(masked)] = masked
        return padded
    return obs

# === Training Parameters ===
episodes = 60
epsilon, decay, min_eps = 0.9, 0.9999, 0.1
rewards_log = []
agent_rewards_log = {a: [] for a in agent_list}

# === Init global video writer ===
# first_frame = env.render()
# frame_h, frame_w, _ = first_frame.shape
# video_writer = cv2.VideoWriter("videos/full_training.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 5, (frame_w, frame_h))

for ep in range(episodes):
    env.reset()
    last_obs = {}
    total_rewards = {a: 0 for a in agent_list}

    full_obs = []
    full_act = []
    messages = {a: 0.0 for a in agent_list}
    agent_list_loop = list(agent_list)

    for agent_idx, agent in enumerate(env.agent_iter()):
        obs, reward, term, trunc, _ = env.last()
        done = term or trunc

        obs = modify_obs(agent, obs)
        if agent not in last_obs:
            last_obs[agent] = obs

        # Receive message from previous agent (circular)
        prev_agent = agent_list_loop[(agent_list_loop.index(agent) - 1) % n_agents]
        message = messages[prev_agent]

        if done:
            action = None
        elif agent == "agent_1" and random.random() < 0.1:
            action = 0
        else:
            action = agents[agent].act(obs, epsilon, message=message, fingerprint=epsilon)

        env.step(action)

        # For centralized critic, keep full_obs as before
        full_obs.append(np.append(obs, message))
        full_act.append(action if action is not None else 0)

        if not done:
            # Store only the individual agent's state (obs), not full_obs
            agents[agent].remember(last_obs[agent], action, reward, obs, done, message, epsilon)
            last_obs[agent] = obs
            total_rewards[agent] += reward
            # Send message to next agent (for demonstration, use reward as message)
            messages[agent] = reward

        # frame = env.render()
        # if frame is not None:
            # bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # video_writer.write(bgr_frame)

    # === IQL Training ===
    for agent in agents.values():
        agent.learn()

    # === CTDE Centralized Critic Update ===
    if len(full_obs) == n_agents:
        # Reshape to (1, total_obs_with_messages)
        obs_cat = torch.FloatTensor(np.array(full_obs).flatten()).unsqueeze(0)
        act_cat = nn.functional.one_hot(torch.LongTensor(full_act), n_actions).view(1, -1).float()
        q_val = critic(obs_cat, act_cat)
        with torch.no_grad():
            next_obs_cat = obs_cat
            next_act_cat = act_cat
            target_q = critic_target(next_obs_cat, next_act_cat)
        loss = crit_criterion(q_val, target_q)
        critic_opt.zero_grad()
        loss.backward()
        critic_opt.step()

        if (ep + 1) % 10 == 0:
            critic_target.load_state_dict(critic.state_dict())

    # === Logging ===
    avg_reward = np.mean(list(total_rewards.values()))
    rewards_log.append(avg_reward)
    # print(f"{rewards_log}")  # Your commented-out debug line
    for agent in agent_list:
        agent_rewards_log[agent].append(total_rewards[agent])
    if (ep + 1) % 50 == 0:
        agent_rewards_str = ', '.join([f'{agent}: {total_rewards[agent]:.2f}' for agent in agent_list])
        print(f"Episode {ep+1}, AvgR {avg_reward:.2f}, Eps {epsilon:.3f}, Agent Rewards: {agent_rewards_str}", flush=True)
        print(f"Agent Rewards Log (Episode {ep+1}): {', '.join([f'{agent}: {agent_rewards_log[agent][-1]:.2f}' for agent in agent_list])}", flush=True)
        print(f"Total Rewards Check: {total_rewards}", flush=True)
        print(f"Agent List: {agent_list}", flush=True)
        print(f"Epsilon Check: {epsilon:.3f} (decay={decay}, min_eps={min_eps})", flush=True)
    epsilon = max(min_eps, epsilon * decay)

# video_writer.release()

# === Save and show reward plot ===
plt.figure(figsize=(10, 5))
plt.plot(rewards_log)
plt.title("Reward Trend")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid()
plt.savefig("plots/average_rewards.png")
plt.show()

# === Plot individual agent rewards with moving average trend ===
plt.figure(figsize=(10, 5))
min_reward = min([min(agent_rewards_log[agent]) for agent in agent_list if agent_rewards_log[agent]] or [0])
max_reward = max([max(agent_rewards_log[agent]) for agent in agent_list if agent_rewards_log[agent]] or [0])

# Plot individual agent rewards
for agent in agent_list:
    if agent_rewards_log[agent]:
        plt.plot(agent_rewards_log[agent], label=f"{agent} Reward")
        print(f"Plotting {agent}: {agent_rewards_log[agent][:5]}... (first 5 episodes)", flush=True)
    else:
        print(f"Warning: No reward data for {agent}", flush=True)

# Compute and plot moving average trend
window_size = 50
all_rewards = [reward for agent in agent_list for reward in agent_rewards_log[agent]]
moving_averages = []
for i in range(len(all_rewards) - window_size + 1):
    window = all_rewards[i:i + window_size]
    moving_averages.append(np.mean(window))
# Pad the beginning with the first value to match the plot length
moving_averages = [all_rewards[0]] * (window_size - 1) + moving_averages
plt.plot(moving_averages, label="General Trend", color="red", linewidth=2)

plt.title("Individual Agent Reward Trends with General Trend")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.ylim(min_reward - 10, max_reward + 10)
plt.grid()
plt.legend()
plt.savefig("plots/individual_agent_rewards.png")
plt.show()
