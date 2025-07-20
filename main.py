# main.py

from pettingzoo.mpe import simple_spread_v3
import numpy as np
from agents.dqn_agent import DQNAgent
import torch
import matplotlib.pyplot as plt
import random

import os
os.makedirs("plots", exist_ok=True)
# === ENV SETUP ===
env = simple_spread_v3.env(N=3, max_cycles=100, render_mode="human")
env.reset(seed=42)

# === SPECIALIZATION: define obs size per agent ===
original_obs_size = env.observation_space(env.agents[0]).shape[0]
agent_obs_sizes = {
    "agent_0": original_obs_size // 2,  # masked obs for specialization
    "agent_1": original_obs_size,
    "agent_2": original_obs_size
}
action_n = env.action_space(env.agents[0]).n

# === AGENT INITIALIZATION ===
agents = {}
for agent_name in env.agents:
    agents[agent_name] = DQNAgent(agent_obs_sizes[agent_name], action_n)

# === SPECIALIZATION: limit observation for agent_0 ===
def modify_obs(agent, obs):
    if agent == "agent_0":
        obs = obs[:agent_obs_sizes[agent]]
    return obs

# === TRAINING SETUP ===
episodes = 500
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.05
rewards_log = []

for ep in range(episodes):
    env.reset()
    total_rewards = {agent: 0 for agent in env.agents}
    last_obs = {}

    for agent in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        done = term or trunc

        obs = modify_obs(agent, obs)
        if agent not in last_obs:
            last_obs[agent] = obs

        # Simulate failure
        if done:
            action = None
        elif agent == "agent_1" and random.random() < 0.1:
            action = 0  # safe dummy action
        else:
            action = agents[agent].act(obs, epsilon)

        env.step(action)

        if action is not None:
            agents[agent].remember(last_obs[agent], action, reward, obs, done)
            last_obs[agent] = obs
            total_rewards[agent] += reward

    # Train agents
    for ag in agents.values():
        ag.learn()
        ag.update_target()

    # Log
    avg_reward = np.mean(list(total_rewards.values()))
    rewards_log.append(avg_reward)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}, Avg reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

# Plot
plt.plot(rewards_log)
plt.title("Average Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.savefig("plots/average_rewards.png")
plt.show()
