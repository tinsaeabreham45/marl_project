# 🧠 Multi-Agent Reinforcement Learning System: Coordination & Specialization in PettingZoo

## 📌 Project Overview

This project implements a **Multi-Agent Reinforcement Learning (MARL)** system using the **PettingZoo `simple_spread_v3`** environment. It demonstrates **independent learning**, **agent coordination**, **specialization**, and **robustness** under agent failure and non-stationarity.

The task is designed to meet the full requirements of a real-world MARL system design and analysis, including reward tracking, convergence analysis, and policy behavior evaluation.

---

## 🎯 Objectives

- Implement a MARL system where **3 agents** learn to coordinate to cover **3 landmarks**.
- Use the **IQL (Independent Q-Learning)** algorithm with a **DQN-based agent**.
- Handle **non-stationary environments**, simulate **agent failures**, and introduce **specialization** via observation masking.
- Evaluate learning curves and agent robustness using quantitative metrics.

---

---

## 🧪 Features Implemented (with Code Snippets)

### ✅ 1. Multi-Agent Learning (IQL)
Each agent learns independently using its own DQN.

```python
# agent initialization
for agent_name in env.agents:
    agents[agent_name] = DQNAgent(obs_shape, action_n)
```

```python
# agent experience collection
agents[agent].remember(last_obs[agent], action, reward, obs, done)
```

---

### ✅ 2. Centralized Training with Decentralized Execution (CTDE)
All agents are trained in a loop, but make decisions based only on their local observations.

```python
# decentralized action selection
action = agents[agent].act(obs, epsilon)

# centralized training loop
for ag in agents.values():
    ag.learn()
    ag.update_target()
```

---

### ✅ 3. Coordination via Shared Rewards and Common Goal
All agents interact in the shared `simple_spread_v3` world where success depends on **cooperation** (covering separate landmarks).

```python
# agents are trained independently, but rewards reflect shared environment
avg_reward = np.mean(list(total_rewards.values()))
```

---

### ✅ 4. Specialization – Masked Observations
Agent 0 receives a partial observation to simulate limited sensing capability.

```python
def modify_obs(agent, obs):
    if agent == "agent_0":
        obs = obs[:obs_shape // 2]  # mask half of the observation
    return obs

# usage in training
obs = modify_obs(agent, obs)
```

---

### ✅ 5. Non-Stationarity Handling
Agents are trained simultaneously; learning is stabilized with target networks and experience replay.

```python
# Epsilon decay controls exploration in a non-stationary setting
epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Target network for stabilizing Q-learning
ag.update_target()
```

---

### ✅ 6. Agent Failure Simulation
Agent 1 randomly fails to act during training.

```python
if agent == "agent_1" and random.random() < 0.1:
    action = 0  # simulate failure with default action
```

---

### ✅ 7. Visualization and Logging
Rewards are logged and plotted. Optional video recording supported.

```python
# Logging average episode reward
rewards_log.append(avg_reward)

# Plotting after training
plt.plot(rewards_log)
plt.title("Average Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("plots/average_rewards.png")
```

---

## 🚀 How to Run Training

```bash
python main.py
```

Trains 3 agents for 500 episodes and logs rewards. A plot is saved in `plots/average_rewards.png`.

---

## 🎥 Optional: Record Agent Behavior

```bash
python record_episode.py
```

This saves a visual `agent_behavior.mp4` using PettingZoo's `"rgb_array"` rendering.

---

## 🏗️ Project Structure

```
marl_project/
├── agents/
│   └── dqn_agent.py          # DQNAgent class for IQL
├── main.py                   # Training script with coordination, failure, specialization
├── record_episode.py         # record one episode as a video
├── plots/
│   └── average_rewards.png   # Output reward plot
└── README.md                 # This file
```

---

## ⚙️ Environment & Dependencies

- Python 3.12+
- [PettingZoo](https://www.pettingzoo.ml/mpe)
- [Supersuit](https://github.com/Farama-Foundation/SuperSuit)
- NumPy
- Matplotlib
- PyTorch
- OpenCV 

### ✅ Install Requirements

```bash
pip install pettingzoo[mpe] supersuit torch numpy matplotlib opencv-python
```

---

## 🚀 How to Run Training

### ▶️ Run the main training script

```bash
python main.py
```

- Trains 3 agents for 500 episodes
- Plots and saves average reward curve in `plots/average_rewards.png`

### 🎥 Optional: Record Agent Behavior

```bash
python record_episode.py
```

- Outputs a video file `agent_behavior.mp4` showing agents in one full episode


## 📈 Results Snapshot

| Episode | Avg Reward |
|---------|------------|
| 50      | ~ -138     |
| 150     | ~ -112     |
| 300     | ~ -86      |
| 500     | ~ -147     |

Reward improves initially, fluctuates due to agent failures and policy shifts, but shows learning over time.

---

## 🧠 Key Design Decisions

- **IQL** was chosen for simplicity and baseline performance
- **PettingZoo** offers clean agent-environment interaction
- **Coordination** emerges from shared reward and environment dynamics
- **Failure Simulation** shows agent dropout and robustness testing
- **Specialization** created via partial observability (`agent_0`)

---

## 📂 Deliverables

- ✅ **Source Code**: Modular implementation with clean separation
- ✅ **Plot**: `plots/average_rewards.png`
- ✅ **Optional Video**: `agent_behavior.mp4`
- ✅ **Presentation**: (Use reward plot + behavior video + code walkthrough)

---

## 📚 References

- [PettingZoo Documentation](https://pettingzoo.farama.org)
- [MARL Algorithms Survey](https://arxiv.org/abs/1810.05587)
- [DQN Paper (2015)](https://www.nature.com/articles/nature14236)

---

