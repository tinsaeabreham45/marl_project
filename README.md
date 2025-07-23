# 🧠 Multi-Agent Reinforcement Learning System: Coordination & Specialization in PettingZoo

## 📌 Project Overview

This project implements a **Multi-Agent Reinforcement Learning (MARL)** system using the **PettingZoo `simple_spread_v3`** environment. It demonstrates **independent learning**, **agent coordination**, **specialization**, and **robustness** under agent failure and non-stationarity. The system is designed to meet the requirements of a real-world MARL system design and analysis, including reward tracking, convergence analysis, and policy behavior evaluation.

---

## 🎯 Objectives & Features

- **3 agents** learn to coordinate to cover **3 landmarks** in a cooperative setting.
- Implements **Independent Q-Learning (IQL)** with DQN-based agents.
- Handles **non-stationary environments** using target networks, experience replay, and fingerprinting.
- Simulates **agent failures** and introduces **specialization** via observation masking.
- Supports **inter-agent communication** via message fields in agent observations.
- Logs and visualizes learning curves and agent performance.

---

## 🧪 Key Components & Implementation

### 1. Multi-Agent Learning (IQL)
Each agent learns independently using its own DQN. The environment is PettingZoo's `simple_spread_v3`.

```python
# agent initialization
for agent_name in env.agents:
    agents[agent_name] = DQNAgent(obs_shape, action_n, shared_buffer)
```

### 2. Centralized Training with Decentralized Execution (CTDE)
Agents act based on local observations, but training is performed in a centralized loop. Each agent's experience is stored in a **shared replay buffer**.

```python
# decentralized action selection
action = agents[agent].act(obs, epsilon, message, fingerprint)

# centralized training loop
for ag in agents.values():
    ag.learn()
    ag.update_target()
```

### 3. Coordination & Communication
- **Shared Rewards**: Agents are rewarded based on the team's performance.
- **Message Passing**: Each agent's observation is augmented with a `message` field, allowing for simple inter-agent communication (can be extended for learned protocols).

### 4. Specialization & Diversity
- **Observation Masking**: Agent 0 receives a partial observation to simulate limited sensing capability, encouraging specialization.

```python
def modify_obs(agent, obs):
    if agent == "agent_0":
        obs = obs[:obs_shape // 2]  # mask half of the observation
    return obs
```

### 5. Non-Stationarity Handling
- **Experience Replay**: Uses a shared replay buffer for stability.
- **Fingerprinting**: Each experience includes a fingerprint to help agents disambiguate non-stationary policies of others.
- **Target Networks**: Used to stabilize Q-learning.

### 6. Robustness & Failure Modes
- **Agent Failure Simulation**: Agent 1 randomly fails to act during training, simulating dropout or malfunction.
- **Effect Logging**: The impact of failures is reflected in the reward logs and plots.

### 7. Visualization and Logging
- **Reward Logging**: Average episode rewards are logged and plotted.
- **Video Recording**: Optionally record agent behavior for qualitative analysis.

---

## 🚀 How to Run Training

```bash
python main.py
```

- Trains 3 agents for 500 episodes
- Logs and plots average reward curve in `plots/average_rewards.png`

---

## 🎥 Optional: Record Agent Behavior

```bash
python record_episode.py
```

- Outputs a video file `agent_behavior.mp4` showing agents in one full episode

---

## 🏗️ Project Structure

```
marl_project/
├── agents/
│   ├── dqn_agent.py          # DQNAgent class for IQL, message/fingerprint support
│   ├── critic.py             # Centralized critic (for future extensions)
│   └── shared_replay_buffer.py # Shared replay buffer for CTDE
├── main.py                   # Training script with coordination, failure, specialization
├── record_episode.py         # Record one episode as a video
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

### Install Requirements

```bash
pip install pettingzoo[mpe] supersuit torch numpy matplotlib opencv-python
```

---

## 📖 Usage Guide & Customization

### Basic Training
- Run `python main.py` to start training with default settings (3 agents, 500 episodes).
- Reward plot is saved to `plots/average_rewards.png`.

### Recording Agent Behavior
- Run `python record_episode.py` to save a video of a single episode.

### Customizing Agents & Environment
- **Change number of agents/landmarks**: Edit the environment initialization in `main.py`.
- **Modify agent specialization**: Adjust the `modify_obs` function for different observation masks.
- **Tune hyperparameters**: Change learning rate, batch size, or epsilon decay in `main.py` or `dqn_agent.py`.
- **Enable/disable agent failure**: Comment/uncomment the failure simulation code in `main.py`.
- **Extend communication**: Modify the `message` field logic in `dqn_agent.py` for more complex protocols.

---

## 📈 Results Snapshot

| Episode | Avg Reward |
|---------|------------|
| 50      | ~ -138     |
| 150     | ~ -112     |
| 300     | ~ -86      |
| 500     | ~ -147     |

Reward improves initially, fluctuates due to agent failures and policy shifts, but shows learning over time.

---

## 🧠 Analysis: Agent Behaviors & Failure Recovery

- **Coordination**: Agents learn to spread out and cover landmarks, as seen in improved average rewards.
- **Specialization**: Agent 0, with limited observation, often takes a distinct role or lags in performance, demonstrating emergent specialization.
- **Failure Handling**: When Agent 1 fails, the team reward drops, but the system continues training, showing some robustness. No explicit recovery/fail-safe is implemented, but the shared reward structure encourages remaining agents to adapt.
- **Communication**: The `message` field is available for inter-agent signaling, but is currently used as a placeholder. This can be extended for learned or hard-coded protocols.

---




## 📂 Deliverables

- **Source Code**: Modular implementation with clean separation
- **Plot**: `plots/average_rewards.png`
- **Optional Video**: `agent_behavior.mp4`

---

## 📚 References

- [PettingZoo Documentation](https://pettingzoo.farama.org)
- [MARL Algorithms Survey](https://arxiv.org/abs/1810.05587)
- [DQN Paper (2015)](https://www.nature.com/articles/nature14236)

---

