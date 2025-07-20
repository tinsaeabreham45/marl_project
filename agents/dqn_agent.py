# agents/dqn_agent.py

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, obs_size, action_size, lr=1e-3, gamma=0.99):
        self.q_net = QNetwork(obs_size, action_size)
        self.target_net = QNetwork(obs_size, action_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.batch_size = 64
        self.action_size = action_size

    def act(self, obs, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Fix: convert to numpy first before torch.FloatTensor
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
        target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
